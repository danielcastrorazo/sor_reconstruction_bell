import dataclasses

import numpy as np
from loguru import logger

from controllers.data_controller import DataController
from controllers.methods_controller import MethodController, Stage
from geometric.kwan import (
    axis_rectification_to_yz_plane,
    rectification_using_x_angle,
    search_angle_rotation_x,
)
from shared.optimization import (
    filter_data_by_percentages,
    filter_function_unique,
    filter_outliers,
    generate_and_evaluate_spline,
    normalize_range,
    optimize_axis_of_symmetry,
    sort_by_distance_to_axis,
)
from shared.projective_utils import (
    apply_transformation,
    check_contour_orientation,
    reflect_points_on_line,
    two_points_to_line,
)


class KwanController(MethodController):
    class ContourFiltering(Stage):
        def _process(self, stage_data):
            return {'output': self.controller.contour[stage_data]}
    
    class PreProcessingAndAxisRectification(Stage):
        def _process(self, stage_data):
            data = stage_data['output']
            logger.info(f'Initial number of points. {len(data)}')
            
            data = data[sort_by_distance_to_axis(data, self.controller.axis)]
            
            orientation = check_contour_orientation(data)
            logger.trace(f"Contour orientation: {'CCW' if orientation else 'CW'}")
            if not orientation:
                data = reflect_points_on_line(data, self.controller.axis)
                logger.info('Applied a reflection over the initial points.')
            
            rot_a, rot_b = axis_rectification_to_yz_plane(self.controller.camera_matrix, self.controller.axis)
            transformation = rot_a @ rot_b @ np.linalg.inv(self.controller.camera_matrix)
            data = apply_transformation(transformation, data)

            data[:, 0] = np.abs(data[:, 0])
            data = data[sort_by_distance_to_axis(data, [1, 0, 0])]

            fixed_idx = set([0, len(data) - 1])
            ellipses_coefficients = []
            for i, ellipse in enumerate(self.controller.ellipses):
                ei = dataclasses.replace(ellipse)
                ei.apply_transformation(transformation)
                a, _, c, _, e, f = ei.get_pol_to_cart()
                ellipses_coefficients.append((a, c, e, f))
                if i == 0:
                    distances = [ei.aprox_distance_to_point(p) for p in data]
                    fixed_idx.add(np.argmin(distances))

            logger.info(f'Looking for rotation angle using {len(self.controller.ellipses)} ellipses.')
            result = search_angle_rotation_x(ellipses_coefficients)
            rotation_angle_x = result.x[0]
            logger.trace(f'Angle opt result. {result.message}')

            fixed_points = np.array([data[i] for i in fixed_idx])

            return {**stage_data, 'output': data, 'rotation_angle_x': rotation_angle_x,
                    'fixed_idx': fixed_idx, 'fixed_points': fixed_points}
        
    class SplineFittingAndNormal(Stage):
        def _process(self, stage_data):
            data = stage_data['output']

            parametrization = self.controller.options['parametrization']
            s = self.controller.options['smoothing_factor'] * len(data)
            t = np.linspace(0.0, 1.0, self.controller.options['sample_size'])

            fixed_idx = stage_data['fixed_idx']
            weights = np.ones(len(data))
            weights[[*fixed_idx]] = 10 ** 6

            data, tangent_vectors = generate_and_evaluate_spline(data, s=s, t=t, parametrization=parametrization, weights=weights)
            data = np.c_[*data, np.ones_like(t)]
            tangent_vectors = np.c_[*tangent_vectors, np.zeros_like(t)]
            tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]

            logger.info(f'Spline fit to contour and sampled. Number of sample points: {len(data)}')

            normals = np.cross(data, tangent_vectors)
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
                
            if not self.controller.options['disable_filter_overall']:
                n_x, n_y = normals[:, :2].T
                mask_values = np.abs(np.divide(n_y, n_x, where=n_x != 0))
                data, normals = filter_outliers(data, normals, mask_values=mask_values, m=self.controller.options['filter_outliers'], msg='Outliers in normals.')

            return {**stage_data, 'output': data, 'normals': normals}
 
    class DepthRecoveryAndNormalization(Stage):
        def _process(self, stage_data):
            data, normals = stage_data['output'], stage_data['normals']
            rotation_angle_x = stage_data['rotation_angle_x']

            x, y, depth = rectification_using_x_angle(data, normals, [rotation_angle_x], dz=1.0)[0]
            if not self.controller.options['disable_filter_overall']:
                x, y, depth = filter_outliers(x, y, depth, mask_values=depth, m=self.controller.options['filter_outliers'], msg='Outliers in depth recovery.')
            
            if self.controller.options['filter_depth'][0] > 0  or self.controller.options['filter_depth'][1] < 1:
                x, y, depth = filter_outliers(x, y, depth, mask=filter_data_by_percentages(depth, *self.controller.options['filter_depth']), msg='Depth.', user=True)
            
            normalized_data = normalize_range(np.c_[x, y], xmin=0.0)
            if normalized_data[0, 1] < normalized_data[-1, 1]:
                normalized_data[:, 1] = 1.0 - normalized_data[:, 1]
            
            verify, sorted_idx = filter_function_unique(normalized_data, 'Enforced bijective mapping.')
            if not verify:
                normalized_data, depth = normalized_data[sorted_idx], depth[sorted_idx]
            
            return {'output': normalized_data, 'depth':depth}
  
    def __init__(self, data_controller: DataController, identifier: str):
        stages = [self.ContourFiltering(self),
                  self.PreProcessingAndAxisRectification(self),
                  self.SplineFittingAndNormal(self),
                  self.DepthRecoveryAndNormalization(self),
                  self.CurveGeneration(self),
                  self.ExportData(self)]
        super().__init__(stages, data_controller, identifier)

    def initialize(self):
        super().initialize()

        if len(self.ellipses) == 2:
            axis = two_points_to_line(self.ellipses[0].center, self.ellipses[1].center)
        else:
            points = [np.array([e.center[0], e.center[1], 1.0]) for e in self.ellipses]
            axis = np.linalg.svd(np.array(points))[2][-1, :]

        self.axis = axis / np.linalg.norm(axis[:2])
        self.axis = optimize_axis_of_symmetry(self.contour, self.axis, self.camera_matrix)
