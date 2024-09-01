import numpy as np
from loguru import logger

from controllers.data_controller import DataController
from controllers.methods_controller import MethodController, Stage
from geometric.colombo import (
    calculate_intersection_of_ellipses,
    create_rectification_matrix,
    generate_entities,
    generate_w_homologies,
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
    get_angle_of_line,
    intersection_between_conic_and_line,
    reflect_points_on_line,
    rotation_matrix,
    to_homogeneous,
)


class ColomboController(MethodController):
    class EntitySelection(Stage):
        def _process(self, combobox_selection_idx):
            axis, _, vanishing_line, ic, jc = self.controller.entities_groups[combobox_selection_idx]

            self.vanishing_line = vanishing_line
            self.controller.axis = axis

            self.controller.axis = optimize_axis_of_symmetry(self.controller.contour, self.controller.axis, self.controller.camera_matrix)
            
            return {'axis': self.controller.axis, 'vanishing_line': vanishing_line, 'ic':ic, 'jc':jc}

    class ContourFiltering(Stage):
        def _process(self, stage_data):
            data = self.controller.contour[self.controller.options['filter_mask']]            
            axis = stage_data['axis']

            data = data[sort_by_distance_to_axis(data, axis)]

            orientation = check_contour_orientation(data)
            logger.trace(f"Contour orientation: {'CCW' if orientation else 'CW'}")
            if not orientation:
                data = reflect_points_on_line(data, axis)
                logger.info('Applied a reflection over the initial points.')
            
            fixed_idx = set([0, len(data) - 1])
            for ellipse in self.controller.ellipses[1:]:
                distances = [ellipse.aprox_distance_to_point(p) for p in data]
                fixed_idx.add(np.argmin(distances))
            fixed_points = np.array([data[i] for i in fixed_idx])

            return {**stage_data, 'output': data, 'fixed_idx': fixed_idx, 'fixed_points': fixed_points, 'top_point': data[0]}

    class SplineFittingAndTangent(Stage):
        def _process(self, stage_data):
            data, axis = stage_data['output'], stage_data['axis']
            logger.info(f'Initial number of points. {len(data)}')
            
            fixed_idx = stage_data['fixed_idx']

            parametrization = self.controller.options['parametrization']
            s = self.controller.options['smoothing_factor'] * len(data)
            t = np.linspace(0.0, 1.0, self.controller.options['sample_size'])
            logger.info(f'Smoothing Factor {s}.')

            weights = np.ones(len(data))
            weights[[*fixed_idx]] = 10 ** 6
            
            data, tangent_vectors = generate_and_evaluate_spline(data, s=s, t=t, parametrization=parametrization, weights=weights)
            data = np.c_[*data, np.ones_like(t)]
            logger.info(f'Spline fit to contour and sampled. Number of sample points: {len(data)}')

            tangent_vectors = np.c_[*tangent_vectors, np.zeros_like(t)]
            tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]

            tangent_lines = np.cross(data, tangent_vectors)
            tangent_lines /= np.linalg.norm(tangent_lines[:, :2], axis=1)[:, np.newaxis]

            theta = get_angle_of_line(axis)
            if not self.controller.options['disable_filter_overall'] and np.abs(theta - np.pi / 2) > 1e-10:
                diff = theta - np.pi / 2 if theta > 0 else theta + np.pi / 2
                rotation_matrix_z = rotation_matrix('z', -diff)

                normals = np.c_[-tangent_vectors[:, 1], tangent_vectors[:, 0], np.zeros_like(tangent_vectors[:, 0])]
                normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

                normals_rectify = apply_transformation(rotation_matrix_z, normals, divide=False)
                normals_rectify /= np.linalg.norm(normals_rectify, axis=1)[:, np.newaxis]
                n_x, n_y = normals_rectify[:, 0], normals_rectify[:, 1]

                mask_values = np.abs(np.divide(n_y, n_x, where=n_x != 0))
                data, tangent_lines = filter_outliers(data, tangent_lines, mask_values=mask_values,
                                                      m=self.controller.options['filter_outliers'], msg='Outliers in normals.')

            return {**stage_data, 'output': data, 'tangent_lines': tangent_lines}

    class HomologyAndCrossRatio(Stage):
        def _process(self, stage_data):
            data, tangent_lines = stage_data['output'], stage_data['tangent_lines']
            axis, vanishing_line = stage_data['axis'], stage_data['vanishing_line']

            w_homologies, w_chars, _, _ = generate_w_homologies(data, tangent_lines, axis, vanishing_line, self.controller.ellipse_base_matrix)

            if not self.controller.options['disable_filter_overall']:
                data, w_homologies, w_chars = filter_outliers(data, w_homologies, w_chars, mask_values=np.abs(w_chars), m=self.controller.options['filter_outliers'],
                                                            msg='Cross-Ratio.')

            if self.controller.options['filter_cross_ratio'][0] > 0  or self.controller.options['filter_cross_ratio'][1] < 1:
                data, w_homologies, w_chars = filter_outliers(data, w_homologies, w_chars, mask=filter_data_by_percentages(w_chars, *self.controller.options['filter_cross_ratio']),
                                                              msg='Cross-Ratio.', user=True)

            return {**stage_data, 'output': data, 'w_homologies': w_homologies, 'cross-ratio': w_chars}
    
    class RectificationAndNormalization(Stage):
        def _process(self, stage_data):
            axis, vanishing_line = stage_data['axis'], stage_data['vanishing_line']
            w_homologies, cross_ratio = stage_data['w_homologies'], stage_data['cross-ratio']

            logger.trace('Searching for a good ellipse point that will be part of the unrectified meridian.')
            if not self.controller.options['user_ellipse']:
                if any(np.abs(np.linalg.eigvals(self.controller.ellipse_base_matrix)) < 1e-15):
                    phi = np.pi / 4
                else:
                    p0, q0 = intersection_between_conic_and_line(self.controller.ellipse_base_matrix, axis)
                    lpq = np.cross(p0, q0)
                    lpq /= np.linalg.norm(lpq[:2])
                    
                    lpq = [-lpq[1], lpq[0], -self.controller.ellipse_base.center[1]]
                    p, q = intersection_between_conic_and_line(self.controller.ellipse_base_matrix, lpq)

                    def find_t(x, y, params):
                        x0, y0 = params.center
                        ap, bp = params.axes
                        phi = params.angle
                        A = x - x0
                        B = y - y0
                        C = ap * np.cos(phi)
                        D = - bp * np.sin(phi)
                        E = ap * np.sin(phi)
                        F = bp * np.cos(phi)

                        denom_cos = C * F - E * D
                        denom_sin = F * C - D * E

                        cos_t = (A * F - B * D) / denom_cos
                        sin_t = (B * C - A * E) / denom_sin
                        return np.arctan2(sin_t, cos_t)
                    if np.sign(np.dot(axis, p)) == np.sign(np.dot(axis, stage_data['output'][0])):
                        phi = find_t(p[0], p[1], self.controller.ellipse_base)
                        phi = np.pi if np.isnan(phi) else phi
                    else:
                        phi = find_t(q[0], q[1], self.controller.ellipse_base)
                        phi = 0.0 if np.isnan(phi) else phi
            else:
                phi = self.controller.options['phi_ellipse']
                logger.info('User Input for the ellipse point.')
                phi = self.controller.options['phi_ellipse']
            ellipse_point_u_meridian = to_homogeneous(self.controller.ellipse_base.get_points(1, phi, phi))[0]
            logger.trace(f'Base point for the unrectified meridian : {ellipse_point_u_meridian[0]:.3f}, {ellipse_point_u_meridian[1]:.3f}.')

            unrectified_meridian = np.array([apply_transformation(w, ellipse_point_u_meridian) for w in w_homologies])

            rectification_matrix = create_rectification_matrix(ellipse_point_u_meridian, self.controller.ellipse_base_matrix, vanishing_line, self.controller.iac)

            rectified_meridian = apply_transformation(rectification_matrix, unrectified_meridian)
            rectified_axis = apply_transformation(np.linalg.inv(rectification_matrix).T, self.controller.axis, divide=False)
            rectified_axis /= np.linalg.norm(rectified_axis[:2])
            
            rotation_matrix_line = rotation_matrix('z', -np.arctan2(rectified_axis[1], rectified_axis[0]) + np.pi / 2.0)

            rectified_meridian = apply_transformation(rotation_matrix_line, rectified_meridian)
            rectified_axis = apply_transformation(rotation_matrix_line, rectified_axis, divide=False)
            
            rectified_meridian[:, 1] = np.abs(rectified_meridian[:, 1] + rectified_axis[2])
            rectified_axis[2] = 0
            
            normalized_data = normalize_range(np.c_[rectified_meridian[:, 1], rectified_meridian[:, 0]], xmin=0.0)

            if normalized_data[0, 1] < normalized_data[-1, 1]:
                normalized_data[:, 1] = 1.0 - normalized_data[:, 1]

            verify, sorted_idx = filter_function_unique(normalized_data, 'Enforced bijective mapping.')
            if not verify:
                normalized_data = normalized_data[sorted_idx]
                w_homologies = w_homologies[sorted_idx]
                unrectified_meridian = unrectified_meridian[sorted_idx]
                cross_ratio = cross_ratio[sorted_idx]
            
            return {**stage_data, 'output': normalized_data, 'w_homologies': w_homologies, 'unrectified_meridian': unrectified_meridian, 'cross-ratio':cross_ratio}
 
    def __init__(self, data_controller: DataController, identifier):
        stages = [self.EntitySelection(self, pause=True),
                  self.ContourFiltering(self),
                  self.SplineFittingAndTangent(self),
                  self.HomologyAndCrossRatio(self),
                  self.RectificationAndNormalization(self),
                  self.CurveGeneration(self),
                  self.ExportData(self)]
        super().__init__(stages, data_controller, identifier)

    def initialize(self):
        super().initialize()
        
        self.iac = np.linalg.inv(self.camera_matrix @ self.camera_matrix.T)
        self.iac /= self.iac[2, 2]

        logger.info('Calculating the intersection of the first and last ellipses.')
        # TODO if 3 >= ellipses there *should* be one solution MLE
        roots = calculate_intersection_of_ellipses(
            self.ellipse_base.get_pol_to_cart(),
            self.ellipses[-1].get_pol_to_cart()
        )
        
        logger.info('Intersection points found:')
        for root in roots:
            logger.info(f'\t{root[0]:.5f}\t{root[1]:.5f}')

        self.entities_groups = generate_entities(roots)
        self.entities_groups = self.entities_groups[:1] + self.entities_groups[-1:]