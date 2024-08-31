import copy
from enum import Enum

from loguru import logger

from controllers.data_controller import (
    CDCState,
    DataController,
    Observable,
)
from model_generator import create_model, generate_model
from shared.projective_utils import calculate_intrinsics_camera_matrix, to_homogeneous


class StageState(Enum):
    FAILURE = 0
    SUCCESS = 1
    READY = 2


class Stage:
    def __init__(self, controller, pause=False):
        self.controller = controller
        self.state = StageState.READY
        self.output = None
        self.pause = pause

    def process(self, input):
        def is_invalid(x):
            if x is None:
                return True
            if 'output' in x:
                if x['output'] is None or len(x['output']) == 0:
                    return True
            return False
        
        self.state = StageState.READY
        try:
            self.output = self._process(input)
            if is_invalid(self.output):
                self.state = StageState.FAILURE
            else:
                self.state = StageState.SUCCESS
        except Exception as e:
            self.state = StageState.FAILURE
            logger.error(f'An error has occurred in stage {self}: {e}.')

        return self.output
        
    def _process(self, input):
        pass


class MethodController(Observable):
    def __init__(self, stages: list[Stage], data_controller: DataController, identifier: str):
        super().__init__()
        self.data_controller = data_controller
        self.data_controller.attach(self)

        self.attach(self.data_controller)
        
        self.stages = stages
        self.identifier = f'geometric_{identifier}'
        self.defaults = ['geometric', f'{self.identifier}']

    def on_change_control(self, key, value):
        if self.data_controller.state != CDCState.DATA_LOADED:
            return
        if key != 'filter_mask' and value == self.options[key]:
            return
        
        logger.trace(f'Change : {key} : {value}')
        self.options[key] = value
        self.reprocess(self._component_stage_mapping[key], value)

    def reprocess(self, start_index, x):
        if start_index > 0:
            x = self.stages[start_index - 1].output

        if x is None:
            return
        logger.trace(f'Reprocessing from : {start_index}')
        for i in range(start_index, len(self.stages)):
            current = self.stages[i]
            x = current.process(x)
            if current.state == StageState.FAILURE:
                logger.error(f'Stage {i} failed.')
                return
            elif current.state == StageState.SUCCESS:
                current.state = StageState.READY
                if i == len(self.stages) - 1:
                    logger.trace(f'Stage {i} was the last one.')
                    return
                self.notify_observers(event="stage_update", stage_id=i, **current.output)

            if current.pause:
                logger.trace(f'Stage {i} is a pause.')
                return

    def update(self, event, *args, **kwargs):
        if self.data_controller.state == CDCState.DATA_LOADED:
            if event == "change_of_state":
                self.initialize()
        elif self.data_controller.state == CDCState.DEFAULT_SYSTEM_LOAD:
            self.default_options = {}
            def load_defaults(s):
                for k in s:
                    for key, value in self.data_controller[k].items():
                        if isinstance(value, list) and len(value) == 2:
                            # HACK
                            if key == 'filter_depth' or key == 'filter_cross_ratio':
                                self.default_options[key] = value
                            else:
                                self.default_options[key] = value[-1]
                        elif isinstance(value, list) and len(value) == 3: # parametrization
                            self.default_options[key] = value[0]
                        else:
                            self.default_options[key] = value
            load_defaults(self.defaults)
            self._component_stage_mapping = self.default_options['stage_mappings']

        self.notify_observers(event)

    def initialize(self):
        for stage in self.stages:
            stage.output = None

        self.options = copy.deepcopy(self.default_options)
        self.rho = self.data_controller['metadata'].scale

        self.height, self.width = self.data_controller['metadata'].image_size[:2]
        self.contour = to_homogeneous(self.data_controller['contour'])

        self.ellipses = self.data_controller['metadata'].ellipses

        self.ellipse_base = self.ellipses[0]
        self.ellipse_base_matrix = self.ellipse_base.get_matrix_form()

        self.camera_matrix = calculate_intrinsics_camera_matrix(self.data_controller['metadata'])
        self.axis = None

    class CurveGeneration(Stage):
        def _process(self, stage_data):
            data = stage_data['output']

            parametrization = self.controller.options['parametrization']
            outer_detail, inner_detail= self.controller.options['outer_curve_size'], self.controller.options['inner_curve_size']

            full_curve, full_tangents, outer_curve, inner_curve, outer_tangents, inner_tangents = generate_model(data, parametrization, outer_detail, inner_detail)

            return {**stage_data, 'output': (full_curve, full_tangents), 'outer_curve': (outer_curve, outer_tangents), 'inner_curve': (inner_curve, inner_tangents)}
        
    class ExportData(Stage):
        def _process(self, stage_data):
            create_obj = self.controller.options['create_obj']
            phi_n = self.controller.options['phi_n']

            if create_obj:
                full_curve, full_tangents = stage_data['output']
                mesh = create_model(full_curve, full_tangents, phi_n)
                self.controller.notify_observers(event="export_mesh", identifier=self.controller.identifier, mesh=mesh)

                event_data = {
                    'inner': stage_data['outer_curve'][0][:, :2],
                    'outer': stage_data['inner_curve'][0][:, :2],
                }

                if 'unrectified_meridian' in stage_data:
                    import numpy as np
                    x, y = stage_data['unrectified_meridian'][:, :2].T
                    x = [xi / self.controller.width for xi in x]
                    y = [(self.controller.height - yi) / self.controller.height for yi in y]
                    event_data['unrectified_meridian'] = np.c_[x, y]

                self.controller.notify_observers(event="write_for_tex", identifier=self.controller.identifier, **event_data)

            return {}