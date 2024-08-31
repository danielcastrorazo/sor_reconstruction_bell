import numpy as np
from matplotlib import pyplot as plt

from controllers.data_controller import CDCState
from shared.projective_utils import intersection_of_line_in_rectangle
from ui._frame import _Frame


class FrameKwan(_Frame):
    def __init__(self, parent, tab_controller, data_controller):
        super().__init__(parent, tab_controller, data_controller)

    def _set_controls_defaults(self):
        super()._set_controls_defaults()

        cfg_k = self.data_controller.get_config(self.controller.identifier)
        filter_depth = cfg_k['filter_depth']
        
        self._set_spline_controls(row=0)
        self._set_filter_controls('Depth Percentage', filter_depth, 'filter_depth', row=1)
        self._set_final_curve_controls(row=2)

        self.controls_frame.rowconfigure((0, 1, 2), weight=1)
        self.controls_frame.columnconfigure(0, weight=1)

    def _set_figure_defaults(self):
        super()._set_figure_defaults()
        self.axes[1, 0].set_xlim(-0.05, 1.05)
        self.axes[1, 0].set_ylim(-0.05, 1.05)

    def _reset_figure(self):
        super()._reset_figure()

    def _reset_controls(self):
        super()._reset_controls()

    def _change_of_state_update(self):
        intersection_points = intersection_of_line_in_rectangle(
            self.controller.axis, self.controller.width, self.controller.height)
        self.axis_line.set_data(zip(*intersection_points))

    def _stage0_update(self, filter_mask):
        self.fc[:, -1] = 0.0
        self.fc[filter_mask, -1] = 1
        self.collection0.set_facecolors(self.fc)

    def update(self, event, *args, **kwargs):
        super().update(event, *args, **kwargs)
        if self.data_controller.state != CDCState.DATA_LOADED:
            return
        if event == 'change_of_state':
            self._change_of_state_update()
        elif event == 'stage_update':
            stage_id = kwargs["stage_id"]
            if stage_id == 0:
                filter_mask = self.controller.options['filter_mask']
                self._stage0_update(filter_mask)
            elif stage_id == 2:
                x, y = kwargs['output'][:, :2].T
                
                x_range, y_range = max(x) - min(x), max(y) - min(y)
                bounding_box_size = max(x_range, y_range)
                midx, midy = (max(x) + min(x)) / 2, (max(y) + min(y)) / 2

                padding = bounding_box_size * 0.1
                self.axes[0, 1].set_xlim(0.0, midx + bounding_box_size / 2 + padding)
                self.axes[0, 1].set_ylim(midy + bounding_box_size / 2 + padding, midy - bounding_box_size / 2 - padding)

                self.line1.set_data(kwargs['output'][:, :2].T)
            elif stage_id == 3:
                x, y = kwargs['output'][:, :2].T
                self.line2.set_data(kwargs['output'][:, :2].T)
                self.c1.set_offsets(kwargs['output'][:, :2])
                self.c1.set_array(kwargs['depth'])
                self.c1.set_norm(plt.Normalize(vmin=np.min(kwargs['depth']), vmax=np.max(kwargs['depth'])))
                self.c1.set_cmap('viridis')
                self.color_bar.ax.yaxis.tick_left()
            elif stage_id == 4:
                if 'inner_curve' in kwargs and 'outer_curve' in kwargs:
                    outer, inner = kwargs['outer_curve'][0], kwargs['inner_curve'][0]
                    inner = np.append([outer[-1]], inner, axis=0)
                    
                    self.outer_curve_line.set_data(outer[:, :2].T)
                    self.inner_curve_line.set_data(inner[:, :2].T)
                else:
                    self.line3.set_data(kwargs['output'][0][:, :2].T)
        self.canvas.draw_idle()
