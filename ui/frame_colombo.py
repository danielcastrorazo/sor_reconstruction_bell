import tkinter as tk
from tkinter import ttk

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from controllers.data_controller import CDCState
from shared.ellipse_utils import get_ellipse_pts_from_matrix, is_ellipse
from shared.projective_utils import intersection_of_line_in_rectangle
from ui._frame import _Frame
from ui.ui_utils import create_dial_control


class FrameColombo(_Frame):
    def __init__(self, parent, tab_controller, data_controller):
        super().__init__(parent, tab_controller, data_controller)

    def _set_controls_defaults(self):
        super()._set_controls_defaults()

        cfg_c = self.data_controller.get_config(self.controller.identifier)

        filter_cross_ratio = cfg_c['filter_cross_ratio']
        phi_ellipse = cfg_c['phi_ellipse']
        user_ellipse = cfg_c['user_ellipse']

        def _set_combination_controls(row):
            self.combination_frame = ttk.LabelFrame(self.controls_frame, text='Combinations imaged circular points')
            self.combination_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
            self.combobox = ttk.Combobox(
                self.combination_frame,
                values=['Option 1', 'Option 2'],
                # values=['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6'],
                state='readonly',
            )
            self.combobox.bind(
                '<<ComboboxSelected>>',
                lambda _: self.controller.on_change_control(
                    'combination', self.combobox.current()
                ),
            )
            self.combobox.grid(row=0, column=0, padx=5, pady=5)
            self.combination_frame.rowconfigure(0, weight=1)
            self.combination_frame.columnconfigure(0, weight=1)

        _set_combination_controls(row=0)
        self._set_spline_controls(row=1)
        self._set_filter_controls('Cross-Ratio Percentage', filter_cross_ratio, 'filter_cross_ratio', row=2)

        def _set_ellipse_angle_controls(row):
            self.ellipse_angle_frame = ttk.LabelFrame(self.controls_frame, text='Imaged Meridian Initial Point')
            self.ellipse_angle_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5) 
            self.checkbox_ellipse_var = tk.BooleanVar(value=user_ellipse)
            self.checkbox_ellipse = ttk.Checkbutton(
                self.ellipse_angle_frame,
                variable=self.checkbox_ellipse_var,
                text='User Input',
                command=lambda: self.controller.on_change_control(
                    'user_ellipse', self.checkbox_ellipse_var.get()
                )
            )
            self.checkbox_ellipse.grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.dial_angle = create_dial_control(
                self.ellipse_angle_frame,
                'Angle',
                phi_ellipse,
                self.controller,
                'phi_ellipse',
                row=0, column=1
            )
            self.ellipse_angle_frame.rowconfigure(0, weight=1)
            self.ellipse_angle_frame.columnconfigure((0, 1), weight=1)

        _set_ellipse_angle_controls(row=3)

        self._set_final_curve_controls(row=4)
        
        self.controls_frame.rowconfigure((0, 1, 2, 3, 4), weight=1)
        self.controls_frame.columnconfigure(0, weight=1)

    def _set_figure_defaults(self):
        super()._set_figure_defaults()
        self.ellipse_lines2 = []
        self.vanishing_line = self.axes[0, 0].add_line(Line2D([], [], color=(0.0, 1.0, 0.0, 0.85)))

    def _reset_figure(self):
        super()._reset_figure()
        self.vanishing_line.set_data([], [])

        self.axes[0, 1].set_xlim(0, self.controller.width)
        self.axes[0, 1].set_ylim(self.controller.height, 0)

        self.axes[1, 0].set_xlim(0, self.controller.width)
        self.axes[1, 0].set_ylim(self.controller.height, 0)

        for line in self.ellipse_lines2:
            line.remove()
        self.ellipse_lines2.clear()

    def _reset_controls(self):
        super()._reset_controls()
        cfg_values = self.data_controller.state_data[
            CDCState.DEFAULT_SYSTEM_LOAD
        ]
        cfg_c = cfg_values[self.controller.identifier]

        phi_ellipse = cfg_c['phi_ellipse']
        user_ellipse = cfg_c['user_ellipse']

        self.combobox.set('')
        self.checkbox_ellipse_var.set(user_ellipse)
        self.dial_angle.set_angle(phi_ellipse)

    def _stage0_update(self, axis, vanishing_line):
        self.axis_line.set_data([], [])
        self.vanishing_line.set_data([], [])

        self.filter_1_line[0].set_data([], [])
        self.filter_1_line[1] = None

        intersection_points = intersection_of_line_in_rectangle(
            axis, self.controller.width, self.controller.height
        )
        if len(intersection_points):
            self.axis_line.set_data(zip(*intersection_points))
        else:
            logger.error("The chosen imaged axis can't be drawn.")

        intersection_points = intersection_of_line_in_rectangle(
            np.real(vanishing_line),
            self.controller.width,
            self.controller.height,
        )
        if len(intersection_points):
            self.vanishing_line.set_data(zip(*intersection_points))
        else:
            logger.info("The chosen vanishing line can't be drawn.")

        self.fc[:, -1] = 1
        self.collection0.set_facecolors(self.fc)

    def _stage1_update(self, filter_mask):
        self.fc[:, -1] = 0.0
        self.fc[filter_mask, -1] = 1
        self.collection0.set_facecolors(self.fc)

    def _stage3_update(self, w_homologies, ellipse_w_matrix):
        for line in self.ellipse_lines2:
            line.remove()
        self.ellipse_lines2.clear()

        resulting_ellipses_points = []
        for iw, w in enumerate(w_homologies):
            ellipse_matrix = np.linalg.inv(w).T @ ellipse_w_matrix @ np.linalg.inv(w)
            if not is_ellipse(ellipse_matrix):
                logger.warning(f'Could not transform the base ellipse using the transformation W_{iw}.')
                continue
            resulting_ellipses_points.append(get_ellipse_pts_from_matrix(ellipse_matrix))

        for data in resulting_ellipses_points:
            line, = self.axes[1, 0].plot(data[0], data[1], alpha=0.5, color='grey', zorder=3)
            self.ellipse_lines2.append(line)

        for i, e in enumerate(self.controller.ellipses):
            data = e.get_points(100, 0, 2 * np.pi)
            if i == 0:
                line, = self.axes[1, 0].plot(data[0], data[1], zorder=4, color='blue')
            else:
                line, = self.axes[1, 0].plot(data[0], data[1], zorder=2, color='black')
            self.ellipse_lines2.append(line)

    def _stage4_update(self, unrectified_meridian, cross_ratio):
        self.line2.set_data(unrectified_meridian[:, :2].T)

        self.c1.set_offsets(unrectified_meridian[:, :2])
        self.c1.set_array(cross_ratio)
        self.c1.set_norm(
            plt.Normalize(vmin=np.min(cross_ratio), vmax=np.max(cross_ratio))
        )
        self.c1.set_cmap('viridis')
        self.color_bar.mappable.set_clim(min(cross_ratio), max(cross_ratio))

    def update(self, event, *args, **kwargs):
        super().update(event, *args, **kwargs)
        if self.data_controller.state != CDCState.DATA_LOADED:
            return
        if event == 'change_of_state':
            self.axes[1, 0].imshow(self.data_controller['image'])
        if event == 'stage_update':
            stage_id = kwargs['stage_id']
            if stage_id == 0:
                axis, vanishing_line = kwargs['axis'], kwargs['vanishing_line']
                self._stage0_update(axis, vanishing_line)
            elif stage_id == 1:
                filter_mask = self.controller.options['filter_mask']
                self._stage1_update(filter_mask)
            elif stage_id == 2:
                self.line1.set_data(kwargs['output'][:, :2].T)
            elif stage_id == 3:
                w_homologies = kwargs['w_homologies']
                self._stage3_update(w_homologies, self.controller.ellipse_base_matrix)
            elif stage_id == 4:
                unrectified_meridian, cross_ratio = (
                    kwargs['unrectified_meridian'],
                    kwargs['cross-ratio'],
                )
                self._stage4_update(unrectified_meridian, cross_ratio)
            elif stage_id == 5:
                if 'inner_curve' in kwargs and 'outer_curve' in kwargs:
                    outer, inner = kwargs['outer_curve'][0], kwargs['inner_curve'][0]
                    inner = np.append([outer[-1]], inner, axis=0)

                    self.outer_curve_line.set_data(outer[:, :2].T)
                    self.inner_curve_line.set_data(inner[:, :2].T)
                else:
                    self.line3.set_data(kwargs['output'][:, :2].T)


        self.canvas.draw_idle()
