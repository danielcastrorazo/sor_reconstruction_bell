import tkinter as tk
from tkinter import ttk

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from controllers.data_controller import CDCState, DataController
from controllers.methods_controller import MethodController
from shared.projective_utils import (
    find_perpendicular_to_axis,
    intersection_of_line_in_rectangle,
)
from ui.ui_utils import create_slider_control, create_slider_control_multiple


class _Frame(ttk.Frame):
    def __init__(
        self,
        parent,
        tab_controller: MethodController,
        data_controller: DataController,
    ):
        super().__init__(parent)
        self.controller = tab_controller
        self.controller.attach(self)
        self.data_controller = data_controller

        self.shift_is_held = False

    def on_key_press(self, event):
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def _on_click_(self, event):
        if event.inaxes != self.axes[0, 0] or event.button not in [1, 2]:
            return
        if self.controller.axis is None:
            logger.warning('The imaged symmetry axis was not found.')
            return

        if event.button == 2:
            if self.shift_is_held:
                center = (event.xdata, event.ydata, 1.0)
            else:
                center = self.controller.ellipse_base.center

            normal = find_perpendicular_to_axis(
                center[0], center[1], self.controller.axis
            )
            intersection_points = intersection_of_line_in_rectangle(
                normal, self.controller.width, self.controller.height
            )
            self.filter_1_line[0].set_data(zip(*intersection_points))
            self.filter_1_line[1] = normal
            self.axes[0, 0].draw_artist(self.filter_1_line[0])

            self.canvas.draw_idle()
            logger.trace(
                'Using the center of the first ellipse (asummed to be the base) as way to filter points.'
            )
        elif event.button == 1:
            if self.filter_1_line[1] is None:
                logger.warning('The filter line was not found.')
                return

            p = np.array([event.xdata, event.ydata, 1.0])
            condition = (
                np.dot([self.controller.axis, self.filter_1_line[1]], p) > 0
            )

            axis_condition = (
                np.dot(self.controller.contour, self.controller.axis) > 0
            )
            normal_1_condition = (
                np.dot(self.controller.contour, self.filter_1_line[1]) > 0
            )

            filter_mask = np.logical_and(
                axis_condition == condition[0],
                normal_1_condition == condition[1],
            )
            self.controller.on_change_control('filter_mask', filter_mask)

    def _base_controls(self):
        self.method_frame = ttk.Frame(self)
        self.method_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        canvas_frame = ttk.Frame(self)
        canvas_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        controls_canvas = tk.Canvas(canvas_frame)
        controls_canvas.grid(row=0, column=0, sticky='nsew')

        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=controls_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns") 
        controls_canvas.configure(yscrollcommand=scrollbar.set)

        self.controls_frame = ttk.LabelFrame(controls_canvas, text='Controls')
        controls_canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")

        self.controls_frame.bind("<Configure>", 
                                lambda e: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")))

        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.columnconfigure(0, weight=8) 
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        controls_canvas.bind("<Configure>", lambda e: controls_canvas.itemconfig('all', width=e.width))


    def _set_spline_controls(self, row):
        cfg_geo = self.data_controller.get_config('geometric')

        sample_size = cfg_geo['sample_size']
        spline_parametrization = cfg_geo['parametrization']
        spline_smoothing_factor = cfg_geo['smoothing_factor']
        
        self.spline_details_frame = ttk.LabelFrame(
            self.controls_frame, text='Spline Details'
        )
        self.spline_details_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)

        self.slider_spline_n, parent_n = create_slider_control(
            self.spline_details_frame,
            'Spline Sample Size',
            sample_size,
            self.controller,
            'sample_size',
            ret_parent=True,
            row=0, column=0
        )
        self.combobox_selected_option = tk.StringVar(
            value=spline_parametrization[0]
        )
        self.combobox_spline_parametrization_change = ttk.Combobox(
            parent_n,
            textvariable=self.combobox_selected_option,
            values=spline_parametrization,
            state='readonly',
        )
        self.combobox_spline_parametrization_change.grid(row=0, column=1, sticky="e", padx=5, pady=4)
        self.combobox_spline_parametrization_change.bind(
            '<<ComboboxSelected>>',
            lambda e: self.controller.on_change_control(
                'parametrization', self.combobox_selected_option.get()
            ),
        )
        self.combobox_spline_parametrization_change.grid(row=0, column=1, sticky="w", padx=5)
        self.spline_details_frame.rowconfigure(0, weight=1)
        self.spline_details_frame.columnconfigure(0, weight=1)

        self.slider_spline_s, parent_s = create_slider_control(
            self.spline_details_frame,
            'Spline Smoothing Factor',
            spline_smoothing_factor,
            self.controller,
            'smoothing_factor',
            ret_parent=True,
            precision=8,
            row=1, column=0
        )

        def value_by_factor(factor):
            current_to_value = self.slider_spline_s.configure('to')[-1]
            current_value = self.slider_spline_s.get()

            smoothing_range = self.data_controller.get_config('smoothing_range')

            new_to_value = current_to_value * factor
            if new_to_value < 10 ** -smoothing_range or new_to_value > 10 ** smoothing_range:
                return
            new_curr_value = new_to_value * current_value / current_to_value
            logger.debug(new_curr_value)
            self.slider_spline_s.configure(to=new_to_value)
            self.slider_spline_s.set(new_curr_value)

        mul_smoothing_button = tk.Button(
            parent_s, text='x', command=lambda: value_by_factor(10)
        )
        div_smoothing_button = tk.Button(
            parent_s, text='/', command=lambda: value_by_factor(0.1)
        )
        mul_smoothing_button.grid(row=0, column=1, sticky="w", padx=5)
        div_smoothing_button.grid(row=0, column=2, sticky="w", padx=5)

        self.spline_details_frame.columnconfigure(0, weight=1) 
        self.spline_details_frame.columnconfigure(1, weight=0) 
        self.spline_details_frame.columnconfigure(2, weight=0) 
        self.spline_details_frame.rowconfigure(0, weight=1)

    def _set_filter_controls(self, label, values, key, row):
        cfg_geo = self.data_controller.get_config('geometric')

        filter_outliers = cfg_geo['filter_outliers']

        self.filter_frame = ttk.LabelFrame(self.controls_frame, text='Filters')
        self.filter_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)

        self.slider_outliers = create_slider_control(
            self.filter_frame,
            'Outliers',
            filter_outliers,
            self.controller,
            'filter_outliers',
            row=0, column=0, precision=2
        )
        self.slider_multiple_filter = create_slider_control_multiple(self.filter_frame, label, values, self.controller, key, row=1, column=0)

        self.filter_frame.columnconfigure(0, weight=1)
        self.filter_frame.rowconfigure(0, weight=1)
        self.filter_frame.rowconfigure(1, weight=1)

    def _set_final_curve_controls(self, row):
        cfg_geo = self.data_controller.get_config('geometric')

        outer_curve_size = cfg_geo['outer_curve_size']
        inner_curve_size = cfg_geo['inner_curve_size']
        create_obj = cfg_geo['create_obj']

        self.inner_outer_frame = ttk.LabelFrame(self.controls_frame, text='Inner / Outer')
        self.inner_outer_frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)

        self.slider_outer = create_slider_control(
            self.inner_outer_frame,
            'Outer n',
            outer_curve_size,
            self.controller,
            'outer_curve_size',
            row=0, column=0
        )
        self.slider_inner = create_slider_control(
            self.inner_outer_frame,
            'Inner n',
            inner_curve_size,
            self.controller,
            'inner_curve_size',
            row=1, column=0
        )
        self.checkbox_create_obj_var = tk.BooleanVar(value=create_obj)
        self.checkbox_create_obj = ttk.Checkbutton(
            self.inner_outer_frame,
            variable=self.checkbox_create_obj_var,
            text='Create .obj file',
            command=lambda: self.controller.on_change_control(
                'create_obj', self.checkbox_create_obj_var.get()
            ),
        )
        self.checkbox_create_obj.grid(row=2, column=0, sticky="w", padx=5, pady=5)  # Use grid

        self.inner_outer_frame.columnconfigure(0, weight=1) 
        self.inner_outer_frame.rowconfigure(0, weight=1) 
        self.inner_outer_frame.rowconfigure(1, weight=1) 
        self.inner_outer_frame.rowconfigure(2, weight=1) 

    def _set_controls_defaults(self):
        self._base_controls()

    def _set_figure_defaults(self):
        self.fig, self.axes = plt.subplots(
            nrows=2, ncols=2, constrained_layout=True, figsize=(8,8), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]} 
        )

        axes = self.axes.flat
        for ax in axes:
            ax.grid(color='gray', linestyle='--', linewidth=0.5)
            ax.set_aspect('equal')

        axes[0].grid(False)

        self.axis_line = axes[0].add_line(Line2D([], []))
        self.filter_1_line = [
            axes[0].add_line(Line2D([], [], color=(1.0, 0.0, 0.0, 0.85))),
            None,
        ]
        self.collection0 = axes[0].scatter([], [], s=1, c='blue')
        self.fc = self.collection0.get_facecolors()
        axes[0].figure.canvas.mpl_connect('button_press_event', self._on_click_)
        axes[0].figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        axes[0].figure.canvas.mpl_connect('key_release_event', self.on_key_release)

        self.line1 = axes[1].add_line(
            Line2D([], [], zorder=10, marker='o', markersize=1)
        )

        self.line2 = axes[2].add_line(
            Line2D([], [], linestyle='--', marker='o', markersize=1, alpha=0.6)
        )
        self.c1 = axes[2].scatter([], [], zorder=314, s=5)
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.5)
        self.color_bar  = plt.colorbar(self.c1, cax=cax)
        self.color_bar.ax.yaxis.tick_left()

        axes[3].grid(True, alpha=0.5)
        axes[3].set_xlim(-0.05, 1.05)
        axes[3].set_ylim(-0.05, 1.05)
        self.line3 = axes[3].add_line(Line2D([], [], zorder=10, marker='.', markersize=1, linewidth=0.5, color='black'))
        self.outer_curve_line = axes[3].add_line(Line2D([], [], zorder=10, marker='.', markersize=1, linewidth=0.5, color='black', label=r'$\rho(z)$'))
        self.inner_curve_line = axes[3].add_line(Line2D([], [], zorder=10, marker='.', markersize=1, linewidth=0.5, color='grey'))
        axes[3].legend()

        self.canvas = FigureCanvasTkAgg(self.fig, self.method_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.method_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")

        self.method_frame.rowconfigure(0, weight=1)  
        self.method_frame.rowconfigure(1, weight=0) 
        self.method_frame.columnconfigure(0, weight=1)

    def _reset_controls(self):
        cfg_values = self.data_controller.state_data[
            CDCState.DEFAULT_SYSTEM_LOAD
        ]
        cfg_geo = cfg_values['geometric']

        sample_size = cfg_geo['sample_size']
        spline_parametrization = cfg_geo['parametrization']
        spline_smoothing_factor = cfg_geo['smoothing_factor']
        filter_outliers = cfg_geo['filter_outliers']
        outer_curve_size = cfg_geo['outer_curve_size']
        inner_curve_size = cfg_geo['inner_curve_size']
        create_obj = cfg_geo['create_obj']
		
        self.slider_spline_n.set(sample_size[-1])
        self.combobox_selected_option.set(spline_parametrization[0])
        self.slider_spline_s.set(spline_smoothing_factor[-1])
        self.combobox_spline_parametrization_change.event_generate("<<ComboboxSelected>>")
        self.slider_outliers.set(filter_outliers[-1])
        self.slider_multiple_filter.reset()
        self.slider_outer.set(outer_curve_size[-1])
        self.slider_inner.set(inner_curve_size[-1])
        self.checkbox_create_obj_var.set(create_obj)

    def _reset_figure(self):
        self.axis_line.set_data([], [])
        self.filter_1_line[0].set_data([], [])
        self.filter_1_line[1] = None
        self.collection0.remove()
        self.collection0 = self.axes[0, 0].scatter(
            [], [], s=1, c='blue', zorder=10
        )
        self.fc = self.collection0.get_facecolors()

        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.line3.set_data([], [])

        self.outer_curve_line.set_data([], [])
        self.inner_curve_line.set_data([], [])

        self.c1.set_offsets(np.c_[[], []])

    def update(self, event, *args, **kwargs):
        if self.data_controller.state == CDCState.NO_DATA:
            return
        elif self.data_controller.state == CDCState.DATA_LOADED:
            if event == 'change_of_state':
                self._reset_controls()
                self._reset_figure()
                self.axes[0, 0].imshow(self.data_controller['image'])

                self.collection0.set_offsets(self.controller.contour[:, :2])
                self.fc = np.tile(
                    self.collection0.get_facecolors(),
                    (len(self.controller.contour), 1),
                )
                self.toolbar.update()
        elif self.data_controller.state == CDCState.DEFAULT_SYSTEM_LOAD:
            self._set_controls_defaults()
            self._set_figure_defaults()
