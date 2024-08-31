import tkinter as tk
from tkinter import ttk

from ui.custom_controls import Dial, DoubleSlider


def create_slider_control(parent_frame, label_text, values, controller, option_name, row, column, delay=500, precision=4, ret_parent=False):
    label_frame = ttk.LabelFrame(parent_frame, text=label_text)
    label_frame.grid(row=row, column=column, sticky="ew", padx=5, pady=5)  

    min_value, max_value = values[:2]
    value = values[-1]

    is_float = any(isinstance(v, float) for v in values)
    if is_float:
        value = float(value)
        label_frame.config(text=f'{label_text}: {value:.{precision}f}')
    else:
        value = int(float(value))
        label_frame.config(text=f'{label_text}: {value}')

    delay_task = None
    def on_slider_change(value):
        nonlocal delay_task
        if is_float:
            value = float(value)
            label_frame.config(text=f'{label_text}: {value:.{precision}e}')
        else:
            value = int(float(value))
            label_frame.config(text=f'{label_text}: {value}')
        if delay_task:
            parent_frame.after_cancel(delay_task)
        delay_task = parent_frame.after(delay, lambda: controller.on_change_control(option_name, value))

    slider = ttk.Scale(label_frame, value=value, from_=min_value, to=max_value, orient=tk.HORIZONTAL, command=on_slider_change)
    if ret_parent:
        slider.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        label_frame.columnconfigure(0, weight=1)
        return slider, label_frame
    
    slider.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    label_frame.columnconfigure(0, weight=1) 
    return slider


def create_slider_control_multiple(parent_frame, label_text, values, controller, option_name, row, column, delay=500):
    label_frame = ttk.LabelFrame(parent_frame, text=label_text)
    label_frame.grid(row=row, column=column, sticky="ew", padx=5, pady=5)

    min_value, max_value = values
    is_float = any(isinstance(v, float) for v in values)
    value = min_value
    if is_float:
        value = float(value)
        label_frame.config(text=f'{label_text}')
    else:
        value = int(float(value))
        label_frame.config(text=f'{label_text}')

    delay_task = None
    def on_slider_change(value):
        nonlocal delay_task
        if delay_task:
            parent_frame.after_cancel(delay_task)
        delay_task = parent_frame.after(delay, lambda: controller.on_change_control(option_name, value))

    slider = DoubleSlider(label_frame, from_=min_value, to=max_value, command=on_slider_change)
    
    slider.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    label_frame.columnconfigure(0, weight=1)
    return slider


def create_dial_control(parent_frame, label_text, value, controller, option_name, row, column, delay=500):
    label_frame = ttk.LabelFrame(parent_frame, text=label_text)
    label_frame.grid(row=row, column=column, sticky="ew", padx=5, pady=5)

    delay_task = None
    def on_dial_change(value):
        nonlocal delay_task
        label_frame.config(text=f'{label_text}: {value:.4f}')
        if delay_task:
            parent_frame.after_cancel(delay_task)
        delay_task = parent_frame.after(delay, lambda: controller.on_change_control(option_name, value))

    dial = Dial(label_frame, value, size=100, command=on_dial_change)
    dial.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    label_frame.columnconfigure(0, weight=1) 
    return dial