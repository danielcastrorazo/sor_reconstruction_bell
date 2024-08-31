import math
import tkinter as tk

import numpy as np


class DoubleSlider(tk.Canvas):
    def __init__(self, parent, from_=0.0, to=1.0, resolution=0.001, command=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.height = 40
        self.width = self.winfo_width()
        self.config(height=self.height)
        
        self.from_ = from_
        self.to = to
        self.resolution = resolution
        
        self.point0 = self.from_
        self.point1 = self.to

        self.padding = 20
        self.slider_radius = 4
        self.text_offset = 15

        self.active_point = None

        self.bind("<B1-Motion>", self.move_slider)
        self.bind("<Button-1>", self.start_move_slider)
        self.bind("<ButtonRelease-1>", self.release_slider)
        self.bind("<Configure>", self.on_resize)

        self.on_slider_change = command

        self.draw_sliders()

    def on_resize(self, event):
        self.width = event.width
        self.height = event.height
        self.draw_sliders()

    def draw_sliders(self):
        h = self.height / 2

        def pp(p):
            d = self.value_to_pixel(p)
            d = max(self.slider_radius + self.padding, min(d, self.width - self.slider_radius - self.padding))
            x0, y0 = d - self.slider_radius, h - self.slider_radius
            x1, y1 = d + self.slider_radius, h + self.slider_radius
            return x0, y0, x1, y1

        self.delete("all")

        self.create_line(self.slider_radius + self.padding, h, self.width - self.slider_radius - self.padding, h, fill="gray", width=2)

        x0, y0, x1, y1 = pp(self.point0)
        self.create_oval(x0, y0, x1, y1, fill='blue', outline='black', tags='slider1')
        text_x = max(self.padding, min(x0 + self.slider_radius, self.width - self.padding)) 
        self.create_text(text_x, h + self.text_offset, text=f'{self.point0:.3f}', font=('Arial', 10))

        x0, y0, x1, y1 = pp(self.point1)
        self.create_oval(x0, y0, x1, y1, fill='red', outline='black', tags='slider2')
        text_x = max(self.padding, min(x0 + self.slider_radius, self.width - self.padding)) 
        self.create_text(text_x, h + self.text_offset, text=f'{self.point1:.3f}', font=('Arial', 10))

    def start_move_slider(self, event):
        x = event.x
        value = self.pixel_to_value(x)
        self.active_point = 0 if abs(value - self.point0) < abs(value - self.point1) else 1
        self.move_slider(event)

    def release_slider(self, event):
        self.active_point = None

    def move_slider(self, event):
        if self.active_point is None:
            return
        
        x = event.x
        if x < 0:
            x = 0
        elif x > self.width:
            x = self.width

        value = self.pixel_to_value(x)
        if self.active_point == 0:
            self.point0 = self.clamp(value)
        else:
            self.point1 = self.clamp(value)

        if self.point0 > self.point1:
            self.point0, self.point1 = self.point1, self.point0

        self.draw_sliders()
        if self.on_slider_change:
            self.on_slider_change((self.point0, self.point1))

    def value_to_pixel(self, value):
        return (value - self.from_) / (self.to - self.from_) * (self.width - self.slider_radius) + self.slider_radius / 2

    def pixel_to_value(self, pixel):
        return (pixel - self.slider_radius / 2) / (self.width - self.slider_radius) * (self.to - self.from_) + self.from_

    def clamp(self, value):
        return max(self.from_, min(value, self.to))
    
    def reset(self):
        self.point0 = self.from_
        self.point1 = self.to
        self.draw_sliders()
        if self.on_slider_change:
            self.on_slider_change((self.point0, self.point1))

class Dial(tk.Canvas):
    def __init__(self, master, angle, command=None, size=200, **kwargs):
        super().__init__(master, width=size, height=size, **kwargs)
        self.size = size
        self.angle = angle
        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_button_motion)

        self.on_dial_change = command

        self.draw_dial()

    def set_angle(self, angle):
        self.angle = angle
        if self.on_dial_change:
            self.on_dial_change(self.angle)
        self.draw_dial()

    def draw_dial(self):
        self.delete("all")

        self.create_oval(10, 10, self.size-10, self.size-10, fill="lightgray", outline="black", width=2)

        for angle in np.linspace(0, 2 * np.pi, 36):
            x1 = self.size/2 + (self.size/2-20) * math.cos(angle)
            y1 = self.size/2 + (self.size/2-20) * math.sin(angle)
            x2 = self.size/2 + (self.size/2-10) * math.cos(angle)
            y2 = self.size/2 + (self.size/2-10) * math.sin(angle)
            self.create_line(x1, y1, x2, y2)

        x = self.size/2 + (self.size/2-30) * math.cos(self.angle)
        y = self.size/2 + (self.size/2-30) * math.sin(self.angle)
        self.create_line(self.size/2, self.size/2, x, y, fill="red", width=3)

    def on_button_press(self, event):
        """Handle mouse button press"""
        self.update_angle(event.x, event.y)

    def on_button_motion(self, event):
        """Handle mouse motion while button is pressed"""
        self.update_angle(event.x, event.y)

    def update_angle(self, x, y):
        """Calculate and update the angle based on mouse position"""
        dx = x - self.size/2
        dy = y - self.size/2
        self.angle = math.atan2(dy, dx)

        # angle_step = 2 * np.pi / 8  # 45 degrees in radians
        # self.angle = round(self.angle / angle_step) * angle_step

        self.angle = self.angle % (2 * np.pi)
        self.draw_dial()

        if self.on_dial_change:
            self.on_dial_change(self.angle)