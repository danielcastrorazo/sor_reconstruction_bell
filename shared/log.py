import time
import tkinter as tk

from loguru import logger


def timeit(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("Function '{}' executed in {:f} s", func.__name__, end - start)
        return result
    return wrapped

def io_process(func):
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.trace(f"Function '{func.__name__}' finished execution")
        return result
    return wrapped

class TextWidgetHandler:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        level = message.record['level'].name.upper()
        color = ColorConfig.get_color(level)
        color = color[0] if isinstance(color, tuple) else color
        tag_name = f"{level}_tag"
        if not self.widget.tag_names().__contains__(tag_name):
            self.widget.tag_configure(tag_name, foreground=color)

        self.widget.insert(tk.END, message, (tag_name,))
        self.widget.see(tk.END)

class ColorConfig:
    COLOR_MAP = {
        "FILTER": ("green", 101),
        "FILTER-USER": ("green", 102),
        "INFO": "black",
        "DEBUG": "blue",
        "ERROR": "red",
        "WARNING": "magenta",
        "TRACE": "grey",
        "INFO-S": ("blue", 100)
    }

    @classmethod
    def get_color(cls, level_name):
        data = cls.COLOR_MAP.get(level_name.upper(), "black") 
        return data[0] if isinstance(data, tuple) else data