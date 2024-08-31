import tkinter as tk
from tkinter import Tk, ttk

from loguru import logger

from controllers.colombo_controller import ColomboController
from controllers.data_controller import (
    CDCState,
    DataController,
    SegmentationSubprocessController,
)
from controllers.input_controller import InputController
from controllers.kwan_controller import KwanController
from shared.log import TextWidgetHandler
from ui.frame_colombo import FrameColombo
from ui.frame_input import FrameInput
from ui.frame_kwan import FrameKwan


class MainGUI(Tk):
    def __init__(self):
        super().__init__()
        self.minsize(1280, 800)
        self.protocol("WM_DELETE_WINDOW", self.quit)

        self.data_controller = DataController()
        self.data_controller.attach(self)

        self.subprocess_controller = SegmentationSubprocessController(self.data_controller)

        self.tab1_controller = InputController(self.data_controller)
        self.tab2_controller = KwanController(self.data_controller, identifier='k')
        self.tab3_controller = ColomboController(self.data_controller, identifier='c')

        self.main_frame = ttk.Frame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew") 

        def set_style():
            self.style = ttk.Style()
            self.style.configure('TNotebook.Tab', background='#adacb5', padding=[10, 5])

            self.style.layout("Tab",
            [('Notebook.tab', {'sticky': 'nswe', 'children':
                [('Notebook.padding', {'side': 'top', 'sticky': 'nswe', 'children':
                        [('Notebook.label', {'side': 'top', 'sticky': ''})],
                })],
            })]
            )
        set_style()

        self.tabs = ttk.Notebook(self.main_frame)
        self.tab1 = FrameInput(self.tabs, self.tab1_controller, self.data_controller, self.subprocess_controller)
        self.tab2 = FrameKwan(self.tabs, self.tab2_controller, self.data_controller)
        self.tab3 = FrameColombo(self.tabs, self.tab3_controller, self.data_controller)
        self.tabs.add(self.tab1)
        self.tabs.add(self.tab2)
        self.tabs.add(self.tab3)
        self.tabs.grid(row=0, column=0, sticky="nsew") 

        entryFrame = ttk.Frame(self.main_frame)
        self.log_textbox = tk.Text(entryFrame, height=10)
        self.log_textbox.grid(row=0, column=0, sticky="nsew")
        self.data_controller.add_text_log_handler(TextWidgetHandler(self.log_textbox))
        
        scrollbar = ttk.Scrollbar(entryFrame, orient="vertical", command=self.log_textbox.yview)
        self.log_textbox['yscrollcommand'] = scrollbar.set
        scrollbar.grid(row=0, column=1, sticky="ns")

        entryFrame.grid(row=1, column=0, sticky="nsew")
        entryFrame.rowconfigure(0, weight=1)
        entryFrame.columnconfigure(0, weight=1)

        self.main_frame.rowconfigure(0, weight=9)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.data_controller.set_system_defaults()

        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")
        self.schedule_controller_mask()

        self.show_instructions()
        
    def update(self, event, *args, **kwargs):
        if self.data_controller.state == CDCState.DEFAULT_SYSTEM_LOAD:
            d = self.data_controller['gui']
            self.title(d['title'])
            
            self.tabs.tab(0, text=d['tab1']['title'])
            self.tabs.tab(1, text=d['tab2']['title'])
            self.tabs.tab(2, text=d['tab3']['title'])
        else:
            self.log_textbox.delete('1.0', tk.END)

    def schedule_controller_mask(self):
        if self.subprocess_controller.finished_flag.is_set():
            self.subprocess_controller.finished_flag.clear()
            self.subprocess_controller.notify_data_controller()
        self.after(1000, self.schedule_controller_mask)

    def show_instructions(self):
        logger.log('INFO-S','Configuration may be done in default_cfg.yaml or config.yaml.')
        logger.info('For saving or editing existing data. Resize and ellipse_angle_mean should be False.')