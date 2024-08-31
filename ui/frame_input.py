import tkinter as tk
from tkinter import Toplevel, filedialog, messagebox, ttk

from loguru import logger
from PIL import Image, ImageTk

from controllers.data_controller import (
    CDCState,
    DataController,
    SegmentationSubprocessController,
)
from controllers.input_controller import InputController
from ui.canvas import CanvasImage


class FrameInput(ttk.Frame):
    def _set_readonly(self, entry_widget, text):
        entry_widget.config(state=tk.NORMAL)
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, text)
        entry_widget.config(state=tk.DISABLED)
 
    def _set_normal(self, entry_widget, text):
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, text)

    def _reset_all_text_entries(self):
        self._set_readonly(self.filepath_entry, '')
        self._set_normal(self.source_entry, '')
        self._set_readonly(self.height_entry, '')
        self._set_readonly(self.width_entry, '')
        self._set_readonly(self.aspect_ratio_entry, '')
        self._set_readonly(self.camera_model_entry, '')
        self._set_normal(self.sensor_width_entry, '')
        self._set_normal(self.sensor_height_entry, '')
        self._set_normal(self.focal_length_entry, '')
        self._set_normal(self.scale_entry, '')

    def _delete_ellipse_from_tree(self):
        selected_item = self.tree.selection()
        if selected_item:
            self.controller.delete_ellipse(self.tree.index(selected_item[0]))

    def _make_selected_ellipse_first(self):
        selected_item = self.tree.selection()
        if selected_item:
            self.controller.make_ellipse_first(self.tree.index(selected_item[0]))

    def _make_selected_ellipse_last(self):
        selected_item = self.tree.selection()
        if selected_item:
            self.controller.make_ellipse_last(self.tree.index(selected_item[0]))

    def _create_ellipse_treeview(self, parent):
        tree = ttk.Treeview(parent, columns=("color","center", "axes"), height=10)
        tree.heading("#0", text="Color", anchor=tk.CENTER)
        tree.heading("#1", text="Center(x, y)", anchor=tk.CENTER)
        tree.heading("#2", text="Axes(a, b)", anchor=tk.CENTER)
        tree.heading("#3", text="Angle", anchor=tk.CENTER)
        tree.column("#0", anchor=tk.CENTER, width=50)
        tree.column("#1", anchor=tk.CENTER)
        tree.column("#2", anchor=tk.CENTER)
        tree.column("#3", anchor=tk.CENTER)
        return tree

    def __init__(self, parent, tab_controller :InputController, data_controller : DataController, subprocess_controller: SegmentationSubprocessController):
        super().__init__(parent)

        self.controller = tab_controller
        self.controller.attach(self)
        self.data_controller = data_controller
        self.segmentation_controller = subprocess_controller
        
        self.image_on_canvas = None
        
        self.canvas_frame = ttk.LabelFrame(self, text="Image")
        self.canvas_frame.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.controls_frame = ttk.Frame(self)
        self.controls_frame.grid(row=0, column=1, sticky='nswe', padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="crosshair")
        self.canvas.bind("<Leave>", lambda e: self.controller.on_leaving_canvas() if e.state == 0 else None)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.top_frame = ttk.Frame(self.controls_frame)
        self.top_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nswe')
        self.bot_frame = ttk.LabelFrame(self.controls_frame, text='Ellipse details')
        self.bot_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nswe')

        self.top_frame.rowconfigure(0, weight=1)
        self.top_frame.columnconfigure(0, weight=8)
        self.top_frame.columnconfigure(1, weight=2)

        self.bot_frame.rowconfigure(0, weight=1)
        self.bot_frame.columnconfigure(0, weight=2)
        self.bot_frame.columnconfigure(1, weight=8)

        self.controls_frame.rowconfigure((0, 2), weight=1)
        self.controls_frame.columnconfigure(0, weight=1)

        self.data_actions_frame = ttk.Frame(self.top_frame)
        self.data_actions_frame.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.load_new_image_button = tk.Button(self.data_actions_frame, text='Segment New\n Image', bg='#b6d7a8', fg='black', width=10, height=3, command=self.segment_new_image)
        self.load_new_image_button.grid(row=0, column=0, padx=5, pady=5)
        self.load_data_button = tk.Button(self.data_actions_frame, text='Load\nImage Data', bg='#b8a8d0', fg='black', width=10, height=3, command=self.load_image_data)
        self.load_data_button.grid(row=1, column=0, padx=5, pady=5)
        self.save_data_button = tk.Button(self.data_actions_frame, text='Save\nImage data', bg='#ea9999', fg='black', width=10, height=3, command=self.save_image_data)
        self.save_data_button.grid(row=2, column=0, padx=5, pady=5)
        self.data_actions_frame.rowconfigure((0, 1, 2), weight=1)
        self.data_actions_frame.columnconfigure(0, weight=1)

        self.image_details_frame = ttk.LabelFrame(self.top_frame, text="Image information")
        self.image_details_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')

        self.filepath_entry = ttk.Entry(self.image_details_frame)
        self.source_entry = ttk.Entry(self.image_details_frame)

        self.height_entry = ttk.Entry(self.image_details_frame)
        self.width_entry = ttk.Entry(self.image_details_frame)
        self.aspect_ratio_entry = ttk.Entry(self.image_details_frame)
        
        self.camera_model_entry = ttk.Entry(self.image_details_frame)
        self.sensor_width_entry = ttk.Entry(self.image_details_frame, textvariable=tk.DoubleVar(value=0.0))
        self.sensor_height_entry = ttk.Entry(self.image_details_frame, textvariable=tk.DoubleVar(value=0.0))
        self.focal_length_entry = ttk.Entry(self.image_details_frame, textvariable=tk.DoubleVar(value=0.0))
        
        self.scale_entry = ttk.Entry(self.image_details_frame, textvariable=tk.DoubleVar(value=1.0))

        self.image_details_frame.columnconfigure((0, 1, 2, 3), weight=1)
        self.image_details_frame.rowconfigure((0,1,2,3,4,5), weight=1)
        label_entries = [
            ("Filepath:", self.filepath_entry, 0, 1, 3),
            ("Source:", self.source_entry, 1, 1, 3),
            ("Height:", self.height_entry, 2, 1, 1),
            ("Width:", self.width_entry, 3, 1, 1),
            ("Aspect Ratio:", self.aspect_ratio_entry, 4, 1, 1),
            ("Sensor Height:", self.sensor_height_entry, 2, 3, 1),
            ("Sensor Width:", self.sensor_width_entry, 3, 3, 1),
            ("Camera Model:", self.camera_model_entry, 4, 3, 1),
            ("Focal Length (mm):", self.focal_length_entry, 6, 1, 1),
            ("Scale:", self.scale_entry, 6, 3, 1)
        ]
        for item in label_entries:
            text, entry, row, column, columnspan = item
            ttk.Label(self.image_details_frame, text=text).grid(row=row, column=column-1, padx=5, pady=5, sticky='w')
            entry.grid(row=row, column=column, columnspan=columnspan, padx=10, pady=5, sticky='ew')

        self.ellipse_actions_frame = ttk.Frame(self.bot_frame)
        self.ellipse_actions_frame.grid(row=1, column=0, sticky='ns')
        self.finish_ellipse_button = tk.Button(self.ellipse_actions_frame, text='Save\nellipse', bg='#7880b5', fg='black', width=10, height=3, command=self.controller.save_new_ellipse)
        self.finish_ellipse_button.grid(row=0, column=0, padx=5, pady=5)
        self.delete_ellipse_button = tk.Button(self.ellipse_actions_frame, text='Delete\nellipse', bg='#2a9d8f', fg='black', width=10, height=3, command=self._delete_ellipse_from_tree)
        self.delete_ellipse_button.grid(row=1, column=0, padx=5, pady=5)
        self.make_first_button = tk.Button(self.ellipse_actions_frame, text='Make\nFirst', bg='#f4a261', fg='black', width=10, height=3, command=self._make_selected_ellipse_first)
        self.make_first_button.grid(row=2, column=0, padx=5, pady=5)
        self.make_last_button = tk.Button(self.ellipse_actions_frame, text='Make\nLast', bg='#e76f51', fg='black', width=10, height=3, command=self._make_selected_ellipse_last)
        self.make_last_button.grid(row=3, column=0, padx=5, pady=5)
        self.ellipse_actions_frame.rowconfigure((0, 1, 2, 3), weight=1)
        self.ellipse_actions_frame.columnconfigure(0, weight=1)

        self.tree = self._create_ellipse_treeview(self.bot_frame)
        self.tree.grid(row=1, column=1, sticky='nswe', padx=5, pady=5)

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=7)
        self.columnconfigure(1, weight=3)

    def update(self, event, *args, **kwargs):
        if self.data_controller.state == CDCState.DEFAULT_SYSTEM_LOAD:
            if event == 'disable_edits':
                self.save_data_button.config(state=tk.DISABLED)
                self.finish_ellipse_button.config(state=tk.DISABLED)
                self.delete_ellipse_button.config(state=tk.DISABLED)
                self.make_first_button.config(state=tk.DISABLED)
                self.make_last_button.config(state=tk.DISABLED)
                self.load_new_image_button.config(state=tk.DISABLED)
            return
        
        if event == "change_of_state":
            self.update_ellipse_details()
            self.update_canvas(True)
            self._reset_all_text_entries()
            if self.data_controller.state == CDCState.DATA_LOADED:
                self.show_information_image_data()
            elif self.data_controller.state == CDCState.NEW_DATA:
                self.show_information_exif()
        elif event == "ellipse_change":
            self.update_ellipse_details()
            self.update_canvas()
        else:
            self.update_canvas()

    def segment_new_image(self):
        initial_dir = self.data_controller.get_config('IMAGE_PATH')
        file_path = filedialog.askopenfilename(defaultextension=".jpg",
            filetypes=[("Image Files", "*.jpg *.jpeg *.tiff *.JPG *.JPEG *.TIFF"), ("All Files", "*.*")], initialdir=initial_dir)

        if not file_path:
            logger.warning('File path not found.')
            return
        self.segmentation_controller.start(file_path)

    def load_image_data(self):
        def open_image_selector(callback, data):
            if not data:
                return

            def on_select(event):
                selected_item = tree.selection()
                if selected_item:
                    item = tree.item(selected_item)
                    value = item['values'][0]
                    callback(value)

            def filter_treeview(event):
                filter_text = filter_entry.get().lower()
                tree.delete(*tree.get_children())
                for d in data:
                    if filter_text in d.lower():
                        tree.insert('', 'end', values=(d,))

            top = Toplevel()
            top.title("Select Image Data")

            filter_entry = tk.Entry(top)
            filter_entry.pack(fill=tk.X, pady=(5, 0))
            filter_entry.bind('<KeyRelease>', filter_treeview)

            tree = ttk.Treeview(top, columns=('Value'), show='headings')
            tree.heading('#1', text='Value')
            tree.column('#1', anchor=tk.W)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar = ttk.Scrollbar(top, orient=tk.VERTICAL, command=tree.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.configure(yscrollcommand=scrollbar.set)

            for d in data:
                tree.insert('', 'end', values=(d,))
            tree.bind('<<TreeviewSelect>>', on_select)

            top.geometry("600x400")
            top.wait_visibility()

        open_image_selector(
            lambda e: self.data_controller.update_with_stored_data(e),
            self.data_controller.available_stored_keys() 
        )

    def save_image_data(self):
        try:
            if self.data_controller.get_config('resize'):
                logger.warning("Can't save if 'resize' is True.")
                return
            if self.data_controller.get_config('ellipse_angle_mean'):
                logger.warning("Can't save if 'ellipse_angle_mean' is True.")
                return
            
            source = self.source_entry.get()
            image_size = [int(self.height_entry.get()), int(self.width_entry.get())]
            scale = float(self.scale_entry.get())
            sensor_dimensions = [float(self.sensor_width_entry.get()), float(self.sensor_height_entry.get())]
            focal_length = float(self.focal_length_entry.get())

            if messagebox.askyesno("Save Confirmation", "Are you sure you want to save?"):
                ellipse_params = [ep[1] for ep in self.controller.loaded_ellipses_points]
                file_id = self.data_controller.save(source, image_size, scale, sensor_dimensions, focal_length, ellipse_params)
                self.data_controller.update_with_stored_data(file_id)
        except Exception as e:
            logger.warning(f"Can't save data. {e}.")

    def show_information_image_data(self):
        metadata = self.data_controller['metadata']
        self._set_readonly(self.filepath_entry, self.data_controller['file_path'])
        self._set_normal(self.source_entry, metadata.source)

        self._set_readonly(self.height_entry, metadata.image_size[0])
        self._set_readonly(self.width_entry, metadata.image_size[1])

        ratio = max(metadata.image_size[0], metadata.image_size[1]) / min(metadata.image_size[0], metadata.image_size[1])
        self._set_readonly(self.aspect_ratio_entry, ratio)

        self._set_normal(self.sensor_width_entry, metadata.sensor_dimensions[0])
        self._set_normal(self.sensor_height_entry, metadata.sensor_dimensions[1])
        self._set_normal(self.focal_length_entry, metadata.focal_length)

        self._set_normal(self.scale_entry, metadata.scale)
        self._set_readonly(self.camera_model_entry, "")

    def show_information_exif(self):
        state_data = self.data_controller.get_current()
        file_path = state_data['file_path']
        height, width = state_data['image'].shape[:2]
        exif = state_data['exif']

        self._set_readonly(self.filepath_entry, file_path)

        self._set_readonly(self.width_entry, width)
        self._set_readonly(self.height_entry, height)
        ratio = max(width, height) / min(width, height)
        self._set_readonly(self.aspect_ratio_entry, ratio)

        self._set_readonly(self.camera_model_entry, exif.get('EXIF:Camera Model Name', 'Unknown'))
        
        focal_length = float(exif.get('EXIF:Focal Length', '0.0').replace('mm', '').strip() or '0.0')
        self._set_normal(self.focal_length_entry, focal_length)

        self._set_normal(self.scale_entry, '')

    def update_ellipse_details(self):
        def create_color_square(color, size=15):
            img = Image.new('RGBA', (size, size), color)
            return ImageTk.PhotoImage(img)

        self.img_ref = []
        self.tree.delete(*self.tree.get_children())

        data_source = self.controller.loaded_ellipses_points

        for i, value in enumerate(data_source):
            xc, yc, ap, bp, angle = value[1]

            img = create_color_square(self.controller.COLORS[i])
            self.tree.insert('', 'end', image=img, values=(
                f'({xc:.2f}, {yc:.2f})', f'({ap:.2f}, {bp:.2f})', f'{angle:.3f}'))
            self.img_ref.append(img)

    def update_canvas(self, reset_canvas=False):
        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)

        def _key_l_callback(is_positive, x, y):
            self.canvas.focus_set()
            if self.image_on_canvas is None:
                return
            self.controller.on_middle_click(x, y)

        def _left_click_callback(is_positive, x, y):
            self.canvas.focus_set()
            if self.image_on_canvas is None:
                return
            self.controller.on_left_click(x, y)

        def _mouse_move_callback(is_positive, x, y):
            self.canvas.focus_set()
            if self.image_on_canvas is None:
                return
            self.controller.on_mouse_move(x, y)

        def _right_click_callback(is_positive, x, y):
            self.canvas.focus_set()
            if self.image_on_canvas is None:
                return
            self.controller.on_right_click()

        self.image_on_canvas.register_click_callback(_left_click_callback)
        self.image_on_canvas.register_mouse_movement(_mouse_move_callback)
        self.image_on_canvas.register_right_click_callback(_right_click_callback)
        self.image_on_canvas.register_middle_click_callback(_key_l_callback)
        
        self.image_on_canvas.reload_image(Image.fromarray(self.controller.bg_image), reset_canvas)
