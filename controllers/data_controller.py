import os
import re
import subprocess
import sys
import threading
from enum import Enum
from subprocess import PIPE, Popen

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from pycocotools import mask as msk

from shared.dataclasses_ import ImageMetadata, _Ellipse
from shared.io import (
    export_for_tex,
    export_mesh,
    load_configuration,
    load_data_keys,
    load_image_metadata,
    save_image_metadata,
)
from shared.log import ColorConfig


class Observable:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def notify_observers(self, event="", *args, **kwargs):
        for observer in self._observers:
            observer.update(event, *args, **kwargs)

class CDCState(Enum):
    NO_DATA = 0
    NEW_DATA = 1
    DATA_LOADED = 2
    DEFAULT_SYSTEM_LOAD = 99

class DataController(Observable):
    def __init__(self):
        super().__init__()
        
        self.state = CDCState.DEFAULT_SYSTEM_LOAD

        self.state_data = {
            CDCState.NO_DATA: {},
            CDCState.NEW_DATA: {},
            CDCState.DATA_LOADED: {},
            CDCState.DEFAULT_SYSTEM_LOAD: {}
        }

        self.state_data[self.state] = load_configuration()

        logger.remove()
        logger.add(sys.stdout, level=self['log_level'])

        for level_name, cn in ColorConfig.COLOR_MAP.items():
            if isinstance(cn, tuple):
                logger.level(level_name, no=cn[1], color=f"<{cn[0]}>") 

    def add_text_log_handler(self, handler):
        if self['log_level'] == 'TRACE' or self['log_level'] == 'DEBUG':
            logger.add(handler, level=self['log_level'])
        else:
            format = '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}'
            logger.add(handler, format=format, level=self['log_level'])

    def __getitem__(self, key):
        return self.state_data[self.state][key]
    
    def get_current(self):
        return self.state_data[self.state]

    def get_config(self, key: str):
        return self.state_data[CDCState.DEFAULT_SYSTEM_LOAD][key]

    def available_stored_keys(self):
        data_path = self.get_config('DATA_PATH')
        image_path = self.get_config('IMAGE_PATH')
        return load_data_keys(data_path, image_path)

    def set_system_defaults(self):
        is_resize = self.get_current()['resize']
        is_ellipses_mean = self.get_current()['ellipse_angle_mean']
        if is_resize or is_ellipses_mean:
            self.notify_observers(event="disable_edits")
        else:
            self.notify_observers()
        self.state = CDCState.NO_DATA

    def update(self, event, *args, **kwargs):
        if event == 'export_mesh':
            self.save_mesh(**kwargs)
        elif event == 'write_for_tex':
            self.save_for_tex(**kwargs, image=self['image'])

    def update_with_subprocess(self, file_path, image, mask, exif:dict[str, str]):
        file_id = os.path.basename(file_path)

        if file_id in self.available_stored_keys():
            logger.warning(f'There exists an entry with the same name as {file_id}. Segmentation will stop.')
            return

        contour = np.squeeze(max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea))
        
        self.state = CDCState.NEW_DATA
        self.state_data[self.state] = {
            "file_path": file_path,
            "image": image,
            "mask": mask,
            "exif": exif,
            "contour": contour
        }
        self.notify_observers(event="change_of_state")

    def update_with_stored_data(self, file_id):
        file_path, metadata = load_image_metadata(file_id, self.get_config('DATA_PATH'), self.get_config('IMAGE_PATH'))
        
        image = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGBA)
        mask = msk.decode(metadata.mask)

        def resize_data():
            nonlocal image, mask
            s =  self.get_config('resize_max')
            h, w = metadata.image_size
            
            s = max(np.ceil(h / s), np.ceil(w / s))
            if s > 1:
                logger.trace(f'Resizing Image. Scale : 1.0 / {s}')
                image = cv2.resize(image, (int(image.shape[1]/s), int(image.shape[0]/s)))
                mask = cv2.resize(mask, (int(mask.shape[1]/s), int(mask.shape[0]/s)))
                metadata.image_size = image.shape[:2]
                for e in metadata.ellipses:
                    e.axes = (e.axes[0] / s, e.axes[1] / s)
                    e.center = (e.center[0] / s, e.center[1] / s)

        def ellipses_mean_angle():
            from scipy.stats import circmean
            t = circmean([e.angle for e in metadata.ellipses], high=np.pi)
            for e in metadata.ellipses:
                e.angle = t

        if self.get_config('resize'):
            resize_data()

        if self.get_config('ellipse_angle_mean'):
            ellipses_mean_angle()

        contour = np.squeeze(max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea))
        
        self.state = CDCState.DATA_LOADED
        self.state_data[self.state] = {
            "file_path": file_path,
            "metadata": metadata,
            "mask": mask,
            "image": image,
            "contour": contour
        }
        self.notify_observers(event="change_of_state")

    def save(self, source, image_size, scale, sensor_dimensions, focal_length, params):
        class ImageSizeMismatchError(Exception):
            def __init__(self, existing_size, new_size):
                super().__init__(f'Image size mismatch. Existing {existing_size}. New {new_size}')

        file_id = os.path.basename(self.get_current()['file_path'])

        if file_id in self.available_stored_keys():
            _, metadata = load_image_metadata(file_id, self.get_config('DATA_PATH'), self.get_config('IMAGE_PATH'))
            if metadata.image_size != image_size:
                raise ImageSizeMismatchError(metadata.image_size, image_size)

        ellipses = []
        for p in params:
            p = tuple(float(x) for x in p)
            xc, yc, ap, bp, angle = p
            ellipses.append(_Ellipse(center=(xc, yc), axes=(ap, bp), angle=angle))

        mask = msk.encode(np.asfortranarray(self.get_current()['mask']))
        datum = ImageMetadata(source, image_size, scale, sensor_dimensions, focal_length, mask, ellipses)

        save_image_metadata(file_id, datum, self.get_config('DATA_PATH'))
        logger.success(f'Saved image metadata {file_id}.')
        return file_id

    def save_mesh(self, **kwargs):
        file_id = os.path.basename(self.get_current()['file_path'])
        output_path = self.get_config('OUTPUT_PATH')
        export_mesh(file_id, output_path, **kwargs)

    def save_for_tex(self, **kwargs):
        file_id = os.path.basename(self.get_current()['file_path'])
        output_path = self.get_config('OUTPUT_PATH')
        export_for_tex(file_id, output_path, **kwargs)

class SegmentationSubprocessController:
    def __init__(self, data_controller : DataController):
        super().__init__()
        self.data_controller = data_controller
        self.data_controller.attach(self)

        self.finished_flag = threading.Event()

    def update(self, event):
        if self.data_controller.state == CDCState.DEFAULT_SYSTEM_LOAD:
            sub = self.data_controller.get_config('subprocess')
            key = sub['segmentation']
            if key == 'sc':
                self.notify_data_controller = self.sc_notify
                self.start = self.start_sc
            elif key == 'rembg':
                self.notify_data_controller = self.rembg_notify
                self.start = self.start_rembg

            key = sub['exif']
            if key == 'pillow':
                self.exif_process = self.exif_data_pillow
            elif key == 'exiftool':
                self.exif_process = self.exif_data_exiftool

    def sc_notify(self):
        file_path = self.output['file_path_bytes'].decode()
        mask_data = self.output['mask_bytes']
        exif = self.exif_process(file_path)

        image = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGBA)
        
        mask_ = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_COLOR)
        mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask_, 128, 255, cv2.THRESH_BINARY)

        self.data_controller.update_with_subprocess(file_path, image, mask, exif)

    def rembg_notify(self):
        file_path = self.output['file_path']
        image = self.output['image']
        mask = self.output['mask']
        exif = self.exif_process(file_path)

        self.data_controller.update_with_subprocess(file_path, image, mask, exif)

    def exif_data_pillow(self, file_path):

        exif = {}
        with Image.open(file_path) as img:
            exif_data = img._getexif() 
            if not exif_data:
                logger.warning('No EXIF data found in the image.')
            else:
                if 272 in exif_data:
                    exif['EXIF:Camera Model Name'] = exif_data[272]
                if 37386 in exif_data:
                    exif['EXIF:Focal Length'] = str(exif_data[37386])
        return exif
    
    def exif_data_exiftool(self, file_path):
        command = ['exiftool', '-G', file_path]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except subprocess.SubprocessError as e:
            logger.error(f'Error running exiftool command: {e}')
        
        if result.returncode != 0:
            raise RuntimeError(f'Error running exiftool: {result.stderr}')
        
        exif = {}
        lines = result.stdout.splitlines()
        for line in lines:
            match = re.match(r'\[(.+?)\]\s+(.+?)\s*:\s*(.*)', line)
            if match:
                group, tag, value = match.groups()
                exif[f"{group}:{tag}"] = value.strip()
        return exif

    def start_rembg(self, path):
        def get_data_with_rembg(file_path):
            try:
                from rembg import remove
                image = cv2.imread(path)
                output = remove(image)
                mask = output[:,:,3]
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                image = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGBA)
                self.output = {'image': image, 'mask': mask, 'file_path': file_path}
                self.finished_flag.set()
            except ImportError as e:
                logger.warning(f'The package was not found. {e}.')

        threading.Thread(target=get_data_with_rembg(path)).start()
    
    def start_sc(self, path):

        def get_data_with_sc():
            try:
                delimiter_end = b'--DELIM-END--'
                file_path = "submodules/SimpleClick/run.sh"

                with Popen([file_path], stdout=PIPE, bufsize=0) as process:
                    buffer = b''
                    waiting_image = True
                    while True:
                        chunk = process.stdout.read(4096)
                        if not chunk and process.poll() is not None:
                            break

                        buffer += chunk
                        
                        if waiting_image:
                            start = buffer.find(b'\x89PNG')
                            if start == -1:
                                continue
                            end = buffer.find(b'\x00\x00\x00\x00IEND\xAE\x42\x60\x82', start)
                            if end == -1:
                                continue
                            
                            end += 12
                            mask_data = buffer[start:end]
                            buffer = buffer[end:]
                            waiting_image = False

                        if not waiting_image:
                            end_delim = buffer.find(delimiter_end)
                            if end_delim == -1:
                                continue
                            file_data = buffer[:end_delim]

                            waiting_image = True
                            buffer = b''

                            self.output = {
                                'mask_bytes': mask_data,
                                'file_path_bytes': file_data
                            }
                            self.finished_flag.set()
            except FileNotFoundError as e:
                logger.warning(f'Could not find the submodule. {e}.' )

        threading.Thread(target=get_data_with_sc).start()
