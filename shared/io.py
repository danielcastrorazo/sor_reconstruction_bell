import os
from dataclasses import asdict
from pathlib import Path

import yaml
from loguru import logger

from shared.dataclasses_ import ImageMetadata, _Ellipse
from shared.log import io_process


def custom_dataclass_representer(dumper, data):
    return dumper.represent_dict(asdict(data))

yaml.add_representer(ImageMetadata, custom_dataclass_representer)
yaml.add_representer(_Ellipse, custom_dataclass_representer)
yaml.add_representer(tuple, lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data))


def merge_cfg(cfg_defaults, cfg_user):
    def _merge_cfg(default, usr):
        for k, v in default.items():
            if k not in usr:
                usr[k] = v
            elif isinstance(v, dict):
                _merge_cfg(v, usr[k])

    def count():
        if len(cfg_defaults) != len(cfg_user):
            return False
        for k in cfg_defaults:
            if isinstance(cfg_defaults[k], dict):
                if len(cfg_defaults[k]) != len(cfg_user[k]):
                    return False
        return True

    _merge_cfg(cfg_defaults, cfg_user)

    if not count():
        logger.warning('Some user cfg settings are missing. Default settings will be used.')
    logger.info('Configuration settings, ' + ''.join(f'{t}:{cfg_user[t]}, ' for t in cfg_user))

def load_configuration(cfg_filename='config.yaml'):
    CFG_DEFAULT_FILENAME='default_cfg.yaml'
    
    cfg_user = load_config(cfg_filename)[1]
    cfg_defaults = load_config(CFG_DEFAULT_FILENAME)[1]
    
    merge_cfg(cfg_defaults, cfg_user)

    for var in ['INPUT_PATH', 'OUTPUT_PATH', 'DATA_PATH', 'CURVE_PATH', 'IMAGE_PATH']:
        cfg_user[var] = Path(cfg_user[var])

    return cfg_user

def load_config(config_name='config.yaml'):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_script_dir)
    config_path = os.path.join(root_dir, config_name)

    if os.path.isfile(config_path):
        with open(config_path) as config_file:
            data = yaml.safe_load(config_file)
            return root_dir, data if data is not None else dict()
    else:
        return root_dir, dict()

def _load_data(filepath: Path):
    try:
        with open(filepath) as stream:
            data = yaml.load(stream, yaml.SafeLoader)
            if data is None:
                return dict()
            return data
    except FileNotFoundError as e:
        logger.critical(f'File {filepath} was  not found. {e}')

def _save_data(filepath: str, data: dict):
    with open(filepath, 'w') as stream:
        yaml.dump(data, stream, default_flow_style=None, sort_keys=False)

def load_config_key(key: str):
    root_dir, cfg = load_config()
    path = os.path.join(root_dir, cfg[key])
    return path

def load_data_keys(data_path: Path, image_path=None):
    input_data = _load_data(data_path)
    if not image_path:
        return input_data.keys()

    result = []
    image_file_paths = list_image_files(image_path)
    for file_id in input_data:
        if file_id in image_file_paths:
            result.append(file_id)
    return result

def save_image_metadata(file_id: str, metadata: ImageMetadata, data_path: Path):
    input_data = _load_data(data_path)

    input_data[file_id] = metadata
    _save_data(data_path, input_data)

def load_image_metadata(file_id: str, data_path: Path, image_path: Path) -> tuple[Path, ImageMetadata]:
    input_data = _load_data(data_path)

    if file_id not in input_data:
        raise KeyError(f"Key '{file_id}' not found in input data.")

    image_file_paths = list_image_files(image_path)

    if file_id not in image_file_paths:
        raise KeyError(f"File '{file_id}' was not found in {image_path}.")

    return image_file_paths[file_id], ImageMetadata(**input_data[file_id])

def list_image_files(image_path: Path, allowed_extensions = None) -> dict[str, Path]:
    if allowed_extensions is None:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'} 

    image_paths = {}
    for entry in image_path.rglob('*'):
        if entry.is_file() and entry.suffix.lower() in allowed_extensions:
            if entry.name in image_paths:
                raise KeyError(f"Key '{entry.name}' was found more than once.")
            image_paths[entry.name] = entry
    
    return image_paths

@io_process
def export_for_tex(file_id: str, output_path: Path, identifier: str, inner, outer, image, unrectified_meridian=None):
    full_output_path_dir = output_path / file_id / identifier
    logger.log('INFO-S', f'Saving data for tex to {full_output_path_dir}')
    try:
        import numpy as np
        full_output_path_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(full_output_path_dir / 'inner.txt', inner)
        np.savetxt(full_output_path_dir / 'outer.txt', outer)
        if unrectified_meridian is not None:
            total_samples = 20
            middle_rows = unrectified_meridian[1:-1, :]
            num_middle_samples = total_samples - 2
            interval = max(1, len(middle_rows) // num_middle_samples)

            sampled_middle_rows = middle_rows[::interval]
            sampled_array = np.concatenate(([unrectified_meridian[0]], sampled_middle_rows, [unrectified_meridian[-1]]))
            np.savetxt(full_output_path_dir / 'u_meridian.txt', sampled_array)

        import cv2
        cv2.imwrite(output_path / file_id / 'image.png', image)
    except Exception as e:
        logger.error(f'Could not save data for tex. {e}.')
    
@io_process
def export_mesh(file_id: str, output_path: Path, identifier: str, mesh):
    full_output_path_dir = output_path / file_id / identifier
    logger.log('INFO-S', f'Saving mesh to {full_output_path_dir}')
    try:
        full_output_path_dir.mkdir(parents=True, exist_ok=True)

        model_file_path = full_output_path_dir / 'model.obj'
        mesh.export(model_file_path)
    except Exception as e:
        logger.error(f'Could not save mesh. {e}.')