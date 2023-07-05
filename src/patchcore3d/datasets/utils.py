from typing import Union

import numpy as np
from connectome import Transform

from dpipe.im.shape_ops import zoom


class ScaleMRI(Transform):
    __inherit__ = True

    _q_min: int = 1
    _q_max: int = 99

    def image(image, _q_min, _q_max):
        image = np.clip(image, *np.percentile(np.float32(image), [_q_min, _q_max]))
        min_val = np.min(image)
        max_val = np.max(image)
        return np.array((image.astype(np.float32) - min_val) / (max_val - min_val), dtype=image.dtype)


class Zoom(Transform):
    __inherit__ = True
    _new_spacing: Union[tuple, float, int] = (None, None, None)
    _order = 3

    def _scale_factor(voxel_spacing, _new_spacing):
        if not isinstance(_new_spacing, (tuple, list, np.ndarray)):
            _new_spacing = np.broadcast_to(_new_spacing, 3)
        return np.nan_to_num(np.float32(voxel_spacing) / np.float32(_new_spacing), nan=1)

    def image(image, _scale_factor, _order):
        return np.array(zoom(image.astype(np.float32), _scale_factor, order=_order))

    def mask(mask, _scale_factor):
        return np.array(zoom(mask.astype(np.float32), _scale_factor) > 0.5, dtype=mask.dtype)


class ZoomImage(Transform):
    __inherit__ = True
    _new_spacing: Union[tuple, float, int] = (None, None, None)
    _order = 3

    def _scale_factor(voxel_spacing, _new_spacing):
        if not isinstance(_new_spacing, (tuple, list, np.ndarray)):
            _new_spacing = np.broadcast_to(_new_spacing, 3)
        return np.nan_to_num(np.float32(voxel_spacing) / np.float32(_new_spacing), nan=1)

    def image(image, _scale_factor, _order):
        return np.array(zoom(image.astype(np.float32), _scale_factor, order=_order))
