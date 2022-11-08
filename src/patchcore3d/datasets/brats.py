import os
from pathlib import Path

import nibabel as nb
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent


class BraTS2021(Source):

    _root: str = None

    @meta
    def ids(_root):
        ids_list = [uid for uid in os.listdir(
            Path(_root)) if uid.startswith('BraTS2021_')]
        return tuple(sorted(set(ids_list)))

    def _image_file(i, _root: Silent):
        return nb.load(Path(_root) / i / f'{i}_t2.nii.gz')

    def voxel_spacing(_image_file):
        return tuple(_image_file.header['pixdim'][1:4])

    def image(_image_file):
        return np.int16(_image_file.get_fdata())

    def mask(i, _root: Silent):
        return np.bool_(nb.load(Path(_root) / i / f'{i}_seg.nii.gz').get_fdata())
