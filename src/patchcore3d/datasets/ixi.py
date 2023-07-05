import os
from pathlib import Path

import nibabel as nb
import numpy as np
from connectome import Source, meta
from connectome.interface.nodes import Silent


class IXI(Source):

    _root: str = None

    @meta
    def ids(_root):
        ids_list = [uid.split('_t2.nii.gz')[0]
                    for uid in os.listdir(_root) if uid.endswith('_t2.nii.gz')]
        return tuple(sorted(set(ids_list)))

    def _image_file(i, _root: Silent):
        return nb.load(Path(_root) / f'{i}_t2.nii.gz')

    def image(_image_file):
        # returns T2 image
        return np.float32(_image_file.get_fdata())

    def voxel_spacing(_image_file):
        return tuple(_image_file.header['pixdim'][1:4])

    def image_t1(i, _root: Silent):
        return np.float32(nb.load(Path(_root) / f'{i}_t1.nii.gz').get_fdata())

    def brain_mask(i, _root: Silent):
        return np.bool_(nb.load(Path(_root) / f'{i}_mask.nii.gz').get_fdata())

    def fold(i):
        return i.split('-')[1]
