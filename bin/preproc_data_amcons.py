import nibabel as nib
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import argparse
import cv2

from dpipe.io import load

np.random.seed(42)
random.seed(42)


def save_slices(img_paths, uids, scan, partition, dir_out, mask_paths=None, nSlices=None):
    for i, img_path in enumerate(tqdm(img_paths)):

        uid = uids[i]

        if mask_paths is not None:
            mask_path = mask_paths[i]

        img = nib.load(img_path)
        img = (img.get_fdata())[:, :, :]
        img = (img / img.max()) * 255
        img = img.astype(np.uint8)

        if mask_paths is not None:
            mask = nib.load(mask_path)
            mask = (mask.get_fdata())
            mask[mask > 0] = 255
            mask = mask.astype(np.uint8)

        if nSlices is None:
            slice_range = np.arange(img.shape[-1])
        else:
            slice_range = np.arange(
                round(img.shape[-1]/2) - nSlices, round(img.shape[-1]/2) + nSlices)

        for iSlice in slice_range:
            filename = f'{uid}_{str(iSlice)}.jpg'

            i_image = img[:, :, iSlice]
            if mask_paths is not None:
                i_mask = mask[:, :, iSlice]

                if np.any(i_mask == 255):
                    label = 'malign'
                    cv2.imwrite(str(dir_out / scan / partition /
                                'ground_truth' / filename), i_mask)
                else:
                    label = 'benign'
            else:
                label = 'benign'

            cv2.imwrite(str(dir_out / scan / partition / label / filename), i_image)


def adequate_BRATS(args):

    dir_dataset_brats = Path(args.dir_dataset_brats)
    dir_dataset_ixi = Path(args.dir_dataset_ixi)
    dir_out = Path(args.dir_out)
    scan = args.scan
    nSlices = args.nSlices
    ixi_ids_split_file = args.ixi_ids_split_file

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    if not os.path.isdir(dir_out / scan):
        os.mkdir(dir_out / scan)

    for cur_partition in ['train', 'test', 'val']:
        if not os.path.isdir(dir_out / scan / cur_partition):
            os.mkdir(dir_out / scan / cur_partition)
        if not os.path.isdir(dir_out / scan / cur_partition / 'benign'):
            os.mkdir(dir_out / scan / cur_partition / 'benign')
        if not os.path.isdir(dir_out / scan / cur_partition / 'malign'):
            os.mkdir(dir_out / scan / cur_partition / 'malign')
        if not os.path.isdir(dir_out / scan / cur_partition / 'ground_truth'):
            os.mkdir(dir_out / scan / cur_partition / 'ground_truth')

    dir_dataset = dir_dataset_brats
    img_files_brats = os.listdir(dir_dataset)
    img_paths_brats = [dir_dataset / iCase /
                       f'{iCase}_{scan}.nii.gz' for iCase in img_files_brats if iCase != '.DS_Store']
    mask_paths_brats = [dir_dataset / iCase /
                        f'{iCase}_seg.nii.gz' for iCase in img_files_brats if iCase != '.DS_Store']
    uids_brats = [iCase for iCase in img_files_brats if iCase != '.DS_Store']
    save_slices(img_paths=img_paths_brats, uids=uids_brats, scan=scan,
                partition='test', dir_out=dir_out, mask_paths=mask_paths_brats, nSlices=None)
    save_slices(img_paths=img_paths_brats[:2], uids=uids_brats, scan=scan,
                partition='val', dir_out=dir_out, mask_paths=mask_paths_brats, nSlices=nSlices)

    dir_dataset = dir_dataset_ixi
    ixi_ids_splits = load(ixi_ids_split_file)
    train_ids = ixi_ids_splits["train"]
    test_ids = ixi_ids_splits["test"]
    mask_paths_ixi = None
    img_files = os.listdir(dir_dataset)
    img_paths_train = [
        dir_dataset / iCase for iCase in img_files if '_t2' in iCase and iCase.split('_t2.nii.gz')[0] in train_ids]
    img_paths_test = [
        dir_dataset / iCase for iCase in img_files if '_t2' in iCase and iCase.split('_t2.nii.gz')[0] in test_ids]

    save_slices(img_paths=img_paths_train, uids=train_ids, scan=scan,
                partition='train', dir_out=dir_out, mask_paths=mask_paths_ixi, nSlices=nSlices)
    # save_slices(img_paths=img_paths_train, uids=train_ids, scan=scan, partition='train', dir_out=dir_out, mask_paths=mask_paths_ixi, nSlices=None)
    save_slices(img_paths=img_paths_test, uids=test_ids, scan=scan,
                partition='test', dir_out=dir_out, mask_paths=mask_paths_ixi, nSlices=None)
    save_slices(img_paths=img_paths_test[:2], uids=test_ids, scan=scan,
                partition='val', dir_out=dir_out, mask_paths=mask_paths_ixi, nSlices=nSlices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_dataset_brats", default='../../brats2021/', type=str)
    parser.add_argument("--dir_dataset_ixi", default='../../ixi/', type=str)
    parser.add_argument("--ixi_ids_split_file", default='../ixi_ids.json', type=str)
    parser.add_argument("--dir_out", default='../brain_10slices/', type=str)
    parser.add_argument("--scan", default='t2', type=str)
    parser.add_argument("--nSlices", default=5, type=int)

    args = parser.parse_args()
    adequate_BRATS(args)
