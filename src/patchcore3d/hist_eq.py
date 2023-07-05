import logging
import tqdm
import os
import time

import numpy as np
from skimage.exposure import equalize_hist
import torch
import torch.nn.functional as F

from dpipe.io import save

import patchcore3d.metrics

LOGGER = logging.getLogger(__name__)


def get_histogram_equalization_predictions(dataset, run_save_path, small_target_size=(70, 70, 45), save_segmentation_images=False):
    """This function provides anomaly scores/maps for histogram equalization method."""

    scores = []
    top_anomaly_patches_dict = {}
    labels_gt = []

    small_masks_gt = []
    small_masks = []
    single_aucs_dict = {}
    single_dices_dict = {}
    single_slice_dices_dict = {}

    if save_segmentation_images:
        LOGGER.info('Save_segmentation_images')

    if save_segmentation_images:
        save_indices = [18,   56,   88,   92,  155,  168,  201,  214,  238,  272,  308,
        319,  354,  375,  432,  465,  484,  522,  558,  605,  610,  645,
        649,  650,  813,  823,  866,  885,  959,  971,  978,  995, 1015,
       1026, 1097, 1148, 1176, 1200, 1280, 1283, 1286, 1330, 1351, 1352]#[0, 1, 1000, 1001, 1002, 1003]
        image_save_path = os.path.join(
            run_save_path, "segmentation_images"
        )
        os.makedirs(image_save_path, exist_ok=True)
    pred_save_path = os.path.join(
        run_save_path, "predictions"
    )
    os.makedirs(pred_save_path, exist_ok=True)

    times = []
    with tqdm.tqdm(range(len(dataset)), desc="Inferring...") as data_iterator:
        for idx in data_iterator:
            
            image = dataset[idx]
            uid = image["uid"]
            is_anomaly = image["is_anomaly"]
            labels_gt.extend([is_anomaly])
            mask_gt = image["mask"]
            small_mask_gt = F.interpolate(
                mask_gt.float(), size=small_target_size, mode="trilinear", align_corners=False
            ) > 0.5
            small_masks_gt.extend(small_mask_gt.numpy().tolist())

            image = image["image"]

            _scores, _masks, _small_masks, _anomaly_patches, time_elapsed = predict_hist_eq(
                image, small_target_size=small_target_size)
            times.append(time_elapsed)
            
            segmentations = np.array(_masks)
            if save_segmentation_images:
                if idx in save_indices:
                    save(segmentations, os.path.join(image_save_path, f'{uid}.npy'))
                    save(_small_masks[0], os.path.join(
                        image_save_path, f'small_{uid}.npy'))
                    save(small_mask_gt, os.path.join(
                        image_save_path, f'small_mask_{uid}.npy'))

            # scale segmentations
            segmentations = segmentations[None, ...]
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            if is_anomaly:
                pixel_scores = patchcore3d.metrics.compute_pixelwise_retrieval_metrics(
                    segmentations,
                    mask_gt.numpy(),
                )
                single_aucs_dict[uid] = pixel_scores["auroc"]
                single_dices_dict[uid] = pixel_scores["dices"][0]
                single_slice_dices_dict[uid] = pixel_scores["slice_dices"][0]

            scores.append(_scores[0])
            top_anomaly_patches_dict[uid] = (
                _anomaly_patches[0] * mask_gt.numpy()).sum() > 1e-9
            small_masks.append(_small_masks[0])

    save(times, os.path.join(run_save_path, 'time.json'))

    return scores, labels_gt, top_anomaly_patches_dict, small_masks, small_masks_gt, single_aucs_dict, single_dices_dict, single_slice_dices_dict


def predict_hist_eq(image, small_target_size):
    start_time = time.time()

    image = image.numpy()

    # Create equalization mask
    mask = np.zeros_like(image)
    mask[image > 1e-9] = 1

    # Equalize
    image = equalize_hist(image, nbins=256, mask=mask)

    # Assure that background still is 0
    image *= mask

    score = np.mean(image)
    time_elapsed = time.time() - start_time

    small_mask = F.interpolate(
        torch.Tensor(image), size=small_target_size, mode="trilinear", align_corners=False
    )[0, 0]

    return [score], [image[0, 0]], [small_mask.numpy()], [False], time_elapsed
