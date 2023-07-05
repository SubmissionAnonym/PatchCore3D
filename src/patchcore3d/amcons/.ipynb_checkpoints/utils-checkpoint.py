import sklearn.metrics
from skimage import exposure
import imutils
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from dpipe.io import save


def inference_dataset_volume_by_volume(method, dataset, results_path, save_segmentation_results=False):

    # Take references and inputs from testing dataset
    X_paths = dataset.X_paths

    # Init variables
    p = len(X_paths)
    s = len(X_paths[0])
    Scores = np.zeros((p, s))

    uids = ['IXI025-Guys-0852', 'IXI012-HH-1211', 'BraTS2021_01302',
            'BraTS2021_01303', 'BraTS2021_01301', 'BraTS2021_01300']

    os.makedirs(results_path / 'Mhat', exist_ok=True)
    if save_segmentation_results:
        os.makedirs(results_path / 'Mhat_full', exist_ok=True)

    times = []
    for iVolume in tqdm(np.arange(0, p)):

        M_hat = []
        vol_time = 0
        for iSlice in np.arange(0, s):
            # Take image
            x, m, y, uid = dataset.get_slice(iVolume, iSlice)
            # Predict anomaly map and score
            if x.mean() == 0:
                score = 0
                time_elapsed = 0
                mhat = np.zeros_like(x)[0]
            else:
                score, mhat, xhat, time_elapsed = method.predict_score(x)
            vol_time += time_elapsed

            Scores[iVolume, iSlice] = score
            M_hat.append(mhat)
        times.append(vol_time)

        M_hat = np.asarray(M_hat, dtype=np.float32).transpose(1, 2, 0)[None, None]
        if save_segmentation_results and uid in uids:
            save(M_hat[0, 0], results_path / 'Mhat_full' /
                 f'{uid}.npy.gz', compression=1)

        M_hat = torch.from_numpy(M_hat)
        small_segm = F.interpolate(M_hat, size=(70, 70, 45),
                                   mode="trilinear", align_corners=False)[0, 0]
        save(small_segm.numpy(), results_path / 'Mhat' / f'{uid}.npy.gz', compression=1)

    save(times, results_path / 'time.json')


def image_normalization(x, shape, norm='max', channels=3, histogram_matching=False, reference_image=None,
                        mask=False, channel_first=True):

    # Histogram matching to reference image
    if histogram_matching:
        x_norm = exposure.match_histograms(x, reference_image)
        x_norm[x == 0] = 0
        x = x_norm

    # image resize
    x = imutils.resize(x, height=shape)
    #x = resize_image_canvas(x, shape)

    # Grayscale image -- add channel dimension
    if len(x.shape) < 3:
        x = np.expand_dims(x, -1)

    if mask:
        x = (x > 200)

    # channel first
    if channel_first:
        x = np.transpose(x, (2, 0, 1))
    if not mask:
        if norm == 'max':
            x = x / 255.0
        elif norm == 'zscore':
            x = (x - 127.5) / 127.5

    # numeric type
    x.astype('float32')
    return x
