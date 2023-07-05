import os
import numpy as np
from deli import load, save
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('--patchcore_experiment_folder', type=str,
                        help='path to patchcore experiment folder, that contains results.csv and small_segmentations.npy')
    parser.add_argument('--second_stage_preds_path', type=str,
                        help='path to predicted segmentations for the second stage of the pipeline (eg. AMCons)')
    parser.add_argument('--output_path', type=str,
                        help='path to store pipeline segmentations')
    parser.add_argument('--patchcore_threshold', type=float, default=None, 
                        help='threshold to binarize patchcore predictions (if not passed, will be taken from results.csv in the patchcore experiment folder)')
    args = parser.parse_args()
    
    new_preds = []
    pred_shape = (70, 70, 45)
    patchcore_preds = load(os.path.join(args.patchcore_experiment_folder, 'small_segmentations.npy'))
    uids = load(os.path.join(args.patchcore_experiment_folder, 'uids.json'))
    threshold = args.patchcore_threshold
    if threshold is None:
        res = load(os.path.join(args.patchcore_experiment_folder, 'results.csv'))
        threshold = res['threshold'].iloc[0]

    for i, uid in tqdm(enumerate(uids)):

        if not os.path.exists(os.path.join(args.second_stage_preds_path, f'{uid}.npy.gz')):
            continue

        final_pred = np.zeros(pred_shape)
        patchcore_mask = patchcore_preds[i] > threshold
        if patchcore_mask.sum() > 0:
            amcons_pred = load(os.path.join(args.second_stage_preds_path, f'{uid}.npy.gz'))
            final_pred[patchcore_mask] = np.copy(amcons_pred[patchcore_mask])    

        save(final_pred, os.path.join(args.output_path, f'{uid}.npy.gz'), compression=1)
        