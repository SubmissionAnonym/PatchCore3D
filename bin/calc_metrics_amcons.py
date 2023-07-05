import logging
import os
import sys
import tqdm
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F

from dpipe.io import save_json, load

import patchcore3d.common
import patchcore3d.metrics
import patchcore3d.utils
from patchcore3d.datasets.brain import BrainMRIDataset, DatasetSplit

LOGGER = logging.getLogger(__name__)


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--saved_predictions_path", type=str, default=None, show_default=True)
@click.option("--preds_in_one_file", is_flag=True)
@click.option("--no_sample_level_metrics", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    seed,
    log_group,
    log_project,
    saved_predictions_path,
    preds_in_one_file,
    no_sample_level_metrics,
):
    methods = {key: item for (key, item) in methods}
    target_size_small = (70, 70, 45)

    run_save_path = patchcore3d.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    datasets = methods["get_dataloaders"]()
    uids = datasets["testing"].all_ids
    dataset_name = datasets["training"].name
    datasets["testing"].return_images = False

    result_collect = []

    saved_predictions_path = Path(saved_predictions_path)
    uids = datasets["testing"].all_ids

    print('uids', len(uids))

    # get ground truths
    labels_gt = []
    small_masks_gt = []
    anomaly_segmentor_small = patchcore3d.common.RescaleSegmentor(
        device='cpu', target_size=target_size_small,
    )

    with tqdm.tqdm(range(len(datasets["testing"])), desc="Collecting ground truths...") as data_iterator:
        for idx in data_iterator:

            image = datasets["testing"][idx]
            uid = image["uid"]
            if not os.path.exists(saved_predictions_path / f'{uid}.npy.gz'):
                continue
            is_anomaly = image["is_anomaly"]
            labels_gt.extend([is_anomaly])
            mask_gt = image["mask"]
            small_mask_gt = F.interpolate(
                mask_gt.float(), size=anomaly_segmentor_small.target_size, mode="trilinear", align_corners=False
            ) > 0.5
            small_masks_gt.extend(small_mask_gt.numpy().tolist())

    # loading predictions
    if preds_in_one_file:
        small_segmentations = load(saved_predictions_path)
    else:
        small_segmentations = []
        for uid in tqdm.tqdm(uids, desc="Loading predictions..."):
            if not os.path.exists(saved_predictions_path / f'{uid}.npy.gz'):
                continue
            fname = f'{uid}.npy.gz'
            small_segm = load(saved_predictions_path / fname)
            small_segmentations.append(small_segm)

    if not no_sample_level_metrics:
        scores = [np.std(segm) for segm in small_segmentations] # for amcons
        # scores = [np.mean(segm) for segm in small_segmentations] 

        scores = np.array(scores)
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        scores_dict = {uid: score for uid, score in zip(uids, scores)}
        
        instance_results = patchcore3d.metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )
        auroc = instance_results["auroc"]
        auprc = instance_results["auprc"]
        LOGGER.info(f'Instance AUROC: {auroc}, instance AUPRC: {auprc}')
    else:
        scores_dict = None
        auroc = None
        auprc = None

    # scale small segmentations
    small_segmentations = np.array(small_segmentations)[None, ...]
    min_scores = (
        small_segmentations.reshape(len(small_segmentations), -1)
        .min(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    max_scores = (
        small_segmentations.reshape(len(small_segmentations), -1)
        .max(axis=-1)
        .reshape(-1, 1, 1, 1)
    )
    small_segmentations = (small_segmentations -
                           min_scores) / (max_scores - min_scores)
    small_segmentations = np.mean(small_segmentations, axis=0)

    LOGGER.info("Computing evaluation metrics.")

    # Compute PRO score & PW Auroc for all images
    print('small_segmentations', small_segmentations.shape)
    pixel_scores = patchcore3d.metrics.compute_pixelwise_retrieval_metrics(
        small_segmentations, small_masks_gt
    )
    full_pixel_auroc = pixel_scores["auroc"]
    full_pixel_auprc = pixel_scores["auprc"]
    threshold = pixel_scores["optimal_threshold"]
    full_pixel_f1 = pixel_scores["f1"]
    small_dices = pixel_scores["dices"]
    small_dices_dict = {uid: dice for uid, dice in zip(uids, small_dices)}

    anomaly_indices = []
    i = 0
    for uid in tqdm.tqdm(uids, desc="Filtering predictions..."):
        if not os.path.exists(saved_predictions_path / f'{uid}.npy.gz'):
            continue
        if not uid.startswith('IXI'):
            anomaly_indices.append(i)
        i += 1

    anomaly_segmentations = small_segmentations[anomaly_indices]
    anomaly_masks = [mask for i, mask in enumerate(small_masks_gt) if i in anomaly_indices]
    print('anomaly_segmentations', anomaly_segmentations.shape, len(anomaly_masks))
    pixel_scores = patchcore3d.metrics.compute_pixelwise_retrieval_metrics(
        anomaly_segmentations, anomaly_masks
    )
    anomaly_pixel_auroc = pixel_scores["auroc"]
    anomaly_pixel_auprc = pixel_scores["auprc"]
    # threshold = pixel_scores["optimal_threshold"]
    anomaly_pixel_f1 = pixel_scores["f1"]
    # small_dices = pixel_scores["dices"]
    # small_dices_dict = {uid: dice for uid, dice in zip(uids, small_dices)}
    

    save_json(uids, os.path.join(run_save_path, 'uids.json'))
    save_json(scores_dict, os.path.join(run_save_path, 'scores.json'))
    # save_json(small_single_anomaly_pixel_auroc, os.path.join(
    #     run_save_path, 'small_single_anomaly_pixel_auroc.json'))
    save_json(small_dices_dict, os.path.join(
        run_save_path, 'small_dices.json'))
    # save_json(small_single_anomaly_dice, os.path.join(
    #     run_save_path, 'small_single_anomaly_dice.json'))

    result_collect.append(
        {
            "dataset_name": dataset_name,
            "instance_auroc": auroc,
            "instance_auprc": auprc,
            "full_pixel_auroc": full_pixel_auroc,
            "full_pixel_auprc": full_pixel_auprc,
            "full_pixel_f1(dice)": full_pixel_f1,
            "threshold": threshold,
            "anomaly_pixel_auroc": anomaly_pixel_auroc,
            "anomaly_pixel_auprc": anomaly_pixel_auprc,
            "anomaly_pixel_f1(dice)": anomaly_pixel_f1,
            "small_mean_dices": np.mean(small_dices),
            # "small_mean_anomaly_dices": np.mean([small_dices[i] for i in sel_idxs]),
            # "small_mean_anomaly_pixel_auroc(single anomaly)": np.mean(list(small_single_anomaly_pixel_auroc.values())),
            # "small_single_dices(anomaly)": np.mean(list(small_single_anomaly_dice.values())),
        }
    )

    print(result_collect)

    # for key, item in result_collect[-1].items():
    #     if key != "dataset_name":
    #         LOGGER.info("{0}: {1:3.5f}".format(key, item))

    # LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    # result_metric_names = list(result_collect[-1].keys())
    # result_dataset_names = [results["dataset_name"] for results in result_collect]
    # result_scores = [list(results.values()) for results in result_collect]

    # print(result_metric_names)
    # print(result_dataset_names)
    # print(result_scores)
    # patchcore3d.utils.compute_and_store_final_results(
    #     run_save_path,
    #     result_scores,
    #     column_names=result_metric_names,
    #     row_names=result_dataset_names,
    # )


@main.command("dataset")
@click.argument("ixi_data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("ood_data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--test_size", type=float, default=0.2, show_default=True)
@click.option("--tumor_type", type=str, default="whole_tumor", show_default=True)
def dataset(
    ixi_data_path,
    ood_data_path,
    test_size,
    tumor_type,
):

    def get_dataloaders():
        train_dataset = BrainMRIDataset(
            ixi_data_path=ixi_data_path,
            split=DatasetSplit.TRAIN,
            test_size=test_size,
            imagesize=(240, 240, 155),
            tumor_type=tumor_type,
        )

        test_dataset = BrainMRIDataset(
            ixi_data_path=ixi_data_path,
            ood_data_path=ood_data_path,
            test_size=test_size,
            imagesize=(240, 240, 155),
            split=DatasetSplit.TEST,
            tumor_type=tumor_type,
        )

        dataloader_dict = {
            "training": train_dataset,
            "testing": test_dataset,
        }

        return dataloader_dict

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
