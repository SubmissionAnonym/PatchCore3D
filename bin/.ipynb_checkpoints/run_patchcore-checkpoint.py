import contextlib
import logging
import os
import sys
import tqdm

import click
import numpy as np
import torch

from dpipe.io import save, save_json, load, load_json

import patchcore3d.backbones
import patchcore3d.common
import patchcore3d.hist_eq
import patchcore3d.metrics
import patchcore3d.patchcore3d
import patchcore3d.sampler
import patchcore3d.sample_histograms
import patchcore3d.utils
from patchcore3d.datasets.brain import BrainMRIDataset, DatasetSplit

LOGGER = logging.getLogger(__name__)


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
@click.option("--saved_features_path", type=str, default=None)
@click.option("--anomaly_method", type=str, default="patchcore")
@click.option("--saved_test_features_path", type=str, default=None, show_default=True)
@click.option("--save_test_features", is_flag=True)
@click.option("--save_result_data", is_flag=True)
@click.option("--saved_result_data_path", type=str, default=None, show_default=True)
@click.option("--save_ground_truths", is_flag=True)
@click.option("--saved_ground_truths_path", type=str, default=None, show_default=True)
@click.option("--n_hist_bins", type=int, default=128, show_default=True)
@click.option("--saved_hist_features_path", type=str, default=None, show_default=True)
@click.option("--saved_test_hist_features_path", type=str, default=None, show_default=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
    saved_features_path,
    anomaly_method,
    saved_test_features_path,
    save_test_features,
    save_result_data,
    saved_result_data_path,
    save_ground_truths,
    saved_ground_truths_path,
    n_hist_bins,
    saved_hist_features_path,
    saved_test_hist_features_path,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore3d.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    datasets = methods["get_dataloaders"]()
    uids = datasets["testing"].all_ids
    dataset_name = datasets["training"].name

    result_collect = []
    if anomaly_method == "histogram_equalization":
        scores, labels_gt, top_anomaly_patches_dict, small_segmentations, small_masks_gt, single_aucs_dict, single_dices_dict, single_slice_dices_dict = patchcore3d.hist_eq.get_histogram_equalization_predictions(
                    datasets["testing"], run_save_path=run_save_path, save_segmentation_images=save_segmentation_images)
        
    elif anomaly_method == "sample_histograms":
        LOGGER.info("Evaluating dataset...")

        if saved_hist_features_path is None:
            features = patchcore3d.sample_histograms.get_histogram_features(
                datasets["training"], n_bins=n_hist_bins)
            save(features, os.path.join(run_save_path, 'features.npy'))
        else:
            features = load(saved_hist_features_path)

        if saved_test_hist_features_path is None:
            scores, labels_gt, test_features = patchcore3d.sample_histograms.get_histogram_ood_scores(
                datasets["testing"], features)
            save(test_features, os.path.join(run_save_path, 'test_features.npy'))
        else:
            test_features = load(saved_test_hist_features_path)
            scores, labels_gt = patchcore3d.sample_histograms.get_histogram_ood_scores_with_features(
                features, test_features)

        scores = (scores - scores.min()) / (scores.max() - scores.min())
        save_json(scores, os.path.join(run_save_path, 'scores.json'))

        results = patchcore3d.metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )
        LOGGER.info(f'Instance AUROC: {results["auroc"]}, instance AUPRC: {results["auprc"]}')
        save_json(results, os.path.join(run_save_path, 'results.json'))
        save_json(uids, os.path.join(run_save_path, 'uids.json'))
        return
        
    elif anomaly_method == "patchcore":
        if saved_result_data_path is None:
            device = patchcore3d.utils.set_torch_device(gpu)
            # Device context here is specifically set and used later
            # because there was GPU memory-bleeding which I could only fix with
            # context managers.
            device_context = (
                torch.cuda.device("cuda:{}".format(device.index))
                if "cuda" in device.type.lower()
                else contextlib.suppress()
            )

            LOGGER.info("Evaluating dataset...")
            patchcore3d.utils.fix_seeds(seed, device)

            with device_context:
                torch.cuda.empty_cache()
                imagesize = datasets["training"].imagesize
                sampler = methods["get_sampler"](device)
                PatchCore = methods["get_patchcore"](imagesize, sampler, device)
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore3d.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info("Training model")
                torch.cuda.empty_cache()
                PatchCore.fit(datasets["training"], run_save_path, saved_features_path)

                # (Optional) Store PatchCore model for later re-use.
                # SAVE all patchcores only if mean_threshold is passed?
                if save_patchcore_model:
                    patchcore_save_path = os.path.join(
                        run_save_path, "models", dataset_name
                    )
                    os.makedirs(patchcore_save_path, exist_ok=True)
                    PatchCore.save_to_path(patchcore_save_path)

                # (Optional) Save test features for later re-use
                if saved_test_features_path is None and save_test_features:
                    torch.cuda.empty_cache()
                    LOGGER.info("Embedding test data with model")
                    PatchCore.save_test_features(datasets["testing"], run_save_path)
                    saved_test_features_path = run_save_path

                torch.cuda.empty_cache()

                scores, labels_gt, top_anomaly_patches_dict, small_segmentations, small_masks_gt, single_aucs_dict, single_dices_dict, single_slice_dices_dict = PatchCore.predict(
                    datasets["testing"], saved_test_features_path=saved_test_features_path, 
                    run_save_path=run_save_path, save_segmentation_images=save_segmentation_images
                )
                if save_result_data:
                    if save_ground_truths:
                        save(np.array(small_masks_gt), os.path.join(
                            run_save_path, 'small_masks_gt.npy'))
                        save_json(labels_gt, os.path.join(
                            run_save_path, 'labels_gt.json'))
                    del small_masks_gt
                    save_json(uids, os.path.join(run_save_path, 'uids.json'))
                    save_json(scores, os.path.join(
                        run_save_path, 'scores_arr.json'))
                    save_json(single_aucs_dict, os.path.join(
                        run_save_path, 'single_anomaly_aucs.json'))
                    save_json(single_dices_dict, os.path.join(
                        run_save_path, 'single_dices.json'))
                    save_json(single_slice_dices_dict, os.path.join(
                        run_save_path, 'single_slice_dices.json'))
                    save_json(top_anomaly_patches_dict, os.path.join(
                        run_save_path, 'top_anomaly_patches_dict.json'))

                    small_segmentations = np.array(small_segmentations).astype(np.float16)
                    save(small_segmentations, os.path.join(run_save_path, f'small_segmentations.npy'))
                    del small_segmentations
                    return
        else:
            uids = load_json(os.path.join(saved_result_data_path, 'uids.json'))
            scores = load_json(os.path.join(saved_result_data_path, 'scores_arr.json'))
            single_aucs_dict = load_json(os.path.join(
                saved_result_data_path, 'single_anomaly_aucs.json'))
            single_dices_dict = load_json(
                os.path.join(saved_result_data_path, 'single_dices.json'))
            single_slice_dices_dict = load_json(
                os.path.join(saved_result_data_path, 'single_slice_dices.json'))
            top_anomaly_patches_dict = load_json(os.path.join(
                saved_result_data_path, 'top_anomaly_patches_dict.json'))

            small_segmentations = load(os.path.join(
                saved_result_data_path, 'small_segmentations.npy')).astype(np.float32)

            if saved_ground_truths_path is None:
                saved_ground_truths_path = run_save_path
            small_masks_gt = load(os.path.join(
                saved_ground_truths_path, 'small_masks_gt.npy'))
            labels_gt = load_json(os.path.join(
                saved_ground_truths_path, 'labels_gt.json'))

    scores = np.array(scores)
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores)
    scores = np.mean(scores, axis=0)

    scores_dict = {uid: score for uid, score in zip(uids, scores)}

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
    
    save(min_scores, os.path.join(run_save_path, 'min_scores.npy'))
    save(max_scores, os.path.join(run_save_path, 'max_scores.npy'))
    save(small_segmentations, os.path.join(run_save_path, 'small_segmentations.npy'))
    save(small_masks_gt, os.path.join(run_save_path, 'small_masks_gt.npy'))

    LOGGER.info("Computing evaluation metrics.")

    instance_results = patchcore3d.metrics.compute_imagewise_retrieval_metrics(
        scores, labels_gt
    )
    auroc = instance_results["auroc"]
    auprc = instance_results["auprc"]
    LOGGER.info(f'Instance AUROC: {auroc}, instance AUPRC: {auprc}, threshold: {instance_results["threshold"]}')
    
    # Compute PRO score & PW Auroc for all images
    pixel_scores = patchcore3d.metrics.compute_pixelwise_retrieval_metrics(
        small_segmentations, small_masks_gt
    )
    full_pixel_auroc = pixel_scores["auroc"]
    full_pixel_auprc = pixel_scores["auprc"]
    threshold = pixel_scores["optimal_threshold"]
    full_pixel_f1 = pixel_scores["f1"]
    small_dices = pixel_scores["dices"]
    small_dices_dict = {uid: dice for uid, dice in zip(uids, small_dices)}

    # Compute PRO score & PW Auroc only images with anomalies
    LOGGER.info("Computing evaluation metrics for single small images")
    sel_idxs = []
    small_single_anomaly_pixel_auroc = {}
    small_single_anomaly_dice = {}
    for i in tqdm.tqdm(range(len(small_masks_gt))):
        if labels_gt[i] == 1:
            sel_idxs.append(i)
            uid = uids[i]
            # compute aucs for small
            pixel_scores = patchcore3d.metrics.compute_pixelwise_retrieval_metrics(
                [small_segmentations[i]],
                [small_masks_gt[i]],
            )
            small_single_anomaly_pixel_auroc[uid] = pixel_scores["auroc"]
            small_single_anomaly_dice[uid] = pixel_scores["dices"][0]

    pixel_scores = patchcore3d.metrics.compute_pixelwise_retrieval_metrics(
        [small_segmentations[i] for i in sel_idxs],
        [small_masks_gt[i] for i in sel_idxs],
    )
    anomaly_pixel_auroc = pixel_scores["auroc"]
    anomaly_pixel_auprc = pixel_scores["auprc"]
    anomaly_pixel_f1 = pixel_scores["f1"]

    anomaly_uids = datasets["testing"].ood_dataset.ids

    save_json(uids, os.path.join(run_save_path, 'uids.json'))
    save_json(scores_dict, os.path.join(run_save_path, 'scores.json'))
    save_json(single_aucs_dict, os.path.join(
        run_save_path, 'single_anomaly_aucs.json'))
    save_json(small_single_anomaly_pixel_auroc, os.path.join(
        run_save_path, 'small_single_anomaly_pixel_auroc.json'))
    save_json(single_dices_dict, os.path.join(
        run_save_path, 'single_dices.json'))
    save_json(single_slice_dices_dict, os.path.join(
        run_save_path, 'single_slice_dices.json'))
    save_json(small_dices_dict, os.path.join(
        run_save_path, 'small_dices.json'))
    save_json(small_single_anomaly_dice, os.path.join(
        run_save_path, 'small_single_anomaly_dice.json'))
    save_json(top_anomaly_patches_dict, os.path.join(
        run_save_path, 'top_anomaly_patches_dict.json'))

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
            "small_mean_anomaly_dices": np.mean([small_dices[i] for i in sel_idxs]),
            "anomaly_patches_top": np.mean([top_anomaly_patches_dict[uid] for uid in anomaly_uids]),
            "mean_anomaly_pixel_auroc(single anomaly)": np.mean(list(single_aucs_dict.values())),
            "small_mean_anomaly_pixel_auroc(single anomaly)": np.mean(list(small_single_anomaly_pixel_auroc.values())),
            "single_dices(anomaly)": np.mean([single_dices_dict[uid] for uid in anomaly_uids]),
            "small_single_dices(anomaly)": np.mean(list(small_single_anomaly_dice.values())),
        }
    )

    for key, item in result_collect[-1].items():
        if key != "dataset_name":
            LOGGER.info("{0}: {1:3.5f}".format(key, item))

    LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore3d.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--pretrained_model_path", type=str, default=None)
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--anomaly_scorer_type", type=str, default="NN")
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    pretrained_model_path,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    anomaly_scorer_num_nn,
    anomaly_scorer_type,
    faiss_on_gpu,
    faiss_num_workers,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )

            nn_method = patchcore3d.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            backbone = patchcore3d.backbones.load(device, pretrained_model_path)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            patchcore_instance = patchcore3d.patchcore3d.PatchCore3D(device)

            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                anomaly_scorer_type=anomaly_scorer_type,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores[0]

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore3d.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore3d.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore3d.sampler.ApproximateGreedyCoresetSampler(percentage, device)
        elif name == "random":
            return patchcore3d.sampler.RandomSampler(percentage)

    return ("get_sampler", get_sampler)


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
