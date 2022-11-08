"""Anomaly metrics."""
import numpy as np
from sklearn import metrics

from dpipe.im.metrics import dice_score


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    precision, recall, thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auprc = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    return {"auroc": auroc, "auprc": auprc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    # fpr, tpr, thresholds = metrics.roc_curve(
    #     flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    # )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auprc = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    dices = []
    slice_dices = []
    for segm, mask_gt in zip(anomaly_segmentations, ground_truth_masks):
        dices.append(dice_score(segm >= optimal_threshold, mask_gt.astype(bool)[0]))
        for slice_ind in range(segm.shape[-1]):
            slice_dices.append(dice_score(
                segm[..., slice_ind] >= optimal_threshold, mask_gt.astype(bool)[0, ..., slice_ind]))
        slice_dices.append(np.mean(slice_dices))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": np.max(F1_scores),
        # "fpr": fpr,
        # "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "dices": dices,
        "slice_dices": slice_dices,
    }
