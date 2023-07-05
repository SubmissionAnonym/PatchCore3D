import time
from tqdm import tqdm

import numpy as np


def get_histogram_features(dataset, n_bins=128):
    
    start_time = time.time()
    features = []
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        if isinstance(item, dict):
            image = item["image"].numpy()

            # scale to 0-1
            image -= np.min(image)
            image /= np.max(image)

            # compute histogram
            histogram, _ = np.histogram(image, bins=n_bins, range=(0, 1), density=True)
            features.append(histogram)

    features = np.array(features)
    print('Training features time', time.time() - start_time)
    print(features.shape)
    return features


def get_histogram_ood_scores(dataset, features):

    start_time = time.time()
    mean_mahalanobis = features.mean(axis=0)
    print('mean', mean_mahalanobis.shape)

    cov_mahal = np.zeros((features.shape[1], features.shape[1]))
    for train_sample in features:
        cov_mahal += np.outer(train_sample - mean_mahalanobis, train_sample - mean_mahalanobis)

    cov_mahal /= len(features)
    inv_covariance_mahalanobis = np.linalg.inv(cov_mahal)
    print("inv_covariance_mahalanobis", inv_covariance_mahalanobis.shape)
    print('Training statistics time', time.time() - start_time)

    start_time = time.time()
    n_bins = features.shape[-1]
    labels_gt = []
    scores = []

    test_features = []
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        if isinstance(item, dict):
            image = item["image"].numpy()
            labels_gt.append(item["is_anomaly"])

            # scale to 0-1
            image -= np.min(image)
            image /= np.max(image)

            # compute histogram
            histogram, _ = np.histogram(image, bins=n_bins, range=(0, 1), density=True)
            test_features.append(histogram)
            anomaly_score = (histogram - mean_mahalanobis) @ inv_covariance_mahalanobis @ \
                            (histogram - mean_mahalanobis)
            scores.append(anomaly_score)

    test_features = np.array(test_features)
    print('Test time', time.time() - start_time)
    print(test_features.shape)

    print(len(scores), len(labels_gt))
    return np.array(scores), np.array(labels_gt), test_features


def get_histogram_ood_scores_with_features(features, test_features):

    mean_mahalanobis = features.mean(axis=0)
    print('mean', mean_mahalanobis.shape)

    cov_mahal = np.zeros((features.shape[1], features.shape[1]))
    for train_sample in features:
        cov_mahal += np.outer(train_sample - mean_mahalanobis, train_sample - mean_mahalanobis)

    cov_mahal /= len(features)
    inv_covariance_mahalanobis = np.linalg.inv(cov_mahal)
    print("inv_covariance_mahalanobis", inv_covariance_mahalanobis.shape)

    labels_gt = []
    scores = []

    for histogram in tqdm(test_features):
        anomaly_score = (histogram - mean_mahalanobis) @ inv_covariance_mahalanobis @ \
                        (histogram - mean_mahalanobis)
        scores.append(anomaly_score)

    print(len(scores), len(labels_gt))
    return np.array(scores), np.array(labels_gt)
