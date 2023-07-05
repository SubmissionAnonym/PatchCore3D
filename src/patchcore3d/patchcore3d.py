"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle
from typing import List
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from dpipe.io import save, load, save_json
from dpipe.im.box import mask2bounding_box

import patchcore3d
import patchcore3d.backbones
import patchcore3d.common
import patchcore3d.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore3D(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore3D, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_scorer_type="NN",
        anomaly_score_num_nn=1,
        featuresampler=patchcore3d.sampler.IdentitySampler(),
        nn_method=patchcore3d.common.FaissNN(False, 4),
        target_size_small=(70, 70, 45),
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker3D(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore3d.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions, patch_shapes = feature_aggregator.feature_dimensions(input_shape)
        self.patch_shapes = patch_shapes

        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore3d.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore3d.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        LOGGER.info(f'Anomaly_scorer_type: {anomaly_scorer_type}')
        if anomaly_scorer_type == "NN":
            self.anomaly_scorer = patchcore3d.common.NearestNeighbourScorer(
                n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
            )
        elif anomaly_scorer_type == "mahalanobis":
            self.anomaly_scorer = patchcore3d.common.MahalanobisDistanceScorer()

        self.anomaly_segmentor = patchcore3d.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-3:]
        )

        self.anomaly_segmentor_small = patchcore3d.common.RescaleSegmentor(
            device=self.device, target_size=target_size_small,
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]

        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]

        features = [x.reshape(-1, *x.shape[-4:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)

        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data, run_save_path, saved_features_path):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data, run_save_path, saved_features_path)

    def _fill_memory_bank(self, input_dataset, run_save_path, saved_features_path):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        if saved_features_path is None:
            features = []
            uids = []
            with tqdm.tqdm(
                range(len(input_dataset)), desc="Computing support features...", position=1, leave=False
            ) as data_iterator:
                for idx in data_iterator:
                    image = input_dataset[idx]
                    if isinstance(image, dict):
                        uids.append(image["uid"])
                        image = image["image"]
                    features.append(_image_to_features(image))
                    if (idx + 1) % 50 == 0 or (idx + 1) == len(data_iterator):
                        features = np.concatenate(features, axis=0)
                        fname = f'{idx + 1}_features.npy'
                        LOGGER.info(f'Saving features to {fname} (shape: {features.shape})')
                        save(features.astype(np.float16), os.path.join(run_save_path, fname))
                        features = []

            save_json(uids, os.path.join(run_save_path, 'uids.json'))
            features = np.vstack([np.load(os.path.join(run_save_path, embedding_file))
                                  for embedding_file in os.listdir(run_save_path) if embedding_file.endswith('_features.npy')])
            save(features.astype(np.float16), os.path.join(run_save_path, 'features_1024.npy'))
            features = features.astype(np.float32)

            for embedding_file in os.listdir(run_save_path):
                if embedding_file.endswith('_features.npy'):
                    os.remove(os.path.join(run_save_path, embedding_file))

        else:
            features = torch.from_numpy(load(saved_features_path).astype(np.float32))
            new_features = torch.zeros((features.shape[0], self.target_embed_dimension))
            # new_features = torch.from_numpy(new_features)

            batch_size = 2025
            for i in tqdm.tqdm(range(int(np.ceil(len(features) / batch_size)))):
                cur_features = self.forward_modules["preprocessing"](
                    features[i * batch_size: (i + 1) * batch_size][None, ...])
                new_features[i * batch_size: (
                    i + 1) * batch_size] = self.forward_modules["preadapt_aggregator"](cur_features)
            features = new_features.numpy().astype(np.float32)
            del new_features

        start = time.time()
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])
        LOGGER.info("Elapsed time (s): {:.2f}".format(time.time() - start))

    def predict(self, data, run_save_path, saved_test_features_path=None, save_segmentation_images=False):
        if isinstance(data, torch.utils.data.Dataset):
            return self._predict_dataset(data, saved_test_features_path=saved_test_features_path, 
                                         run_save_path=run_save_path, save_segmentation_images=save_segmentation_images)
        return self._predict(data)

    def _predict_dataset(self, dataset, run_save_path, saved_test_features_path=None, save_segmentation_images=False):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        top_anomaly_patches_dict = {}
        labels_gt = []

        small_masks_gt = []
        small_masks = []
        single_aucs_dict = {}
        single_dices_dict = {}
        single_slice_dices_dict = {}

        if save_segmentation_images:
            save_indices = [0, 1, 1000, 1001, 1002, 1003]
            image_save_path = os.path.join(
                run_save_path, "segmentation_images"
            )
            os.makedirs(image_save_path, exist_ok=True)

        pred_save_path = os.path.join(
            run_save_path, "predictions"
        )
        os.makedirs(pred_save_path, exist_ok=True)

        has_features = False
        if saved_test_features_path is not None:
            has_features = True
            dataset.return_images = False

        times = []
        mask_bboxes = {}
        with tqdm.tqdm(range(len(dataset)), desc="Inferring...") as data_iterator:
            for idx in data_iterator:
                
                image = dataset[idx]
                uid = image["uid"]
                                
                is_anomaly = image["is_anomaly"]
                labels_gt.extend([is_anomaly])
                mask_gt = image["mask"]
                small_mask_gt = F.interpolate(
                    mask_gt.float(), size=self.anomaly_segmentor_small.target_size, mode="trilinear", align_corners=False
                ) > 0.5
                small_masks_gt.extend(small_mask_gt.numpy().tolist())

                if has_features:
                    # load new chunk of features
                    chunk_size = 50
                    if idx % chunk_size == 0:
                        chunk_file = f'test_features_{idx}.npy'
                        chunk_features = load(os.path.join(
                            saved_test_features_path, chunk_file)).astype(np.float32)

                        # preproc features to target dim
                        chunk_features = torch.from_numpy(chunk_features)
                        len_chunk = len(chunk_features)
                        chunk_features = chunk_features.reshape(
                            -1, chunk_features.shape[-1])

                        new_features = np.zeros(
                            (chunk_features.shape[0], self.target_embed_dimension))
                        new_features = torch.from_numpy(new_features)

                        batch_size = 2025
                        for i in range(int(np.ceil(len(chunk_features) / batch_size))):
                            cur_features = self.forward_modules["preprocessing"](
                                chunk_features[i * batch_size: (i + 1) * batch_size][None, ...])
                            new_features[i * batch_size: (i + 1) * batch_size] = self.forward_modules["preadapt_aggregator"](cur_features)
                        chunk_features = new_features.numpy().astype(np.float32)
                        del new_features

                        chunk_features = chunk_features.reshape(
                            (len_chunk, -1, chunk_features.shape[-1]))

                    image = None
                    image_features = chunk_features[idx % chunk_size]  # [None, ...]
                else:
                    image = image["image"]
                    image_features = None

                _scores, _masks, _small_masks, _anomaly_patches, time_elapsed = self._predict(
                    image, features=image_features)
                times.append(time_elapsed)
                
                segmentations = np.array(_masks)
                if save_segmentation_images:
                    if idx in save_indices:
                        save(segmentations, os.path.join(image_save_path, f'{uid}.npy'))
                        save(_small_masks[0], os.path.join(image_save_path, f'small_{uid}.npy'))
                        save(small_mask_gt, os.path.join(image_save_path, f'small_mask_{uid}.npy'))

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
                
                _small_masks = np.array(_small_masks)
                min_ = load('../results_test/ixi_t2/patch3_sp1/data461_sample1_dim128_with_preds/min_scores.npy')[0, 0, 0, 0]
                max_ = load('../results_test/ixi_t2/patch3_sp1/data461_sample1_dim128_with_preds/max_scores.npy')[0, 0, 0, 0]
                threshold = 0.006821 * (max_ - min_) + min_
                if (_small_masks > threshold).sum() == 0:
                    mask_bbox = None
                else:
                    mask_bbox = mask2bounding_box(_small_masks > threshold)
                mask_bboxes[uid] = mask_bbox

                # save(np.array(_small_masks), os.path.join(pred_save_path, f'small_{uid}.npy'))

                if is_anomaly:
                    pixel_scores = patchcore3d.metrics.compute_pixelwise_retrieval_metrics(
                        segmentations,
                        mask_gt.numpy(),
                    )
                    single_aucs_dict[uid] = pixel_scores["auroc"]
                    single_dices_dict[uid] = pixel_scores["dices"][0]
                    single_slice_dices_dict[uid] = pixel_scores["slice_dices"]

                scores.append(_scores[0])
                top_anomaly_patches_dict[uid] = (_anomaly_patches[0] * mask_gt.numpy()).sum() > 1e-9
                small_masks.append(_small_masks[0])

        save(times, os.path.join(run_save_path, 'time.json'))
        dataset.return_images = True
        
        save(mask_bboxes, os.path.join(run_save_path, 'mask_bboxes.json'))
        
        return scores, labels_gt, top_anomaly_patches_dict, small_masks, small_masks_gt, single_aucs_dict, single_dices_dict, single_slice_dices_dict

    def _predict(self, images, features=None):
        """Infer score and mask for a batch of images."""
        start_time = time.time()
        if features is None:
            images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        with torch.no_grad():
            if features is None:
                features, patch_shapes = self._embed(images, provide_patch_shapes=True)
                features = np.asarray(features)
                batchsize = images.shape[0]
            else:
                patch_shapes = self.patch_shapes
                batchsize = 1

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )

            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(
                batchsize, scales[0], scales[1], scales[2])

            image_scores, highest_anomalies_masks = self.patch_maker.score(image_scores, return_max_mask=True,
                                                                           patch_scores=patch_scores,
                                                                           input_shape=self.anomaly_segmentor.target_size)

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            time_elapsed = time.time() - start_time
            small_masks = self.anomaly_segmentor_small.convert_to_segmentation(
                patch_scores)

        return [sc for sc in image_scores], [m for m in masks], [m for m in small_masks], [an_mask for an_mask in highest_anomalies_masks], time_elapsed

    def save_test_features(self, input_dataset, run_save_path):
        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        chunk_size = 50
        with tqdm.tqdm(range(len(input_dataset)), desc="Saving test features...") as data_iterator:
            for idx in data_iterator:
                image = input_dataset[idx]
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
                if (idx + 1) % chunk_size == 0 or (idx + 1) == len(input_dataset):
                    chunk_file = f'test_features_{idx // chunk_size * chunk_size}.npy'
                    # features = np.concatenate(features, axis=0)
                    features = np.stack(features)
                    save(features.astype(np.float16), os.path.join(run_save_path, chunk_file))
                    LOGGER.info('Saving test features with shape {features.shape}')
                    features = []

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore3d.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore3d.backbones.load(device)
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker3D:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            features: [torch.Tensor, bs x c x w x h x d]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride * d//stride, c, patchsize,
            patchsize, patchsize]
        """
        padding = int((self.patchsize - 1) / 2)

        unfolded_features = self.im2col3d(
            input=features,
            kernel_size=[self.patchsize, self.patchsize, self.patchsize],
            dilation=[1, 1, 1],
            padding=[padding, padding, padding],
            stride=[self.stride, self.stride, self.stride],
        )

        number_of_total_patches = []
        for s in features.shape[-3:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 5, 1, 2, 3, 4)
        # unfolded_features = unfolded_features.reshape(-1, *unfolded_features.shape[2:])

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def im2col3d(
        self,
        input: torch.Tensor,
        kernel_size: List[int],
        dilation: List[int],
        padding: List[int],
        stride: List[int],
    ) -> torch.Tensor:
        ''' 
        Function to perform unfolding of a 5D tensor (not implemented in pytorch) 
        '''

        batch_dim = input.size(0)
        channel_dim = input.size(1)
        input_h = input.size(2)
        input_w = input.size(3)
        input_d = input.size(4)

        stride_h, stride_w, stride_d = stride[0], stride[1], stride[2]
        padding_h, padding_w, padding_d = padding[0], padding[1], padding[2]
        dilation_h, dilation_w, dilation_d = dilation[0], dilation[1], dilation[2]
        kernel_h, kernel_w, kernel_d = kernel_size[0], kernel_size[1], kernel_size[2]

        def _get_im2col_indices_along_dim(
            input_d, kernel_d, dilation_d, padding_d, stride_d
        ):
            blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)

            # Stride kernel over input and find starting indices along dim d
            blocks_d_indices = torch.arange(
                0, blocks_d, stride_d, dtype=torch.int64, device=input.device
            ).unsqueeze(0)
            num_blocks = (blocks_d - 1) // stride_d + 1

            # Apply dilation on kernel and find its indices along dim d
            kernel_grid = torch.arange(
                0, kernel_d * dilation_d, dilation_d, dtype=torch.int64, device=input.device
            ).unsqueeze(-1)

            # Broadcast and add kernel staring positions (indices) with
            # kernel_grid along dim d, to get block indices along dim d
            block_mask = blocks_d_indices + kernel_grid

            return block_mask, num_blocks

        blocks_row_indices, num_blocks_row = _get_im2col_indices_along_dim(
            input_h, kernel_h, dilation_h, padding_h, stride_h
        )
        blocks_col_indices, num_blocks_col = _get_im2col_indices_along_dim(
            input_w, kernel_w, dilation_w, padding_w, stride_w
        )
        blocks_depth_indices, num_blocks_depth = _get_im2col_indices_along_dim(
            input_d, kernel_d, dilation_d, padding_d, stride_d
        )

        padded_input = F.pad(input, (padding_d, padding_d,
                             padding_w, padding_w, padding_h, padding_h))

        blocks_row_indices = blocks_row_indices.unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        blocks_col_indices = blocks_col_indices.unsqueeze(-1).unsqueeze(-1)
        output = padded_input[:, :, blocks_row_indices,
                              blocks_col_indices, blocks_depth_indices]
        output = output.permute(0, 1, 2, 4, 6, 3, 5, 7)
        return output.reshape(
            batch_dim, channel_dim * kernel_h * kernel_w *
            kernel_d, num_blocks_row * num_blocks_col * num_blocks_depth
        )

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x, return_max_mask=False, patch_scores=None, input_shape=None):
        was_numpy = False

        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
            patch_scores = torch.from_numpy(patch_scores)

        one_hot_mask = torch.zeros_like(patch_scores)

        while x.ndim > 1:
            x = torch.max(x, dim=-1).values

        if return_max_mask:
            flat_indices = torch.stack([torch.argmax(patch_scores[ind])
                                       for ind in range(patch_scores.shape[0])])

            one_hot_mask.reshape(
                one_hot_mask.shape[0], -1)[range(one_hot_mask.shape[0]), flat_indices] = 1
            one_hot_mask = torch.stack([F.interpolate(one_hot_mask[ind][None, None, ...],
                                                      size=input_shape, mode="trilinear", align_corners=False)
                                        for ind in range(one_hot_mask.shape[0])])[:, 0, 0]
            one_hot_mask = one_hot_mask.bool()

        if return_max_mask:
            if was_numpy:
                return x.numpy(), one_hot_mask.numpy()
            return x, one_hot_mask
        if was_numpy:
            return x.numpy()
        return x
