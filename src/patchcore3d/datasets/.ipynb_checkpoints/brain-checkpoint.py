from enum import Enum

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from connectome import Chain, Filter
from patchcore3d.datasets.brats import BraTS2021
from patchcore3d.datasets.ixi import IXI
from patchcore3d.datasets.utils import ScaleMRI


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class BrainMRIDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Brain MRI.
    """

    def __init__(
        self,
        ixi_data_path,
        ood_data_path=None,
        imagesize=(240, 240, 155),
        split=DatasetSplit.TRAIN,
        test_size=0.2,
        return_images=True,
        tumor_type='whole_tumor',
    ):
        """
        Args:
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.split = split
        self.return_images = return_images
        self.tumor_type = tumor_type
        print('Dataset tumor type:', self.tumor_type)

        self.id_dataset = IXI(ixi_data_path)

        self.imagesize = imagesize
        self.name = 'brainMRI'

        folds = [self.id_dataset.fold(uid) for uid in self.id_dataset.ids]
        train_ids, test_ids = train_test_split(self.id_dataset.ids, test_size=test_size,
                                               random_state=0, stratify=folds)
        
        if self.split == DatasetSplit.TRAIN:
            self.id_dataset = Chain(
                self.id_dataset,
                Filter(lambda id: id in set(train_ids)),
                ScaleMRI(),
            )
            self.all_ids = list(self.id_dataset.ids)

        else:
            self.id_dataset = Chain(
                self.id_dataset,
                Filter(lambda id: id in set(test_ids)),
                ScaleMRI(),
            )
            self.ood_dataset = Chain(
                BraTS2021(ood_data_path),
                ScaleMRI(),
            )

            id_test_ids = self.id_dataset.ids
            ood_test_ids = self.ood_dataset.ids

            self.id_dataset = self.id_dataset >> Filter(
                lambda id: id in set(id_test_ids))
            self.ood_dataset = self.ood_dataset >> Filter(
                lambda id: id in set(ood_test_ids))
            self.all_ids = list(self.id_dataset.ids) + list(self.ood_dataset.ids)

    def __getitem__(self, idx):
        uid = self.all_ids[idx]
        is_anomaly = int(uid.startswith('BraTS2021_'))

        if is_anomaly:
            if self.return_images:
                image = self.ood_dataset.image(uid)
            if self.tumor_type == 'whole_tumor':
                load_mask = self.ood_dataset.mask
            elif self.tumor_type == 'tumor_core':
                load_mask = self.ood_dataset.tumor_core
            elif self.tumor_type == 'enhancing_tumor':
                load_mask = self.ood_dataset.enhancing_tumor
            else:
                # incorrect tumor_type
                load_mask = None
                
            mask = load_mask(uid)[None, :]
        else:
            if self.return_images:
                image = self.id_dataset.image(uid)
            mask = np.zeros([1, *self.imagesize])

        if self.return_images:
            return {
                "image": torch.from_numpy(image[None, None, ...]),
                "mask": torch.from_numpy(mask)[None, ...],
                "is_anomaly": is_anomaly,
                "uid": uid,
            }
        return {
            "mask": torch.from_numpy(mask)[None, ...],
            "is_anomaly": is_anomaly,
            "uid": uid,
        }

    def __len__(self):
        return len(self.all_ids)
