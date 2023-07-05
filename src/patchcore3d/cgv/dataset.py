import os

from torch.utils.data import Dataset
from deli import load
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from utils import *


class MOODDataset(Dataset):
    """MICCAI 2021 MOOD Challange Dataset""" 
    def __init__(self,
                 dataset_path,
                 subset='train',
                 category='brain',
                 pretrain=False):

        self.idx = 1
        
        self.dataset_path = os.path.join(dataset_path)
        self.subset = subset
        self.category = category
        self.pretrain = pretrain
        
        train_ids = load('train_ids.json')
        if self.subset == 'train':
            subset_ids = train_ids[:-5]
        else: 
            # self.subset == 'valid'
            subset_ids = train_ids[-5:]
        self.img_names = [f'{f}_t2.nii.gz' for f in subset_ids]
        self.img_names.sort()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = self.GetImage(self.img_names[idx])
                   
        if self.category == 'brain':
            if self.pretrain :
                img_patch, imgGT_patch, orig_patch = recon_task(img, self.category)
            else :
                img_patch, img_2d, imgGT_patch, img_cls, aug_patch, aug_2d, augGT_patch, aug_cls, augGT_2d  = extract_brain_patch(img,                                                
                                                                self.idx,
                                                                self.subset)
        elif self.category == 'abdom':
            if self.pretrain :
                img_patch, imgGT_patch, orig_patch = recon_task(img, self.category)
            else:
                img_patch, img_2d, imgGT_patch, img_cls, aug_patch, aug_2d, augGT_patch, aug_cls, augGT_2d  = extract_abdom_patch(img, 
                                                                self.idx,
                                                                self.subset)
        if self.pretrain:
            img_patch[img_patch<0.0] = 0.0
            img_patch[img_patch>1.0] = 1.0
            self.idx += 1
            return img_patch, imgGT_patch, orig_patch
        else :
            img_patch[img_patch<0.0] = 0.0
            img_patch[img_patch>1.0] = 1.0
            self.idx += 1
            aug_patch[aug_patch<0.0] = 0.0
            aug_patch[aug_patch>1.0] = 1.0

            augGT_2d_cls = self.MakeCls(augGT_2d)
            augGT_2d[augGT_2d<0.0] = 0.0
            augGT_2d[augGT_2d >1.0] = 1.0

            return img_patch, img_2d, imgGT_patch, img_cls, aug_patch, aug_2d, augGT_patch, aug_cls, augGT_2d_cls, augGT_2d.copy()
     
    def GetImage(self, image_name):
        image_path = os.path.join(self.dataset_path, image_name)
        img  = nib.load(image_path)
        img_data = img.get_fdata()
        img_data = np.array(img_data, dtype = np.float32)
        target_shape = [128 / img_data.shape[0], 128 / img_data.shape[1], 
                            128 / img_data.shape[2]]
        img_data = zoom(img_data, target_shape, order = 0)
        return img_data

    def MakeCls(self, gt):
        aug = np.count_nonzero(gt, axis=(1,2))
        aug = np.array(aug, dtype = np.float16)
        aug = np.reshape(aug, (-1))

        for i in range (aug.shape[0]):
            if aug[i] > 0.0 and aug[i] <= 6.0:
                aug[i] = 0.0
            elif aug[i] > 3.0 and aug[i] <= 12.0 :
                aug[i] = 0.75
            elif aug[i] > 6.0 :
                aug[i] = 1.0

        return aug
