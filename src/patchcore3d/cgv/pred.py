import os
import argparse
import numpy as np
import nibabel as nib

from deli import load, save
from torch import nn
from tqdm import tqdm

from networks import Unet_brain
from utils import *

def inference_brain(model, device, test_img, output_path, eval):
    model.eval()
    patch_size = 64

    img_orig = nib.load(test_img)
    target_size = patch_size
    resized_img, orientation = resize_brain_nib(img_orig, target_size) 
        
    # img : B x C x X x Y x Z
    img = np.array(resized_img, dtype = np.float32)
    img= np.expand_dims(img, axis = 0)
    img = np.expand_dims(img, axis = 0)
    img = torch.tensor(img).to(device)
    
    with torch.no_grad():
        S_img, pred_img_cls, _ = model(img.float())
        pred_img_cls = torch.flatten(pred_img_cls)

    S_img = S_img.to('cpu').numpy()
    pred_img_cls = pred_img_cls.to('cpu').numpy()

    cls_img = pred_img_cls[0]

    test_img_name = os.path.split(test_img)[1]
    if eval == 'sample':
        f = open(os.path.join(output_path, eval, test_img_name + '.txt'), 'w')
        f.write(str(cls_img))
        f.close()
        
    elif eval == 'pixel':
        pixel = recon(S_img[0,0,:,:,:], img_orig.shape, orientation)
        
        selected_uids = ['IXI012-HH-1211', 'IXI025-Guys-0852', 'BraTS2021_01300',
                         'BraTS2021_01301', 'BraTS2021_01302', 'BraTS2021_01303']
        
        if test_img_name.split('_t2')[0] in selected_uids:
            save(pixel, os.path.join(output_path, 'full_segmentations', 
                 f"{test_img_name.split('_t2')[0]}.npy.gz"), compression=1)
            
        small_segmentation = F.interpolate(
            torch.from_numpy(pixel)[None, None, :], size=(70, 70, 45), mode="trilinear", align_corners=False
        )[0, 0]
        save(small_segmentation.numpy(), os.path.join(output_path, eval, 
                                                      f"{test_img_name.split('_t2')[0]}.npy.gz"), compression=1)
        # nii_img = nib.Nifti1Image(pixel, img_orig.affine, img_orig.header)
        # nib.save(nii_img, os.path.join(output_path, eval, test_img_name))

def inference_abdom(model, device, test_img, input_path, output_path, eval):
    model.eval()
    patch_size = 64
    stride = int(patch_size/2)

    img_orig = nib.load(os.path.join(input_path, test_img))
    target_size = patch_size * 2
    resized_img, orientation = resize_abdom(img_orig, target_size)   

    img = np.array(resized_img, dtype = np.float32)
    img = np.expand_dims(img, axis = 0)
    img = np.expand_dims(img, axis = 0)
    img = torch.tensor(img)

    # img : B x C x X x Y x Z
    S_img = np.zeros_like(img)
    cls_img = 0
    overlapped_cnt = np.zeros_like(img)
    
    for position in range (27):
        x = position // 9
        y = (position - 9*x) // 3
        z = (position - 9*x - 3*y) % 3
        x = stride * x
        y = stride * y
        z = stride * z

        img_patch = img[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size]
        overlapped_cnt[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size] += 1.0
        
        label = np.zeros((1, 27, patch_size, patch_size, patch_size))
        label[:, position, :, :, :] = 1
        img_patch = np.concatenate((img_patch, label), axis=1)
        img_patch = torch.from_numpy(img_patch).to(device)
        
        with torch.no_grad():
            S_img_patch, pred_img_cls, _ = model(img_patch.float())
            pred_img_cls = torch.flatten(pred_img_cls)

        S_img_patch = S_img_patch.to('cpu').numpy()
        pred_img_cls = pred_img_cls.to('cpu').numpy()

        #S_img[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size] += S_img_patch
        S_img_temp = S_img[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size]
        S_img[:,:,x: x + patch_size, y: y + patch_size, z: z + patch_size] = np.maximum(S_img_temp, S_img_patch)
        
        if cls_img  < pred_img_cls[0]:
            cls_img = pred_img_cls[0]

    #S_img = S_img / overlapped_cnt


    if eval == 'sample':
        f = open(os.path.join(output_path, test_img + '.txt'), 'w')
        f.write(str(cls_img))
        f.close()
        
    elif eval == 'pixel':
        pixel = recon(S_img[0,0,:,:,:], img_orig.shape, orientation)
        nii_img = nib.Nifti1Image(pixel, img_orig.affine, img_orig.header)
        nib.save(nii_img, os.path.join(output_path, test_img))



if __name__ == "__main__":
    # Training args
    parser = argparse.ArgumentParser(description='#### MOOD-CGV TEAM ####')
    parser.add_argument('--e', type=str, default='sample', metavar='N',
                        help='choose the evaluation between sample and pixel')
    parser.add_argument('--ixi_path', type=str, metavar='N',
                        help='ixi directory')
    parser.add_argument('--brats_path', type=str, metavar='N',
                        help='brats directory')
    parser.add_argument('--test_ids_path', type=str, metavar='N',
                        help='test ids file (test_ids.json)')
    parser.add_argument('--model_path', type=str, default='/mnt/model', metavar='N',
                        help='Trained model path')
    parser.add_argument('--o', type=str, default='/mnt/pred', metavar='N',
                        help='Output directory')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (default 0)')
    args = parser.parse_args()

    # Use GPU if it is available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    model = Unet_brain(num_channels=64).to(device)
    model = nn.DataParallel(model, device_ids=[device])
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'latest_checkpoints_all.pth'), 
                                        map_location=device)['model_state_dict'])

    ixi_test_ids = load(args.test_ids_file)
    test_imgs = [os.path.join(args.ixi_path, f'{fname}_t2.nii.gz') for fname in ixi_test_ids if fname.startswith('IXI')] + [os.path.join(args.brats_path, f'{fname}/{fname}_t2.nii.gz') for fname in ixi_test_ids if fname.startswith('BraTS2021_')]

    for test_img in tqdm(test_imgs):
        inference_brain(model, device, test_img, args.o, args.e)
    