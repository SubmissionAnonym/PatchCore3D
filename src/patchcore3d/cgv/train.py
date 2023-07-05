import os
import time
import argparse
import numpy as np
import logging
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Segloss, ContrastiveLoss

from dataset import MOODDataset
from networks import Unet_brain, resnet50

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)


def pretask(model, device, scatter_patch, orig_patch, optimizer, scheduler, patch_size, iteration, test=False):
    if test:
        model.eval()
    else:
        model.train()
        optimizer.zero_grad()

    correct = 0.0
    scatter_patch, orig_patch = scatter_patch.to(device), orig_patch.to(device)
    
    if test:
        with torch.no_grad():
            pred_seg, _, _ = model(scatter_patch.float())
    else:
        pred_seg, _, _ = model(scatter_patch.float())

    # calculate the loss
    BCELoss = nn.BCELoss()
    reconLoss = BCELoss(pred_seg, orig_patch.float())
    total_loss = reconLoss
    if iteration % 50 == 0:
        logging.info(f"Recon Loss: {total_loss.item():.8f}")

    if not test:
        # optimize the parameters
        total_loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss.item()


def train(model, pre_model, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls, optimizer, scheduler, patch_size, beta, pre_optimizer, pre_scheduler, rotImage, rotAug, pre_augGT, lamda, preaugGT_patch, iteration):
    model.train()
    pre_model.train()
    optimizer.zero_grad()
    pre_optimizer.zero_grad()

    correct = 0.0
    aug_index = np.transpose(np.nonzero(augGT_patch[0, :, :, :, :]))
    img_patch, imgGT_patch, img_cls, rotImage = img_patch.to(device), imgGT_patch.to(device), img_cls.to(device), rotImage.to(device)
    aug_patch, augGT_patch, aug_cls, rotAug = aug_patch.to(device), augGT_patch.to(device), aug_cls.to(device), rotAug.to(device)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name + str(input[0].device)] = output
        return hook
    
    def merge_activation(name, device):
        activation_list = []
        for key in activation.keys():
            if name in key:
                activation_list.append(activation[key].to(device))
        activation[name] = torch.cat(activation_list, 0)

    latent_size = 4
    batch_size = img_patch.shape[0]


    #Normal Image: 3D Network
    pred_normal_seg, pred_normal_cls, mood_latent_img = model(img_patch.float())
    #Normal Image: 2D Network
    sliceImgs = torch.reshape(rotImage, (batch_size * patch_size * 2, rotImage.shape[1], patch_size*2, patch_size*2))
    outImg, latent_img = pre_model(sliceImgs.float())
    latent_img = torch.reshape(latent_img, (batch_size, patch_size * 2, latent_img.shape[1], latent_size, latent_size))
    latent_img = torch.transpose(latent_img, 1,2)
    latent_img = latent_img.reshape(-1, 512, latent_img.shape[1]//512, 4, latent_img.shape[2]//4, latent_size, latent_size).permute(0,1,3,2,4,5,6)
    latent_img = latent_img.reshape(-1, latent_img.shape[3] * latent_img.shape[4], latent_size * latent_size).permute(0,2,1)
    latent_img = latent_img.reshape(-1, latent_img.shape[2])
    reshape_latent_img = torch.amax(latent_img, dim=1).reshape(-1, 512, 4, latent_size, latent_size)


    #Abnormal Image: 3D Network
    pred_aug_seg, pred_aug_cls, mood_latent_aug = model(aug_patch.float())
    #Abnormal Image: 2D Network
    sliceAugs = torch.reshape(rotAug, (batch_size * patch_size * 2, rotAug.shape[1], patch_size*2, patch_size*2))
    outAug, latent_aug = pre_model(sliceAugs.float())
    latent_aug = torch.reshape(latent_aug, (batch_size, patch_size * 2, latent_aug.shape[1], latent_size, latent_size))
    latent_aug = torch.transpose(latent_aug, 1,2)
    latent_aug = latent_aug.reshape(-1, 512, latent_aug.shape[1]//512, 4, latent_aug.shape[2]//4, latent_size, latent_size).permute(0,1,3,2,4,5,6)
    latent_aug = latent_aug.reshape(-1, latent_aug.shape[3] * latent_aug.shape[4], latent_size * latent_size).permute(0,2,1)
    latent_aug = latent_aug.reshape(-1, latent_aug.shape[2])
    reshape_latent_aug = torch.amax(latent_aug, dim=1).reshape(-1, 512, 4, latent_size, latent_size)


    #Similarity Loss
    rep_2d = torch.cat([reshape_latent_img, reshape_latent_aug, reshape_latent_img, reshape_latent_aug], dim = 0)
    rep_3d = torch.cat([mood_latent_img, mood_latent_aug, mood_latent_aug, mood_latent_img], dim = 0)
    rep_2d, rep_3d = torch.reshape(rep_2d, (batch_size*4, -1)), torch.reshape(rep_3d, (batch_size*4, -1))
    SimLoss_ = ContrastiveLoss(rep_2d.to(device), rep_3d.to(device), batch_size*4)
    SimLoss = SimLoss_ * (100 - lamda) / 8000.0

    # calculate the loss
    bceLoss = nn.BCELoss()

    pred_normal_cls = torch.flatten(pred_normal_cls)
    pred_aug_cls = torch.flatten(pred_aug_cls)
    pred_2d_normal_cls = torch.flatten(outImg)
    pred_2d_aug_cls = torch.flatten(outAug)

    pre_augGT = torch.flatten(pre_augGT.float())
    pre_imgGT = img_cls.repeat(patch_size)

    normal_seg_loss = bceLoss(pred_normal_seg, imgGT_patch.float())
    aug_seg_loss = bceLoss(pred_aug_seg, augGT_patch.float())
    seg_loss = normal_seg_loss + aug_seg_loss
    normal_cls_loss = bceLoss(pred_normal_cls, img_cls.float())
    aug_cls_loss = bceLoss(pred_aug_cls, aug_cls.float())
    cls_loss = normal_cls_loss + aug_cls_loss

    countAug = np.count_nonzero(pre_augGT)
    pos_weight = torch.ones([1]) * ((pre_augGT.shape[0] - countAug) / (countAug))
    bceLossW = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to('cuda:0'))
    aug_cls_loss_2d = bceLossW(pred_2d_aug_cls.to('cuda:0'), pre_augGT.to('cuda:0'))
    cls_loss_2d = aug_cls_loss_2d


    # aviod the trivial solution
    FP, FN = Segloss(pred_aug_seg, augGT_patch.float(), patch_size)

    total_loss = seg_loss + (FN + FP) * beta + cls_loss + cls_loss_2d + SimLoss

    if iteration % 50 == 0:
        logging.info(f"Cls BCE Loss: {cls_loss.item():.6f} | Seg BCE Loss: {seg_loss.item():.6f} | FN+FP loss {(FN + FP).item() * beta:.6f}  | Sim loss {SimLoss.item():.6f} | 2d Cls loss {cls_loss_2d.item() :.6f} | Pos weight {pos_weight.item() :.4f}")

    for i in range (pred_normal_cls.shape[0]):
        if pred_normal_cls[i].round() == img_cls[i]:
            correct += 0.5
        if pred_aug_cls[i].round() == aug_cls[i]:
            correct += 0.5
  
    # optimize the parameters
    total_loss.backward()
    optimizer.step()
    pre_optimizer.step()
    scheduler.step()
    pre_scheduler.step()

    return total_loss.item(), correct

def test(model, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls):
    torch.cuda.empty_cache()
    model.eval()

    correct = 0.0
    img_patch, imgGT_patch, img_cls = img_patch.to(device), imgGT_patch.to(device), img_cls.to(device)
    aug_patch, augGT_patch, aug_cls = aug_patch.to(device), augGT_patch.to(device), aug_cls.to(device)

    with torch.no_grad():
        # pred_normal_seg, pred_normal_cls = model(img_patch.float())
        # pred_aug_seg, pred_aug_cls = model(aug_patch.float())
        pred_normal_seg, pred_normal_cls, mood_latent_img = model(img_patch.float())
        pred_aug_seg, pred_aug_cls, mood_latent_aug = model(aug_patch.float())
        pred_normal_cls = torch.flatten(pred_normal_cls)
        pred_aug_cls = torch.flatten(pred_aug_cls)

    # calculate the loss
    bceLoss = nn.BCELoss()
    normal_seg_loss = bceLoss(pred_normal_seg, imgGT_patch.float())
    aug_seg_loss = bceLoss(pred_aug_seg, augGT_patch.float())
    seg_loss = normal_seg_loss + aug_seg_loss

    normal_cls_loss = bceLoss(pred_normal_cls, img_cls.float())
    aug_cls_loss = bceLoss(pred_aug_cls, aug_cls.float())
    cls_loss = normal_cls_loss + aug_cls_loss

    total_loss = seg_loss + cls_loss
    logging.info(f"*TEST* Cls BCE Loss: {cls_loss.item():.6f} | Seg BCE Loss: {seg_loss.item():.6f}")
    
    
    for i in range (pred_normal_cls.shape[0]):
        if pred_normal_cls[i].round() == img_cls[i]:
            correct += 0.5
        if pred_aug_cls[i].round() == aug_cls[i]:
            correct += 0.5

    torch.cuda.empty_cache()
    return seg_loss.item(), cls_loss.item(), total_loss.item(), correct
    

if __name__ == "__main__":
    # Version of Pytorch
    logging.info("Pytorch Version:%s" % torch.__version__)

    # Training args
    """##############################################################################################################################
    Brain = Put the resized into 128 x 128 x 128 images (2D Network: 128 x 128 images, 3D Network: 64 x 64 x 64 with patch size 64)
    Abdom = Put the resized into 256 x 256 x 256 images (2D Network: 256 x 256 images, 3D Network: 128 x 128 x 128 volumes with patch size 64)
    ###################################################################################################################################################"""

    parser = argparse.ArgumentParser(description='MOOD training')
    parser.add_argument('--dataset', type=str, default='/root/MOOD2021/dataset/brain_train',
                        help='path of processed dataset')
    parser.add_argument('--weight', type=str, default='./weights',
                        help='path of the weights folder')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epoches', type=int, default=100, metavar='N',
                        help='number of epoches to train (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of iterations to log (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', metavar='N',
                        help='Select the GPU (default 0)')
    parser.add_argument('--category', type=str, default='brain', metavar='N',
                        help='Select the category brain or abdom (defualt brain)')
    parser.add_argument('--patch-size', type=int, default=128, metavar='N',
                        help='patch-size (default=128')
    parser.add_argument('--pretrain', action='store_true',
                        help='pretraining step')
    parser.add_argument('--pretrained_checkpoint_path', type=str, default=None,
                        help='pretraining step')
    parser.add_argument('--identifier', type=str, default='all', metavar='N',
                        help='model identifier')
    args = parser.parse_args()
    
    os.makedirs(args.checkpoints, exist_ok=False)
    os.makedirs(args.weight, exist_ok=False)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set random seed for reproducibility
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Torch use the device: %s" % device)

    if args.category == 'brain':
        model = Unet_brain(num_channels=64).to(device)
        model_2d = resnet50().to(device)
        patch_size = 64
    elif args.category == 'abdom':
        model = Unet_abdom(num_channels=64).to(device)
        model_2d = resnet50().to(device)
        patch_size = 64
    else:
        logging.info("Choose the Correct Category")

    model = nn.DataParallel(model)
    model_2d = nn.DataParallel(model_2d)
    
    if args.pretrained_checkpoint_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.pretrained_checkpoint_path, 'latest_checkpoints_all.pth'), 
                                         map_location=device)['model_state_dict'])
        logging.info("Checkpoint loaded")
 
    batch_size = args.batch_size

    if args.pretrain:
        train_dataset = MOODDataset(args.dataset, subset='train', category = args.category, pretrain = args.pretrain)
        test_dataset = MOODDataset(args.dataset, subset='valid', category = args.category, pretrain = args.pretrain)
    else:
        train_dataset = MOODDataset(args.dataset, subset='train', category = args.category)
        test_dataset = MOODDataset(args.dataset, subset='valid', category = args.category)
   
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=generator)
    train_iterator = iter(train_loader)

    total_iteration = args.epoches * len(train_loader)
    train_interval = args.log_interval * len(train_loader) 

    logging.info(f"total iter: {total_iteration}")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_2d = optim.Adam(model_2d.parameters(), lr=args.lr)

    # iteration = 1
    best_train_loss, best_test_loss = float('inf'), float('inf')

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*10, steps_per_epoch=len(train_loader), epochs=args.epoches, anneal_strategy='linear')
    scheduler_2d = optim.lr_scheduler.OneCycleLR(optimizer_2d, max_lr=args.lr*10, steps_per_epoch=len(train_loader), epochs=args.epoches, anneal_strategy='linear')
 
    epoch_train_loss = []
    epoch_test_loss = []
    epoch_test_segloss = []
    epoch_test_clsloss = []
    correct_train_count = 0
    correct_test_count = 0
    epoch_train_accuracy = 0.
    epoch_test_accuracy = 0.
    v_clsloss = 0.
    start_time = time.time()

    # Seed initializaiton for patch reproducibility
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # while iteration <= total_iteration:
    for iteration in tqdm(range(total_iteration)):
        # 0 epoch: 1.0 ~ max epoch* 0.3: 0.05 (linear)
        beta = 100 * (1.0 - iteration / (args.epoches * 0.3 * len(train_loader)))
        beta = 0.05 if beta < 0 else beta
        lamda = int(iteration / args.epoches)

        try:
            if args.pretrain:
                img_patch, imgGT_patch, orig_patch = next(train_iterator)
            else:
                # Samples the batch
                img_patch, img_2d, imgGT_patch, img_cls, aug_patch, aug_2d, augGT_patch, aug_cls, augGT_2d_cls, augGT_2d = next(train_iterator)

        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            train_iterator = iter(train_loader)
            if args.pretrain:
                img_patch, imgGT_patch, orig_patch = next(train_iterator)
            else:
                img_patch, img_2d, imgGT_patch, img_cls, aug_patch, aug_2d, augGT_patch, aug_cls, augGT_2d_cls, augGT_2d = next(train_iterator)
    
        if args.pretrain:
            t_total_loss = pretask(model, device, img_patch, orig_patch, optimizer, scheduler, patch_size, iteration)
            t_correct = 0
        else:
            t_total_loss, t_correct = train(model, model_2d, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls, optimizer, scheduler, patch_size, beta, optimizer_2d, scheduler_2d, img_2d, aug_2d, augGT_2d_cls, lamda, augGT_2d, iteration)

        epoch_train_loss.append(t_total_loss)
        if (iteration % train_interval == 0):
            avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            epoch_train_accuracy = (t_correct / (args.log_interval * len(train_dataset))) * 100

            logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval}: \t Loss: {avg_train_loss:.6f}\t')

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                logging.info(f'--- Saving model at Avg Train Loss:{avg_train_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weight, 'mood_best_train_' + args.identifier +'.pth'))

            # validation process
            test_iterator = iter(test_loader)
            for i in range(len(test_loader)):
                if args.pretrain:
                    v_total_loss = pretask(model, device, img_patch, orig_patch, optimizer, scheduler, patch_size, iteration, test=True)
                    epoch_test_loss.append(v_total_loss)
                else:
                    img_patch, img_2d, imgGT_patch, img_cls, aug_patch, aug_2d, augGT_patch, aug_cls, augGT_2d_cls, augGT_2d = next(test_iterator)

                    v_segloss, v_clsloss, v_total_loss, v_correct = test(model, device, img_patch, imgGT_patch, img_cls, aug_patch, augGT_patch, aug_cls)

                    epoch_test_loss.append(v_total_loss)
                    epoch_test_segloss.append(v_segloss)
                    epoch_test_clsloss.append(v_clsloss)
                    correct_test_count += v_correct

            avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
            if not args.pretrain:
                avg_test_segloss = sum(epoch_test_segloss) / len(epoch_test_segloss)
                avg_test_clsloss = sum(epoch_test_clsloss) / len(epoch_test_clsloss)
                epoch_test_accuracy = (correct_test_count / len(test_dataset)) * 100

                logging.info(f'Iter {iteration / train_interval}-{total_iteration / train_interval} eval: \t Loss: {avg_test_loss:.6f}\t')

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                logging.info(f'--- Saving model at Avg Valid Loss:{avg_test_loss:.6f}  ---')
                torch.save(model.state_dict(), os.path.join(args.weight, 'mood_best_valid_' + args.identifier +'.pth'))

            # save snapshot for resume training
            logging.info('--- Saving snapshot ---')
            torch.save({
                'iteration': iteration+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_train_loss': best_train_loss,
                'best_test_loss': best_test_loss,
            },
                os.path.join(args.checkpoints, 'latest_checkpoints_' + args.identifier +'.pth'))

            logging.info(f"--- {time.time() - start_time} seconds ---")

            epoch_train_loss = []
            epoch_test_loss = []
            epoch_test_segloss = []
            epoch_test_clsloss = []
            correct_train_count = 0
            correct_test_count = 0
            start_time = time.time()
        # iteration += 1