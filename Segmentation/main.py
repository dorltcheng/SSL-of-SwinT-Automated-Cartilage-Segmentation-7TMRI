import os
import json
import torch
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import pickle
import helpers
import io
import numpy as np
from pathlib import Path
import collections.abc
from time import time
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from sklearn.model_selection import KFold

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.config import print_config
from monai.transforms import AsDiscrete

from monai.data import decollate_batch

from data_utils import *
from model import SwinUNETR


def main():

    # function to save checkpoints
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    # training
    def train(foldnum, trainData, valData, model_args, pretrainW_path,
                batch_size, num_workers, scaler, lr, decay, epochs, local_rank,
                loss_function, acc_dice, acc_iou, val_post_label, val_post_pred, cp_logdir):


        # define new model
        model = SwinUNETR(
                img_size=model_args['img_size'],
                in_channels=model_args['in_channels'], # grayscale
                out_channels=model_args['out_channels'], # single class seg
                feature_size =model_args['feature_size'], # embedding size (default: 24), TBC
                use_checkpoint=model_args['use_checkpoint'],
                spatial_dims=model_args['spatial_dims']
            ).cuda()

        # define optimizer
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=decay)

        # load pretrained weights
        ############# Uncomment for purely supervised training 
        ptrain_state_dict = torch.load(pretrainW_path)
        new_state_dict = model.state_dict()
        for name, param in ptrain_state_dict.items():
            if name in new_state_dict:
                new_state_dict[name].copy_(param)
        model.load_state_dict(new_state_dict)
        #############

        # Dataloader
        train_loader, val_loader = get_dataloader_seg(trainData, valData, batch_size, num_workers)


        val_best = 0.0
        val_best_iou = 0.0
        fold_train_losses = [] # store loss at each epoch at one specific fold
        fold_val_dices = [] # store dice score at each validation epoch at one specific fold
        fold_val_ious = []

        print('Fold ', foldnum, ' starts:')
        for epoch in range(epochs):
            print('Epoch ', epoch)
            t1 = time()

            model.train()

            epoch_train_loss = []

            for step, batch in enumerate(train_loader):
                data, target = batch
                data, target = data.cuda(local_rank), target.cuda(local_rank)

                with autocast(enabled=True):
                    logits = model(data)
                    loss = loss_function(logits, target)

                epoch_train_loss.append(loss.item())

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # scheduler.step()
                optimizer.zero_grad()

            fold_train_losses.append(np.mean(epoch_train_loss))
            print("Epoch: {}/{}, Loss: {:.4f}".format(epoch, epochs, np.mean(epoch_train_loss)))


            print('Training done, Time:{:.4f}'.format(time()-t1))
            print('Validation:')
            model, dice_score_val, iou_val = validation(model, val_loader, acc_dice, acc_iou, val_post_label, val_post_pred)
            fold_val_dices.append(dice_score_val)
            fold_val_ious.append(iou_val)
            
            if iou_val > val_best_iou:
                val_best_iou = iou_val

            if dice_score_val > val_best:
                val_best = dice_score_val
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                    }
                save_ckp(checkpoint, cp_logdir+"fold"+str(foldnum)+"_model_bestVal_ckp.pt")
                print("Model saved! Best Avg. Dice Score:{:.4f}, Current Avg. Dice Score:{:.4f}, Current Avg. IoU:{:.4f}".format(val_best, dice_score_val, iou_val))
            else:
                print("Model not saved. Best Avg. Dice Score:{:.4f}, Current Avg. Dice Score:{:.4f}, Current Avg. IoU:{:.4f}".format(val_best, dice_score_val, iou_val))

            print('Epoch {}/{} done'.format(epoch, epochs))


        print('Fold ', foldnum, ' finishes.')

        return fold_train_losses, fold_val_dices, fold_val_ious, val_best, val_best_iou, model, optimizer.state_dict()

    # Validation
    def validation(model, valDataloader, dice_function, iou_function, post_label, post_pred):
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(valDataloader):
                data, target = batch
                data, target = data.cuda(local_rank), target.cuda(local_rank)

                with autocast(enabled=True):
                    # no need sliding_window_inference
                    val_logits = model(data)

                # if not val_logits.is_cuda:
                # target = target.cpu()
                # val_logits = val_logits.cpu()

                # target_list = decollate_batch(target)
                # target_convert = [post_label(target_tensor) for target_tensor in target_list]
                # output_list = decollate_batch(val_logits)
                # output_convert = [post_pred(output_tensor) for output_tensor in output_list]
                # dice_function.reset()
                # dice_function(y_pred=output_convert, y=target_convert)
                
                val_pred = torch.sigmoid(val_logits)
                val_pred = (val_pred > 0.5).float()

                target = (target > 0).float()

                dice_function(y_pred=val_pred, y=target)
                iou_function(y_pred=val_pred, y=target)

            mean_dice_score = dice_function.aggregate().item()
            mean_iou = iou_function.aggregate().item()
            dice_function.reset()
            iou_function.reset()

        return model, mean_dice_score, mean_iou


    ### KFold data sorting => return train_list, val_list: len(5), each item is the list of data for that fold
    batch_size = 32
    num_workers = 4
    dataJsonPath = 'OAIdataset/3TMRI_dataset_1.json'
    trainData, valData = load_dataset(dataJsonPath)

    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    train_list, val_list = kf_datasetSort(kf, trainData)


    ### Define Hyperparemters and parameters ###
    max_epochs = 60
    lr = 1e-3
    decay = 1e-5
    # warmup_steps = 500

    loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True) # softmax is for multi-class
    dice_accuracy = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    iou_accuracy = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    post_label = AsDiscrete(to_onehot=1)
    post_pred = AsDiscrete(argmax=True, to_onehot=1)

    # Scaler
    # scaler = None
    scaler = torch.cuda.amp.GradScaler()

    # local rank (about torch.device, see utils.data_ops)
    local_rank = 0

    best_val = 0.0 # starting validation loss (very large)

    # model arguments
    args = {
        'img_size': (256, 256),
        'in_channels': 1,
        'out_channels': 1,
        'feature_size': 4,
        'use_checkpoint': True,
        'spatial_dims': 2
    }

    # pretrained weights
    pretrainW_path = '/rds/general/user/dlc19/home/SSLPretrainAll/log/05_0217/final_SSLmodel.pth'
    # pretrainW_path = None ### Set to None if purely supervised

    # saving directory
    cp_logdir = 'log_ckp/11_0330/'
    logdir = 'log/11_0330/'
    resultsdir = 'Results/11_0330/'

    ### Training (5-fold cross validation)
    train_losses = []
    val_dices = []
    val_ious = []
    best_vals = []
    best_vals_iou = []
    models = []
    opt_statedicts = []

    # 5-fold cv loop
    for fold in range(num_folds):
        # call train function
        train_loss, val_dice, val_iou, bestval, bestval_iou, model, opt_statedict = train(fold, train_list[fold], val_list[fold], args, pretrainW_path,
                                                                                    batch_size, num_workers, scaler, lr, decay, max_epochs, local_rank,
                                                                                    loss_function, dice_accuracy, iou_accuracy, post_label, post_pred, cp_logdir)
        train_losses.append(train_loss)
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        best_vals.append(bestval)
        best_vals_iou.append(bestval_iou)
        models.append(model)
        opt_statedicts.append(opt_statedict)

    bestmodel_idx = best_vals.index(max(best_vals))
    best_model = models[bestmodel_idx]
    best_opt = opt_statedicts[bestmodel_idx]

    print()
    for fold in range(num_folds):
        print('Fold ', fold, ' best validation dice: ', best_vals[fold], ' best validation IoU: ', best_vals_iou[fold])

    # save best val dice scores:
    with open('bestValDices.pkl', 'wb') as f:
        pickle.dump(best_vals, f)

    # save best model
    checkpoint = {"epoch": max_epochs, "state_dict": best_model.state_dict(), "optimizer": best_opt}
    torch.save(best_model.state_dict(), logdir+'best_model_fold'+str(bestmodel_idx)+'.pth')
    save_ckp(checkpoint, logdir+'best_model_fold'+str(bestmodel_idx)+'_epoch.pt')

    # Plot training loss curve (Plot best one and average)
    avg_losses = [sum([fold[i] for fold in train_losses])/len(train_losses) for i in range(max_epochs)]
    plt.plot(avg_losses, label='Avg training loss')
    plt.plot(train_losses[bestmodel_idx], label='Best training loss (Fold '+str(bestmodel_idx)+')')
    plt.xlabel('Epochs')
    plt.ylabel('DiceCELoss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(resultsdir+'Traininglosses.png', dpi=200)

    plt.clf()

    # Plot validation score curve (Plot best one and average)
    avg_valdices = [sum([fold[i] for fold in val_dices])/len(val_dices) for i in range(max_epochs)]
    plt.plot(avg_valdices, label='Avg val dice score')
    plt.plot(val_dices[bestmodel_idx], label='Best val dice score (Fold '+str(bestmodel_idx)+')')
    plt.xlabel('Epochs')
    plt.ylabel('Dice score')
    plt.title('Validation Dice Scores')
    plt.legend()
    plt.savefig(resultsdir+'ValidationDices.png', dpi=200)
    
    plt.clf()
    
    # Plot validation IoU curve (Plot best one and average)
    avg_valIous = [sum([fold[i] for fold in val_ious])/len(val_ious) for i in range(max_epochs)]
    plt.plot(avg_valIous, label='Avg val IoU')
    plt.plot(val_ious[bestmodel_idx], label='Best IoU (Fold '+str(bestmodel_idx)+')')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.savefig(resultsdir+'ValidationIou.png', dpi=200)



if __name__ == '__main__':
    main()
