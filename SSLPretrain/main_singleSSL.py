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


from utils.data_utils import *
from utils.data_ops import patch_rand_drop2D, aug_rand, rot_rand
from model import SSLHead2DSim 

def main():

    # Define model and model_args
    model_args = { "spatial_dims": 2,   # 2D
                "in_channels": 1,     # 1 for grayscale
                "feature_size": 4,
                "dropout_path_rate": 0.0,
                "use_checkpoint": True,
                } # to be determined

    model = SSLHead2DSim(model_args)
    model.cuda()

    # function to save checkpoints
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    # training function
    def train(train_loader, val_loader, val_best, scaler, cp_logdir,
            loss_function, optimizer, epochs):

        all_train_losses = []
        all_val_losses = []

        for epoch in range(epochs):
            print('Epoch ', epoch)
            t1 = time()
            pbar = tqdm(total=len(train_loader))

            model.train()
            losses_train_recon = [] # reconstruction loss @training
            imgs_list = []

            for step, batch in enumerate(train_loader):
                x = batch.cuda()
                x_aug = aug_rand(local_rank, x)

                with autocast(enabled=True):
                    rec_x = model(x_aug)
                    recon_loss = loss_function(rec_x, x)

                losses_train_recon.append(recon_loss.item())

                if scaler is not None:
                    scaler.scale(recon_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon_loss.backward()
                    optimizer.step()

                # scheduler.step()
                optimizer.zero_grad()

                pbar.update(1)

            all_train_losses.append(np.mean(losses_train_recon))
            print("Epoch: {}/{}, (Recon) Loss:{:.4f}".format(epoch, epochs, np.mean(losses_train_recon)))
            pbar.close()

            print('Training done, Time:{:.4f}'.format(time()-t1))
            print('Validation:')
            val_loss, img_list = validation(val_loader)
            imgs_list.append(img_list)
            all_val_losses.append(val_loss)

            if val_loss < val_best:
                val_best = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                    }
                save_ckp(checkpoint, cp_logdir+"model_bestValRMSE.pt")
                print("Model saved! Best Recon Val loss:{:.4f}, Recon Val loss:{:.4f}".format(val_best, val_loss))
            else:
                print("Model not saved. Best Recon Val loss:{:.4f}, Recon Val loss:{:.4f}".format(val_best, val_loss))

            print('Epoch {}/{} done'.format(epoch, epochs))

        return imgs_list, all_train_losses, all_val_losses, val_best 

    def validation(val_loader):

        model.eval()
        losses_val_recon = []
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                val_x = batch.cuda()
                x_aug = aug_rand(local_rank, val_x)

                with autocast(enabled=True):
                    rec_x = model(x_aug)
                    recon_loss = loss_function(rec_x, val_x)

                losses_val_recon.append(recon_loss.item())

                # save the GT image, augmented image, recon image
                x_gt = val_x.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
                xgt = x_gt

                x_aug = x_aug.detach().cpu().numpy()
                x_aug = (x_aug - np.min(x_aug)) / (np.max(x_aug) - np.min(x_aug))
                xaug = x_aug

                rec_x = rec_x.detach().cpu().numpy()
                rec_x = (rec_x - np.min(rec_x)) / (np.max(rec_x) - np.min(rec_x))
                recon = rec_x

                img_list = [xgt, xaug, recon]
                print("Validation step:{}, (Recon) Loss:{:.4f}".format(step, recon_loss))

        return np.mean(losses_val_recon), img_list


    ### get dataloaders
    batch_size = 32
    num_workers = 4
    dataJsonPath = './OAI_all/dataset.json'
    train_loader, val_loader = get_dataloader(dataJsonPath, batch_size, num_workers)

    ### Define Hyperparemters and parameters ###
    max_epochs = 200
    lr = 1e-3
    decay = 1e-5
    warmup_steps = 500

    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=decay)

    loss_function = nn.L1Loss().cuda() # just for reconstruction loss

    # Scaler
    scaler = None  # GradScaler() or None

    # local rank (about torch.device, see utils.data_ops)
    local_rank = 0

    best_val = 1e8 # starting validation loss (very large)


    # saving directory
    cp_logdir = './log_ckp/'
    logdir = './log/'
    resultsdir = './Results/'

    ### Training
    images_list, losses_train, losses_val, val_best = train(train_loader, val_loader,
                                                            best_val, scaler, cp_logdir, loss_function,
                                                            optimizer, max_epochs)

    # save images_list (list) as pickle file
    with open(resultsdir+'img_augImg_RecIm_list.pkl', 'wb') as f:
        pickle.dump(images_list, f)

    # save train loss
    with open(resultsdir+'losses_train.pkl', 'wb') as f:
        pickle.dump(losses_train, f)

    # save val loss
    with open(resultsdir+'losses_val.pkl', 'wb') as f:
        pickle.dump(losses_val, f)

    # plot losses
    plt.plot(losses_train, label='Training loss')
    plt.plot(losses_val, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.title('Training and Validation Losses')
    plt.savefig(resultsdir+'losses.png', dpi=200)


    # save final model and ckp
    checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(model.state_dict(), logdir+'final_SSLmodel.pth')
    save_ckp(checkpoint, logdir+'final_SSLmodel_epoch.pt')



if __name__ == '__main__':
    main()
