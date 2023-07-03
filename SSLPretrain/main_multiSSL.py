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
from loss import * 
from model import SSLHead2D 

def main():

    # Define model and model_args
    model_args = { "spatial_dims": 2,   # 2D
                "in_channels": 1,     # 1 for grayscale
                "feature_size": 4,
                "dropout_path_rate": 0.0,
                "use_checkpoint": True,
                } # to be determined

    model = SSLHead2D(model_args)
    model.cuda()

    # function to save checkpoints
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    # training function
    def train(train_loader, val_loader, val_best, scaler, cp_logdir,
            loss_function, optimizer, epochs):

        all_train_losses = []
        all_train_losses_recon = []
        all_train_losses_rot = []
        all_train_losses_con = []
        
        all_val_losses = []
        all_val_losses_recon = []
        all_val_losses_rot = []
        all_val_losses_con = []

        for epoch in range(epochs):
            print('Epoch ', epoch)
            t1 = time()

            model.train()
            
            losses = [] # total loss
            losses_train_recon = [] # reconstruction loss @training
            losses_train_rot = [] # rotation loss @training
            losses_train_con = [] # contrastive loss @training
            
            imgs_list = []

            for step, batch in enumerate(train_loader):
                x = batch.cuda()
                x1, rot1 = rot_rand(x) 
                x2, rot2 = rot_rand(x)
                
                x1_aug = aug_rand(local_rank, x1)
                x2_aug = aug_rand(local_rank, x2)


                with autocast(enabled=True):
                    rec_x1, rot_x1, con_x1 = model(x1_aug)
                    rec_x2, rot_x2, con_x2 = model(x2_aug)
                    
                    rot_pred = torch.cat([rot_x1, rot_x2], dim=0)
                    rot_true = torch.cat([rot1, rot2], dim=0)
                    
                    rec_pred = torch.cat([rec_x1, rec_x2], dim=0)
                    rec_true = torch.cat([x1, x2], dim=0)
                    
                    loss, task_losses = loss_function(rot_pred, rot_true, con_x1, con_x2, rec_pred, rec_true)
                    
                losses.append(loss.item()) # total loss 
                losses_train_rot.append(task_losses[0].item())
                losses_train_con.append(task_losses[1].item())
                losses_train_recon.append(task_losses[2].item())
                

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # scheduler.step()
                optimizer.zero_grad()


            all_train_losses.append(np.mean(losses))
            all_train_losses_rot.append(np.mean(losses_train_rot))
            all_train_losses_con.append(np.mean(losses_train_con))
            all_train_losses_recon.append(np.mean(losses_train_recon))
            
            print("Epoch: {}/{}, Loss:{:.4f} (Recon:{:.4f}, Rot:{:.4f}, Contrast:{:.4f})".format(epoch, epochs, np.mean(losses), np.mean(losses_train_recon), np.mean(losses_train_rot), np.mean(losses_train_con)))

            print('Training done, Time:{:.4f}'.format(time()-t1))
            
            print('Validation:')
            val_loss, val_loss_recon, val_loss_rot, val_loss_con, img_list = validation(val_loader)
            print("Validation: Loss:{:.4f} (Recon:{:.4f}, Rot:{:.4f}, Contrast:{:.4f})".format(val_loss, val_loss_recon, val_loss_rot, val_loss_con))
            imgs_list.append(img_list)
            all_val_losses.append(val_loss)
            all_val_losses_recon.append(val_loss_recon)
            all_val_losses_rot.append(val_loss_rot)
            all_val_losses_con.append(val_loss_con)

            if val_loss < val_best:
                val_best = val_loss
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                    }
                save_ckp(checkpoint, cp_logdir+"model_bestValRMSE.pt")
                print("Model saved! Best Val loss:{:.4f}, Current Val loss:{:.4f}".format(val_best, val_loss))
            else:
                print("Model not saved. Best Val loss:{:.4f}, Current Val loss:{:.4f}".format(val_best, val_loss))

            print('Epoch {}/{} done'.format(epoch, epochs))
            
            
        all_train_losses_tuple = (all_train_losses_rot, all_train_losses_con, all_train_losses_recon)
        all_val_losses_tuple = (all_val_losses_rot, all_val_losses_con, all_val_losses_recon)

        return imgs_list, all_train_losses, all_train_losses_tuple, all_val_losses, all_val_losses_tuple, val_best 

    def validation(val_loader):

        model.eval()
        
        losses_val = []
        losses_val_recon = []
        losses_val_rot = []
        losses_val_con = []
        
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                val_x = batch.cuda()
                
                x1, rot1 = rot_rand(val_x)
                x2, rot2 = rot_rand(val_x)
                x1_aug = aug_rand(local_rank, x1)
                x2_aug = aug_rand(local_rank, x2)

                with autocast(enabled=True):
                    rec_x1, rot_x1, con_x1 = model(x1_aug)
                    rec_x2, rot_x2, con_x2 = model(x2_aug)
                    
                    rot_pred = torch.cat([rot_x1, rot_x2], dim=0)
                    rot_true = torch.cat([rot1, rot2], dim=0)
                    
                    rec_pred = torch.cat([rec_x1, rec_x2], dim=0)
                    rec_true = torch.cat([x1, x2], dim=0)
                    
                    loss, task_losses = loss_function(rot_pred, rot_true, con_x1, con_x2, rec_pred, rec_true)

                losses_val.append(loss.item()) # total loss 
                losses_val_rot.append(task_losses[0].item())
                losses_val_con.append(task_losses[1].item())
                losses_val_recon.append(task_losses[2].item())

                # save the GT image, rotated image, augmented image, recon image
                x_gt = val_x.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
                xgt = x_gt
                
                x_rot = x1.detach().cpu().numpy()
                x_rot = (x_rot - np.min(x_rot)) / (np.max(x_rot) - np.min(x_rot))
                xrot = x_rot

                x_aug = x1_aug.detach().cpu().numpy()
                x_aug = (x_aug - np.min(x_aug)) / (np.max(x_aug) - np.min(x_aug))
                xaug = x_aug

                rec_x = rec_x1.detach().cpu().numpy()
                rec_x = (rec_x - np.min(rec_x)) / (np.max(rec_x) - np.min(rec_x))
                recon = rec_x

                img_list = [xgt, xrot, xaug, recon]

        return np.mean(losses_val), np.mean(losses_val_recon), np.mean(losses_val_rot), np.mean(losses_val_con), img_list


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
    
    loss_function = Loss(batch_size) 

    # Scaler
    scaler = None  # GradScaler() or None

    # local rank (about torch.device, see utils.data_ops)
    local_rank = 0

    best_val = 1e8 # starting validation loss (very large)


    # saving directory
    cp_logdir = './log_ckp/MT01_0407/'
    logdir = './log/MT01_0407/'
    resultsdir = './Results/MT01_0407/'

    ### Training
    images_list, losses_train, losses_train_tuple, losses_val, losses_val_tuple, val_best = train(train_loader, val_loader, best_val, scaler, cp_logdir, loss_function, optimizer, max_epochs)
    
    # save final model and ckp
    checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(model.state_dict(), logdir+'final_SSLmodel.pth')
    save_ckp(checkpoint, logdir+'final_SSLmodel_epoch.pt')

    # save images_list (list) as pickle file
    with open(resultsdir+'img_augImg_RecIm_list.pkl', 'wb') as f:
        pickle.dump(images_list, f)

    # save train loss
    with open(resultsdir+'losses_train.pkl', 'wb') as f:
        pickle.dump(losses_train, f)
        
    # save train loss (all)
    with open(resultsdir+'losses_train_tuple.pkl', 'wb') as f:
        pickle.dump(losses_train_tuple, f)

    # save val loss
    with open(resultsdir+'losses_val.pkl', 'wb') as f:
        pickle.dump(losses_val, f)
    
    # save val loss (all)
    with open(resultsdir+'losses_val_tuple.pkl', 'wb') as f:
        pickle.dump(losses_val_tuple, f)

    # plot losses
    plt.plot(losses_train, label='Training loss')
    plt.plot(losses_val, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Combined Loss')
    plt.title('Training and Validation Losses')
    plt.savefig(resultsdir+'losses.png', dpi=200)
    
    plt.clf()
    
    # plot reconstruction loss - 2
    plt.plot(losses_train_tuple[2], label='Training recon loss')
    plt.plot(losses_val_tuple[2], label='Validation recon loss')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.title('Training and Validation Reconstruction Losses')
    plt.savefig(resultsdir+'losses_recon.png', dpi=200)
    
    plt.clf()
    
    # plot contrastive loss - 1
    plt.plot(losses_train_tuple[1], label='Training contrast loss')
    plt.plot(losses_val_tuple[1], label='Validation contrast loss')
    plt.xlabel('Epochs')
    plt.ylabel('Contrastive Loss')
    plt.title('Training and Validation Contrastive Losses')
    plt.savefig(resultsdir+'losses_con.png', dpi=200)
    
    plt.clf()
    
    # plot rotational loss - 0
    plt.plot(losses_train_tuple[0], label='Training rotation loss')
    plt.plot(losses_val_tuple[0], label='Validation rotation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training and Validation Rotation Losses')
    plt.savefig(resultsdir+'losses_rot.png', dpi=200)



if __name__ == '__main__':
    main()
