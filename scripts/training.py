#!/usr/bin/env python
# coding: utf-8

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

from dataloader import *
from models import Merlin
import load_config as cfg
import argparse
from utils import stream
def np_now(x): return x.detach().cpu().numpy()

def exp_lr_scheduler(optimizer, epoch, init_lr=0.002, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train_loop(buffer_size, lab_dim, num_epochs, warmup_epoch,  model, optimiser, criterion, reduce_lr, model_path, checkpoint_every_n, lr, duration = True):
    device = next(model.parameters()).device

    min_validation_loss = 2.0
    validation_counter = 0
    batch_size = 2300
    # This gets the valid set

    if not duration:
        print("\n Loading validation data ... \n")
        dataset = get_tts_dataset(cfg.valid_list,cfg.acoustic_bin_no_sil_norm, cfg.bin_acoustic_feats,None, lab_dim, cfg.cmp_dim, None)
    else:

        print("\n Loading validation data ... \n")
        dataset = get_tts_dataset(cfg.valid_list,cfg.bin_no_sil_norm, cfg.bin_acoustic_feats,cfg.dur_no_sil_norm, cfg.lab_dim, cfg.cmp_dim, cfg.dur_dim)


    valid_dataloader = DataLoader(dataset, batch_size=1,collate_fn=lambda batch : collate_tts(batch),
                            shuffle=True, num_workers=1)

    start_train = time.time()
    for epoch in range(num_epochs):

        msg = f'Epoch {epoch+1}/{num_epochs} \n'
        #stream(msg)

        start_time = time.time()
        batch_losses = []

        print("Loading training data ... \n")

        # This gets the training set
        if not duration:
            train_dataset = get_tts_dataset(cfg.train_list,cfg.acoustic_bin_no_sil_norm, cfg.bin_acoustic_feats,None, lab_dim, cfg.cmp_dim,None)
        else:
            # This gets the training set
            train_dataset = get_tts_dataset(cfg.train_list,cfg.bin_no_sil_norm, cfg.bin_acoustic_feats,cfg.dur_no_sil_norm, cfg.lab_dim, cfg.cmp_dim, cfg.dur_dim)


        train_dataloader = DataLoader(train_dataset, batch_size=2300,collate_fn=lambda batch : collate_tts(batch),
                            shuffle=True, num_workers=1)



        for idx, (x, t,ids, d, frames, dur_len) in enumerate(train_dataloader):

            msg =f'\n Batch {idx+1}/{len(train_dataloader)}\n'
            stream(msg)

            model.train()

            n_frames = 256
            training_losses = []
            iters = int(len(x) / n_frames)
            print(iters)


            for i in range(iters):

                optimiser.zero_grad()
                start = i*n_frames
                end = (i+1)*n_frames

                if end > x.shape[0]:
                    end = x.shape[0]
                    print("end frames", i)

                x_in = x[start:end,:]



                if duration:
                    d_in = d[start:end,:]
                    lab, dur = x_in.to(device), d_in.to(device)
                else:
                    t_in = t[start:end,:]
                    lab, targ = x_in.to(device), t_in.to(device)

                #print(x_in.shape)
                #print(t.shape)

                # Forward pass
                y_pred = model(lab)
                #print(y_pred.shape)
                # Compute Loss
                if duration:
                    mse_loss = criterion(y_pred, dur)
                else:
                    #mse_loss = criterion(y_pred, targ)
                    #print("mse torch", mse_loss)
                    finetune = torch.mean(torch.sum((y_pred - targ)**2, dim=1))



                #loss =  mse_loss
                loss = finetune

                # Backward pass
                loss.backward()
                optimiser.step()

                training_losses.append(np_now(finetune))

                end_time = time.time()
                #print("iter time", end_time-start_time)

            print(training_losses)
            this_batch_loss = np.mean(training_losses)

        if epoch > warmup_epoch:
            reduce_lr = True

        if reduce_lr:
            lr = lr * 0.5
            optimiser = optim.Adam(model.parameters(), lr)

        validation_losses = []
        with torch.no_grad():
            print("Validating ...")

            total = 0
            for idx, (x, t,ids, d, frames, dur_len) in enumerate(valid_dataloader):

                if duration:
                    val_in_x, val_d = x.to(device), d.to(device)
                else:
                    val_in_x, val_t = x.to(device), t.to(device)

                model.eval()
                # Forward pass
                val_prediction = model(val_in_x)

                # Compute Loss
                if duration:
                    mse_loss = criterion(val_prediction, val_d)
                else:
                    #mse_loss = criterion(val_prediction, val_t)
                    finetune = torch.mean(torch.sum((val_prediction - val_t)**2, dim=1))


                val_loss =  finetune

                validation_losses.append(np_now(finetune))


        checkpoint = f'{model_path}/latest_model.pyt'
        model.save(checkpoint)
        #print("validation losses",len(validation_losses))
        this_validation_loss = np.mean(validation_losses)
        msg = f'\nEpoch {epoch}: mean val loss: {this_validation_loss} \n'
        stream(msg)




        end_time = time.time()

        epoch_time = end_time - start_time
        msg  = f'\nEpoch {epoch}: train loss: {this_batch_loss} time: {epoch_time}\n'
        stream(msg)
        #optimiser = exp_lr_scheduler(optimiser, epoch)



    end_train = time.time()
    total_time = end_train - start_train
    msg = f'Total training time: {total_time}'
    stream(msg)
    model.save(checkpoint)


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Merlin')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    args = parser.parse_args()


    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.cuda.set_device(1)

    if device.type == 'cuda':
        print(torch.cuda.current_device())



    print('Using device:', device)

    num_epochs = cfg.num_epochs
    print("\n Num epochs:", num_epochs)


    if not os.path.exists(cfg.models_path):
        os.system("mkdir -p %s" %cfg.models_path)
        os.system("mkdir -p %s/duration_model" %cfg.models_path)
        os.system("mkdir -p %s/acoustic_model" %cfg.models_path)

    duration_model_path = os.path.join(cfg.models_path, "duration_model")
    acoustic_model_path = os.path.join(cfg.models_path, "acoustic_model")

    criterion = torch.nn.MSELoss(reduction='mean')

    if cfg.TRAIN_DURATION:
        print('\nInitialising Duration Model...\n')

        # Instantiate Duration Model
        duration_model = Merlin(cfg.input_size, cfg.hidden_size, cfg.dur_dim).to(device=device)
        optimiser = optim.Adam(duration_model.parameters(), cfg.lr)


        print('\n Starting training ...\n')
        train_loop(cfg.buffer_size, cfg.lab_dim, num_epochs, cfg.warmup_epoch , duration_model, optimiser, criterion, cfg.reduce_lr, duration_model_path, cfg.checkpoint_every_n, cfg.lr )

    if cfg.TRAIN_ACOUSTIC:
        print('\nInitialising Acoustic Model...\n')
        #Instantiate Acoustic Model


        lab_dim = cfg.lab_dim + cfg.frame_feat_dim
        print(lab_dim)



        acoustic_model = Merlin(lab_dim, cfg.hidden_size, cfg.cmp_dim).to(device=device)
        #optimiser = optim.SGD(acoustic_model.parameters(), lr=0.01, weight_decay=0.01)


        optimiser = optim.Adam(acoustic_model.parameters(), cfg.lr)
        print(acoustic_model)

        train_loop(cfg.buffer_size, lab_dim, num_epochs, cfg.warmup_epoch,  acoustic_model, optimiser, criterion, cfg.reduce_lr, acoustic_model_path, cfg.checkpoint_every_n, cfg.lr, duration = False )
