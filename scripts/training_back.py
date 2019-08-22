#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time

from dataloader import *
from models import Merlin
import load_config_test as cfg
import argparse

def np_now(x): return x.detach().cpu().numpy()

def train_duration_loop(buffer_size, lab_dim, num_epochs, warmup_epoch, train_dataloader, valid_dataloader, duration_model, optimiser, criterion, reduce_lr, model_path, checkpoint_every_n, lr ):
    device = next(duration_model.parameters()).device
    #Lets just work with one batch and later it needs to loop through all batched
    temp_x = np.zeros((buffer_size, lab_dim))
    temp_dur = np.zeros((buffer_size))
    frame_index = 0
    dur_index = 0

    start_train = time.time()
    for epoch in range(num_epochs):

        msg = f'Epoch {epoch+1}/{num_epochs} '
        stream(msg)

        start_time = time.time()
        batch_training_losses = []


        for idx, (x, t,ids, d, frames, dur_len) in enumerate(train_dataloader):

            msg =f'\n Batch {idx+1}/{len(train_dataloader)}'
            stream(msg)

            duration_model.train()

            x_in, d = x.to(device), d.to(device)

            optimiser.zero_grad()
            # Forward pass
            y_pred = duration_model(x_in)

            # Compute Loss
            mse_loss = criterion(y_pred.squeeze(), d)

            loss =  mse_loss

            batch_training_losses.append(loss.item())

            # Backward pass
            loss.backward()
            optimiser.step()

            end_time = time.time()


        if epoch > warmup_epoch:
            reduce_lr = True

        if reduce_lr:
            lr = lr * 0.5
            optimiser = optim.Adam(duration_model.parameters(), cfg.lr)



        if epoch % checkpoint_every_n == 0:
            validation_losses = []
            total = 0
            for idx, (x, t,ids, d, frames, dur_len) in enumerate(valid_dataloader):


                val_in_x, val_d = x.to(device), d.to(device)
                # Forward pass
                val_prediction = duration_model(val_in_x)

                # Compute Loss
                #l1_loss = F.l1_loss(val_prediction.squeeze(), d)
                mse_loss = criterion(val_prediction.squeeze(), val_d)

                val_loss =  mse_loss

                validation_losses.append(val_loss.item())


            checkpoint = f'{model_path}/latest_model.pyt'
            duration_model.save(checkpoint)
            this_validation_loss = np.mean(validation_losses)
            msg = f'Epoch {epoch}: mean val loss: {this_validation_loss}'
            stream(msg)




        end_time = time.time()
        this_train_loss = np.mean((batch_training_losses))
        epoch_time = end_time - start_time
        msg  = f'Epoch {epoch}: train loss: {this_train_loss} time: {epoch_time}'
        stream(msg)

    end_train = time.time()
    total_time = end_train - start_train
    msg = f'Total training time: {total_time}'
    stream(msg)
    duration_model.save(checkpoint)


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Merlin')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    args = parser.parse_args()


    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')



    if device.type == 'cuda':
        print(torch.cuda.current_device())

    print('Using device:', device)

    print("Loading training data ... \n")

    # This gets the training set
    train_dataset = get_tts_dataset(cfg.train_list,cfg.bin_no_sil_norm, cfg.bin_acoustic_feats,cfg.dur_no_sil_norm, cfg.lab_dim, cfg.cmp_dim, cfg.dur_dim)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size,collate_fn=lambda batch : collate_tts(batch),
                            shuffle=True, num_workers=4)

    # This gets the valid set

    print("\n Loading validation data ... \n")
    valid_dataset = get_tts_dataset(cfg.valid_list,cfg.bin_no_sil_norm, cfg.bin_acoustic_feats,cfg.dur_no_sil_norm, cfg.lab_dim, cfg.cmp_dim, cfg.dur_dim)

    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size,collate_fn=lambda batch : collate_tts(batch),
                            shuffle=True, num_workers=4)


    num_epochs = cfg.num_epochs
    print("\n Num epochs:", num_epochs)


    if not os.path.exists(cfg.models_path):
        os.system("mkdir -p %s" %cfg.models_path)
        os.system("mkdir -p %s/duration_model" %cfg.models_path)
        os.system("mkdir -p %s/acoustic_model" %cfg.models_path)

    duration_model_path = os.path.join(cfg.models_path, "duration_model")
    acoustic_model_path = os.path.join(cfg.models_path, "acoustic_model")

    criterion = torch.nn.MSELoss()

    if cfg.TRAIN_DURATION:
        print('\nInitialising Duration Model...\n')

        # Instantiate Duration Model
        duration_model = Merlin(cfg.input_size, cfg.hidden_size, cfg.dur_dim).to(device=device)
        optimiser = optim.Adam(duration_model.parameters(), cfg.lr)
        print('\n Starting training ...\n')
        train_duration_loop(cfg.buffer_size, cfg.lab_dim, num_epochs, cfg.warmup_epoch , train_dataloader, valid_dataloader, duration_model, optimiser, criterion, cfg.reduce_lr, duration_model_path, cfg.checkpoint_every_n, cfg.lr )

    if cfg.TRAIN_ACOUSTIC:
        print('\nInitialising Acoustic Model...\n')
        #Instantiate Acoustic Model
        acoustic_model = Merlin(cfg.input_size, cfg.hidden_size, cfg.cmp_dim)
        optimiser = optim.Adam(acoustic_model.parameters(), cfg.lr)

        train_acoustic_loop(cfg.buffer_size, cfg.lab_dim, num_epochs, cfg.warmup_epoch, train_dataloader, valid_dataloader, acoustic_model, optimiser, criterion, cfg.reduce_lr, acoustic_model_path, cfg.checkpoint_every_n )
