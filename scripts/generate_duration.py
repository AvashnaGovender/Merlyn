#!/usr/bin/env python
# coding: utf-8
import os, re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import argparse
from dataloader import *
from utils import *
from models import Merlin
import load_config_test as cfg
from prepare_inputs import check_silence_pattern, load_binary_file


def np_now(x): return x.detach().cpu().numpy()


def denormalise(features):

        fid = open(cfg.dur_norm_info, 'rb')
        cmp_min_max = np.fromfile(fid, dtype=np.float32)
        fid.close()
        cmp_min_max = cmp_min_max.reshape((2, -1))
        cmp_min_vector = cmp_min_max[0, ]
        cmp_max_vector = cmp_min_max[1, ]

        assert  cmp_min_vector.size == cfg.dur_dim and cmp_max_vector.size == cfg.dur_dim

        frame_number = features.size // cfg.dur_dim

        mean_matrix = np.tile(cmp_min_vector, (frame_number, 1))
        std_matrix = np.tile(cmp_max_vector   , (frame_number, 1))
        norm_features = features * std_matrix + mean_matrix

        return norm_features


def prepare_label_with_durations(label_file_name, gen_lab_file_name, dur_features):

        frame_number = dur_features.size // cfg.dur_dim
        dur_features = dur_features[:(cfg.dur_dim * frame_number)]
        dur_features = dur_features.reshape((-1, cfg.dur_dim))

        # Open original label file (assuming no duration information)
        fid = open(label_file_name)
        utt_labels = fid.readlines()
        fid.close()

        label_number = len(utt_labels)
        out_fid = open(gen_lab_file_name, 'w')

        current_index = 0
        prev_end_time = 0
        state_number = cfg.n_states

        for idx, line in enumerate(utt_labels):
            line = line.strip()

            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list)==1:
                start_time = 0
                end_time = 600000 ## hard-coded silence duration
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

                full_label_length = len(full_label) - 3  # remove state information [k]
                state_index = full_label[full_label_length + 1]
                state_index = int(state_index) - 1

            label_binary_flag = check_silence_pattern(full_label)

            if len(temp_list)==1:
                for state_index in range(1, state_number+1):
                    if label_binary_flag == 1:
                        current_state_dur = end_time - start_time
                    else:
                        pred_state_dur = dur_features[current_index, state_index-1]
                        current_state_dur = int(pred_state_dur)*5*10000
                    out_fid.write(str(prev_end_time)+' '+str(prev_end_time+current_state_dur)+' '+full_label+'['+str(state_index+1)+']\n')
                    prev_end_time = prev_end_time + current_state_dur
            else:
                if label_binary_flag == 1:
                    current_state_dur = end_time - start_time
                else:
                    pred_state_dur = dur_features[current_index, state_index-1]
                    current_state_dur = int(pred_state_dur)*5*10000
                out_fid.write(str(prev_end_time)+' '+str(prev_end_time+current_state_dur)+' '+full_label+'\n')
                prev_end_time = prev_end_time + current_state_dur

            if state_index == state_number and label_binary_flag!=1:
                current_index += 1




def save_binary(filename, data):

    save = np.array(data, 'float32')
    fid = open(filename, 'wb')
    save.tofile(fid)
    fid.close()


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
        torch.cuda.set_device(3)
        print(torch.cuda.current_device())

    print('Using device:', device)

    if not os.path.exists(cfg.gen_path):
        os.system("mkdir -p %s" %cfg.gen_path)


    duration_model = Merlin(cfg.input_size, cfg.hidden_size, cfg.dur_dim).to(device=device)
    model_restore_path = cfg.dur_latest_weights

    duration_model.restore(model_restore_path)

    print("\n Loading eval data ... \n")
    eval_dataset = get_tts_dataset(cfg.test_list,cfg.bin_no_sil_norm, None,cfg.dur_no_sil_norm, cfg.lab_dim, None, cfg.dur_dim)

    eval_dataloader = DataLoader(eval_dataset, batch_size=1,collate_fn=lambda batch : collate_tts(batch),
                            shuffle=True, num_workers=4)

    device = next(duration_model.parameters()).device

    ref_all_files_data = np.reshape(np.array([]), (-1,1))
    gen_all_files_data = np.reshape(np.array([]), (-1,1))

    correlation = []
    for idx, (x, _ , ids , d , frames, _) in enumerate(eval_dataloader):

        criterion = torch.nn.MSELoss()

        duration_model.eval()
        pred_dur = duration_model(x.cuda())


        r = np.sum(np_now(d.squeeze()), axis=1)
        ref_data = np.reshape(r, (-1, 1))
        ref_all_files_data = np.concatenate((ref_all_files_data, ref_data), axis=0)

        g = np.sum(np_now(pred_dur.squeeze()), axis=1)
        gen_data = np.reshape(g, (-1, 1))
        gen_all_files_data = np.concatenate((gen_all_files_data, gen_data), axis=0)

        denormalised_feats = denormalise(np_now(pred_dur))
        gen_features = np.int32(np.round(denormalised_feats))
        gen_features[gen_features<1]=1

        gen_lab_file_name = os.path.join(cfg.gen_path, str(ids[0]) + cfg.lab_extension)
        lab_filename = os.path.join(cfg.test_labels_dir, str(ids[0]) + cfg.lab_extension)
        filename = os.path.join(cfg.gen_path, str(ids[0]) + cfg.dur_extension)
        prepare_label_with_durations(lab_filename, gen_lab_file_name, gen_features)

        save_binary(filename, gen_features)

    rmse = compute_rmse(ref_all_files_data, gen_all_files_data)

    r = ref_all_files_data.squeeze()
    h = gen_all_files_data.squeeze()
    corr  = compute_corr(r, h)
    msg  = f'RMSE: {rmse} CORR: {corr}'
    print(msg)


    print('\n\nDone.\n')
