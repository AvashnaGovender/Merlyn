#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

def load_binary_file(file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        features = features[:(dimension * (features.size // dimension))]
        features = features.reshape((-1, dimension))

        return  features


def get_tts_dataset(file_list,labels_path, acoustic_path,duration_path, lab_dim, cmp_dim, dur_dim):

    dataset = TTSDataset(file_list,labels_path, acoustic_path,duration_path, lab_dim, cmp_dim, dur_dim)
    bar = ''
    for i in range(len(dataset)):

        labels, targets, ids, durations, frames, dur_lens = dataset[i]

    return dataset


def load_binary_file_frame(file_name, dimension):

        fid_lab = open(file_name, 'rb')
        features = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        frame_number = features.size // dimension
        features = features[:(dimension * frame_number)]
        features = features.reshape((-1, dimension))
        return features



class TTSDataset(Dataset):
    def __init__(self, file_list, labels_path,  acoustic_path, duration_path, lab_dim, cmp_dim, dur_dim):

        self.labels_path = labels_path
        self.acoustic_path = acoustic_path
        self.durations_path = duration_path
        self.lab_dim = lab_dim
        self.cmp_dim = cmp_dim
        self.dur_dim = dur_dim

        fid = open(file_list)
        all_filenames = fid.readlines()
        filenames = [x.strip() for x in all_filenames]
        self.file_list = filenames


    def __getitem__(self, idx):

        ids = self.file_list[idx]
        label_file = os.path.join(self.labels_path, ids+'.lab')



        labels = load_binary_file_frame(label_file, self.lab_dim)
        frames = labels.size // self.lab_dim

        targets = None
        durations = None
        dur_len = None

        if self.acoustic_path is not None:
            target_file = os.path.join(self.acoustic_path, ids+'.cmp')
            targets = load_binary_file_frame(target_file, self.cmp_dim)

        if self.durations_path is not None:
            duration_file = os.path.join(self.durations_path, ids+'.dur')
            durations = load_binary_file(duration_file, self.dur_dim)
            dur_len = durations.size

        return labels, targets, ids, durations, frames,  dur_len

    def __len__(self):
        return len(self.file_list)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    padded = np.pad(x, ((0, max_len-x.shape[0]),(0,0)), mode='constant')
    return padded


def collate_tts(batch):

        ids = [x[2] for x in batch]

        x_lens = [len(x[0]) for x in batch]
        max_x_len = max(x_lens)
        labels = [pad2d(x[0], max_x_len) for x in batch]
        labels = np.stack(labels)
        labels = torch.tensor(labels).clone().detach()

        targets = None
        durations = None
        dur_lens = None

        if batch[0][1] is not None:
            t_lens = [len(x[1]) for x in batch]
            max_t_len = max(t_lens)
            targets = [pad2d(x[1], max_t_len) for x in batch]
            targets = np.stack(targets)
            targets = torch.tensor(targets).clone().detach()

        if batch[0][3] is not None:
            d_lens = [len(x[3]) for x in batch]
            max_d_len = max(d_lens)
            durations = [pad2d(x[3], max_d_len) for x in batch]
            durations = np.stack(durations)
            durations = torch.tensor(durations).clone().detach()
            dur_lens = [x[5] for x in batch]


        frames = [x[4] for x in batch]


        return labels, targets,ids,durations, frames,dur_lens
