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
import load_config as cfg
from prepare_inputs import check_silence_pattern
from mlpg_fast import MLParameterGenerationFast as MLParameterGeneration

def np_now(x): return x.detach().cpu().numpy()

def equal_frames(ref_frame_number, gen_frame_number, ref_data, gen_data):
    #print("Ref frames:", ref_frame_number)
    #print("Gen frames:", gen_frame_number)

    if ref_frame_number == gen_frame_number:
        #print("equal")
        return ref_data, gen_data

    if abs(ref_frame_number - gen_frame_number) <= 2:
                ref_frame_number = min(ref_frame_number, gen_frame_number)
                gen_frame_number = min(ref_frame_number, gen_frame_number)
                ref_data = ref_data[0:ref_frame_number, ]
                gen_data = gen_data[0:gen_frame_number, ]
    else:
        print("frame mismatch!")

    return ref_data, gen_data

def calculate_objective_scores(filenames):

    mgc_mse = 0
    total_frames = 0
    voiced_frames = 0
    vuv_error = 0
    lf0_rmse = 0
    bap_mse = 0

    ref_all_files_data = np.reshape(np.array([]), (-1,1))
    gen_all_files_data = np.reshape(np.array([]), (-1,1))

    for filename in filenames:
        print("OBJECTIVE METRICS")
        print(filename)

        gen_mgc = os.path.join(cfg.gen_path, filename+".mgc" )
        gen_bap = os.path.join(cfg.gen_path, filename+".bap" )
        gen_lf0 = os.path.join(cfg.gen_path, filename+".lf0" )

        ref_mgc = os.path.join(cfg.mgc_dir, filename+".mgc" )
        ref_bap = os.path.join(cfg.bap_dir, filename+".bap" )
        ref_lf0 = os.path.join(cfg.lf0_dir, filename+".lf0" )

        # compute MCD
        r_mgc, frames1 = load_binary_file_frame(ref_mgc, cfg.dmgc_dim//3)
        g_mgc, frames2 = load_binary_file_frame(gen_mgc, cfg.dmgc_dim//3)


        indices_filename = os.path.join(cfg.acoustic_indices_filepath, filename+".npy")
        fid_lab = open(indices_filename, 'rb')
        nonsilence_indices = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        nonsilence_indices = nonsilence_indices.astype(int)
        #print("indices", len(nonsilence_indices))

        indices = [ix for ix in nonsilence_indices if ix < frames1]
        r_mgc = r_mgc[indices,]
        #print("ref feats nonsilence only", len(r_mgc))

        r_mgc, g_mgc = equal_frames(r_mgc.shape[0], g_mgc.shape[0], r_mgc, g_mgc)

        #print("MGC")
        #print(r_mgc.shape, g_mgc.shape)
        mse = compute_mse(r_mgc, g_mgc)
        mgc_mse += mse
        total_frames = total_frames + r_mgc.shape[0]


        #compute BAP
        r_bap, frames1 = load_binary_file_frame(ref_bap, cfg.dbap_dim//3)
        g_bap, frames2 = load_binary_file_frame(gen_bap, cfg.dbap_dim//3)


        #print("bap ref", r_bap.shape)
        #print("bap gen", g_bap.shape)


        r_bap = r_bap[indices,]

        r_bap, g_bap = equal_frames(r_bap.shape[0], g_bap.shape[0], r_bap, g_bap)

        #print("BAP")
        #print(r_bap.shape, g_bap.shape)
        comp_bap_mse = compute_mse(r_bap, g_bap)
        bap_mse += comp_bap_mse

        #compute RMSE
        r_lf0, f1 = load_binary_file_frame(ref_lf0, cfg.dlf0_dim//3)
        g_lf0, f2 = load_binary_file_frame(gen_lf0, cfg.dlf0_dim//3)

        r_lf0 = r_lf0[indices,]

        r_lf0, g_lf0 = equal_frames(r_lf0.shape[0], g_lf0.shape[0], r_lf0, g_lf0)
        print("Lf0")
        print(filename, r_lf0.shape, g_lf0.shape)
        #ref_all_files_data = np.concatenate((ref_all_files_data, ref_feats), axis=0)
        #gen_all_files_data = np.concatenate((gen_all_files_data, gen_feats), axis=0)

        f_mse, temp_vuv_error, voiced_frame_number = compute_f0_mse(r_lf0, g_lf0)

        lf0_rmse += f_mse
        voiced_frames += voiced_frame_number
        vuv_error += temp_vuv_error

    if voiced_frames != 0:
        lf0_rmse /= float(voiced_frames)
        vuv_error  /= float(total_frames)
        lf0_rmse = np.sqrt(lf0_rmse)

        #r = ref_all_files_data.squeeze()
        #h = gen_all_files_data.squeeze()
        #f0_corr = compute_f0_corr(r, h)

        vuv_error = vuv_error*100.
    else:
        lf0_rmse = None
        f0_corr = None
        vuv_error = None



    f0_corr = None

    mgc_mse /= float(total_frames)
    mgc_mse *= (10 / np.log(10)) * np.sqrt(2.0)

    bap_mse /=  float(total_frames)
    bap_mse = bap_mse / 10.0

    return lf0_rmse, vuv_error, f0_corr, mgc_mse, bap_mse





def acoustic_decomposition(features, frames, file_id, var_dict, stream_start_index, mlpg = True):
    mlpg_algo = MLParameterGeneration()

    #print(features.shape)
    #print("frames:",frames)
    #print(cfg.feats)
    #print(stream_start_index)

    for feature_name in stream_start_index:

            #print("Not VUV!")
            #print(feature_name)

            end = stream_start_index[feature_name]+cfg.feats[feature_name]
            #print(end)
            current_features = features[:, stream_start_index[feature_name]:end]

            #print("Avashna 0 - Before MLPG")
            #print(current_features.shape)
            #print("Merlin debug - Before MLPG")
            #print(current_features[:10])

            var = var_dict[feature_name]

            #print("Debugging!", feature_name)

            var = np.transpose(np.tile(var,frames))
            #print("var", var)
            if mlpg and feature_name not in ['vuv']:
                #print("mlpg ")
                gen_features = mlpg_algo.generation(current_features, var, cfg.feats[feature_name]//3)
                filename = os.path.join(cfg.gen_path, file_id + "."+feature_name)
            else:
                #print("no mlpg")
                gen_features = current_features
                filename = os.path.join(cfg.gen_path, file_id + "."+feature_name)
#
            #print("Avashna 1 - After MLPG")
            #print(gen_features[:10]*-1.0e-9)


            print(' feature dimensions: %d by %d' %(gen_features.shape[0], gen_features.shape[1]))

            if feature_name in ['lf0', 'F0']:

                    #print(gen_features[:100,])

                    #print("IN THIS LOOP")
                    if 'vuv' in stream_start_index:
                        #print("vuv present")
                        end = stream_start_index['vuv']+cfg.feats['vuv']
                        vuv_feature = features[:, stream_start_index['vuv']:end]
                        #print("vuv",vuv_feature.shape)
                        #print(frames)
                        count = 0
                        for i in range(frames):
                            if vuv_feature[i, 0] < 0.5 or gen_features[i, 0] < np.log(20):
                                gen_features[i, 0] = -1.0e+10
                                count = count +1
                        #print(count)

                    #print(gen_features[:100,])


            #print("Avashna 2")
            #print(gen_features)
            save_binary(filename, gen_features)
            print(' wrote to file %s' % filename)



def denormalise(features):

        fid = open(cfg.ac_norm_info, 'rb')
        cmp_min_max = np.fromfile(fid, dtype=np.float32)
        fid.close()
        cmp_min_max = cmp_min_max.reshape((2, -1))
        cmp_min_vector = cmp_min_max[0, ]
        cmp_max_vector = cmp_min_max[1, ]


        assert  cmp_min_vector.size == cfg.cmp_dim and cmp_max_vector.size == cfg.cmp_dim

        frame_number = features.size // cfg.cmp_dim

        mean_matrix = np.tile(cmp_min_vector, (frame_number, 1))
        std_matrix = np.tile(cmp_max_vector   , (frame_number, 1))

        denorm_features = features * std_matrix + mean_matrix

        return denorm_features



if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Merlin')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--validation', '-v', action='store_true', help='Computes objective metrics')
    args = parser.parse_args()

    val_mode = False

    if args.validation:
        val_mode = True


    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        print(torch.cuda.current_device())

    print('Using device:', device)

    if not os.path.exists(cfg.gen_path):
        os.system("mkdir -p %s" %cfg.gen_path)
        os.system("mkdir -p %s" %cfg.acoustic_no_sil)

    lab_dim = cfg.lab_dim + cfg.frame_feat_dim
    acoustic_model = Merlin(lab_dim, cfg.hidden_size, cfg.cmp_dim).to(device=device)
    model_restore_path = cfg.acoustic_latest_weights

    acoustic_model.restore(model_restore_path)
    lab_dim = cfg.lab_dim + cfg.frame_feat_dim
    print("\n Loading eval data ... \n")
    if val_mode:
        eval_dataset = get_tts_dataset(cfg.test_list,cfg.acoustic_bin_no_sil_norm, cfg.bin_acoustic_feats   , None, lab_dim, cfg.cmp_dim, None)

        eval_dataloader = DataLoader(eval_dataset, batch_size=1,collate_fn=lambda batch : collate_tts(batch),
                                shuffle=True, num_workers=4)
    else:
        eval_dataset = get_tts_dataset(cfg.test_list,cfg.acoustic_bin_no_sil_norm, None,None, lab_dim,cfg.cmp_dim, None)

        eval_dataloader = DataLoader(eval_dataset, batch_size=1,collate_fn=lambda batch : collate_tts(batch),
                                shuffle=True, num_workers=4)


    device = next(acoustic_model.parameters()).device

    fid = open(cfg.test_list)
    all_filenames = fid.readlines()
    filenames = [x.strip() for x in all_filenames]

    ref_all_files_data = np.reshape(np.array([]), (-1,1))
    gen_all_files_data = np.reshape(np.array([]), (-1,1))

    correlation = []
    stream_start_index = {}
    for idx, (x, t , ids , _ , frames, _ ) in enumerate(eval_dataloader):

            acoustic_model.eval()
            pred = acoustic_model(x.cuda())
            #print(np_now(pred).squeeze().shape)
            filename = os.path.join(cfg.gen_path, str(ids[0]) + cfg.cmp_extension)
            print(filename)
            save_binary(filename, np_now(pred).squeeze())


            denormalised_feats = denormalise(np_now(pred).squeeze())
            #
            filename = os.path.join(cfg.gen_path, str(ids[0]) + cfg.cmp_extension)
            save_binary(filename, denormalised_feats)
            #
            feats = cfg.feats
            var_dict = load_covariance(feats, cfg.var)
            #
            stream_start_index = {}
            dimension_index = 0

            for feature_name in list(feats.keys()):
                    stream_start_index[feature_name] = dimension_index
                    dimension_index += feats[feature_name]
            #print(stream_start_index)


            acoustic_decomposition(denormalised_feats, frames[0], ids[0], var_dict, stream_start_index)

    if val_mode:
        lf0_rmse, vuv_error, f0_corr, mgc_mse, bap_mse = calculate_objective_scores(filenames)
        msg  = f'RMSE: {lf0_rmse} F0_CORR: {f0_corr}, VUV_ERROR: {vuv_error} MCD: {mgc_mse} BAP:{bap_mse}'

        stream(msg)
    print('\n\nDone.\n')
