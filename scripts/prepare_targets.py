#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import sys
import multiprocessing
from multiprocessing import Pool, cpu_count
import load_config_test as cfg
from utils import save_binary, load_binary_file_frame


def compute_dynamic_vector(vector, dynamic_win, frame_number):

        vector = np.reshape(vector, (frame_number, 1))

        win_length = len(dynamic_win)
        win_width = int(win_length/2)
        temp_vector = np.zeros((frame_number + 2 * win_width, 1))
        delta_vector = np.zeros((frame_number, 1))

        temp_vector[win_width:frame_number+win_width] = vector
        for w in range(win_width):
            temp_vector[w, 0] = vector[0, 0]
            temp_vector[frame_number+win_width+w, 0] = vector[frame_number-1, 0]

        for i in range(frame_number):
            for w in range(win_length):
                delta_vector[i] += temp_vector[i+w, 0] * dynamic_win[w]

        return  delta_vector


def compute_dynamic_matrix(features, delta_win, frame_number, dimension):
    dynamic_matrix = np.zeros((frame_number, dimension))
    ###compute dynamic feature dimension by dimension
    for dim in range(dimension):
        dynamic_matrix[:, dim:dim+1] = compute_dynamic_vector(features[:, dim], delta_win, frame_number)

    return  dynamic_matrix


def interpolate_f0(data):

    data = np.reshape(data, (data.size, 1))

    vuv_vector = np.zeros((data.size, 1))
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i+1
            for j in range(i+1, frame_number):
                if data[j] > 0.0:
                    break
            if j < frame_number-1:
                if last_value > 0.0:
                    step = (data[j] - data[i-1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i-1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            ip_data[i] = data[i]
            last_value = data[i]

    return  ip_data, vuv_vector



def compute_mean(file_list, start_index, end_index):

        local_feature_dimension = end_index - start_index

        mean_vector = np.zeros((1, local_feature_dimension))

        all_frame_number = 0

        for file_name in file_list:
            features, current_frame_number = load_binary_file_frame(file_name, end_index)

            mean_vector += np.reshape(np.sum(features[:, start_index:end_index], axis=0), (1, local_feature_dimension))

            all_frame_number += current_frame_number

        mean_vector /= float(all_frame_number)

        print('computed mean vector')


        return  mean_vector



def compute_std(file_list, mean_vector, start_index, end_index):

        local_feature_dimension = end_index - start_index

        std_vector = np.zeros((1, local_feature_dimension))

        all_frame_number = 0

        for file_name in file_list:
            features, current_frame_number = load_binary_file_frame(file_name, end_index)

            mean_matrix = np.tile(mean_vector, (current_frame_number, 1))

            std_vector += np.reshape(np.sum((features[:, start_index:end_index] - mean_matrix) ** 2, axis=0), (1, local_feature_dimension))


            all_frame_number += current_frame_number

        std_vector /= float(all_frame_number)

        std_vector = std_vector ** 0.5
        print('computed std vector of length ')


        return  std_vector

def feature_normalisation(filename, out_file, mean_vector,std_vector, dim ):

    features, current_frame_number = load_binary_file_frame(filename, dim)

    mean_matrix = np.tile(mean_vector, (current_frame_number, 1))
    std_matrix = np.tile(std_vector, (current_frame_number, 1))

    norm_features = (features - mean_matrix) / std_matrix

    save_binary(out_file, norm_features)

def prepare_feats(out_file, idx, stream_dim_index, stream_start_index):

    out_data_matrix = None
    out_frame_number = 0


    for stream_name in cfg.feats.keys() :

        if stream_name == 'mgc':
            file_names = [os.path.join(cfg.mgc_dir, x + cfg.mgc_extension) for x in filenames]
        elif stream_name == 'bap':
            file_names = [os.path.join(cfg.bap_dir, x +cfg.bap_extension) for x in filenames]
        elif stream_name == 'lf0':
            file_names = [os.path.join(cfg.lf0_dir, x +cfg.lf0_extension) for x in filenames]

        if stream_name not in ['vuv']:
            in_file_name   = file_names[idx]
            in_feature_dim = cfg.feats_in[stream_name]

            features, frame_number = load_binary_file_frame(in_file_name, in_feature_dim)

            if out_frame_number == 0:
                        out_frame_number = frame_number
                        out_data_matrix = np.zeros((out_frame_number, stream_dim_index))

            if frame_number > out_frame_number:
                features = features[0:out_frame_number, ]
                frame_number = out_frame_number

            try:
                assert  out_frame_number == frame_number
            except AssertionError:
                print('the frame number of data stream %s is not consistent with others: current %d others %d'
                                     %(stream_name, out_frame_number, frame_number))
                raise

            dim_index = stream_start_index[stream_name]


            out_data_matrix[0:out_frame_number, dim_index:dim_index+in_feature_dim] = features
            dim_index = dim_index+in_feature_dim

            if stream_name in ['lf0', 'F0']:   ## F0 added for GlottHMM
                features, vuv_vector = interpolate_f0(features)

                ### if vuv information to be recorded, store it in corresponding column
                if cfg.record_vuv:
                    out_data_matrix[0:out_frame_number, stream_start_index['vuv']:stream_start_index['vuv']+1] = vuv_vector

            if cfg.compute_dynamic:

                delta_features = compute_dynamic_matrix(features, cfg.delta_win, frame_number, in_feature_dim)
                acc_features   = compute_dynamic_matrix(features, cfg.acc_win, frame_number, in_feature_dim)


                out_data_matrix[0:out_frame_number, dim_index:dim_index+in_feature_dim] = delta_features
                dim_index = dim_index+in_feature_dim

                out_data_matrix[0:out_frame_number, dim_index:dim_index+in_feature_dim] = acc_features


    ### write data to file
    save_binary(out_file, out_data_matrix)
    print('Wrote %d frames of features' %out_frame_number )

def remove_sil_acoustic(idx,indices_filenames, out_filenames):
    print(out_filenames[idx])
    fid_lab = open(indices_filenames[idx], 'rb')
    nonsilence_indices = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    nonsilence_indices = nonsilence_indices.astype(int)

    ori_cmp_data, frame_number = load_binary_file_frame(out_filenames[idx], cfg.cmp_dim)
    indices = [ix for ix in nonsilence_indices if ix < ori_cmp_data.shape[0]]
    new_cmp_data = ori_cmp_data[indices,]
    save_binary(out_filenames[idx], new_cmp_data)


if __name__ == '__main__':

    #Configuration

    prepare_features = False
    remove_silence = False
    normalisation = False
    dur_normalisation = True

    fid = open(cfg.file_id_list)
    all_filenames = fid.readlines()
    filenames = [x.strip() for x in all_filenames]
    dur_filenames = [os.path.join(cfg.duration_path, x + cfg.dur_extension) for x in filenames]
    lf0_filenames = [os.path.join(cfg.lf0_dir, x + cfg.lf0_extension) for x in filenames]
    out_filenames = [os.path.join(cfg.output_acoustic, x + cfg.cmp_extension) for x in filenames]
    indices_filenames = [os.path.join(cfg.indices_filepath, x + '.npy') for x in filenames]
    norm_filenames = [os.path.join(cfg.bin_acoustic_feats, x + cfg.cmp_extension )for x in filenames]
    dur_no_sil = [os.path.join(cfg.bin_no_sil_dur, x + cfg.dur_extension )for x in filenames]
    dur_norm_filenames = [os.path.join(cfg.dur_no_sil_norm, x + cfg.dur_extension )for x in filenames]

    if not os.path.exists(cfg.output_acoustic):
        os.system("mkdir -p %s" %cfg.output_acoustic)
        os.system("mkdir -p %s" %cfg.bin_acoustic_feats)

    jobs = []
    pool = Pool(processes=cpu_count())

    if prepare_features:
        # Load duration and lf0 files
        for idx,utt in enumerate(dur_filenames):

            print(f'Processing {idx+1}/{len(dur_filenames)}....')

            duration_feats, dur_frame_number = load_binary_file_frame(dur_filenames[idx], cfg.dur_dim)
            lf0_feats , lf0_frame_number = load_binary_file_frame(lf0_filenames[idx], cfg.lf0_dim)
            print(dur_filenames[idx])

            target_features = np.zeros((lf0_frame_number, cfg.dur_dim))
            if dur_frame_number == lf0_frame_number:
                continue
            elif dur_frame_number > lf0_frame_number:
                target_features[0:lf0_frame_number, ] = duration_feats[0:lf0_frame_number, ]
                save_binary(dur_filenames[idx], target_features)
            elif dur_frame_number < lf0_frame_number:
                target_features[0:dur_frame_number, ] = duration_feats[0:dur_frame_number, ]
                save_binary(dur_filenames[idx], target_features)

        print('Finished! Number of frames in dur and lf0 are equal')


        n_files = len(dur_filenames)

        stream_start_index = {}
        stream_dim_index = 0
        data_stream_number = 0

        for stream_name in cfg.feats.keys():
            if stream_name not in stream_start_index:
                stream_start_index[stream_name] = stream_dim_index

            stream_dim_index += cfg.feats[stream_name]

        print('Starting to prepare feats jobs ....')
        for idx, out_file in enumerate(out_filenames):

            job_process = pool.apply_async(prepare_feats, (out_file, idx, stream_dim_index, stream_start_index ))
            jobs.append(job_process)

        for job in jobs:
            job.get()


    # Remove the silece frames from the acoustic features
    job_remove_sil = []
    #Read the non-silence indices file
    if remove_silence:
        print('Stating remove silence jobs ....')
        for idx in range(len(filenames)):

            job_sil = pool.apply_async(remove_sil_acoustic, (idx, indices_filenames, out_filenames ))
            job_remove_sil.append(job_sil)

        for j in job_remove_sil:
            j.get()


    if normalisation:

        #Compute the mean and std vector for normlisation
        global_mean_vector = compute_mean(out_filenames, 0, cfg.feat_dimension)
        global_std_vector = compute_std(out_filenames, global_mean_vector, 0, cfg.feat_dimension)

        #Perform acoustic normalisation
        print('Stating normalisation job ....')
        jobs_norm = []
        for idx, filename in enumerate(out_filenames):

            job_norm = pool.apply_async(feature_normalisation, (filename, norm_filenames[idx] ,global_mean_vector, global_std_vector,cfg.feat_dimension ))
            jobs_norm.append(job_norm)

        for j in jobs_norm:
            j.get()

        cmp_norm_info = np.concatenate((global_mean_vector, global_std_vector), axis=0)
        cmp_norm_info = np.array(cmp_norm_info, 'float32')

        save_binary(cfg.norm_info, cmp_norm_info)

    if dur_normalisation:

        #Compute the mean and std vector for normlisation
        dur_global_mean_vector = compute_mean(dur_no_sil, 0, cfg.dur_dim)
        dur_global_std_vector = compute_std(dur_no_sil, dur_global_mean_vector, 0, cfg.dur_dim)

        #Perform acoustic normalisation
        print('Stating duration normalisation job ....')
        dur_jobs_norm = []
        for idx, filename in enumerate(dur_no_sil):

            dur_job_norm = pool.apply_async(feature_normalisation, (filename, dur_norm_filenames[idx] ,dur_global_mean_vector, dur_global_std_vector, cfg.dur_dim))
            dur_jobs_norm.append(dur_job_norm)

        for j in dur_jobs_norm:
            j.get()

        dur_norm_info = np.concatenate((dur_global_mean_vector, dur_global_std_vector), axis=0)
        dur_norm_info = np.array(dur_norm_info, 'float32')

        save_binary(cfg.dur_norm_info, dur_norm_info)
