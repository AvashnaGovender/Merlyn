#!/usr/bin/env python
# coding: utf-8

import re
import sys
import os
import numpy as np
from dataloader 
import load_config_test as cfg
import multiprocessing
from multiprocessing import Pool, cpu_count
from utils import check_silence_pattern, save_binary, load_binary_file

# HTS Normalisation Class converts HTS labels files into continuous or binary values.
# Time alignment is expected to be present.
#
# It is assumed that the HTS label files have the following format:
#
#     3050000 3100000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[2]
#
#     3100000 3150000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[3]
#
#     3150000 3250000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[4]
#
#     3250000 3350000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[5]
#
#     3350000 3900000 xx~#-p+l=i:1_4/A/0_0_0/B/1-1-4:1-1&1-4#1-3$1-4>0-1<0-1|i/C/1+1+3/D/0_0/E/content+1:1+3&1+2#0+1/F/content_1/G/0_0/H/4=3:1=1&L-L%/I/0_0/J/4+3-1[6]
#
#     The starting and ending time are: 305000 310000
#     [2], [3], [4], [5], [6] signify the HMM state index.
#

def wildcards2regex(question, convert_number_pattern=False):
        """
        Convert HTK-style question into regular expression for searching labels.
        If convert_number_pattern, keep the following sequences unescaped for
        extracting continuous values):
            (\d+)       -- handles digit without decimal point
            ([\d\.]+)   -- handles digits with and without decimal point
        """

        ## handle HTK wildcards (and lack of them) at ends of label:
        prefix = ""
        postfix = ""
        if '*' in question:
            if not question.startswith('*'):
                prefix = "\A"
            if not question.endswith('*'):
                postfix = "\Z"
        question = question.strip('*')
        question = re.escape(question)
        ## convert remaining HTK wildcards * and ? to equivalent regex:
        question = question.replace('\\*', '.*')
        question = question.replace('\\?', '.')
        question = prefix + question + postfix

        if convert_number_pattern:
            question = question.replace('\\(\\\\d\\+\\)', '(\d+)')
            question = question.replace('\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
        return question



# Step 1 is to load the information from the questions file

# Populates the dictionary of all the regular expressions in the questions file
# QS: Contextual features, CQS: continous numerical features
# Returns a dictionaries with indexes 1 - N
# the len of both dictionaries should equal to the number of questions in the questionfile

#Initalise variables
def create_questions_query(questions):

    qs_index = 0
    continuous_qs_index = 0
    questions_dict = {}
    continuous_dict = {}

    LL=re.compile(re.escape('LL-'))


    for line in questions:
        line = line.replace('\n', '').replace('\t', ' ')
        if len(line) > 5:

            temp_line = line.split('{')[1]
            temp_line = temp_line.split('}')[0]
            temp_line = temp_line.strip()
            question_list = temp_line.split(',')

            split = line.split(" ")
            question_key = split[1]

            if split[0] == 'CQS':
                        assert len(question_list) == 1
                        processed_question = wildcards2regex(question_list[0], convert_number_pattern=True)
                        continuous_dict[str(continuous_qs_index)] = re.compile(processed_question) #save pre-compiled regular expression
                        continuous_qs_index = continuous_qs_index + 1

            elif split[0] == 'QS':
                re_list = []
                for temp_question in question_list:
                    processed_question = wildcards2regex(temp_question)

                    if LL.search(question_key):
                        processed_question = '^'+processed_question
                    re_list.append(re.compile(processed_question))

                questions_dict[str(qs_index)] = re_list
                qs_index = qs_index + 1
        else:
            print("Error in QS file")

    assert len(questions_dict) + len(continuous_dict) == n_questions

    return questions_dict, continuous_dict


# Step 2: Prepare the labels by extracting the linguistic features from the standard HTS format (NORMLAB) and transforming them into 1-hot vectors. Each label file is saved as a binary file.

# Step 3: Silence Remover - silence is removed and typically these files are saved in nn_no_silence_lab_481

# One hot encoding for each question in the question file

def pattern_matching_binary(label, questions_dict, continous = False ):
        dim = len(questions_dict)
        lab_binary_vector = np.zeros((1, dim))

        if not continous:
            for i in range(dim):
                current_question_list = questions_dict[str(i)]
                binary_flag = 0
                for iq in range(len(current_question_list)):
                    current_compiled = current_question_list[iq]
                    ms = current_compiled.search(label)
                    if ms is not None:
                        binary_flag = 1
                        break
                lab_binary_vector[0, i] = binary_flag
        else:
            for i in range(dim):
                continuous_value = -1.0
                current_compiled = questions_dict[str(i)]
                ms = current_compiled.search(label)
                if ms is not None:
                    continuous_value = ms.group(1)
                lab_binary_vector[0, i] = continuous_value


        return  lab_binary_vector


def load_aligned_labels(input_file, questions_dict, continuous_dict):


    state_number = cfg.n_states

    label_feature_index = 0

    if cfg.add_frame_features:
         full_dim = cfg.lab_dim + 9

    else:
         full_dim = cfg.label_dimension

    label_feature_matrix = np.empty((cfg.max_dim, full_dim))

    # Initialisation for duration info
    if cfg.state_aligned:
        dur_dim = cfg.n_states

    if cfg.feature_type == "numerical" and cfg.unit_size == "state":
            dur_feature_matrix = np.empty((cfg.max_dim, dur_dim))
            current_dur_array = np.zeros((dur_dim, 1))
    else:
            dur_feature_matrix = np.empty((cfg.max_dim, 1))

    dur_feature_index = 0


    #Initialisation for linguistic features

    current_index = 0

    # Remove Silences
    nonsilence_frame_index_list = []
    base_frame_index = 0
    total_frames = 0


    # Read file list
    fid = open(input_file)
    utt_labels = fid.readlines()
    fid.close()

    label_number = len(utt_labels)
    print('loaded %s, %3d labels' % (input_file, label_number) )

    # Additional feats
    phone_duration = 0
    state_duration_base = 0

    for line in utt_labels:
        line = line.strip()

        if len(line) < 1:
            continue

        file_list = re.split('\s+', line)

        if len(file_list)==1:
            frame_number = 0
            state_index = 1
            full_label = file_list[0]
        else:

            start_time = int(file_list[0])
            end_time = int(file_list[1])
            frame_number = int(end_time/cfg.frame_length) - int(start_time/cfg.frame_length)
            total_frames = total_frames + frame_number
            full_label = file_list[2]


            full_label_length = len(full_label) - 3  # remove state information [k]
            state_index = full_label[full_label_length + 1]

            state_index = int(state_index) - 1
            state_index_backward = cfg.n_states + 1 - state_index
            full_label = full_label[0:full_label_length]
            current_phone = full_label[full_label.index('-') + 1:full_label.index('+')]


        if state_index == 1:
                current_frame_number = 0
                phone_duration = frame_number
                state_duration_base = 0

                # Binary vectors for the discrete features
                label_binary_vector = pattern_matching_binary(full_label, questions_dict)


                # Binary vectors for the continuous features
                label_continuous_vector = pattern_matching_binary(full_label, continuous_dict, True)


                # Concatenate them - Can also summate them
                label_vector = np.concatenate([label_binary_vector, label_continuous_vector], axis = 1)

                if len(file_list)==1:
                    state_index = state_number
                else:

                    for i in range(state_number - 1):
                        line = utt_labels[current_index + i + 1].strip()
                        file_list = re.split('\s+', line)
                        phone_duration += int((int(file_list[1]) - int(file_list[0]))/cfg.frame_length)


        if cfg.add_frame_features:

            dimension = cfg.label_dimension
            current_block_binary_array = np.zeros((frame_number, full_dim))
            for i in range(frame_number):
                current_block_binary_array[i, 0:dimension] = label_vector

                ## Zhizheng's original 9 subphone features:
                current_block_binary_array[i, dimension+0] = float(i+1) / float(frame_number)   ## fraction through state (forwards)
                current_block_binary_array[i, dimension+1] = float(frame_number - i) / float(frame_number)  ## fraction through state (backwards)
                current_block_binary_array[i, dimension+2] = float(frame_number)  ## length of state in frames
                current_block_binary_array[i, dimension+3] = float(state_index)   ## state index (counting forwards)
                current_block_binary_array[i, dimension+4] = float(state_index_backward) ## state index (counting backwards)

                current_block_binary_array[i, dimension+5] = float(phone_duration)   ## length of phone in frames
                current_block_binary_array[i, dimension+6] = float(frame_number) / float(phone_duration)   ## fraction of the phone made up by current state
                current_block_binary_array[i, dimension+7] = float(phone_duration - i - state_duration_base) / float(phone_duration) ## fraction through phone (backwards)
                current_block_binary_array[i, dimension+8] = float(state_duration_base + i + 1) / float(phone_duration)  ## fraction through phone (forwards)


            label_feature_matrix[label_feature_index:label_feature_index+frame_number,] = current_block_binary_array
            label_feature_index = label_feature_index + frame_number
        elif state_index == state_number:
            current_block_binary_array = label_vector
            label_feature_matrix[label_feature_index:label_feature_index+1,] = current_block_binary_array


            if cfg.remove_silence:

                silence_binary_flag = check_silence_pattern(full_label)

                if silence_binary_flag == 0:
                    nonsilence_frame_index_list.append(label_feature_index)

            label_feature_index = label_feature_index + 1
                    #base_frame_index = base_frame_index + 1


        if cfg.feature_type == "binary":
            current_block_array = np.zeros((frame_number, 1))
            if cfg.unit_size == "state":
                current_block_array[-1] = 1
            elif cfg.unit_size == "phoneme":
                if state_index == state_number:
                    current_block_array[-1] = 1
            else:
                print("Unknown unit size: %s \n Please use one of the following: state, phoneme\n" %(unit_size))

        elif cfg.feature_type == "numerical":
            if cfg.unit_size == "state":
                current_dur_array[current_index%cfg.n_states] = frame_number
                if cfg.feat_size == "phoneme" and state_index == state_number:
                    current_block_array =  current_dur_array.transpose()

            elif state_index == state_number:
                if cfg.unit_size == "phoneme":
                    current_block_array = np.array([phone_duration])
        ### writing into dur_feature_matrix ###
        if cfg.feat_size == "frame":
            dur_feature_matrix[dur_feature_index:dur_feature_index+frame_number,] = current_block_array
            dur_feature_index = dur_feature_index + frame_number
        elif state_index == state_number:
            if cfg.feat_size == "phoneme":
                dur_feature_matrix[dur_feature_index:dur_feature_index+1,] = current_block_array
                dur_feature_index = dur_feature_index + 1


        state_duration_base += frame_number
        current_index += 1




    label_feature_matrix = label_feature_matrix[0:label_feature_index,]
    print('made label matrix of %d frames x %d labels' % label_feature_matrix.shape )
    dur_feature_matrix = dur_feature_matrix[0:dur_feature_index,]
    print('made duration matrix of %d frames x %d features' % dur_feature_matrix.shape )


    if len(nonsilence_frame_index_list) == total_frames:
        print('WARNING: no silence found!')

    nonsilence_indices = [ix for ix in nonsilence_frame_index_list if ix < total_frames]
    print(nonsilence_indices)
    print("silence frames", total_frames - len(nonsilence_frame_index_list))
    no_silence = label_feature_matrix[nonsilence_indices,]
    print('made label matrix with no sil of %d frames x %d labels' % no_silence.shape )
    dur_no_silence = dur_feature_matrix[nonsilence_indices,]


    return label_feature_matrix, no_silence, dur_feature_matrix, nonsilence_indices, dur_no_silence



# Step 4: Min-max normalisation - Input is the file from remove silence directory

def get_min_max(min_value_matrix, max_value_matrix, full_dim):

    min_vector = np.amin(min_value_matrix, axis = 0)
    max_vector = np.amax(max_value_matrix, axis = 0)
    min_vector = np.reshape(min_vector, (1, full_dim))
    max_vector = np.reshape(max_vector, (1, full_dim))


    return min_vector, max_vector


def process(filename, questions_dict,continuous_dict):

        basename_ext = filename.split("/")[2]
        output_file_name = os.path.join(cfg.bin_labels_path , basename_ext)
        output_no_sil_file = os.path.join(cfg.bin_no_sil, basename_ext)
        basename = basename_ext[:-4]
        output_dur_filename = os.path.join(cfg.duration_path,  basename + cfg.dur_extension)
        indices_filename = os.path.join(cfg.indices_filepath, basename + '.npy')
        dur_no_sil_file = os.path.join(cfg.bin_no_sil_dur, basename + cfg.dur_extension)

        label_feature_matrix, no_silence, durations, nonsilence_indices, dur_no_silence = load_aligned_labels(filename, questions_dict,continuous_dict)

        # Binary label features
        # label_feature_matrix = load_state_aligned_labels(filename, state=False) #for phone aligned labels
        save_binary(output_file_name, label_feature_matrix)

        save_binary(indices_filename, nonsilence_indices)

        # Durations
        save_binary(output_dur_filename, durations)

        # No silence
        if cfg.remove_silence:
            print("Saving no silence labs and durs")
            save_binary(output_no_sil_file, no_silence)
            save_binary(dur_no_sil_file, dur_no_silence)


def normalise(filename, full_dim, min_vector, max_vector):

    no_sil_filename = os.path.join(cfg.bin_no_sil, filename + cfg.lab_extension)

    fea_max_min_diff = max_vector - min_vector
    diff_value = cfg.target_max_value - cfg.target_min_value
    fea_max_min_diff = np.reshape(fea_max_min_diff, (1, full_dim))
    target_max_min_diff = np.zeros((1, full_dim))
    target_max_min_diff.fill(diff_value)
    target_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
    fea_max_min_diff[fea_max_min_diff <= 0.0] = 1.0

    features = load_binary_file(no_sil_filename, full_dim)
    frame_number = features.size // full_dim
    fea_min_matrix = np.tile(min_vector, (frame_number, 1))
    target_min_matrix = np.tile(cfg.target_min_value, (frame_number, full_dim))

    fea_diff_matrix = np.tile(fea_max_min_diff, (frame_number, 1))
    diff_norm_matrix = np.tile(target_max_min_diff, (frame_number, 1)) / fea_diff_matrix

    norm_features = diff_norm_matrix * (features - fea_min_matrix) + target_min_matrix
    out_file = os.path.join(cfg.bin_no_sil_norm, filename + cfg.lab_extension)
    save_binary(out_file, norm_features)



#### MAIN FUNCTION ###
if __name__ == '__main__':

# Assumes only QS type question file for now
    with open(cfg.question_filepath, "r") as f:
        questions = f.readlines()

    n_questions = len(questions)

    if cfg.add_frame_features:
            full_dim = cfg.lab_dim + 9 # 5 state features + 4 phone feature
    else:
        full_dim = cfg.lab_dim

    questions_dict, continuous_dict = create_questions_query(questions)

    fid = open(cfg.file_id_list)
    all_filenames = fid.readlines()
    filenames = [x.strip() for x in all_filenames]
    lab_filenames = [cfg.labels_dir + x + cfg.lab_extension for x in filenames]
    file_number = len(all_filenames)

    if cfg.min_max_normalisation:
        min_value_matrix = np.zeros((file_number, full_dim))
        max_value_matrix = np.zeros((file_number, full_dim))


    if not os.path.exists(cfg.bin_labels_path):
        os.system("mkdir -p %s" %cfg.bin_labels_path)
        os.system("mkdir -p %s" %cfg.indices_filepath)
        os.system("mkdir -p %s" %cfg.bin_no_sil)
        os.system("mkdir -p %s" %cfg.bin_no_sil_norm)
        os.system("mkdir -p %s" %cfg.duration_path)
        os.system("mkdir -p %s" %cfg.bin_no_sil_dur)
        os.system("mkdir -p %s" %cfg.dur_no_sil_norm)


    jobs = []
    pool = Pool(processes=cpu_count())

    for idx,filename in enumerate(lab_filenames):

        job_process = pool.apply_async(process, (filename,questions_dict, continuous_dict ))
        jobs.append(job_process)

    for job in jobs:
        job.get()


    for idx,filename in enumerate(filenames):

        no_sil_filename = os.path.join(cfg.bin_no_sil, filename + cfg.lab_extension)
        #Load no silence
        no_silence = load_binary_file(no_sil_filename, full_dim)

        #Extarct min-max
        if cfg.min_max_normalisation:
            temp_min = np.amin(no_silence, axis = 0)
            temp_max = np.amax(no_silence, axis = 0)
            min_value_matrix[idx, ] = temp_min;
            max_value_matrix[idx, ] = temp_max;


    if cfg.min_max_normalisation:


        min_vector, max_vector = get_min_max(min_value_matrix,max_value_matrix, full_dim)

        label_norm_info = np.concatenate((min_vector, max_vector), axis=0)
        # Save min max normalisation

        save_binary(cfg.norm_info, label_norm_info)
        print('saved %s vectors to %s' %(min_vector.size, cfg.norm_info))


        norm_jobs = []
        for idx,filename in enumerate(filenames):

            job_normalise = pool.apply_async(normalise, (filename,full_dim, min_vector, max_vector ))
            norm_jobs.append(job_normalise)


        for job in norm_jobs:
            job.get()

    print("/n Done! /n")
