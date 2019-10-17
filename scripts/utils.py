import numpy as np
from scipy.stats import pearsonr
import sys
import os

def compute_rmse(ref_data, gen_data):
        diff = (ref_data - gen_data) ** 2
        total_frame_number = ref_data.size
        sum_diff = np.sum(diff)
        rmse = np.sqrt(sum_diff/total_frame_number)

        return rmse

def compute_mse(ref_data, gen_data):
    diff = (ref_data - gen_data) ** 2
    sum_diff = np.sum(diff, axis=1)
    sum_diff = np.sqrt(sum_diff)       # ** 0.5
    sum_diff = np.sum(sum_diff, axis=0)

    #return ((gen_data - ref_data) ** 2).mean()
    return  sum_diff


def compute_corr(ref_data, gen_data):
    corr_coef = pearsonr(ref_data, gen_data)

    return corr_coef[0]

def check_silence_pattern(label, silence_pattern):
    current_pattern = silence_pattern.strip('*')

    if current_pattern in label:

                return 1
    return 0

def save_binary(filename, data):

    save = np.array(data, 'float32')
    fid = open(filename, 'wb')
    save.tofile(fid)
    fid.close()

def compute_f0_mse(ref_data, gen_data):
        print("f0_mse computations")

        ref_vuv_vector = np.zeros((ref_data.size, 1))
        gen_vuv_vector = np.zeros((ref_data.size, 1))

        ref_vuv_vector[ref_data > 0.0] = 1.0
        gen_vuv_vector[gen_data > 0.0] = 1.0

        sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        voiced_frame_number = voiced_gen_data.size
        print("voiced:", voiced_frame_number)
        f0_mse = (np.exp(voiced_ref_data) - np.exp(voiced_gen_data)) ** 2
        f0_mse = np.sum((f0_mse))

        vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
        vuv_error = np.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])



        #print(gen_data)
        #sys.exit()
        # voiced_ref_vuv_vector = [val for val in ref_data if val > 0.0]
        # unvoiced_ref_vuv_vector = [val for val in ref_data if val < 0.0]
        #
        # voiced_gen_vuv_vector = [val for val in gen_data if val > 0.0]
        # unvoiced_gen_vuv_vector = [val for val in gen_data if val < 0.0]
        #
        # ref_vuv_vector = np.zeros((ref_data.size, 1))
        # gen_vuv_vector = np.zeros((ref_data.size, 1))
        #
        # ref_vuv_vector[ref_data > 0.0] = 1.0
        # gen_vuv_vector[gen_data > 0.0] = 1.0
        #
        # sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        # voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        # voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        # voiced_frame_number = voiced_gen_data.size
        #
        # print("uv ref:",len(unvoiced_ref_vuv_vector))
        # print("voiced ref",len(voiced_ref_vuv_vector))
        #print(len(voiced_ref_vuv_vector) + len(unvoiced_ref_vuv_vector))

        # print("uv gen:",len(unvoiced_gen_vuv_vector))
        # print("voiced",len(voiced_gen_vuv_vector))
        #print(len(voiced_gen_vuv_vector) + len(unvoiced_gen_vuv_vector))

        # print(len(voiced_gen_data))
        #
        #
        # print("DIFF:")
        # print(voiced_ref_data)
        # print(voiced_gen_data)
        #
        # f0_mse = (np.exp(voiced_ref_data) - np.exp(voiced_gen_data)) ** 2
        #
        # f0_mse = np.sum((f0_mse))
        #print("1. f0mse",f0_mse)

        #
        # vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
        # vuv_error = np.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])
        #
        #

        #print("vuv error", vuv_error)
        #print("voiced",voiced_frame_number)

        return  f0_mse, vuv_error, voiced_frame_number


def load_binary_file(file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = np.fromfile(fid_lab, dtype=np.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        features = features[:(dimension * (features.size // dimension))]
        features = features.reshape((-1, dimension))

        return  features

def load_binary_file_frame(file_name, dimension):

    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()

    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    frame_number = features.size // dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))

    return  features, frame_number

def stream(message):
    sys.stdout.write(f"\r{message}")

def load_covariance(var_file_dict, var_dir):
        print("Loading covariance")
        var = {}
        print(var_file_dict)

        for feature_name in list(var_file_dict.keys()):
            var_file = os.path.join(var_dir, feature_name + ".var" )
            var_values, dimension = load_binary_file_frame(var_file, 1)

            var_values = np.reshape(var_values, (var_file_dict[feature_name], 1))
            print(var_values.shape)
            var[feature_name] = var_values

        return var


def compute_f0_corr(ref_data, gen_data):
    ref_vuv_vector = np.zeros((ref_data.size, 1))
    gen_vuv_vector = np.zeros((ref_data.size, 1))

    ref_vuv_vector[ref_data > 0.0] = 1.0
    gen_vuv_vector[gen_data > 0.0] = 1.0

    sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
    voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
    voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
    f0_corr = compute_corr(np.exp(voiced_ref_data), np.exp(voiced_gen_data))

    return f0_corr
