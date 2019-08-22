import numpy as np
from scipy.stats import pearsonr


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

    return  sum_diff


def compute_corr(ref_data, gen_data):
    corr_coef = pearsonr(ref_data, gen_data)

    return corr_coef[0]

def check_silence_pattern(label):
    current_pattern = cfg.silence_pattern.strip('*')

    if current_pattern in label:

                return 1
    return 0

def save_binary(filename, data):

    save = np.array(data, 'float32')
    fid = open(filename, 'wb')
    save.tofile(fid)
    fid.close()


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
