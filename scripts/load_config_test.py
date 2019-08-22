
import json
import os, os.path
from pathlib import Path

with open('config_test.json') as config_file:
    data = json.load(config_file)

model_id= data['model_id']
root = Path(os.path.dirname(os.getcwd()))
file_id_list= data['file_id_list']
train_list= data['train_list']
valid_list= data['valid_list']
test_list = data['test_list']
labels_dir= data['labels_dir']
test_labels_dir = data['test_labels_dir']
lf0_dir = data['lf0_dir']
mgc_dir = data['mgc_dir']
bap_dir = data['bap_dir']

label_feats_dir = 'label_feats'
acoustic_feats_dir = 'acoustic_feats'
duration_feats_dir = 'duration_feats'
question_filepath = data['question_filepath']

models_path=  os.path.join(root, model_id, "models" )

bin_no_sil = os.path.join(root, model_id, label_feats_dir, "nn_no_sil_481") # output dir for binary labels without silence
acoustic_bin_no_sil = os.path.join(root, model_id, label_feats_dir, "nn_no_sil_490") # output dir for binary labels without silence


bin_labels_path = os.path.join(root, model_id, label_feats_dir, "binary_label_481")
acoustic_bin_labels_path = os.path.join(root, model_id, label_feats_dir, "binary_label_490")

bin_no_sil_norm = os.path.join(root, model_id, label_feats_dir,"nn_no_silence_lab_norm_481")
acoustic_bin_no_sil_norm = os.path.join(root, model_id, label_feats_dir,"nn_no_silence_lab_norm_490 ")

indices_filepath = os.path.join(root, model_id, label_feats_dir,"non_silence_indices")
bin_acoustic_feats = os.path.join(root, model_id, acoustic_feats_dir,"norm_feats")
output_acoustic = os.path.join(root, model_id, acoustic_feats_dir,"nn_mgc_lf0_vuv_bap_187") # mgc, lf0, vuv, bap feats with 180+3+1+3
norm_info = os.path.join(root, model_id, acoustic_feats_dir,"norm_info.norm")


duration_path = os.path.join(root,model_id,duration_feats_dir, "durations")            # durations output directory
bin_no_sil_dur =  os.path.join(root,model_id,duration_feats_dir, "dur_no_sil")
dur_no_sil_norm = os.path.join(root,model_id,duration_feats_dir, "dur_norm")
dur_norm_info = os.path.join(root,model_id,duration_feats_dir, "dur_norm_info.norm")

gen_path = os.path.join(root,model_id, "gen")

dur_latest_weights = os.path.join(root, model_id, "models", "duration_model", "latest_model.pyt" )

lab_extension = ".lab"
dur_extension = ".dur"
lf0_extension = '.lf0'
mgc_extension = '.mgc'
bap_extension = '.bap'
cmp_extension = '.cmp'

lab_dim= data['lab_dim']
cmp_dim= data['cmp_dim']
dur_dim=data['dur_dim']

input_size= lab_dim
output_size= cmp_dim
max_dim = 100000

hidden_size= data['hidden_size']
lstm_hidden_size = data['lstm_hidden_size']

batch_size  = data['batch_size']
buffer= data['buffer']
buffer_size = int( buffer / batch_size) * batch_size
num_epochs = data['num_epochs']


n_states = data['n_states']
state_aligned = data['state_aligned']
feature_type= data['feature_type']
unit_size = data['unit_size']
feat_size = data['feat_size']
frame_length= data['frame_length']
label_dimension= data['label_dimension']
silence_pattern= data['silence_pattern']

add_frame_features = data['add_frame_features']
frame_feat_dim = data['frame_feat_dim']

remove_silence = data['remove_silence']
min_max_normalisation = data['min_max_normalisation']
target_max_value = data['target_max_value']
target_min_value = data['target_min_value']

delta_win = data['delta_win']
acc_win= data['acc_win']
mgc_dim= data['mgc_dim']
vuv_dim= data['vuv_dim']
lf0_dim= data['lf0_dim']
bap_dim= data['bap_dim']

dmgc_dim = mgc_dim * 3
dbap_dim =  bap_dim * 3
dlf0_dim = lf0_dim * 3

feats = {'mgc':dmgc_dim,'vuv':vuv_dim,'bap':dbap_dim,'lf0':dlf0_dim}
feats_in = {'mgc':mgc_dim,'vuv':vuv_dim,'bap':bap_dim,'lf0':lf0_dim}

feat_dimension = dmgc_dim + dbap_dim + dlf0_dim + 1
compute_dynamic = True
record_vuv = True

lr  = data['lr']
l1_reg = data['l1_reg']
dropout_rate= data['dropout_rate']
warmup_epoch= data['warmup_epoch']
reduce_lr= data['reduce_lr']
checkpoint_every_n = data['checkpoint_every_n']

TRAIN_DURATION = data['TRAIN_DURATION']
TRAIN_ACOUSTIC = data['TRAIN_ACOUSTIC']
