from pathlib import Path
import numpy as np
from . import xdf
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler

try:
    from . import preprocessing_neural
except:
    print('failed to load preprocessing_neural')
try:
    from . import preprocessing_audio
except:
    print('failed to load preprocessing_audio')
try:
    from . import preprocessing_generic
except:
    print('failed to load preprocessing_generic')
try:
    from . import preprocessing_markers
except:
    print('failed to load preprocessing_markers')



def fast_load(path_file, avoid_fast=False):
    path_file = Path(path_file)
    names = ["eeg", "eeg_sr", "audio", "audio_sr", "ch_names", "markers"]
    pre_loaded_path = Path(path_file.parent / str(path_file.stem + "_preloaded"))

    if pre_loaded_path.exists() and not avoid_fast:
        data = []
        # load the pre-loaded data
        for d_name in names:
            with open(pre_loaded_path / str(d_name + ".npy"), "rb") as f:
                # some data was pickled because it's a complex data structure, so the flag must be set
                data.append(np.load(f, allow_pickle=True))
        eeg, eeg_sr, audio, audio_sr, ch_names, markers = data[0], int(data[1]), data[2], int(data[3]), list(data[4]), data[5].item()
 

    else:
        # load the data inside the xdf file
        data = xdf.load_data(str(path_file))

        # save the data in the pre_loaded_data folder
        for d_name, dataset in zip(names, data):
            pre_loaded_path.mkdir(parents=True, exist_ok=True)
            with open(pre_loaded_path / str(d_name + ".npy"), "wb") as f:
                np.save(f, dataset)
        eeg, eeg_sr, audio, audio_sr, ch_names, markers = data[0], data[1], data[2], data[3], data[4], data[5]

    audio_wav_file = Path(pre_loaded_path / 'audio.wav')
    if not audio_wav_file.exists(): 
        wav.write(filename=audio_wav_file, rate=int(audio_sr), data=audio)

    return eeg, eeg_sr, audio, audio_sr, audio_wav_file, ch_names, markers


def save_data_to_file(data, full_file_path):
    full_file_path = Path(full_file_path)
    full_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_file_path, "wb") as f:
        np.save(f, data)


def load_data_from_file(full_file_path):
    with open(full_file_path, "rb") as f:
        output_data = np.load(f, allow_pickle=True)
    return output_data


def load_data(config):
    main_path = config['main_path']
    p_id = config['p_id']
    dataset_name = config['dataset']
    eeg_stacking = config['feature_extraction']['eeg_stacking']
    feat_ext_audio_method = config['feature_extraction']['method']
    num_feats = config['feature_extraction']['num_feats']
    audio_reconstruction_method = config['audio_reconstruction']['method']
    compensate_eeg_audio_markers = config['feature_extraction']['compensate_eeg_audio_markers']
    eeg_scaling = config['feature_extraction']['eeg_scaling']

    # get raw
    eeg, eeg_sr, audio, audio_sr, audio_wav_file, org_ch_names, markers = fast_load(Path(f'sentences_data\\sentences\\{p_id}_{dataset_name}_sentences.xdf'))

    # clean markers
    markers_time_series_stripped_small, markers_time_stamps_norm_small = preprocessing_markers.clean_markers(markers.get('time_series'), markers.get('time_stamps'))

    # compensate different sizes in the eeg, audio and markers data
    if compensate_eeg_audio_markers:
        eeg, audio = preprocessing_markers.compensate_eeg_audio_with_markers(eeg, eeg_sr, audio, audio_sr, markers_time_stamps_norm_small)


    # get feats
    p_eeg_feats = Path(main_path/'preloaded_features'/f"EEGfeats_{p_id}_{dataset_name}_{eeg_stacking}_{eeg_scaling}.npy")
    p_audio_feats = Path(main_path/'preloaded_features'/f"AUDIOfeats_{p_id}_{dataset_name}_{feat_ext_audio_method}_{num_feats}.npy")
    p_ch_names = Path(main_path/'preloaded_features'/f"used_ch_names_{p_id}_{dataset_name}_{eeg_stacking}.npy")
    if p_eeg_feats.exists() and p_ch_names.exists() and p_audio_feats.exists():
        eeg_feats = load_data_from_file(p_eeg_feats)
        audio_feats = load_data_from_file(p_audio_feats)
        eeg_ch_names = load_data_from_file(p_ch_names)
    else:
        eeg, eeg_feats, eeg_ch_names = preprocessing_neural.extract_eeg_features(eeg, eeg_sr, org_ch_names, stacking=eeg_stacking)
        audio_feats = preprocessing_audio.extract_audio_features(audio, audio_sr, feat_ext_audio_method, num_feats)
        

        # Z-Scale the eeg data ensuring the feature distributions have mean = 0 and std = 1
        if eeg_scaling:
            scaler = StandardScaler()
            scaler.fit(eeg_feats)
            eeg_feats = scaler.transform(eeg_feats)


        p_eeg_feats.parent.mkdir(parents=True, exist_ok=True)
        p_audio_feats.parent.mkdir(parents=True, exist_ok=True)
        p_ch_names.parent.mkdir(parents=True, exist_ok=True)
        
        # align features
        eeg_feats, audio_feats = preprocessing_generic.align_features(eeg_feats, audio_feats)
        

        np.save(p_eeg_feats, eeg_feats)
        np.save(p_audio_feats, audio_feats)
        np.save(p_ch_names, eeg_ch_names)

    baseline_audio = Path(main_path/'preloaded_features'/f"{p_id}_baseline_{dataset_name}_feat{feat_ext_audio_method}_rec{audio_reconstruction_method}")
    if not baseline_audio.with_suffix('.wav').exists():
        print(f'reconstructing baseline audio')
        preprocessing_audio.reconstruct_and_save_audio(main_path/'preloaded_features', baseline_audio.stem, audio_feats, feat_ext_audio_method, audio_reconstruction_method, window_size=0.0464375, window_shift=0.0115625, sampling_rate=16000)
    

    return eeg_feats, audio_feats, eeg_ch_names, eeg, eeg_sr, audio, audio_sr, audio_wav_file, markers