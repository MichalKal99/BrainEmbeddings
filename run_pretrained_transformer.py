from training_files import train_model_latent, train_pretrained_transformer
import latent_viz

import sys 
from pathlib import Path 
import seaborn as sns
import os 
from preprocessing.audio_utils import get_speech_labels
from preprocessing.loading_files import load_data
import numpy as np

# Create a list of unique colors for each string
colors = sns.color_palette("husl", len(["p00", "p01", "p06",  "p07",  "p08", "p09", "p10", "p11", "p12", "p16"]))
pt_colors = {string: color for string, color in zip(["p00", "p01", "p06",  "p07",  "p08", "p09", "p10", "p11", "p12", "p16"], colors)}

config = {
        "main_path":Path('.'),
        'dataset': 'ses1',
        'feature_extraction': {
            'method': 'VocGan', # MFCC / MelSpec / VocGan NOTE not a good idea to use mfcc as features the audio-mfcc-audio reconstruction is not good quality
            'num_feats': 80, # number of logmels / MFCCs
            'eeg_stacking': False,
            'compensate_eeg_audio_markers': False,
            'eeg_scaling': True # normalize eeg
        },
        'feature_selection': {
            'apply': False,
            'num_features': 150
        },
        'audio_reconstruction': {
            'method': 'vocgan', # griffin_lim / vocgan
        },
        'pca': {
            'apply': False,
            'num_components': 20
        },
        'hyperparameter_optimization': {
            'apply': False
        },
        'training': {
            'num_rands': 1000
        },
        "training_epochs_autoencoder":100,
        "autoencoder_lr":1e-2,
        "batch_size":512,
        "latent_dim":64,
        "pt_colors":pt_colors
    }

config['TR'] = {
    "latent_transformer":False,
    "number_reconstructions":40,
    "lr":1e-4,
    "training_epochs_transformer":1,
    "teacher_forcing":False,
    "sch_sampling":False,
    "noamOpt":False,
    "warmup":1600,
    "eps":1e-9,
    "betas":(0.9, 0.98),
    "encoder_seq_len":100,
    'context_length': 100,  # Adjusted sequence length
    'context_offset': 5, # int(0.050 * config['feature_extraction']['eeg_sr']),
    'embedding_size': 256,  # embedding size
    'hidden_size': 256,  # Number of output values
    'num_heads': 8,
    'criterion': 'mse', # mcd / mse
    'dropout': 0.5,
    'shuffle_dataset': False,
    'encoder_layers': 3,
    "decoder_layers":3
    }


# pt_arr = ["p00", "p06",  "p07",  "p08", "p09", "p10", "p11", "p12", "p16"]
# pt_arr = ["p00", "p01", "p06",  "p07",  "p08"]
# pt_arr = ["p00", "p01", "p06"]
# pt_arr = ["p00", "p06"]
pt_arr = ["p06"]

if __name__ == '__main__':
    print(config)
    print(f"\nPatients: {pt_arr}")
    
    # Access the argument passed from the bash script
    if len(sys.argv) > 1:
        new_dir_path = sys.argv[1]
        print(new_dir_path)
    else:
        new_dir_path = "."

    # autoencoder dirs
    os.makedirs(os.path.join(new_dir_path, "latent_data"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "reconstructions_autoencoder"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "clf_performance"), exist_ok=True)

    # transformer dirs
    os.makedirs(os.path.join(new_dir_path, "reconstructions_final"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "reconstructions_training"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "labeled_speech"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "results_images"), exist_ok=True)

    pt_data = {pt_id:{} for pt_id in pt_arr}

    # use latent transformer
    if config["TR"]["latent_transformer"]:

        print("\n","#"*20, "RUNNING AUTOENCODER", "#"*20)
        train_model_latent.main(new_dir_path, config, pt_arr)

        # for each patient prepare their dataset
        for pt_id in pt_arr:
            train_src = np.load(f"{new_dir_path}/latent_data/train_source_encoded_{pt_id}.npy")
            train_tgt = np.load(f"{new_dir_path}/latent_data/train_target_encoded_{pt_id}.npy")
            val_src = np.load(f"{new_dir_path}/latent_data/val_source_encoded_{pt_id}.npy")
            val_tgt = np.load(f"{new_dir_path}/latent_data/val_target_encoded_{pt_id}.npy")
            test_src = np.load(f"{new_dir_path}/latent_data/test_source_encoded_{pt_id}.npy")
            test_tgt = np.load(f"{new_dir_path}/latent_data/test_target_encoded_{pt_id}.npy")

            pt_data[pt_id]["train_src"] = train_src
            pt_data[pt_id]["train_tgt"] = train_tgt
            pt_data[pt_id]["val_src"] = val_src
            pt_data[pt_id]["val_tgt"] = val_tgt
            pt_data[pt_id]["test_src"] = test_src
            pt_data[pt_id]["test_tgt"] = test_tgt

            config["TR"]['num_features'] = config["latent_dim"]

        # run model training
        print("\n","#"*20, "RUNNING TRANSFORMER", "#"*20)
        train_pretrained_transformer.main(new_dir_path, config, pt_data, transformer_type="latent_pretrained_")
        
    # use normal transformer
    else:
        # for each patient run their dataset
        for pt_id in pt_arr:
            config["p_id"] = pt_id
            src, tgt, _, _, _, _, _, _, _ = load_data(config)

            speech_labels = get_speech_labels(tgt)

            print(f"src: {src.shape}")
            print(f"tgt: {tgt.shape}")
            print(f"speech_labels: {speech_labels.shape}")
            
            pt_data[pt_id]["source"] = src
            pt_data[pt_id]["target"] = tgt
            pt_data[pt_id]["speech_labels"] = speech_labels
        
        train_pretrained_transformer.main(new_dir_path, config, pt_data, transformer_type="pretrained_")


    # make some dirs for latent space viz. 
    os.makedirs(os.path.join(new_dir_path, "clustering"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "frames"), exist_ok=True)
    os.makedirs(os.path.join(new_dir_path, "tsne"), exist_ok=True)
    
    # run latent space visualziation
    print("\n","#"*20, "RUNNING CLUSTERING", "#"*20)
    latent_viz.main(config["TR"]["embedding_size"], new_dir_path, config, pt_arr, 
                      latent_data_filename="test_source_encoded_transformer", speech_labels_filename="sorted_test_speech_labels")

    print("\n","#"*20, "DONE!", "#"*20)

