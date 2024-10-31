from training_files import train_autoencoder
import latent_viz
import sys 
from pathlib import Path 
import seaborn as sns
import os 

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
        'audio_reconstruction': {
            'method': 'vocgan', # griffin_lim / vocgan
        },
        "autoencoder_lr":1e-2,
        "training_epochs_autoencoder":20,
        "batch_size":512,
        "latent_dim":64,
        "pt_colors":pt_colors,
        "plot_latent":False,
    }

config['TR'] = {
    "latent_transformer":False,
    "number_reconstructions":60,
    "plot_gifs":True,
    "lr":1e-4,
    "training_epochs_transformer":50,
    "teacher_forcing":True,
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

# pt_arr = ["p00", "p01", "p06",  "p07",  "p08", "p09", "p10", "p11", "p12", "p16"]
# pt_arr = ["p00", "p01", "p06",  "p07",  "p08"]
# pt_arr = ["p00", "p01", "p06"]
# pt_arr = ["p00", "p01"]
pt_arr = ["p01"]

if __name__ == '__main__':

    print(f"\nPatients: {pt_arr}")

    # Access the argument passed from the bash script
    if len(sys.argv) > 1:
        run_dir_path = sys.argv[1]
    else:
        run_dir_path = "."

    # autoencoder dirs
    os.makedirs(os.path.join(run_dir_path, "latent_data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir_path, "reconstructions_autoencoder"), exist_ok=True)
    os.makedirs(os.path.join(run_dir_path, "clf_performance"), exist_ok=True)

    # general dirs
    os.makedirs(os.path.join(run_dir_path, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir_path, "labeled_speech"), exist_ok=True)

    # run model training
    print("\n","#"*20, "RUNNING AUTOENCODER", "#"*20)
    train_autoencoder.main(run_dir_path, config, pt_arr)

    # run latent space visualization
    if config["plot_latent"]:
        os.makedirs(os.path.join(run_dir_path, "clustering"), exist_ok=True)
        os.makedirs(os.path.join(run_dir_path, "frames"), exist_ok=True)
        os.makedirs(os.path.join(run_dir_path, "tsne"), exist_ok=True)
        print("\n","#"*20, "RUNNING CLUSTERING", "#"*20)
        latent_viz.main(config["latent_dim"], run_dir_path, config, pt_arr, 
                        latent_data_filename="source_encoded", speech_labels_filename="sorted_speech_labels")

    print("\n","#"*20, "DONE!", "#"*20)

