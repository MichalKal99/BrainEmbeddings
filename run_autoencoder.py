import train_autoencoder
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
        "training_epochs_autoencoder":100,
        "batch_size":512,
        "latent_dim":64,
        "pt_colors":pt_colors,
        "plot_latent":False,
    }

pt_arr = ["p00", "p01", "p06",  "p07",  "p08", "p09", "p10", "p11", "p12", "p16"]
# pt_arr = ["p00", "p01", "p06",  "p07",  "p08"]
# pt_arr = ["p00", "p01", "p06"]
# pt_arr = ["p00", "p01"]
# pt_arr = ["p01"]

config["clustering"]= {"n_classes":[len(pt_arr),16,32,64]}

if __name__ == '__main__':

    print(f"\nPatients: {pt_arr}")

    # Access the argument passed from the bash script
    if len(sys.argv) > 1:
        new_dir_path = sys.argv[1]
    else:
        new_dir_path = "."

    print("\n","#"*20, "RUNNING AUTOENCODER", "#"*20)
    train_autoencoder.main(new_dir_path, config, pt_arr)

    if config["plot_latent"]:
        print("\n","#"*20, "RUNNING CLUSTERING", "#"*20)
        latent_viz.main(config["latent_dim"], new_dir_path, config, pt_arr, 
                        latent_data_filename="source_encoded", speech_labels_filename="sorted_speech_labels")

    print("\n","#"*20, "DONE!", "#"*20)

