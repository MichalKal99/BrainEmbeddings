import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns

from models.transformer import *
from models.model_utils import *
from preprocessing.loading_files import load_data
from preprocessing.audio_utils import get_speech_labels, sort_speech_labels
from sklearn.model_selection import train_test_split
from models.dataset import TimeSeriesDataset, BatchDataset
import latent_viz

import sys 
import os 
from pathlib import Path 

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
        "training_epochs_autoencoder":100,
        "autoencoder_lr":1e-2,
        "batch_size":512,
        "latent_dim":64,
        "pt_colors":pt_colors,
        "plot_latent":False,
        "reconstruct_audio":True
    }

config['TR'] = {
    "joined_transformer":False,
    "latent_transformer":False,
    "number_reconstructions":60,
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
    'embedding_size': 128,  # embedding size
    'hidden_size': 128,  # Number of output values
    'num_heads': 8,
    'criterion': 'mse', # mcd / mse
    'dropout': 0.5,
    'shuffle_dataset': False,
    'encoder_layers': 3,
    "decoder_layers":3
    }



def reconstruct_sequence_from_predictions(predictions, original_length, sequence_length=100, offset=5):
    batch_size, seq_len, feature_size = predictions.shape
    
    # Initialize tensors to accumulate values and counts
    reconstructed_sequence = torch.zeros([original_length-offset, feature_size])
    
    # Iterate over each prediction in the batch
    for i in range(batch_size):
        start_idx = i * offset
        end_idx = start_idx + sequence_length
        
        # Add the window to the reconstructed sequence tensor
        reconstructed_sequence[start_idx:end_idx] = predictions[i]
    
    return reconstructed_sequence


# pt_arr = ["p00", "p01", "p06",  "p07",  "p08", "p09", "p10", "p11", "p12", "p16"]
pt_arr = ["p01"]

models_paths = {
    # "transformer":"./runs_transformer/2024-09-05_19-53-27",
    # "transformer":"./runs_transformer/2024-09-20_10-15-53",
    "transformer":"./runs_transformer/2024-09-18_10-13-16",
    "latent_transformer":"",
    "joined_transformer":"./runs_transformer/2024-08-30_18-30-19",
    "joined_latent_transformer":"./runs_transformer/2024-08-30_18-29-58"}


if __name__ == '__main__':
    criterion = nn.MSELoss()

    if len(sys.argv) > 1:
        new_dir_path = sys.argv[1]
        print(new_dir_path)
    else:
        new_dir_path = "."
        
    with open(f"{new_dir_path}/transformer_config.txt", "w") as config_file:
        # Loop through each key-value pair in the dictionary
        for key, value in config["TR"].items():
            # Write each key-value pair as 'key:value' on a separate line
            config_file.write(f"{key}:{value}\n")
        config_file.close()

    
    os.makedirs(os.path.join(new_dir_path, "latent_data"), exist_ok=True)
    bar_plot_results = {pt_id:{"test_corr":np.empty((0,80)), "test_mse":[]} for pt_id in pt_arr}
    for pt_id in pt_arr:
        print(f"\nEvaluating: {pt_id}")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        config["p_id"] = pt_id

        if config["TR"]["latent_transformer"]:
            # latent joined transformer
            if config["TR"]["joined_transformer"]:
                model_dir = models_paths["joined_latent_transformer"]
            
            # latent transformer
            else: 
                model_dir = models_paths["latent_transformer"]
            src, tgt, _, _, _, _, _, _, _ = load_data(config)

            X_test = np.load(f"{model_dir}/latent_data/test_source_encoded_{pt_id}.npy")
            y_test = np.load(f"{model_dir}/latent_data/test_target_encoded_{pt_id}.npy")

            test_set = BatchDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), device)
            config["TR"]['num_features'] = X_test.shape[-1]  # Number of features in your dataset

            _, X_test, y_train, y_test = train_test_split(src, tgt, test_size=0.30, shuffle=False, random_state=0)
            _, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.50, shuffle=False, random_state=0)
            test_seq_len = len(y_test)

            y_test_speech_labels_sorted = np.load(f"{model_dir}/latent_data/sorted_test_speech_labels_{pt_id}.npy")
            np.save(f"{new_dir_path}/latent_data/sorted_test_speech_labels_{pt_id}.npy", y_test_speech_labels_sorted)

        else:
            src, tgt, _, _, _, _, _, _, _ = load_data(config)

            speech_labels = get_speech_labels(tgt)  
            
            # We only care about test sets 
            _, X_test, y_train, y_test = train_test_split(src, tgt, test_size=0.30, shuffle=False, random_state=0)
            X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.50, shuffle=False, random_state=0)
            test_seq_len = len(y_test)
            config["TR"]['num_features'] = X_test.shape[-1]  # Number of features in your dataset
                    
            test_set = TimeSeriesDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), device, config["TR"]["context_length"], config["TR"]["context_offset"])
            
            y_test_speech_labels = speech_labels[y_train.shape[0]:(y_train.shape[0]+y_test.shape[0])]
            y_test_speech_labels_sorted = sort_speech_labels(y_test_speech_labels, len(test_set), config["TR"]["context_length"], config["TR"]["context_offset"]) # [ n_test_points x seq_len ]
            np.save(f"{new_dir_path}/latent_data/sorted_test_speech_labels_{pt_id}.npy", y_test_speech_labels_sorted)
        
        test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False) 
        
        pt_decoder = SimpleDecoder(num_feats=config["feature_extraction"]["num_feats"],
                            num_heads=config["TR"]["num_heads"], 
                            num_layers = config["TR"]["decoder_layers"],
                            embedding_size = config["TR"]["embedding_size"],
                            hidden_dim = config["TR"]["hidden_size"],
                            dropout = config["TR"]["dropout"],
                            seq_len = config["TR"]["context_length"])
        
        pt_encoder = SimpleEncoder(num_features=config["TR"]['num_features'],
                                num_heads=config["TR"]["num_heads"],
                                num_layers=config["TR"]["encoder_layers"],
                                embedding_size = config["TR"]["embedding_size"],
                                hidden_dim = config["TR"]["hidden_size"],
                                dropout = config["TR"]["dropout"],
                                seq_len = config["TR"]["encoder_seq_len"])
        
        pt_model = Transformer(encoder=pt_encoder, decoder=pt_decoder).to(device)
        
        # joined latent transformer 
        if config["TR"]["latent_transformer"] & config["TR"]["joined_transformer"]:
            pt_model.load_state_dict(torch.load(f"{models_paths['joined_latent_transformer']}/saved_models/model_{pt_id}.pt"))

        # joined transformer    
        elif (not config["TR"]["latent_transformer"]) & config["TR"]["joined_transformer"]:
            pt_model.load_state_dict(torch.load(f"{models_paths['joined_transformer']}/saved_models/model_{pt_id}.pt"))

        # latent transformer 
        elif config["TR"]["latent_transformer"] & (not config["TR"]["joined_transformer"]):
            pt_model.load_state_dict(torch.load(f"{models_paths['latent_transformer']}/saved_models/model_{pt_id}.pt"))

        # transformer    
        elif (not config["TR"]["latent_transformer"]) & (not config["TR"]["joined_transformer"]):
            pt_model.load_state_dict(torch.load(f"{models_paths['transformer']}/saved_models/model_{pt_id}.pt"))
        
        pt_model.eval()
        
        source_encoded = encode_data_transformer(pt_model, test_loader)
        np.save(f"{new_dir_path}/latent_data/test_source_encoded_transformer_{pt_id}", source_encoded)

        all_pred = torch.zeros(0, config["TR"]["context_length"], y_test.shape[-1])

        for eeg_test, mel_test in test_loader:

            # grab the current batch
            eeg_test, mel_test = eeg_test.to(device), mel_test.to(device)
            
            mel_pred = inference(eeg_test, pt_model, max_length=config["TR"]['context_length'],target_num_feats=config["feature_extraction"]["num_feats"])
            test_corr = calculate_pearsonr(mel_pred, mel_test).cpu().detach().numpy()
            test_mse = criterion(mel_pred, mel_test).detach().item()

            bar_plot_results[pt_id]["test_mse"].append(test_mse)
            bar_plot_results[pt_id]["test_corr"] = np.concatenate([bar_plot_results[pt_id]["test_corr"], test_corr], axis=0)
            
            all_pred = torch.concatenate([all_pred, mel_pred.cpu()],axis=0)

        test_bins_averaged = np.mean(bar_plot_results[pt_id]["test_corr"], axis=1)
        bar_plot_results[pt_id]["test_mean_corr"] = np.mean(test_bins_averaged)
        bar_plot_results[pt_id]["test_sem_corr"] = np.std(test_bins_averaged) / np.sqrt(all_pred.shape[0])
        
        all_test_pred = reconstruct_sequence_from_predictions(all_pred, test_seq_len, config["TR"]["context_length"], config["TR"]["context_offset"])
        all_test_pred = all_test_pred.unsqueeze(0)
        y_test = torch.from_numpy(y_test[:all_test_pred.shape[1]]).unsqueeze(0)

        # Final reconstructions - STITCHED DATA
        os.makedirs(f"{new_dir_path}/reconstructions_final/stitched/{pt_id}")

        test_loss = criterion(all_test_pred, y_test).detach().item()
        test_corr = calculate_pearsonr(all_test_pred, y_test).cpu().detach().numpy()

        plot_reconstruction(all_test_pred[0].cpu().detach().numpy(), y_test[0].cpu().detach().numpy(), 
                            title=f"Mel Spectrogram reconstruction for {pt_id}", 
                            save_fig=True, file_name=f"{new_dir_path}/reconstructions_final/stitched/{pt_id}/mel_spectrogram_reconstruction_{pt_id}")
        
        if config["reconstruct_audio"]:
            wave_path = Path(f"{new_dir_path}/reconstructions_final/stitched/{pt_id}/")
            waveform_true = reconstruct_and_save_audio(wave_path,f"true_audio_{pt_id}", y_test[0])
            waveform_pred = reconstruct_and_save_audio(wave_path,f"rec_audio_{pt_id}", all_test_pred[0])
            plot_audio_signal(waveform_true, waveform_pred, title=f"Audio reconstruction for {pt_id}", file_name=f"{new_dir_path}/reconstructions_final/stitched/{pt_id}/audio_signal_{pt_id}",save_fig=True)
        
        # Final reconstructions - WINDOWED DATA
        n_rec_samples = config["TR"]["number_reconstructions"]

        os.makedirs(f"{new_dir_path}/reconstructions_final/window/mel_spectrograms/{pt_id}")
        os.makedirs(f"{new_dir_path}/reconstructions_final/window/audio/{pt_id}")
        os.makedirs(f"{new_dir_path}/reconstructions_final/window/audio_signals/{pt_id}")
        
        wave_path = Path(f"{new_dir_path}/reconstructions_final/window/audio/{pt_id}")
        idx_to_plot = np.random.choice(range(len(test_set)), n_rec_samples, replace=False)

        for i, data_idx in enumerate(idx_to_plot):
            eeg_input = test_set[data_idx][0][None,:,:]
            mel_true = test_set[data_idx][1][None,:,:]
            mel_pred = inference(eeg_input, pt_model, max_length=config["TR"]["context_length"],target_num_feats=config["feature_extraction"]["num_feats"])
            
            test_corr = np.round(np.average(calculate_pearsonr(mel_pred, mel_true).cpu().detach().numpy()), 4)
            test_mse = np.round(np.average(criterion(mel_pred, mel_true).detach().item()),4)
            
            plot_reconstruction(mel_pred.cpu().detach().numpy(), mel_true.cpu().detach().numpy(), 
                                title=f"Mel Spectrogram reconstruction for {pt_id}\nMSE: {test_mse:.4f}, Corr: {test_corr:.4f}", save_fig=True, 
                                file_name=f"{new_dir_path}/reconstructions_final/window/mel_spectrograms/{pt_id}/mel_spectrogram_reconstruction_{i}")

            if config["reconstruct_audio"]:
                waveform_true = reconstruct_and_save_audio(wave_path,f"true_audio_{i}", mel_true[0])
                waveform_pred = reconstruct_and_save_audio(wave_path,f"rec_audio_{i}", mel_pred[0])
                plot_audio_signal(waveform_true, waveform_pred, title=f"Audio reconstruction for {pt_id}", 
                                  file_name=f"{new_dir_path}/reconstructions_final/window/audio_signals/{pt_id}/audio_signal_{i}",save_fig=True)
            
    
    pt_ids = list(bar_plot_results.keys())
    test_mean_corr = [bar_plot_results[pt]['test_mean_corr'] for pt in pt_ids]
    test_sem_corr = [bar_plot_results[pt]['test_sem_corr'] for pt in pt_ids]

    fig, ax = plt.subplots()
    ax.bar(pt_ids, test_mean_corr, yerr=test_sem_corr, capsize=5)

    # Adding labels and title
    ax.set_ylabel('Mean corr. (%)')
    ax.set_xlabel('patient')
    ax.set_title('Results over test set')

    ax.grid(axis="y")
    ax.set_ylim([-0.05,1.0])

    plt.savefig(f"{new_dir_path}/model_results.png")

    if config["plot_latent"]:
        latent_viz.main(config["TR"]["embedding_size"], new_dir_path, config, pt_arr, 
                        latent_data_filename="test_source_encoded_transformer", speech_labels_filename="sorted_test_speech_labels")
    print("Done!")

