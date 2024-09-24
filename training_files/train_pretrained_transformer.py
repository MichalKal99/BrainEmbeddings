import numpy as np
from model_utils import *
from audio_utils import get_speech_labels, sort_speech_labels
from model_joined_transformer import * 
from dataset import TimeSeriesDataset, BatchDataset
from loading_files import *


# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Pathlib 
from pathlib import Path

# typing
from typing import Any

# Library for progress bars in loops
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import sys
import os  
import matplotlib.pyplot as plt
from datetime import datetime 
import seaborn as sns
import random
import pickle
from scipy.io import wavfile

def main(new_dir_path, config, pt_data, transformer_type=""):
    print(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    criterion = nn.MSELoss()

    # Open a file for writing
    with open(f"{new_dir_path}/transformer_config.txt", "w") as config_file:
        # Loop through each key-value pair in the dictionary
        for key, value in config["TR"].items():
            # Write each key-value pair as 'key:value' on a separate line
            config_file.write(f"{key}:{value}\n")
        config_file.close()

    train_batch_iterations = 0 

    pretrained_decoder = SimpleDecoder(num_feats=config["feature_extraction"]["num_feats"],
                                num_heads=config["TR"]["num_heads"], 
                                num_layers = config["TR"]["decoder_layers"],
                                embedding_size = config["TR"]["embedding_size"],
                                hidden_dim = config["TR"]["hidden_size"],
                                dropout = config["TR"]["dropout"],
                                seq_len = config["TR"]["context_length"])

    # load the weights 
    if config["TR"]["latent_transformer"]:
        pretrained_model_param = torch.load(f"./runs_transformer/2024-09-05_19-53-15/p01/saved_models/model_p01.pt")
    else:
        pretrained_model_param = torch.load(f"./runs_transformer/2024-09-05_19-53-27/p01/saved_models/model_p01.pt")
    
    # prepare the weights dictionary 
    pretrained_decoder_weights = {k: v for k, v in pretrained_model_param.items() if "decoder" in k}    
    pretrained_decoder_weights = {k[len("decoder."):] if k.startswith("decoder.") else k: v for k, v in pretrained_decoder_weights.items()}
    pretrained_decoder.load_state_dict(pretrained_decoder_weights)

    for pt_id, data in pt_data.items():
        config["p_id"] = pt_id
        
        if config["TR"]["latent_transformer"]:
            train_set = BatchDataset(torch.from_numpy(data["train_src"]), torch.from_numpy(data["train_tgt"]), device)
            test_set = BatchDataset(torch.from_numpy(data["test_src"]), torch.from_numpy(data["test_tgt"]), device)
            val_set = BatchDataset(torch.from_numpy(data["val_src"]), torch.from_numpy(data["val_tgt"]), device)
        
        else:
            # NO SHUFFLE
            X_train, X_test, y_train, y_test = train_test_split(data["source"], data["target"], test_size=0.30, shuffle=False, random_state=0)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, shuffle=False, random_state=0)

            y_test_speech_labels = data["speech_labels"][y_train.shape[0]:(y_train.shape[0]+y_test.shape[0])]
            

            train_set = TimeSeriesDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), device, config["TR"]["context_length"], config["TR"]["context_offset"])
            test_set = TimeSeriesDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), device, config["TR"]["context_length"], config["TR"]["context_offset"])
            val_set = TimeSeriesDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), device, config["TR"]["context_length"], config["TR"]["context_offset"])
            
            config["TR"]['num_features'] = X_train.shape[-1]  # Number of features in your dataset

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False) 
        test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False) 
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False) 

        if not config["TR"]["latent_transformer"]:
            spec_plot = np.empty((0,config["TR"]["context_length"],config["feature_extraction"]["num_feats"]))
            for _, tgt in test_loader:
                spec_plot = np.concatenate([spec_plot, tgt.cpu().detach().numpy()], axis=0)

            y_test_speech_labels_sorted = sort_speech_labels(y_test_speech_labels, len(test_set), config["TR"]["context_length"], config["TR"]["context_offset"]) # [ n_test_points x seq_len ]

            idx_to_plot = np.random.choice(range(y_test_speech_labels_sorted.shape[0]), 25, replace=False)
            for i in idx_to_plot:
                spec = spec_plot[i]
                labels = y_test_speech_labels_sorted[i]
                plot_spectrogram(spec, labels, file_name=f"{new_dir_path}/labeled_speech/mel_spec_labelled_{pt_id}_{i}.png")
            np.save(f"{new_dir_path}/latent_data/sorted_test_speech_labels_{pt_id}.npy", y_test_speech_labels_sorted)


        pt_encoder = SimpleEncoder(num_features=config["TR"]['num_features'],
                                num_heads=config["TR"]["num_heads"],
                                num_layers=config["TR"]["encoder_layers"],
                                embedding_size = config["TR"]["embedding_size"],
                                hidden_dim = config["TR"]["hidden_size"],
                                dropout = config["TR"]["dropout"],
                                seq_len = config["TR"]["encoder_seq_len"])

        
        pt_model = Transformer(encoder=pt_encoder,
                            decoder=pretrained_decoder).to(device)
                
        if config["TR"]["noamOpt"]:
            pt_optimizer = NoamOpt(model_size=config["TR"]["embedding_size"], factor=1, warmup=config["TR"]["warmup"],
                                optimizer = torch.optim.Adam(pt_model.parameters(), lr=config["TR"]["lr"], betas=(0.9, 0.98), eps=1e-9))
        else:
            pt_optimizer = optim.Adam(pt_model.parameters(), lr=config['TR']["lr"], weight_decay=1e-5)

        print(f"Number of model parameters: {count_model_parameters(pt_model)}")
        print(pt_model)

        pt_data[pt_id] = {
            "test_set":test_set,
            "val_set":val_set,
            "train_loader":train_loader,
            "test_loader":test_loader,
            "val_loader":val_loader,
            "model":pt_model,
            "optimizer":pt_optimizer
        }

        if train_batch_iterations < len(train_loader):
            train_batch_iterations = len(train_loader)

    history_train_ppt = {pt:[] for pt in pt_data.keys()}
    history_val_ppt = {pt:[] for pt in pt_data.keys()}
    history_train_corr_ppt = {pt:[] for pt in pt_data.keys()}
    history_val_corr_ppt = {pt:[] for pt in pt_data.keys()}

    history_train = []
    history_train_corr = []
    history_val= []
    history_val_corr = []

    start_time = datetime.now()
    start_token = torch.zeros((1, 1, 80)).to(device)  # Example start token

    pt_loader_counter = {pt:0 for pt in pt_data.keys()}

    best_val_corr = -1.0
    best_epoch = 0

    for pt in pt_data.keys():
        pt_data[pt]["train_iter"] = iter(pt_data[pt]["train_loader"])

    for epoch in range(config['TR']["training_epochs_transformer"]):
        print("#"*40)
        batch_bar = tqdm(total=train_batch_iterations, dynamic_ncols=True, 
                        leave=False, position=0, desc=f"Epoch {epoch+1}/{config['TR']['training_epochs_transformer']} Train")
        
        epoch_training_loss_ppt = {pt:[] for pt in pt_data.keys()}
        epoch_train_correlation_ppt = {pt:[] for pt in pt_data.keys()}
        epoch_training_loss = []
        epoch_train_correlation = []

        for batch_idx in range(train_batch_iterations):
            for pt in pt_data.keys():
                current_model = pt_data[pt]["model"]
                current_optimizer = pt_data[pt]["optimizer"]
                current_model.train()

                # grab the current batch
                eeg, mel = next(pt_data[pt]["train_iter"])
                eeg, mel = eeg.to(device), mel.to(device)
                pt_loader_counter[pt] += 1

                if config["TR"]["teacher_forcing"]:
                
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(mel.shape[1]).to(device)
                    mel_shf = create_decoder_input(mel, start_token)
                    mel_pred = current_model(eeg, mel_shf, tgt_mask, tgt_is_causal=True)

                    batch_loss = criterion(mel_pred, mel)
                    batch_train_corr = calculate_pearsonr(mel_pred, mel).cpu().detach().numpy()


                    current_optimizer.zero_grad()
                    batch_loss.backward()
                    current_optimizer.step()

                    batch_loss = batch_loss.detach().item()
                    batch_corr_mean = batch_train_corr.mean()

                    del mel_pred, batch_train_corr
                        
                else:

                    current_optimizer.zero_grad()

                    # Step 1: Encode EEG Data
                    encoded_eeg = current_model.encode(eeg)

                    # Initialize decoder input with the start token
                    decoder_input = torch.zeros((mel.size(0), config['TR']["context_length"]+1, 80)).to(device)
                    for t in range(config["TR"]["context_length"]):
                        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
                        mel_pred = current_model.decode(decoder_input[:,:t+1,:], encoded_eeg, tgt_mask, True)
                        predicted_frame = mel_pred[:, -1, :].detach()

                        # Scheduled sampling: decide whether to use ground truth or prediction as next input
                        if (config["TR"]["sch_sampling"]) & (random.random() < scheduled_sampling_rate(epoch)):
                            decoder_input[:,t+1,:] = mel[:, t, :]
                        else:
                            decoder_input[:,t+1,:] = predicted_frame
                        del tgt_mask

                    batch_loss = criterion(mel_pred, mel)
                    batch_train_corr = calculate_pearsonr(mel_pred, mel).cpu().detach().numpy()
                    
                    batch_loss.backward()
                    current_optimizer.step()
                    
                    batch_loss = batch_loss.detach().item()
                    batch_corr_mean = batch_train_corr.mean()

                    del mel_pred, decoder_input, batch_train_corr, encoded_eeg

                epoch_training_loss_ppt[pt].append(batch_loss)
                epoch_train_correlation_ppt[pt].append(batch_corr_mean)
                epoch_training_loss.append(batch_loss)
                epoch_train_correlation.append(batch_corr_mean)

                del batch_loss, batch_corr_mean

                # Reset the batch iterator
                if (pt_loader_counter[pt] == len(pt_data[pt]["train_loader"])) | ((batch_idx+1) == train_batch_iterations):
                    pt_data[pt]["train_iter"] = iter(pt_data[pt]["train_loader"])
                    pt_loader_counter[pt] = 0

            batch_bar.update() 

        history_train.append(np.mean(epoch_training_loss))
        history_train_corr.append(np.mean(epoch_train_correlation))
        
        for pt in pt_data.keys():
            history_train_ppt[pt].append(np.mean(epoch_training_loss_ppt[pt]))
            history_train_corr_ppt[pt].append(np.average(epoch_train_correlation_ppt[pt]))

        # VALIDATION
        epoch_val_loss_ppt = {pt:[] for pt in pt_data.keys()}
        epoch_val_correlation_ppt = {pt:[] for pt in pt_data.keys()}
        epoch_val_loss = []
        epoch_val_correlation = []

        val_batch_bar = tqdm(total=len(pt_data.keys()), dynamic_ncols=True, 
                        leave=False, position=0, desc=f"Epoch {epoch+1}/{config['TR']['training_epochs_transformer']} Validation...")
        for pt in pt_data.keys():
            current_val_loader = pt_data[pt]["val_loader"]
            current_model = pt_data[pt]["model"]
            current_model.eval()

            for eeg_val, mel_val in current_val_loader:
                
                eeg_val, mel_val = eeg_val.to(device), mel_val.to(device)
                
                mel_pred = inference(eeg_val, current_model, max_length=config["TR"]['context_length'])
                var_batch_loss = criterion(mel_pred, mel_val).detach().item()
                val_batch_corr = calculate_pearsonr(mel_pred, mel_val).cpu().detach().numpy()

                epoch_val_loss_ppt[pt].append(var_batch_loss)
                epoch_val_correlation_ppt[pt].append(val_batch_corr.mean())
                epoch_val_loss.append(var_batch_loss)
                epoch_val_correlation.append(val_batch_corr.mean())
            val_batch_bar.update()
        mean_val_corr = np.mean(epoch_val_correlation)
        history_val.append(np.mean(epoch_val_loss))
        history_val_corr.append(mean_val_corr)

        for pt in pt_data.keys():
            history_val_ppt[pt].append(np.mean(epoch_val_loss_ppt[pt]))
            history_val_corr_ppt[pt].append(np.mean(epoch_val_correlation_ppt[pt]))

        if mean_val_corr > best_val_corr:
            best_val_corr = mean_val_corr
            best_epoch = epoch
            for pt in pt_data.keys():
                pt_model = pt_data[pt]["model"]
                torch.save(pt_model.state_dict(), f'{new_dir_path}/saved_models/model_{pt}.pt') 

        if (epoch % 10 == 0) | (epoch == 0):
            print("\nReconstructing...")
            for pt in pt_data.keys():
                current_model = pt_data[pt]["model"]

                rec_path = f"{new_dir_path}/reconstructions_training/{pt}/epoch_{epoch}"
                if not os.path.exists(rec_path):
                    # Create the directory
                    os.makedirs(rec_path)
                
                pt_val_set = pt_data[pt]["val_set"]

                n_rec_samples = 5
                idx_to_plot = np.random.choice(range(len(pt_val_set)), n_rec_samples, replace=False)
                current_model.eval()
                for i, data_idx in enumerate(idx_to_plot):
                    eeg_input = pt_val_set[data_idx][0][None,:,:]
                    spec_output = pt_val_set[data_idx][1][None,:,:]
                    reconstruction = inference(eeg_input, current_model, max_length=config["TR"]["context_length"]).cpu().detach().numpy()
                    plot_reconstruction(reconstruction, spec_output.cpu().detach().numpy(), title=f"Mel Spectrogram reconstruction for {pt}", save_fig=True, file_name=f"{new_dir_path}/reconstructions_training/{pt}/epoch_{epoch}/e{epoch}_mel_spectrogram_reconstruction_{i}")
        
        print(f"\nEpoch [{epoch+1}/{config['TR']['training_epochs_transformer']}], Training Loss: {np.mean(epoch_training_loss):.4f}, Training Corr: {np.mean(epoch_train_correlation):.4f}")
        print(f"Epoch [{epoch+1}/{config['TR']['training_epochs_transformer']}], Val. Loss: {np.mean(epoch_val_loss):.4f}, Val. Corr: {mean_val_corr:.4f}")
    
    end_time = datetime.now()
    print(f"\nTraining runtime: {end_time-start_time}")

    fig, axs = plt.subplots(2,2, figsize=(20,20))
    fig.suptitle(f"Joined training of: {pt_data.keys()}", y=0.94, fontsize=25)
    axs = axs.reshape(-1)

    sns.lineplot(history_train, color="blue",  ax=axs[0])
    sns.lineplot(history_val, color="orange",linestyle="--",ax=axs[0])

    sns.lineplot(history_train_corr, color="blue", label="train loss/corr.", ax=axs[1], legend=True)
    sns.lineplot(history_val_corr, color="orange", linestyle="--", label="val. loss/corr.", ax=axs[1], legend=True)

    axs[0].set_ylabel("MSE Loss", fontsize=20)
    axs[1].set_ylabel("Correlation (%)", fontsize=20)

    axs[0].set_xlabel("Epochs", fontsize=20)
    axs[1].set_xlabel("Epochs", fontsize=20)

    axs[0].set_title("Loss overview - averaged", fontsize=25)
    axs[1].set_title("Corr. overview- averaged", fontsize=25)

    for pt in pt_data.keys():
        sns.lineplot(history_train_ppt[pt], color=config["pt_colors"][pt], ax=axs[2])
        sns.lineplot(history_val_ppt[pt], color=config["pt_colors"][pt], linestyle="--", ax=axs[2])

        sns.lineplot(history_train_corr_ppt[pt], color=config["pt_colors"][pt], label=f"{pt} train loss/corr.", ax=axs[3], legend=False)
        sns.lineplot(history_val_corr_ppt[pt], color=config["pt_colors"][pt], linestyle="--", label=f"{pt} val. loss/corr.", ax=axs[3], legend=False)
        
        axs[2].set_title("Loss overview - per patient", fontsize=25)
        axs[2].set_xlabel("Epochs", fontsize=20)
        axs[2].set_ylabel("MSE Loss", fontsize=20)
        axs[3].set_title("Corr. overview - per patient", fontsize=25)
        axs[3].set_xlabel("Epochs", fontsize=20)
        axs[3].set_ylabel("Correlation (%)", fontsize=20)

    axs[1].legend(loc='upper right', bbox_to_anchor=(1.34, 1),fontsize=15)
    axs[3].legend(loc='upper right', bbox_to_anchor=(1.40, 1),fontsize=15)

    axs[0].tick_params(axis='both', which='major', labelsize=15)
    axs[1].tick_params(axis='both', which='major', labelsize=15)
    axs[2].tick_params(axis='both', which='major', labelsize=15)
    axs[3].tick_params(axis='both', which='major', labelsize=15)


    for ax in (axs):
        ax.grid()

    plt.close()
    fig.savefig(f"{new_dir_path}/transformer_training_overview.png", bbox_inches = 'tight')

    print(f"\nBest val. corr. of {best_val_corr} achieved at epoch: {best_epoch}")
    
    # Evaluation on test set 
    evaluation_metrics_ppt = {pt_id:{"test_corr":np.empty((0,80)), "test_mse":[]} for pt_id in pt_data.keys()}
    distances_ppt = {pt_id:[] for pt_id in pt_data.keys()}
    all_distances = []
    all_test_points = []
    all_speech_labels = []

    for pt_id in pt_data.keys():

        os.makedirs(f"{new_dir_path}/results_images/{pt_id}")
        
        current_test_loder = pt_data[pt_id]["test_loader"]
        
        current_model = pt_data[pt_id]["model"] 
        current_model.load_state_dict(torch.load(f"{new_dir_path}/saved_models/model_{pt_id}.pt"))
        current_model.eval()
        
        for eeg_test, mel_test in current_test_loder:

            # grab the current batch
            eeg_test, mel_test = eeg_test.to(device), mel_test.to(device)
            
            mel_pred = inference(eeg_test, current_model, max_length=config["TR"]['context_length'],target_num_feats=config["feature_extraction"]["num_feats"])

            # calculate evaluation metrics
            test_batch_loss = criterion(mel_pred, mel_test).detach().item()
            test_batch_corr = calculate_pearsonr(mel_pred, mel_test).cpu().detach().numpy()
            print(f"test_batch_corr: {test_batch_corr.shape}")

            # calculate distance 
            test_dist = calculate_distance_spectrograms(mel_pred.cpu().detach().numpy(), mel_test.cpu().detach().numpy()) # [bs x win_size]
            distances_ppt[pt_id].append(test_dist)
            all_distances.append(test_dist)
            evaluation_metrics_ppt[pt_id]["test_mse"].append(test_batch_loss)
            evaluation_metrics_ppt[pt_id]["test_corr"] = np.concatenate([evaluation_metrics_ppt[pt_id]["test_corr"], test_batch_corr], axis=0)

        
        test_bins_averaged = np.mean(evaluation_metrics_ppt[pt_id]["test_corr"], axis=1)
        test_points_averaged = np.mean(evaluation_metrics_ppt[pt_id]["test_corr"], axis=0)
        sorted_speech_labels = np.load(f"{new_dir_path}/latent_data/sorted_test_speech_labels_{pt_id}.npy")
        distances_ppt[pt_id] = np.concatenate(distances_ppt[pt_id], axis=0)
        all_test_points.append(evaluation_metrics_ppt[pt_id]["test_corr"])
        all_speech_labels.append(sorted_speech_labels)
        
        plot_corr_avg(test_bins_averaged, f"{new_dir_path}/results_images/{pt_id}", filename=f"{pt_id}_test_corr_dist")
        plot_corr_spec_bins(test_points_averaged, f"{new_dir_path}/results_images/{pt_id}", filename=f"{pt_id}_corr_spec_bins")
        plot_corr_speech_ratio(test_bins_averaged, sorted_speech_labels, f"{new_dir_path}/results_images/{pt_id}", filename=f"{pt_id}_corr_dist_speech_ratio")
        plot_dist_heatmap(distances_ppt[pt_id], f"{new_dir_path}/results_images/{pt_id}", filename=f"{pt_id}_heatmap", num_bins=100)

        evaluation_metrics_ppt[pt_id]["test_mean_mse"] = round(np.mean(evaluation_metrics_ppt[pt_id]["test_mse"]), 4)
        evaluation_metrics_ppt[pt_id]["test_mean_corr"] = round(np.mean(test_bins_averaged), 4)

        evaluation_metrics_ppt[pt_id]["test_std_mse"] = round(np.std(evaluation_metrics_ppt[pt_id]["test_mse"]), 4)
        evaluation_metrics_ppt[pt_id]["test_std_corr"] = round(np.std(test_bins_averaged), 4)

        evaluation_metrics_ppt[pt_id]["test_standard_error_mean"] = evaluation_metrics_ppt[pt_id]["test_std_corr"] / np.sqrt(len(pt_data[pt_id]["test_set"]))
        evaluation_metrics_ppt[pt_id]["n_test_points"] = len(pt_data[pt_id]["test_set"])

        source_encoded = encode_data_transformer(current_model, current_test_loder)
        np.save(f"{new_dir_path}/latent_data/test_source_encoded_transformer_{pt_id}", source_encoded)

    all_distances = np.concatenate(all_distances, axis=0)  # Shape should be [all_test_points x 100]
    all_test_points = np.concatenate(all_test_points, axis=0) # concatenate all test points together [all_test_points x 80]
    all_speech_labels = np.concatenate(all_speech_labels, axis=0) # concatenate all speech labels [all_test_points x 100]
    all_test_bins_averaged = np.mean(all_test_points, axis=1) # average across points [1 x 80]
    all_test_points_averaged = np.mean(all_test_points, axis=0) # average across bins [all_test_points x 1]
    results_filename = f"{len(pt_data.keys())}_alt_{transformer_type}transformer_results_dict.pkl" if len(pt_data.keys()) > 1 else f"{list(pt_data.keys())[0]}_{transformer_type}transformer_results_dict.pkl"
    print(f"Filename: {results_filename}")
    if len(pt_data.keys()) > 1:
        plot_dist_heatmap(all_distances, f"{new_dir_path}/results_images", filename=f"heatmap",num_bins=100)
        plot_corr_avg(all_test_bins_averaged, f"{new_dir_path}/results_images", filename=f"test_corr_dist")
        plot_corr_spec_bins(all_test_points_averaged, f"{new_dir_path}/results_images", filename=f"corr_spec_bins")
        plot_corr_speech_ratio(all_test_bins_averaged, all_speech_labels, f"{new_dir_path}/results_images", filename=f"corr_dist_speech_ratio")

    with open(f'./saved_results/{results_filename}', 'wb') as f:
        pickle.dump(evaluation_metrics_ppt, f)
    f.close()

    for pt_id in evaluation_metrics_ppt.keys():
        print(f"Patient: {pt_id}")
        for metric in evaluation_metrics_ppt[pt_id].keys():
            print(f"\tMetric: {metric}")
            print("\t",evaluation_metrics_ppt[pt_id][metric])

    # Final reconstructions
    n_rec_samples = config["TR"]["number_reconstructions"]
    for pt_id in pt_data.keys():
        os.makedirs(f"{new_dir_path}/reconstructions_final/mel_spectrograms/{pt_id}")
        os.makedirs(f"{new_dir_path}/reconstructions_final/audio/{pt_id}")
        os.makedirs(f"{new_dir_path}/reconstructions_final/audio_signals/{pt_id}")

        wave_path = Path(f"{new_dir_path}/reconstructions_final/audio/{pt_id}")
        current_test_set = pt_data[pt_id]["test_set"]
        idx_to_plot = np.random.choice(range(len(current_test_set)), n_rec_samples, replace=False)

        current_model = pt_data[pt_id]["model"] 
        current_model.load_state_dict(torch.load(f"{new_dir_path}/saved_models/model_{pt_id}.pt"))
        current_model.eval()
        
        for i, data_idx in enumerate(idx_to_plot):
            eeg_input = current_test_set[data_idx][0][None,:,:]
            mel_true = current_test_set[data_idx][1][None,:,:]
            mel_pred = inference(eeg_input, current_model, max_length=config["TR"]["context_length"],target_num_feats=config["feature_extraction"]["num_feats"])
            test_corr = np.round(np.average(calculate_pearsonr(mel_pred, mel_true).cpu().detach().numpy()), 4)
            test_mse = np.round(np.average(criterion(mel_pred, mel_true).detach().item()),4)
            plot_reconstruction(mel_pred.cpu().detach().numpy(), mel_true.cpu().detach().numpy(), title=f"Mel Spectrogram reconstruction for {pt_id}\nMSE: {test_mse:.4f}, Corr: {test_corr:.4f}", save_fig=True, file_name=f"{new_dir_path}/reconstructions_final/mel_spectrograms/{pt_id}/mel_spectrogram_reconstruction_{i}")

            waveform_true = reconstruct_and_save_audio(wave_path,f"true_audio_{i}", mel_true[0])
            waveform_pred = reconstruct_and_save_audio(wave_path,f"rec_audio_{i}", mel_pred[0])
            plot_audio_signal(waveform_true, waveform_pred, title=f"Audio reconstruction for {pt_id}", file_name=f"{new_dir_path}/reconstructions_final/audio_signals/{pt_id}/audio_signal_{i}",save_fig=True)
            
    print("Training complete!")