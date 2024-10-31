import numpy as np
from models.model_utils import *
from models.transformer import * 
from preprocessing.audio_utils import get_speech_labels, sort_speech_labels
from preprocessing.loading_files import *
from models.dataset import TimeSeriesDataset, BatchDataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Pathlib 
from pathlib import Path

# Library for progress bars in loops
from tqdm import tqdm

import os  
import pickle
from datetime import datetime 

from PIL import Image
import random

def create_gif_from_pngs(png_folder, output_gif_name, selected_index):
    # Get all filenames in the folder
    filenames = os.listdir(png_folder)

    # Filter the filenames to only those that match the selected index
    png_files = sorted([
        os.path.join(png_folder, f) for f in filenames
        if f.endswith(f"_mel_spectrogram_reconstruction_{selected_index}.png")
    ], key=lambda x: int(os.path.basename(x).split('_')[0][1:])) # Sort by number part (e_{number})

    if not png_files:
        print(f"No PNG files found for index {selected_index}")
        return

    # Open all the images
    images = [Image.open(png) for png in png_files]

    # Save the images as a GIF
    images[0].save(
        f"{output_gif_name}/ani_{selected_index}.gif",
        save_all=True,
        append_images=images[1:],
        duration=300,  # Duration between frames in milliseconds
        loop=0  # Loop forever
    )
    print(f"GIF saved as {output_gif_name}")

def main(new_dir_path, config, pt_data,transformer_type=""):

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

    print(pt_data.keys())

    shared_decoder = SimpleDecoder(num_feats=config["feature_extraction"]["num_feats"],
                                num_heads=config["TR"]["num_heads"], 
                                num_layers = config["TR"]["decoder_layers"],
                                embedding_size = config["TR"]["embedding_size"],
                                hidden_dim = config["TR"]["hidden_size"],
                                dropout = config["TR"]["dropout"],
                                seq_len = config["TR"]["context_length"])

    train_batch_iterations = 0 
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
                            decoder=shared_decoder).to(device)
        print(pt_model)

        if config["TR"]["noamOpt"]:
            pt_optimizer = NoamOpt(model_size=config["TR"]["embedding_size"], factor=1, warmup=config["TR"]["warmup"],
                                optimizer = torch.optim.Adam(pt_model.parameters(), lr=config["TR"]["lr"], betas=(0.9, 0.98), eps=1e-9))
        else:
            pt_optimizer = optim.Adam(pt_model.parameters(), lr=config['TR']["lr"], weight_decay=1e-5)
        
        print(f"Number of model parameters: {count_model_parameters(pt_model)}")

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

    history_train_ppt = {pt_id:[] for pt_id in pt_data.keys()}
    history_val_ppt = {pt_id:[] for pt_id in pt_data.keys()}
    history_train_corr_ppt = {pt_id:[] for pt_id in pt_data.keys()}
    history_val_corr_ppt = {pt_id:[] for pt_id in pt_data.keys()}
    history_train_mcd_ppt = {pt_id:[] for pt_id in pt_data.keys()}
    history_val_mcd_ppt = {pt_id:[] for pt_id in pt_data.keys()}

    history_train = []
    history_train_corr = []
    history_train_mcd = []
    history_val = []
    history_val_corr = []
    history_val_mcd = []

    start_time = datetime.now()
    start_token = torch.zeros((1, 1, config["feature_extraction"]["num_feats"])).to(device)  # Example start token

    pt_loader_counter = {pt_id:0 for pt_id in pt_data.keys()}

    best_val_corr = -1.0
    best_epoch = 0

    # monitor correlation
    # early_stopper = EarlyStopper(patience=20, min_delta=0.01)
    n_rec_samples = 20
    for pt_id in pt_data.keys():
        pt_data[pt_id]["train_iter"] = iter(pt_data[pt_id]["train_loader"])
        pt_data[pt_id]["val_points"] = np.random.choice(range(len(pt_data[pt_id]["val_set"])), n_rec_samples, replace=False)
    
    for epoch in range(config['TR']["training_epochs_transformer"]):
        print("#"*40)
        batch_bar = tqdm(total=train_batch_iterations, dynamic_ncols=True, 
                        leave=False, position=0, desc=f"Epoch {epoch+1}/{config['TR']['training_epochs_transformer']} Train")
        
        epoch_training_loss_ppt = {pt_id:[] for pt_id in pt_data.keys()}
        epoch_train_correlation_ppt = {pt_id:[] for pt_id in pt_data.keys()}
        epoch_train_mcd_ppt = {pt_id:[] for pt_id in pt_data.keys()}
        epoch_training_loss = []
        epoch_train_correlation = []
        epoch_train_mcd = []

        for batch_idx in range(train_batch_iterations):
            for pt_id in pt_data.keys():
                current_model = pt_data[pt_id]["model"]
                current_optimizer = pt_data[pt_id]["optimizer"]
                current_model.train()

                # grab the current batch
                eeg, mel = next(pt_data[pt_id]["train_iter"])
                eeg, mel = eeg.to(device), mel.to(device)

                pt_loader_counter[pt_id] += 1

                if config["TR"]["teacher_forcing"]:
                
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(mel.shape[1]).to(device)
                    mel_shf = create_decoder_input(mel, start_token)
                    mel_pred = current_model(eeg, mel_shf, tgt_mask, tgt_is_causal=True)

                    batch_loss = criterion(mel_pred, mel)
                    batch_train_corr = calculate_pearsonr(mel_pred, mel).cpu().detach().numpy()
                    batch_train_mcd = calculate_mcd(mel_pred, mel).cpu().detach().numpy()

                    if config["TR"]["noamOpt"]:
                        current_optimizer.optimizer.zero_grad()
                    else:
                        current_optimizer.zero_grad()

                    batch_loss.backward()
                    if config["TR"]["noamOpt"]: 
                        current_optimizer.optimizer.step()
                    else:
                        current_optimizer.step()

                    batch_loss = batch_loss.detach().item()
                    batch_corr_mean = batch_train_corr.mean()
                    batch_mcd_mean = batch_train_mcd.mean()
                    
                    del mel_pred, batch_train_corr, batch_train_mcd
                        
                else:
                    if config["TR"]["noamOpt"]: 
                        current_optimizer.optimizer.zero_grad()
                    else:
                        current_optimizer.zero_grad()

                    # Step 1: Encode EEG Data
                    encoded_eeg = current_model.encode(eeg)

                    # Initialize decoder input with the start token
                    decoder_input = torch.zeros((mel.size(0), config['TR']["context_length"]+1, config["feature_extraction"]["num_feats"])).to(device)
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
                    # print(f"Mel pred: {mel_pred.shape}")
                    batch_loss = criterion(mel_pred, mel)
                    batch_train_corr = calculate_pearsonr(mel_pred, mel).cpu().detach().numpy()
                    batch_train_mcd = calculate_mcd(mel_pred, mel).cpu().detach().numpy()
                    
                    batch_loss.backward()
                    if config["TR"]["noamOpt"]: 
                        current_optimizer.optimizer.step()
                    else:
                        current_optimizer.step()
                        
                    batch_loss = batch_loss.detach().item()
                    batch_corr_mean = batch_train_corr.mean()
                    batch_mcd_mean = batch_train_mcd.mean()

                    del mel_pred, decoder_input, batch_train_corr, batch_train_mcd, encoded_eeg

                epoch_training_loss_ppt[pt_id].append(batch_loss)
                epoch_train_correlation_ppt[pt_id].append(batch_corr_mean)
                epoch_train_mcd_ppt[pt_id].append(batch_mcd_mean)
                epoch_training_loss.append(batch_loss)
                epoch_train_correlation.append(batch_corr_mean)
                epoch_train_mcd.append(batch_mcd_mean)

                del batch_loss, batch_corr_mean, batch_mcd_mean

                # Reset the batch iterator
                if (pt_loader_counter[pt_id] == len(pt_data[pt_id]["train_loader"])) | ((batch_idx+1) == train_batch_iterations):
                    pt_data[pt_id]["train_iter"] = iter(pt_data[pt_id]["train_loader"])
                    pt_loader_counter[pt_id] = 0

            batch_bar.update() 

        history_train.append(np.mean(epoch_training_loss))
        history_train_corr.append(np.mean(epoch_train_correlation))
        history_train_mcd.append(np.mean(epoch_train_mcd))

        for pt_id in pt_data.keys():
            history_train_ppt[pt_id].append(np.mean(epoch_training_loss_ppt[pt_id]))
            history_train_corr_ppt[pt_id].append(np.average(epoch_train_correlation_ppt[pt_id]))
            history_train_mcd_ppt[pt_id].append(np.average(epoch_train_mcd_ppt[pt_id]))

        # VALIDATION
        epoch_val_loss_ppt = {pt_id:[] for pt_id in pt_data.keys()}
        epoch_val_correlation_ppt = {pt_id:[] for pt_id in pt_data.keys()}
        epoch_val_mcd_ppt = {pt_id:[] for pt_id in pt_data.keys()}
        epoch_val_loss = []
        epoch_val_correlation = []
        epoch_val_mcd = []

        val_batch_bar = tqdm(total=len(pt_data.keys()), dynamic_ncols=True, 
                        leave=False, position=0, desc=f"Epoch {epoch+1}/{config['TR']['training_epochs_transformer']} Validation...")
        for pt_id in pt_data.keys():
            current_val_loader = pt_data[pt_id]["val_loader"]
            current_model = pt_data[pt_id]["model"]
            current_model.eval()

            for eeg_val, mel_val in current_val_loader:
                
                eeg_val, mel_val = eeg_val.to(device), mel_val.to(device)
                
                mel_pred = inference(eeg_val, current_model, max_length=config["TR"]['context_length'],target_num_feats=config["feature_extraction"]["num_feats"])
                var_batch_loss = criterion(mel_pred, mel_val).detach().item()
                val_batch_corr = calculate_pearsonr(mel_pred, mel_val).cpu().detach().numpy()
                val_batch_mcd = calculate_mcd(mel_pred, mel_val).cpu().detach().numpy()

                epoch_val_loss_ppt[pt_id].append(var_batch_loss)
                epoch_val_correlation_ppt[pt_id].append(val_batch_corr.mean())
                epoch_val_mcd_ppt[pt_id].append(val_batch_mcd.mean())
                epoch_val_loss.append(var_batch_loss)
                epoch_val_correlation.append(val_batch_corr.mean())
                epoch_val_mcd.append(val_batch_mcd.mean())

            val_batch_bar.update()
        mean_val_corr = np.mean(epoch_val_correlation)
        mean_val_mcd = np.mean(epoch_val_mcd)

        history_val.append(np.mean(epoch_val_loss))
        history_val_corr.append(mean_val_corr)
        history_val_mcd.append(mean_val_mcd)

        for pt_id in pt_data.keys():
            history_val_ppt[pt_id].append(np.mean(epoch_val_loss_ppt[pt_id]))
            history_val_corr_ppt[pt_id].append(np.mean(epoch_val_correlation_ppt[pt_id]))
            history_val_mcd_ppt[pt_id].append(np.mean(epoch_val_correlation_ppt[pt_id]))

        if mean_val_corr > best_val_corr:
            best_val_corr = mean_val_corr
            best_epoch = epoch
            for pt_id in pt_data.keys():
                pt_model = pt_data[pt_id]["model"]
                torch.save(pt_model.state_dict(), f'{new_dir_path}/saved_models/model_{pt_id}.pt') 

        if (epoch % 10 == 0) | (epoch == 0):
            print("\nReconstructing...")
            for pt_id in pt_data.keys():
                current_model = pt_data[pt_id]["model"]

                rec_path = f"{new_dir_path}/reconstructions_training/{pt_id}"
                if not os.path.exists(rec_path):
                    # Create the directory
                    os.makedirs(rec_path)
                
                pt_val_set = pt_data[pt_id]["val_set"]
                idx_to_plot = pt_data[pt_id]["val_points"]
                current_model.eval()
                for i, data_idx in enumerate(idx_to_plot):
                    eeg_input = pt_val_set[data_idx][0][None,:,:]
                    spec_output = pt_val_set[data_idx][1][None,:,:]
                    reconstruction = inference(eeg_input, current_model, max_length=config["TR"]["context_length"],target_num_feats=config["feature_extraction"]["num_feats"])

                    rec_val_loss = np.average(criterion(reconstruction, spec_output).detach().item())
                    rec_val_corr = np.average(calculate_pearsonr(reconstruction, spec_output).cpu().detach().numpy())
                    rec_val_mcd = np.average(calculate_mcd(reconstruction, spec_output).cpu().detach())

                    plot_reconstruction(reconstruction.cpu().detach().numpy(), spec_output.cpu().detach().numpy(), 
                                        title=f"Mel Spectrogram reconstruction for {pt_id}, epoch:{epoch}\nMSE: {rec_val_loss:.4f}, Corr: {rec_val_corr:.4f}, MCD: {rec_val_mcd:.4f}",
                                        save_fig=True, file_name=f"{new_dir_path}/reconstructions_training/{pt_id}/e{epoch}_mel_spectrogram_reconstruction_{i}")
        
        print(f"\nEpoch [{epoch+1}/{config['TR']['training_epochs_transformer']}], Training Loss: {np.mean(epoch_training_loss):.4f}, Training Corr: {np.mean(epoch_train_correlation):.4f}")
        print(f"Epoch [{epoch+1}/{config['TR']['training_epochs_transformer']}], Val. Loss: {np.mean(epoch_val_loss):.4f}, Val. Corr: {mean_val_corr:.4f}")


    end_time = datetime.now()
    print(f"\nTraining runtime: {end_time-start_time}")

    # joined model
    if len(pt_data.keys()) > 1:
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

        for pt_id in pt_data.keys():
            sns.lineplot(history_train_ppt[pt_id], color=config["pt_colors"][pt_id], ax=axs[2])
            sns.lineplot(history_val_ppt[pt_id], color=config["pt_colors"][pt_id], linestyle="--", ax=axs[2])

            sns.lineplot(history_train_corr_ppt[pt_id], color=config["pt_colors"][pt_id], label=f"{pt_id} train loss/corr.", ax=axs[3], legend=False)
            sns.lineplot(history_val_corr_ppt[pt_id], color=config["pt_colors"][pt_id], linestyle="--", label=f"{pt_id} val. loss/corr.", ax=axs[3], legend=False)
            
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
    
    # individual models 
    elif len(pt_data.keys()) == 1:
        fig, axs = plt.subplots(2,3, figsize=(30,10), tight_layout=True)
        axs = axs.reshape(-1)
        fig.suptitle(f"Training of: {list(pt_data.keys())[0]}", y=0.98, x=0.45,fontsize=25)

        sns.lineplot(history_train, color="blue",  ax=axs[0])
        sns.lineplot(history_val, color="orange",linestyle="--",ax=axs[0])
        sns.lineplot(history_train, color="blue",  ax=axs[3])
        sns.lineplot(history_val, color="orange",linestyle="--",ax=axs[3])

        
        sns.lineplot(history_train_corr, color="blue", ax=axs[1])
        sns.lineplot(history_val_corr, color="orange", linestyle="--", ax=axs[1])

        sns.lineplot(history_train_mcd, color="blue", label="train loss/corr./mcd", ax=axs[2], legend=True)
        sns.lineplot(history_val_mcd, color="orange", linestyle="--", label="val. loss/corr./mcd", ax=axs[2], legend=True)
        sns.lineplot(history_train_mcd, color="blue", label="train loss/corr./mcd", ax=axs[5], legend=True)
        sns.lineplot(history_val_mcd, color="orange", linestyle="--", label="val. loss/corr./mcd", ax=axs[5], legend=True)

        axs[0].set_ylabel("MSE Loss", fontsize=18)
        axs[1].set_ylabel("Correlation (%)", fontsize=18)
        axs[2].set_ylabel("MCD", fontsize=18)

        axs[0].set_xlabel("Epochs", fontsize=18)
        axs[1].set_xlabel("Epochs", fontsize=18)
        axs[2].set_xlabel("Epochs", fontsize=18)

        axs[0].set_title("MSE loss overview", fontsize=18)
        axs[1].set_title("Corr. overview", fontsize=18)
        axs[2].set_title("MCD overview", fontsize=18)

        axs[2].legend(loc='upper right', bbox_to_anchor=(1.35, 1),fontsize=15)

        axs[0].tick_params(axis='both', which='major', labelsize=15)
        axs[1].tick_params(axis='both', which='major', labelsize=15)
        axs[2].tick_params(axis='both', which='major', labelsize=15)

        axs[3].set_yscale('log')
        axs[5].set_yscale('log')

    for ax in (axs):
        ax.grid()

    plt.close()
    fig.savefig(f"{new_dir_path}/transformer_training_overview.png", bbox_inches = 'tight')

    print(f"\nBest val. corr. of {best_val_corr} achieved at epoch: {best_epoch}")

    for model_idx in range(2):
        # Evaluation on test set 
        evaluation_metrics_ppt = {pt_id:{"test_corr":np.empty((0,80)), "test_mse":[], "test_mcd":[]} for pt_id in pt_data.keys()}
        distances_ppt = {pt_id:[] for pt_id in pt_data.keys()}
        all_distances = []
        all_test_points = []
        all_speech_labels = []

        for pt_id in pt_data.keys():
            current_test_loder = pt_data[pt_id]["test_loader"]
            if model_idx == 1:
                model_name = "best"
                current_model = pt_data[pt_id]["model"] 
                current_model.load_state_dict(torch.load(f"{new_dir_path}/saved_models/model_{pt_id}.pt"))
            elif model_idx == 0:
                model_name = "recent"
                current_model = pt_data[pt_id]["model"]

            os.makedirs(f"{new_dir_path}/results_images/{model_name}/{pt_id}")
            current_model.eval()
            
            for eeg_test, mel_test in current_test_loder:

                # grab the current batch
                eeg_test, mel_test = eeg_test.to(device), mel_test.to(device)
                
                mel_pred = inference(eeg_test, current_model, max_length=config["TR"]['context_length'],target_num_feats=config["feature_extraction"]["num_feats"])

                # calculate evaluation metrics
                test_batch_loss = criterion(mel_pred, mel_test).detach().item()
                test_batch_corr = calculate_pearsonr(mel_pred, mel_test).cpu().detach().numpy()
                test_batch_mcd = calculate_mcd(mel_pred, mel_test).cpu().detach().numpy()

                # calculate distance 
                test_dist = calculate_distance_spectrograms(mel_pred.cpu().detach().numpy(), mel_test.cpu().detach().numpy()) # [bs x win_size]
                distances_ppt[pt_id].append(test_dist)
                all_distances.append(test_dist)
                evaluation_metrics_ppt[pt_id]["test_mse"].append(test_batch_loss)
                evaluation_metrics_ppt[pt_id]["test_corr"] = np.concatenate([evaluation_metrics_ppt[pt_id]["test_corr"], test_batch_corr], axis=0)
                evaluation_metrics_ppt[pt_id]["test_mcd"].extend(test_batch_mcd)

            
            test_bins_averaged = np.mean(evaluation_metrics_ppt[pt_id]["test_corr"], axis=1)
            test_points_averaged = np.mean(evaluation_metrics_ppt[pt_id]["test_corr"], axis=0)
            test_points_mcd_averaged = np.mean(evaluation_metrics_ppt[pt_id]["test_mcd"])
            sorted_speech_labels = np.load(f"{new_dir_path}/latent_data/sorted_test_speech_labels_{pt_id}.npy")
            distances_ppt[pt_id] = np.concatenate(distances_ppt[pt_id], axis=0)
            all_test_points.append(evaluation_metrics_ppt[pt_id]["test_corr"])
            all_speech_labels.append(sorted_speech_labels)
            
            plot_corr_avg(test_bins_averaged, f"{new_dir_path}/results_images/{model_name}/{pt_id}", filename=f"{pt_id}_test_corr_dist")
            plot_corr_spec_bins(test_points_averaged, f"{new_dir_path}/results_images/{model_name}/{pt_id}", filename=f"{pt_id}_corr_spec_bins")
            plot_corr_speech_ratio(test_bins_averaged, sorted_speech_labels, f"{new_dir_path}/results_images/{model_name}/{pt_id}", filename=f"{pt_id}_corr_dist_speech_ratio")
            plot_dist_heatmap(distances_ppt[pt_id], f"{new_dir_path}/results_images/{model_name}/{pt_id}", filename=f"{pt_id}_heatmap", num_bins=100)

            evaluation_metrics_ppt[pt_id]["test_mean_mse"] = round(np.mean(evaluation_metrics_ppt[pt_id]["test_mse"]), 4)
            evaluation_metrics_ppt[pt_id]["test_mean_corr"] = round(np.mean(test_bins_averaged), 4)
            evaluation_metrics_ppt[pt_id]["test_mean_mcd"] = np.mean(evaluation_metrics_ppt[pt_id]["test_mcd"])

            evaluation_metrics_ppt[pt_id]["test_std_mse"] = round(np.std(evaluation_metrics_ppt[pt_id]["test_mse"]), 4)
            evaluation_metrics_ppt[pt_id]["test_std_corr"] = round(np.std(test_bins_averaged), 4)
            evaluation_metrics_ppt[pt_id]["test_std_mcd"] = np.std(evaluation_metrics_ppt[pt_id]["test_mcd"])

            evaluation_metrics_ppt[pt_id]["test_standard_error_mean"] = evaluation_metrics_ppt[pt_id]["test_std_corr"] / np.sqrt(len(pt_data[pt_id]["test_set"]))
            evaluation_metrics_ppt[pt_id]["test_standard_error_mean_mcd"] = evaluation_metrics_ppt[pt_id]["test_std_mcd"] / np.sqrt(len(pt_data[pt_id]["test_set"]))
            evaluation_metrics_ppt[pt_id]["n_test_points"] = len(pt_data[pt_id]["test_set"])

            source_encoded = encode_data_transformer(current_model, current_test_loder)
            np.save(f"{new_dir_path}/latent_data/test_source_encoded_transformer_{pt_id}", source_encoded)

        all_distances = np.concatenate(all_distances, axis=0)  # Shape should be [all_test_points x 100]
        all_test_points = np.concatenate(all_test_points, axis=0) # concatenate all test points together [all_test_points x 80]
        all_speech_labels = np.concatenate(all_speech_labels, axis=0) # concatenate all speech labels [all_test_points x 100]
        all_test_bins_averaged = np.mean(all_test_points, axis=1) # average across points [1 x 80]
        all_test_points_averaged = np.mean(all_test_points, axis=0) # average across bins [all_test_points x 1]
        results_filename = f"{len(pt_data.keys())}_{transformer_type}transformer_results_dict.pkl" if len(pt_data.keys()) > 1 else f"{list(pt_data.keys())[0]}_{transformer_type}transformer_results_dict.pkl"
        
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
            print(f"\tTest mean corr.: {evaluation_metrics_ppt[pt_id]['test_mean_corr']}")
            print(f"\tTest mean mcd: {evaluation_metrics_ppt[pt_id]['test_mean_mcd']}")

        # Final reconstructions
        n_rec_samples = config["TR"]["number_reconstructions"]
        for pt_id in pt_data.keys():

            if model_idx == 1:
                model_name = "best"
                current_model = pt_data[pt_id]["model"] 
                current_model.load_state_dict(torch.load(f"{new_dir_path}/saved_models/model_{pt_id}.pt"))
            elif model_idx == 0:
                model_name = "recent"
                current_model = pt_data[pt_id]["model"]

            os.makedirs(f"{new_dir_path}/reconstructions_final/{model_name}/mel_spectrograms/{pt_id}")
            os.makedirs(f"{new_dir_path}/reconstructions_final/{model_name}/audio/{pt_id}")
            os.makedirs(f"{new_dir_path}/reconstructions_final/{model_name}/audio_signals/{pt_id}")

            wave_path = Path(f"{new_dir_path}/reconstructions_final/{model_name}/audio/{pt_id}")
            current_test_set = pt_data[pt_id]["test_set"]
            idx_to_plot = np.random.choice(range(len(current_test_set)), n_rec_samples, replace=False)

            current_model.eval()
            
            for i, data_idx in enumerate(idx_to_plot):
                eeg_input = current_test_set[data_idx][0][None,:,:]
                mel_true = current_test_set[data_idx][1][None,:,:]
                mel_pred = inference(eeg_input, current_model, max_length=config["TR"]["context_length"],target_num_feats=config["feature_extraction"]["num_feats"])
                
                test_corr = np.average(calculate_pearsonr(mel_pred, mel_true).cpu().detach().numpy())
                test_mse = np.average(criterion(mel_pred, mel_true).detach().item())
                test_mcd = np.average(calculate_mcd(mel_pred, mel_true).detach().item())

                plot_reconstruction(mel_pred.cpu().detach().numpy(), mel_true.cpu().detach().numpy(), 
                                    title=f"Mel Spectrogram reconstruction for {pt_id}\nMSE: {test_mse:.4f}, Corr: {test_corr:.4f}, MCD: {test_mcd:.4f}", save_fig=True, 
                                    file_name=f"{new_dir_path}/reconstructions_final/{model_name}/mel_spectrograms/{pt_id}/mel_spectrogram_reconstruction_{i}")

                waveform_true = reconstruct_and_save_audio(wave_path,f"true_audio_{i}", mel_true[0])
                waveform_pred = reconstruct_and_save_audio(wave_path,f"rec_audio_{i}", mel_pred[0])
                plot_audio_signal(waveform_true, waveform_pred, title=f"Audio reconstruction for {pt_id}", file_name=f"{new_dir_path}/reconstructions_final/{model_name}/audio_signals/{pt_id}/audio_signal_{i}",save_fig=True)
    
    if config["TR"]["plot_gifs"]:            
        png_folder = f"{new_dir_path}/reconstructions_training/{pt_id}"
        output_gif_name = f"{new_dir_path}/training_progress"
        os.makedirs(output_gif_name)
        for i in range(20):
            create_gif_from_pngs(png_folder, output_gif_name, i)
            
    print("Training complete!")