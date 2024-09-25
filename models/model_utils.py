from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import pearsonr, spearmanr

from StreamingVocGan.streaming_voc_gan import StreamingVocGan
from pathlib import Path
simplefilter("ignore", category=ConvergenceWarning)
import os 

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def train_model(model, dataloader, loss_fn, optimizer, epoch, total_epochs, beta=1):
        
    model.train()
    
    # Process tqdm bar, helpful for monitoring training process
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, 
                     leave=False, position=0, desc=f"[EPOCH {epoch+1}/{total_epochs}] training...")
    epoch_training_loss = []
    for batch_idx, (X_train, _) in enumerate(dataloader):
        
        X_train = X_train[:,None,:,:] # add depth dimension
        X_train = X_train.to(DEVICE)

        if model.model_type == "vae":
            X_rec, mean, log_var = model(X_train)
            batch_loss, mse, kld = calculate_vae_loss(X_rec, X_train, mean, log_var,beta=beta)
        else:
            X_rec = model(X_train)
            batch_loss = loss_fn(X_rec, X_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        epoch_training_loss.append(batch_loss.item())

        batch_bar.set_postfix(
            loss=f"{batch_loss.item()}",
            lr = f"{optimizer.param_groups[0]['lr']:.6f}"
        )
        batch_bar.update()             
        
        # remove unnecessary cache in CUDA memory
        torch.cuda.empty_cache()
        del X_train, X_rec

    return np.mean(epoch_training_loss)


def validate_model(model, dataloader, loss_fc, optimizer, scheduler, epoch, total_epochs, beta=1):
    model.eval()
    progress_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0,  desc=f"[EPOCH {epoch+1}/{total_epochs}] validating...")
    epoch_val_loss = []
    for batch_idx, (X_val, _)  in enumerate(dataloader):
        X_val = X_val[:,None,:,:]
        X_val = X_val.to(DEVICE)
        with torch.no_grad(): # we don't need gradients in validation
            if model.model_type == "vae":
                X_rec, mean, log_var = model(X_val)
                batch_loss, mse, kld = calculate_vae_loss(X_rec, X_val, mean, log_var,beta=beta)
            else:
                X_rec = model(X_val)
                batch_loss = loss_fc(X_rec, X_val)

        epoch_val_loss.append(batch_loss.item())
        progress_bar.set_postfix(
            loss=f"{batch_loss.item()}",
            lr = f"{optimizer.param_groups[0]['lr']:.4f}"
        )
        progress_bar.update()
        
        torch.cuda.empty_cache()
        del X_val, X_rec
    
    # progress_bar.close()
    scheduler.step(np.mean(epoch_val_loss))
    return np.mean(epoch_val_loss)

def get_baseline_data(data_loader):
    source = []
    targets = []

    for batch_x, batch_y in data_loader:
        source.append(batch_x)
        targets.append(batch_y)

    source = torch.cat(source, dim=0)
    targets = torch.cat(targets, dim=0)
    return source.cpu().detach().numpy(), targets.cpu().detach().numpy()

def create_val_dataset(source, targets):

    source = source.reshape(-1, source.shape[3])
    targets = targets.reshape(-1, targets.shape[3])

    return source, targets

def encode_data(model, data_loader, source_seq_len, target_seq_len, latent_dim, target_num_feats):
    model.eval()

    # Initialize an empty list to store the encoded data
    print("Encoding data..")
    batch_source_arr = np.empty((0, source_seq_len, latent_dim))
    batch_target_arr = np.empty((0, target_seq_len, target_num_feats))

    # Iterate over batches in the DataLoader
    for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
        # Add the depth dimension
        data = batch_X[:, None, :, :]
        data = data.to(DEVICE)
        
        # Encode the data
        with torch.no_grad():
            if model.model_type == "vae":
                data_enc = model.encode(data)[-1]
            else:
                data_enc = model.encode(data).squeeze(1)

        # batch_source_arr[batch_idx,:,:,:] = data_enc.cpu().numpy()
        # batch_target_arr[batch_idx,:,:,:] = batch_y.cpu().numpy()
        batch_source_arr = np.concatenate([batch_source_arr, data_enc.cpu().numpy()], axis=0)
        batch_target_arr = np.concatenate([batch_target_arr, batch_y.cpu().numpy()], axis=0)

    return batch_source_arr, batch_target_arr

def reconstruct_data(model, data_loader, features_in, seq_len, target_num_feats):
    model.eval()

    print("Reconstructing data..")
    batch_source_arr = np.empty((0, seq_len, features_in))
    batch_target_arr = np.empty((0, seq_len, target_num_feats))

    # Add the depth dimension
    for batch_idx, (batch_X, batch_y) in enumerate(data_loader):
        data_in = batch_X[:,None,:,:]
        data_in = data_in.to(DEVICE)

        # Encode the data
        with torch.no_grad():
            if model.model_type == "vae":
                data_out = model(data_in)[0]
            else:
                data_out = model(data_in).squeeze(1)

        batch_source_arr = np.concatenate([batch_source_arr, data_out.cpu().numpy()], axis=0)
        batch_target_arr = np.concatenate([batch_target_arr, batch_y.cpu().numpy()], axis=0)

    return batch_source_arr, batch_target_arr

def plot_cm(folds_cm, model_name, fig_title="Confusion matricies", save_fig=False, file_name=None):
    fig, axs = plt.subplots(3,2,figsize=(10,10), tight_layout=True)
    fig.suptitle(f"{fig_title}")
    axs=axs.reshape(-1) 
    for fold, results in folds_cm.items():
        cm = results[model_name]["confusion_matrix"]
        sns.heatmap(cm, annot=True, ax=axs[fold], fmt="")
        axs[fold].set_title(f"Fold: {fold}")
        axs[fold].set_ylabel("Actual Values")
        axs[fold].set_xlabel("Predicted Values")
    
    if save_fig:
        plt.close()
        fig.savefig(file_name)

def evaluate_classifcation(clf_data, clf_labels, models):
    nfolds = 5
    fold_scores = {}
    kf = KFold(nfolds, shuffle=False)
    accuracy_average = {type(clf).__name__ : [] for clf in models}
    f1_neg_average = {type(clf).__name__ : [] for clf in models}
    f1_pos_average = {type(clf).__name__ : [] for clf in models}
    print("Classifying data")
    for k, (train_indicies, test_indicies) in enumerate(kf.split(clf_data)):
        
        print(f"Fold: {k+1}/{nfolds}")
        # create training and testing datasets
        X_train_clf = clf_data[train_indicies, :]
        X_test_clf = clf_data[test_indicies, :]

        
        model_results = {}

        for clf in models:
            clf = clf.fit(X_train_clf, clf_labels[train_indicies])
            clf_name = type(clf).__name__
            y_pred = clf.predict(X_test_clf)

            f1_negative = round(f1_score(clf_labels[test_indicies], y_pred, pos_label=0),2) 
            f1_positive = round(f1_score(clf_labels[test_indicies], y_pred, pos_label=1),2)
            acc = round(accuracy_score(clf_labels[test_indicies], y_pred),2)
            cm = confusion_matrix(clf_labels[test_indicies], y_pred)
            # print(f"Class 0 F1: {f1_negative}\nClass 1 F1: {f1_positive}\nOverall Accuracy: {acc}")

            accuracy_average[clf_name].append(acc)
            f1_neg_average[clf_name].append(f1_negative)
            f1_pos_average[clf_name].append(f1_positive)
            model_results[clf_name] = {"accuracy":acc,
                                                   "confusion_matrix":cm,
                                                   "f1_negative_class":f1_negative,
                                                   "f1_positive":f1_positive}
        fold_scores[k] = model_results

    return fold_scores, {"acc":{key: np.mean(array) for key, array in accuracy_average.items()},
                         "f1_neg_class":{key: np.mean(array) for key, array in f1_neg_average.items()},
                         "f1_pos_class":{key: np.mean(array) for key, array in f1_pos_average.items()}}

def evaluate_regression(X_test, y_test, est):
    test_corr = []
    test_mse = []
    for test_src, test_tgt in zip(X_test, y_test):
        test_point_bin_corr = []

        y_pred = est.predict(test_src)

        test_mse.append(mean_squared_error(test_tgt, y_pred))
        for specBin in range(test_tgt.shape[1]):
            r, _ = pearsonr(y_pred[:,specBin], test_tgt[:,specBin])
            if np.isnan(r):
                continue
            test_point_bin_corr.append(r)
        if len(test_point_bin_corr) > 0:
            test_corr.append(np.mean(test_point_bin_corr))

    return np.average(test_corr), np.average(test_mse), np.std(test_corr)


# def evaluate_regression(source, targets, est):
#     nfolds = 5
#     fold_scores = {}
#     kf = KFold(nfolds, shuffle=False)
#     rs = np.zeros((nfolds, targets.shape[1]))
#     folds_mse = []
#     folds_corr = []
    
#     print("Performing regression...")
#     for k, (train, test) in enumerate(kf.split(source)):
#         bin_corr = []
#         print(f"Fold: {k+1}/{nfolds}")
#         # create training and testing datasets
#         X_train = source[train, :]
#         X_test = source[test, :]

#         # get targets
#         y_train = targets[train, :] 
#         y_test = targets[test,: ]
        
#         # Fit the regression model
#         est.fit(X_train, y_train)

#         #Predict the reconstructed spectrogram for the test data
#         y_pred = est.predict(X_test)
#         folds_mse.append(mean_squared_error(y_pred, y_test))

#         for specBin in range(targets.shape[1]):
#             r, p = pearsonr(y_test[:,specBin], y_pred[:,specBin])
#             bin_corr.append(r)

#         folds_corr.append(np.average(bin_corr))

#     return np.average(folds_mse), np.average(folds_corr)


def calculate_vae_loss(X_rec, X_batch, mu, logvar, beta=1):
    reconstruction_function = nn.MSELoss()
    MSE = reconstruction_function(X_rec, X_batch)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + beta*KLD, MSE, KLD


def plot_training(epochs, epoch_loss, epoch_loss_val, title="Training loss over time", file_name=None, save_fig=False):
    fig, ax = plt.subplots(1,1, figsize=(10,7.5))
    fig.suptitle(title)

    sns.lineplot(x=epochs, y=epoch_loss, ax=ax, label="Training loss")
    sns.lineplot(x=epochs, y=epoch_loss_val, ax=ax, label="Validation loss", linestyle='--')

    ax.set_xlabel("Epoch");
    ax.set_ylabel("Loss");
    ax.grid();
    ax.set_yscale("log")

    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')

def update_performance_dictionary(clf_performance_dict, metric_dict):
    for metric in metric_dict.keys():
        for model, value in metric_dict[metric].items():
            clf_performance_dict[model][metric].append(value)
    return clf_performance_dict

def plot_classifier_scores(clf_average_performance, epochs_arr, baseline_metrics, fig_title="Classifier performance over training", file_name=None, save_fig=False):
    
    # Generate unique colors using seaborn color palette
    colors = sns.color_palette('husl', n_colors=len(clf_average_performance.keys()))
    colors_dict = {clf: colors[i] for i, clf in enumerate(clf_average_performance.keys())}

    fig, axs = plt.subplots(1,1, figsize=(7.5,5), tight_layout=True)

    axs.axhline(y=[*baseline_metrics["acc"].values()], label="baseline accuracy")
    axs.axhline(y=[*baseline_metrics["f1_neg_class"].values()], color="red", label="baseline f1 negative class")
    axs.axhline(y=[*baseline_metrics["f1_pos_class"].values()], color="green",label="baseline f1 positive class")

    for clf, performance in clf_average_performance.items():

        clf_lines = sns.lineplot(x=epochs_arr, y=performance["acc"], ax=axs,color="orange", label=clf, marker="o",legend=False)
        l1=sns.lineplot(x=epochs_arr, y=performance["f1_neg_class"], linestyle=":", linewidth=2,ax=axs,color="red", label="F1 negative class", legend=False, marker="X")
        l2=sns.lineplot(x=epochs_arr, y=performance["f1_pos_class"], linestyle=":", ax=axs,color="green", label="F1 positive class",legend=False, marker="X")

    l = l2.get_lines()
    labels = [line.get_label() for line in l[:6]]

    axs.grid()
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Accuracy")
    axs.set_ybound([-0.05,1]);

    
    fig.legend(l, labels, loc="center", bbox_to_anchor=(1.15, 0.825))
    fig.suptitle(fig_title)
    
    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')

def plot_regression_results(baseline_results, latent_results, rec_results, epochs_arr, title=None, file_name=None, save_fig=False):
    

    fig, axs = plt.subplots(1,1, figsize=(7.5,5), tight_layout=True)
    fig.suptitle(title)
    sns.lineplot(x=epochs_arr, y=baseline_results, ax=axs,color="green", label="Baseline - original",)
    sns.lineplot(x=epochs_arr, y=latent_results, ax=axs, color="orange", marker="o", label="Latent",)
    sns.lineplot(x=epochs_arr, y=rec_results, ax=axs, color="orange", linestyle=":", marker="o", label="Reconstruction",)
    
    axs.grid()
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Corr. (%)")
    axs.set_ybound([-0.05,1]);
    axs.legend(title="Data types")
    
    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')

def plot_reconstruction_autoencoder(X_rec, X_true, ch_names, limit_display=True, fig_title="Signal reconstruction", file_name=None, save_fig=False):

    # Limit the plot to 10 channels only
    if limit_display:
        n_channels = 12
        channels_to_display = np.random.choice(np.arange(X_true.shape[3]), n_channels, replace=False)
    else:
        n_channels = X_true.shape[3]

    n_cols = 3
    n_rows = int(np.ceil(n_channels/n_cols))

    fig, axs = plt.subplots(n_rows,n_cols,figsize=(n_cols*10,n_rows*2.5), tight_layout=True)
    fig.suptitle(f"{fig_title}")
    axs = axs.reshape(-1)
    for ch_idx in range(n_channels):
        if limit_display:
            sns.lineplot(X_true[0,0,:, channels_to_display[ch_idx]], ax=axs[ch_idx], label="Original");
            sns.lineplot(X_rec[0,0,:,channels_to_display[ch_idx]], ax=axs[ch_idx], label="Reconstructed", linestyle='--');
            axs[ch_idx].set_title(f"Channel: {ch_names[channels_to_display[ch_idx]]}")
        else:   
            sns.lineplot(X_true[0,0,:,ch_idx], ax=axs[ch_idx], label="Original");
            sns.lineplot(X_rec[0,0,:,ch_idx], ax=axs[ch_idx], label="Reconstructed", linestyle='--');
            axs[ch_idx].set_title(f"Channel: {ch_names[ch_idx]}")
        axs[ch_idx].grid()
        axs[ch_idx].set_xlabel("Time")
        axs[ch_idx].set_xlabel("Amplitude")
    
    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_corr = 0

    def early_stop(self, validation_corr):
        if validation_corr > self.max_validation_corr:
            self.max_validation_corr = validation_corr
            self.counter = 0
        elif validation_corr < (self.max_validation_corr + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EarlyStopperAE:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def count_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

# def calculate_pearsonr(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
#     """
#     Evaluate the correlation between two data sets using Pearson correlation coefficient.

#     Parameters:
#     data1 (numpy.ndarray): First data set.
#     data2 (numpy.ndarray): Second data set.

#     Returns:
#     numpy.ndarray: Array of Pearson correlation coefficients for each column.
#     """
#     batch_size, _, spec_bin_count = data1.shape
#     correlation_coefficients = np.zeros((batch_size, spec_bin_count))
#     print('Calculating Pearson...')
    
#     for batch in range(batch_size):
#         for spec_bin in range(spec_bin_count):
#             r, _ = pearsonr(data1[batch, :, spec_bin], data2[batch, :, spec_bin])
#             correlation_coefficients[batch, spec_bin] = r
#     return correlation_coefficients

# def calculate_spearmanr(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
#     """
#     Evaluate the correlation between two data sets using Spearman rank correlation coefficient.

#     Parameters:
#     data1 (numpy.ndarray): First data set.
#     data2 (numpy.ndarray): Second data set.

#     Returns:
#     numpy.ndarray: Array of Spearman correlation coefficients for each column.
#     """
#     batch_size, _, spec_bin_count = data1.shape
#     correlation_coefficients = np.zeros((batch_size, spec_bin_count))
#     print('Calculating spearmanr...')
#     for batch in range(batch_size):
#         for spec_bin in range(spec_bin_count):
#             r, _ = spearmanr(data1[batch, :, spec_bin], data2[batch, :, spec_bin])
#             correlation_coefficients[batch, spec_bin] = r
#     return correlation_coefficients


def calculate_pearsonr(logits, y):
    pearson_r = cosine_similarity(y-torch.mean(y,dim=1).unsqueeze(1),logits-torch.mean(logits,dim=1).unsqueeze(1),dim=1) 
    return pearson_r

def calculate_pearsonr_np(logits, y):

    # Calculate the mean of y and logits
    y_mean = np.mean(y, axis=0, keepdims=True)
    logits_mean = np.mean(logits, axis=0, keepdims=True)
    
    # Subtract the mean from y and logits
    y_centered = y - y_mean
    logits_centered = logits - logits_mean
    
    # Calculate the Pearson correlation using cosine similarity
    pearson_r = np.sum(y_centered * logits_centered, axis=0) / (
        np.sqrt(np.sum(y_centered ** 2, axis=0)) * np.sqrt(np.sum(logits_centered ** 2, axis=0))
    )
    
    return pearson_r

def custom_inference(model, src, src_seq_len=100, tgt_seq_len=100, tgt_feats=80):

    pos_eeg = torch.arange(1, src_seq_len+1).unsqueeze(0).to(DEVICE)

    # Initialize decoder input with the start token
    decoder_input = torch.zeros((src.shape[0], tgt_seq_len+1, tgt_feats)).to(DEVICE)
    
    pbar = tqdm(range(tgt_seq_len), desc=f"Validating...", position=0, leave=False)
    with torch.no_grad():
        for t in pbar:
            pos_mel = torch.arange(1,t+2).unsqueeze(0).to(DEVICE)
            mel_pred, attn, attn_enc, attn_dec = model(src, decoder_input[:,:t+1,:], pos_eeg, pos_mel)
            
            decoder_input[:,t+1,:] = mel_pred[:, -1, :].detach()

    return mel_pred, attn, attn_enc, attn_dec



def inference(eeg_data, model, max_length=200,target_num_feats=80):    
    # # Hook to save attention outputs
    # save_output = SaveOutput()

    # # Patch and register hooks for all self-attention layers in the decoder
    # for layer in model.decoder.transformer_decoder.layers:
    #     # Patch the self-attention layer so it always returns attention weights
    #     patch_attention(layer.self_attn)
    #     # Register the hook to save the attention weights
    #     layer.self_attn.register_forward_hook(save_output)

    with torch.no_grad():
        # Step 1: Encode EEG Data
        encoded_eeg = model.encode(eeg_data)
        
        # Step 2: Initialize Decoder with start token or initial frame
        decoder_input = torch.zeros((eeg_data.size(0), 1, target_num_feats)).to(eeg_data.device)
        for _ in range(max_length):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1])

            # Step 3: Autoregressive Decoding
            mel_lin = model.decode(decoder_input, encoded_eeg, tgt_mask=tgt_mask, tgt_is_causal=True)
           
            # Select the last frame in the sequence (most recent prediction)
            predicted_frame = mel_lin[:, -1:, :]  # Shape: (batch_size, 1, mel_spec_frame_dim)
            
            # Update decoder input to be the predicted frame
            decoder_input = torch.cat([decoder_input, predicted_frame], dim=1)  # Append new frame to input sequence
    # print(save_output.outputs)
    return mel_lin


def plot_spectrogram(spec, speech_labels, file_name):
    fig, axs = plt.subplots(2,1, figsize=(10,5), tight_layout=True, sharex=True)
    fig.suptitle("Mel-spectrogram")
    cm='viridis'

    # plot the test mel spectrogram
    axs[0].imshow(np.flipud(spec.T), cmap=cm, interpolation=None,aspect='auto')
    axs[0].set_ylabel('Log Mel-Spec Bin')
    axs[0].set_title("Ground truth")
    axs[0].grid()

    
    # Iterate over the data and plot horizontal spans
    for i in range(len(speech_labels)):
        if speech_labels[i] == 1:
            axs[1].axhspan(0, 1, xmin=i/len(speech_labels), xmax=(i+1)/len(speech_labels), facecolor='blue')
        else:
            axs[1].axhspan(0, 1, xmin=i/len(speech_labels), xmax=(i+1)/len(speech_labels), facecolor='red')

    # Create empty plot objects with labels for legend
    axs[1].set_xlabel("Time")
    
    plt.close()
    fig.savefig(file_name, bbox_inches = 'tight')





def plot_reconstruction(y_pred, y_test, title="Mel Spectrogram reconstruction", save_fig=False, file_name=None):

    fig, axs = plt.subplots(2,1, figsize=(10,5), tight_layout=True, sharex=True)
    fig.suptitle(title, x=0.52)
    cm='viridis'

    # plot the test mel spectrogram
    axs[0].imshow(np.flipud(y_test.T), cmap=cm, interpolation=None,aspect='auto')
    axs[0].set_ylabel('Log Mel-Spec Bin')
    axs[0].set_title("Original Mel Spectrogram")
    axs[0].grid()

    # plot the reconstruction
    axs[1].imshow(np.flipud(y_pred.T), cmap=cm, interpolation=None,aspect='auto')
    axs[1].set_title("Predicted Mel Spectrogram")
    axs[1].grid()

    # Create empty plot objects with labels for legend
    axs[1].set_xlabel("Time")
    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')

def scheduled_sampling_rate(epoch, initial_rate=1.0, decay=0.995):
    return initial_rate * (decay ** epoch)


def create_decoder_input(target_seq, start_token):
    batch_size = target_seq.size(0)
    start_tokens = start_token.repeat(batch_size, 1, 1).to(target_seq.device)  # Shape: (batch_size, 1, mel_spec_frame_dim)
    shifted_target = torch.cat([start_tokens, target_seq[:, :-1, :]], dim=1)  # Shifted right by one position
    return shifted_target


def adjust_learning_rate(optimizer, step_num, init_lr, warmup_step=4000):
    lr = init_lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def compare_train_val(mel_train_pred, mel_val_pred, mel_true, p_id, save_fig=False, file_name=None):
    fig, axs = plt.subplots(3,1, figsize=(10,5), tight_layout=True, sharex=True)
    fig.suptitle(f"Mel reconstructions, p_id: {p_id}", x=0.52)
    cm='viridis'

    # plot the test mel spectrogram
    axs[0].imshow(np.flipud(mel_true.T), cmap=cm, interpolation=None,aspect='auto')
    axs[0].set_ylabel('Log Mel-Spec Bin')
    axs[0].set_title("Ground truth")
    axs[0].grid()

    # plot the reconstruction
    axs[1].imshow(np.flipud(mel_train_pred.T), cmap=cm, interpolation=None,aspect='auto')
    axs[1].set_title("Training")
    axs[1].grid()

    # plot the reconstruction
    axs[2].imshow(np.flipud(mel_val_pred.T), cmap=cm, interpolation=None,aspect='auto')
    axs[2].set_title("Validation (on training data)")
    axs[2].grid()

    # Create empty plot objects with labels for legend
    axs[2].set_xlabel("Time")
    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')



def reconstruct_and_save_audio(output_path: str, 
                               file_name: str, 
                               data: np.ndarray,
                               feat_type: str = 'MFCC') -> None:
    """
    Reconstruct audio from Mel Spectrogram and save it.

    Parameters:
    output_path (str): Path to save the audio file.
    file_name (str): Name of the audio file.
    data (numpy.ndarray): Mel Spectrogram data.
    """

    output_path_standard = output_path / str(file_name+'.wav')

    # Load model
    standard_model = StreamingVocGan(is_streaming=False,model_path=Path("./StreamingVocGan/vctk_pretrained_model_3180.pt"))
    
    # Convert
    waveform_standard, processing_time = standard_model.mel_spectrogram_to_waveform(mel_spectrogram=torch.Tensor(data).to('cuda').T)

    # Save
    StreamingVocGan.save(waveform=waveform_standard.cpu(), file_path=output_path_standard)
    
    return waveform_standard


def plot_audio_signal(wave_true, wave_pred,title="Audio reconstruction",file_name=None,save_fig=False):
    fig, ax = plt.subplots(2,1,figsize=(10,10))
    fig.suptitle(title, y=0.94, x=0.52)

    ax[0].plot(wave_true.cpu())
    ax[1].plot(wave_pred.cpu())

    ax[0].grid()
    ax[1].grid()

    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")

    ax[0].set_xlabel("Time")
    ax[1].set_xlabel("Time")

    ax[0].set_title("Original audio signal")
    ax[1].set_title("Predicted audio signal")

    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')



class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


def calculate_distance_spectrograms(mel_pred, mel_true):

  # Efficient vectorized computation of Euclidean distance
  # Calculate the squared difference between corresponding vectors
  squared_diff = np.square(mel_pred - mel_true)

  # Sum the squared differences along the last axis (which is the 80-dimensional vector)
  sum_squared_diff = np.sum(squared_diff, axis=-1)

  # Take the square root to get the Euclidean distance
  dist = np.sqrt(sum_squared_diff)

  return dist


def plot_dist_heatmap(all_dist, output_path, filename="heatmap", num_bins=50):

    # Define the number of bins for distance values (Y-axis)
    min_dist = np.min(all_dist)
    max_dist = np.max(all_dist)

    # Create bins for the distances
    bins = np.linspace(min_dist, max_dist, num_bins + 1)

    # Prepare an array to store frequency counts for each time point
    heatmap_data = np.zeros((num_bins, all_dist.shape[1]))  # shape [num_bins, 100 time points]

    # Count the frequency of distances within each bin for each time point
    for t in range(all_dist.shape[1]):  # Loop over timepoints (100)
        hist, _ = np.histogram(all_dist[:, t], bins=bins)
        heatmap_data[:, t] = hist

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower', 
            extent=[0, all_dist.shape[1], min_dist, max_dist])
    cbar = plt.colorbar()
    cbar.set_label('Frequency', fontsize=15)
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Distance',fontsize=15)
    plt.title('Frequency of distances over timepoints',fontsize=20)
    plt.savefig(f"{output_path}/{filename}.png")
    plt.close()

def plot_corr_spec_bins(data, save_path, filename="corr_spec_bins"):
    fig, ax = plt.subplots(1,1,figsize=(12, 8))
    fig.suptitle("Average correlation across spectral bins",fontsize=20)
    ax.bar(np.arange(len(data)), data)
    ax.grid(axis="y")
    ax.set_ylim([-0.05, 1.0])
    ax.set_ylabel("Correlation (%)",fontsize=15)
    ax.set_xlabel("Spectral Bins",fontsize=15)
    plt.close()
    fig.savefig(f"{save_path}/{filename}.png")

def plot_corr_speech_ratio(data, speech_labels, save_path, filename="corr_dist_speech_ratio"):
    # sppech_labels: array [number of points x sequence_length] holds a speech label (0 or 1) for each timepoint in the sequence

    speech_ratio = np.sum(speech_labels, axis=1) / speech_labels.shape[-1]
    x_bins = np.linspace(-1, 1, 40)
    y_bins = np.linspace(0, 1, 20) 
    heatmap, _, _ = np.histogram2d(data, speech_ratio, bins=[x_bins, y_bins])

    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap.T, origin='lower', aspect='auto', cmap='viridis',
    extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])

    cbar = plt.colorbar()
    cbar.set_label('Frequency', fontsize=15)
    plt.ylabel('Speech percentage (%)',fontsize=15)
    plt.xlabel('Correlation (%)',fontsize=15)
    plt.title('Correlation over the amount of speech in a testing window',fontsize=20)
    plt.savefig(f"{save_path}/{filename}.png")
    plt.close()

def plot_corr_avg(data, save_path, filename="test_corr_dist"):
    fig, ax = plt.subplots(1,1,figsize=(12, 8))
    fig.suptitle("Average correlation distribution",fontsize=20)
    ax.hist(data, bins=50, edgecolor='black')
    ax.grid(axis="y")
    ax.set_ylabel("Frequency",fontsize=15)
    ax.set_xlabel("Correlation (%)",fontsize=15)
    plt.close()
    fig.savefig(f"{save_path}/{filename}.png")

def encode_data_transformer(model, data_loader):
    model.eval()

    # Initialize an empty list to store the encoded data
    print("Encoding data..")
    # batch_source_arr = np.empty(num_test_samples, source_seq_len, latent_dim)
    batch_source_arr = []

    # Iterate over batches in the DataLoader
    for batch_idx, (batch_eeg, _) in enumerate(data_loader):
        batch_eeg = batch_eeg.to(DEVICE)

        with torch.no_grad():
            batch_eeg_encoded = model.encode(batch_eeg)

        batch_source_arr.append(batch_eeg_encoded.cpu().numpy())
        # batch_source_arr[batch_idx*batch_size:batch_idx*batch_size,:,:,:] = batch_eeg_encoded.cpu().numpy()
    batch_source_arr = np.concatenate(batch_source_arr, axis=0)
    return batch_source_arr


def encode_data_custom_transformer(model, data_loader, src_seq_len=100):
    model.eval()

    pos_eeg = torch.arange(1, src_seq_len+1).unsqueeze(0).to(DEVICE)

    # Initialize an empty list to store the encoded data
    print("Encoding data..")
    batch_source_arr = []

    # Iterate over batches in the DataLoader
    for _, (batch_eeg, _) in enumerate(data_loader):
        batch_eeg = batch_eeg.to(DEVICE)

        with torch.no_grad():
            batch_eeg_encoded, _, attn_enc = model.encoder(batch_eeg, pos_eeg)

        batch_source_arr.append(batch_eeg_encoded.cpu().numpy())
    batch_source_arr = np.concatenate(batch_source_arr, axis=0)
    return batch_source_arr



def plot_attn(attn, new_dir_output, num_h=8, src_seq_len=100, tgt_seq_len=100,title="Attention matrix", save_filename="attn"):

    # Initialize an array to store the average attention matrices for each head in the last layer
    average_attention_per_head = np.zeros((num_h, tgt_seq_len, src_seq_len))

    # Compute the average attention matrix for each head in the last layer
    for head in range(num_h):
        # Extract all testing examples for this particular head in the last layer
        head_matrices = attn[head::num_h]
        
        # Compute the average attention matrix across all testing examples for this head
        average_attention_per_head[head] = np.mean(head_matrices, axis=0)

    # Create a figure to summarize the information of all 8 heads in the last layer
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), tight_layout=True)  # 2 rows, 4 columns grid for the 8 heads
    fig.suptitle(title, fontsize=16)

    # Plot the average attention matrix for each head in the last layer
    for head in range(num_h):
        ax = axes[head // 4, head % 4]  # Organize into 2 rows and 4 columns
        sns.heatmap(average_attention_per_head[head].T, cmap="viridis", cbar=False, ax=ax)
        ax.set_title(f'Head {head+1}')
        ax.set_xlabel("Token Index (Column)")
        ax.set_ylabel("Token Index (Row)")

    # Add an overall title for the figure

    # Display the figure
    plt.close()
    fig.savefig(f"{new_dir_output}/{save_filename}.png")




def get_signal_indicies(label_array, output_path, pt_id):
    # Initialize the lists for the indices
    speech = []
    silence = []
    speech_ratio_arr = []

    # Loop through each row by index
    for i in range(label_array.shape[0]):
        # Calculate the number of 1s and 0s
        num_ones = np.sum(label_array[i] == 1)
        num_zeros = np.sum(label_array[i] == 0)
        
        # Calculate the ratio of 1s and 0s
        ratio_ones = num_ones / label_array.shape[1]
        ratio_zeros = num_zeros / label_array.shape[1]

        speech_ratio_arr.append(ratio_ones)

        # Classify based on the ratio
        if ratio_ones > 0.9:
            speech.append(i)  
        elif ratio_ones <= 0.1:
            silence.append(i)  
    
    plt.figure(figsize=(8, 6))
    plt.hist(speech_ratio_arr, bins=50, edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Speech Ratios Histogram')

    # Save the plot to a file
    plt.savefig(f'{output_path}/{pt_id}_ratios_hist.png', dpi=300, bbox_inches='tight')

    # speech and silence now contain the indices
    return speech, silence, speech_ratio_arr


def average_sequences_by_category(arr1, arr2):
    """
    Function to compute the average sequences for each category (0 and 1).
    
    Parameters:
    arr1: np.ndarray of shape (n_sequences, sequence_length, n_features)
          The array containing sequences with their features.
    arr2: np.ndarray of shape (n_sequences, sequence_length)
          The array containing classifications (0 or 1) for each time point in each sequence.
          
    Returns:
    avg_0: np.ndarray of shape (sequence_length, n_features)
           The average sequence for category 0.
    avg_1: np.ndarray of shape (sequence_length, n_features)
           The average sequence for category 1.
    """
    
    # Ensure the input arrays have compatible shapes
    assert arr1.shape[:2] == arr2.shape, "Shapes of arr1 and arr2 must match in the first two dimensions."
    
    # Determine the dimensions
    n_sequences, sequence_length, n_features = arr1.shape
    
    # Create masks for category 0 and 1
    mask_0 = (arr2 == 0)  # Mask for category 0 points
    mask_1 = (arr2 == 1)  # Mask for category 1 points

    # Initialize sums and counts for both categories
    sum_0 = np.zeros((sequence_length, n_features))
    sum_1 = np.zeros((sequence_length, n_features))
    
    # Initialize counts for both categories
    count_0 = np.zeros(sequence_length)
    count_1 = np.zeros(sequence_length)

    # Efficiently calculate the sum of points for each category using broadcasting
    for i in range(n_sequences):
        sum_0 += np.where(mask_0[i][:, np.newaxis], arr1[i], 0)
        sum_1 += np.where(mask_1[i][:, np.newaxis], arr1[i], 0)
        count_0 += mask_0[i]
        count_1 += mask_1[i]

    # Compute the average sequences for each category, avoiding division by zero
    avg_0 = np.divide(sum_0, count_0[:, None], where=(count_0[:, None] != 0))
    avg_1 = np.divide(sum_1, count_1[:, None], where=(count_1[:, None] != 0))

    return avg_0, avg_1



def calculate_mcd(mfccs1: torch.Tensor, mfccs2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mel Cepstral Distortion (MCD) between two MFCC sets using PyTorch tensors.

    Parameters:
    mfccs1 (torch.Tensor): First set of MFCCs.
    mfccs2 (torch.Tensor): Second set of MFCCs.

    Returns:
    torch.Tensor: The MCD value.

    Raises:
    AssertionError: If the dimensions of the input MFCCs do not match.
    """
    # Exclude the zeroth coefficient and ensure the dimensions match
    mfccs1, mfccs2 = mfccs1[:, :, 1:], mfccs2[:, :, 1:]
    assert mfccs1.shape == mfccs2.shape, 'Dimensions do not match'

    # Calculate the squared differences
    diff = torch.sum((mfccs1 - mfccs2) ** 2, dim=2)

    # Calculate the MCD
    mcd = torch.mean(torch.sqrt(diff), dim=1) * (10 / torch.log(torch.tensor(10.0)) * torch.sqrt(torch.tensor(2.0)))
    
    return mcd