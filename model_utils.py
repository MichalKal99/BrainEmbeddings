from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_model(model, dataloader, loss_fn, optimizer, batch_size=1024, beta=1):
        
    model.train()
    
    # Process tqdm bar, helpful for monitoring training process
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, 
                     leave=False, position=0, desc="Train")
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
            lr = f"{optimizer.param_groups[0]['lr']:.4f}"
        )
        batch_bar.update()             
        
        # remove unnecessary cache in CUDA memory
        torch.cuda.empty_cache()
        del X_train, X_rec
    # scheduler.step()
    # batch_bar.close()

    return np.mean(epoch_training_loss)


def validate_model(model, dataloader, loss_fc, optimizer, scheduler, batch_size=1024, beta=1):
    model.eval()
    progress_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Validation")
    epoch_val_loss = []
    for batch_idx, (X_val, _)  in enumerate(dataloader):
        X_val = X_val[:,None,:,:]
        X_val = X_val.to(DEVICE)
        with torch.no_grad(): # we don't need gradients in validation
            if model.model_type == "vae":
                X_rec, mean, log_var = model(X_val)
                batch_loss, mse, kld = calculate_vae_loss(X_rec, X_val, mean, log_var, beta=beta)
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

def encode_data(model, data_loader):
    model.to("cpu")
    model.eval()

    # Initialize an empty list to store the encoded data
    encoded_data = []

    print("Encoding data..")

    # Iterate over batches in the DataLoader
    for batch_X, _ in data_loader:
        # Add the depth dimension
        data = batch_X[:, None, :, :]

        # Encode the data
        with torch.no_grad():
            if model.model_type == "vae":
                data_enc = model.encode(data)[-1]
            else:
                data_enc = model.encoder(data)
                print(data_enc.shape)
        # Flatten the encoded data
        data_enc = data_enc.view(data_enc.size(0), -1)

        # Append the encoded batch to the list
        encoded_data.append(data_enc)

    # Concatenate the encoded batches along the sample dimension
    encoded_data = torch.cat(encoded_data, dim=0)

    model.to(DEVICE)
    return encoded_data

def reconstruct_data(model, data_loader):

    model.to("cpu")
    model.eval()
    data_rec = []

    print("Reconstructing data..")
    # Add the depth dimension
    for batch_X, _ in data_loader:
        data_in = batch_X[:,None,:,:]

        # Encode the data
        with torch.no_grad():
            if model.model_type == "vae":
                data_out = model(data_in)[0]
            else:
                data_out = model(data_in)

        data_rec.append(data_out)

    # Concatenate the encoded batches along the sample dimension
    data_rec = torch.cat(data_rec, dim=0)

    # Flatten the encoded data
    data_rec = data_rec.view(data_rec.size(0), data_rec.size(2), data_rec.size(3))
    model.to(DEVICE)

    return data_rec

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
        
        # create training and testing datasets
        X_train_clf = clf_data[train_indicies, :]
        X_test_clf = clf_data[test_indicies, :]

        
        model_results = {}

        for clf in models:
            clf = clf.fit(X_train_clf, clf_labels[train_indicies])
            clf_name = type(clf).__name__
            # print(clf_name)
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



def calculate_vae_loss(X_rec, X_batch, mu, logvar, beta=1):
    reconstruction_function = nn.MSELoss()
    MSE = reconstruction_function(X_rec, X_batch)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + beta*KLD, MSE, KLD

def plot_training(epochs, epoch_loss, epoch_loss_val, clf_latent_scores=None, clf_rec_scores=None, clf_epochs=None, clf_original_acc=None, fig_name=None, save_fig=False):
    fig, ax = plt.subplots(1,1, figsize=(10,7.5))
    fig.suptitle("Training loss over time")
    ax2 = ax.twinx()
    if clf_original_acc:
        l0 = ax2.axhline(y=clf_original_acc, linestyle="-", color="green", label="Baseline classification accuracy")
    l1 = sns.lineplot(x=epochs, y=epoch_loss, ax=ax, label="Training loss", legend=False)
    l2 = sns.lineplot(x=epochs, y=epoch_loss_val, ax=ax, label="Validation loss", linestyle='--', legend=False)
    
    if clf_latent_scores:
        for clf in clf_latent_scores:
            clf_line = sns.lineplot(x=clf_epochs,y=clf_latent_scores[clf]["acc"], ax=ax2, label=f"{clf} latent accuracy", marker="o",legend=False)
    
    if clf_rec_scores:
        for clf in clf_rec_scores:
            clf_line = sns.lineplot(x=clf_epochs,y=clf_rec_scores[clf]["acc"], ax=ax2, label=f"{clf} reconstruction accuracy", marker="o",legend=False)

    # added these three lines
    l = clf_line.get_lines() + l1.get_lines()
    labs = [l.get_label() for l in l]
    ax.legend(l, labs, fontsize=7);

    ax.set_xlabel("Epoch");
    ax.set_ylabel("Loss");
    ax.grid();
    ax2.set_ybound([0.45,1]);
    ax2.set_ylabel("Accuracy");

    if save_fig:
        plt.close()
        fig.savefig(fig_name, bbox_inches = 'tight')

def update_performance_dictionary(clf_performance_dict, metric_dict):
    for metric in metric_dict.keys():
        for model, value in metric_dict[metric].items():
            clf_performance_dict[model][metric].append(value)
    return clf_performance_dict

def plot_classifier_scores(clf_average_performance, epochs_arr, baseline_metrics, fig_name="Classifier performance over training", save_fig=False):
    
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
    fig.suptitle(fig_name)
    
    if save_fig:
        plt.close()
        fig.savefig(fig_name, bbox_inches = 'tight')


def plot_reconstruction(X_rec, X_true, ch_names, limit_display=True, fig_title="Signal reconstruction", file_name=None, save_fig=False):

    # Limit the plot to 10 channels only
    if limit_display:
        n_channels = 12
        channels_to_display = np.random.choice(np.arange(X_true.shape[2]), n_channels, replace=False)
    else:
        n_channels = X_true.shape[2]

    n_cols = 3
    n_rows = int(np.ceil(n_channels/n_cols))

    fig, axs = plt.subplots(n_rows,n_cols,figsize=(n_cols*10,n_rows*2.5), tight_layout=True)
    fig.suptitle(f"{fig_title}")
    axs = axs.reshape(-1)
    for ch_idx in range(n_channels):
        if limit_display:
            sns.lineplot(X_true[0,0,channels_to_display[ch_idx],:], ax=axs[ch_idx], label="Original");
            sns.lineplot(X_rec[0,0,channels_to_display[ch_idx],:], ax=axs[ch_idx], label="Reconstructed", linestyle='--');
        else:   
            sns.lineplot(X_true[0,0,ch_idx,:], ax=axs[ch_idx], label="Original");
            sns.lineplot(X_rec[0,0,ch_idx,:], ax=axs[ch_idx], label="Reconstructed", linestyle='--');
        axs[ch_idx].set_title(f"Channel: {ch_names[ch_idx][0]}")
        axs[ch_idx].grid()
    
    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')


class EarlyStopper:
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

def calculate_pearsonr(logits, y):
    pearson_r = cosine_similarity(y-torch.mean(y,dim=1).unsqueeze(1),logits-torch.mean(logits,dim=1).unsqueeze(1),dim=1) 
    return pearson_r



def plot_reconstruction(y_pred, y_test, save_fig=False, file_name=None):

    fig, axs = plt.subplots(2,1, figsize=(10,5), tight_layout=True, sharex=True)
    fig.suptitle("Mel Spectrogram reconstruction")
    cm='viridis'

    # plot the test mel spectrogram
    axs[0].imshow(np.flipud(y_test.T), cmap=cm, interpolation=None,aspect='auto')
    axs[0].set_ylabel('Log Mel-Spec Bin')
    axs[0].set_title("Ground truth")
    axs[0].grid()

    # plot the reconstruction
    axs[1].imshow(np.flipud(y_pred.T), cmap=cm, interpolation=None,aspect='auto')
    axs[1].set_title("Prediction")
    axs[1].grid()

    # Create empty plot objects with labels for legend
    axs[1].set_xlabel("Time (seconds)")
    if save_fig:
        plt.close()
        fig.savefig(file_name, bbox_inches = 'tight')
