from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from pathlib import Path
import seaborn as sns
from preprocessing.loading_files import load_data
from models.model_utils import plot_reconstruction
import pickle 
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt 
import os 
from models.model_utils import calculate_pearsonr, calculate_pearsonr_np
from models.dataset import TimeSeriesDataset
import torch 
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")

# pt_arr = ["p00", "p01", "p06",  "p07",  "p08", "p09", "p10", "p11", "p12", "p16"]
pt_arr = ["p00", "p01"]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Create a list of unique colors for each string
colors = sns.color_palette("husl", len(pt_arr))
pt_colors = {string: color for string, color in zip(pt_arr, colors)}

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
        "batch_size":512,
        "pt_colors":pt_colors
    }

evaluation_metrics_ppt = {pt_id:{} for pt_id in pt_arr}

for pt_id in pt_arr:
    print(f"\nPatient: {pt_id}")
    if not os.path.exists(f"./linear_regression/{pt_id}"):
        # If it doesn't exist, create it
        os.makedirs(f"./linear_regression/{pt_id}")

    config["p_id"] = pt_id
    eeg_feat, audio_feat, ch_names, _, _, _, _, _, _ = load_data(config)
    
    # NO SHUFFLE
    X_train, X_test, y_train, y_test = train_test_split(eeg_feat, audio_feat, test_size=0.30, shuffle=False, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, shuffle=False, random_state=0)
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")

    train_set = TimeSeriesDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), device, 100, 5)
    test_set = TimeSeriesDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), device, 100, 5)
    val_set = TimeSeriesDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), device, 100, 5)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=False) 
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False) 
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False) 

    train_set_x = np.empty((0,100, X_train.shape[-1]))
    train_set_y = np.empty((0,100, config["feature_extraction"]["num_feats"]))

    for train_batch_x, train_batch_y in train_loader:
        train_set_x = np.concatenate([train_set_x, train_batch_x.cpu().detach().numpy()], axis=0)
        train_set_y = np.concatenate([train_set_y, train_batch_y.cpu().detach().numpy()], axis=0)

    X_train = train_set_x.reshape(train_set_x.shape[0]*train_set_x.shape[1],train_set_x.shape[-1])
    y_train = train_set_y.reshape(train_set_y.shape[0]*train_set_y.shape[1],train_set_y.shape[-1])

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    X_test = np.empty((0,100, X_train.shape[-1]))
    y_test = np.empty((0,100, 80))
    for batch_x, batch_y in test_loader:
        X_test = np.concatenate([X_test, batch_x.cpu().detach().numpy()])
        y_test = np.concatenate([y_test, batch_y.cpu().detach().numpy()])

    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    est = LinearRegression(n_jobs=5)

    # Save the correlation coefficients for each fold
    test_corr = []
    test_mse = []

    # Fit the regression model
    est.fit(X_train, y_train)
    
    # Predict the reconstructed spectrogram for the test data
    for idx, (test_src, test_tgt) in enumerate(zip(X_test, y_test)):
        test_pred = est.predict(test_src)
        test_point_bin_corr = []

        test_mse.append(mean_squared_error(test_tgt, test_pred))
        for specBin in range(test_tgt.shape[1]):
            r, p = pearsonr(test_pred[:,specBin], test_tgt[:,specBin])
            if np.isnan(r):
                continue
            test_point_bin_corr.append(r)
        if len(test_point_bin_corr) > 0:
            test_corr.append(np.mean(test_point_bin_corr))
    
    idx_to_plot = np.random.choice(range(X_test.shape[0]), 40, replace=False)
    for idx, test_idx in enumerate(idx_to_plot):
        test_pred = est.predict(X_test[test_idx])
        plot_reconstruction(test_pred, y_test[test_idx], title=f"Mel Spectrogram reconstruction for {pt_id}", save_fig=True, file_name=f"linear_regression/{pt_id}/spectrogram_reconstruction_{idx}")

    print("Average corr: ", np.average(test_corr))
    print("Average mse: ", np.average(test_mse))

    evaluation_metrics_ppt[pt_id]["test_mean_mse"] = np.mean(test_mse)
    evaluation_metrics_ppt[pt_id]["test_mean_corr"] = np.mean(test_corr)
    evaluation_metrics_ppt[pt_id]["test_std_corr"] = np.std(test_corr)
    evaluation_metrics_ppt[pt_id]["test_sem_corr"] = np.std(test_corr)/np.sqrt(len(test_corr))
    evaluation_metrics_ppt[pt_id]["n_test_points"] = len(test_corr)

# Extract the keys, metric1 values, and metric2 values for plotting
patients = list(evaluation_metrics_ppt.keys())
test_mean_corr_values = [evaluation_metrics_ppt[pt_id]['folds_corr_mean'] for pt_id in patients]  # test_mean_corr as bar height
test_sem_corr_values = [np.std(evaluation_metrics_ppt[pt_id]['folds_corr'])/np.sqrt(len(evaluation_metrics_ppt[key]['folds_corr'])) for pt_id in patients]    # test_std_corr as error bars

# Plot the vertical bar plot with error bars
plt.figure(figsize=(10, 6))
plt.bar(patients, test_mean_corr_values, yerr=test_sem_corr_values, capsize=5, color='skyblue', edgecolor='black')

# Set plot labels and title
plt.xlabel('Patient')
plt.ylabel('Test Correlation (%)')
plt.title('Performance of EEG2MelSpectrogram transformer')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent label clipping
plt.grid(axis="y")
plt.savefig("./linear_regression/lr_results_barplot.png", format='png', dpi=300)

results_filename = f"lr_results_dict.pkl"
with open(f'./saved_results/{results_filename}', 'wb') as f:
    pickle.dump(evaluation_metrics_ppt, f)
f.close()

print("Done!")