import numpy as np

from datetime import datetime
from models.dataset import TimeSeriesDataset

import pickle 

from sklearn.model_selection import train_test_split


import torch

from torch.utils.data import DataLoader

from preprocessing.loading_files import *
from preprocessing.audio_utils import *

from models.autoencoder import *
from models.model_utils import *
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")

def main(new_dir_path, config, pt_arr):
    print(f"Starting script at: {datetime.now()}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_loss_ppt = {pt_id: [] for pt_id in pt_arr}
    for _, pt_id in enumerate(pt_arr):
        print(f"Patient: {pt_id}")
        config["p_id"] = pt_id
        eeg_feat, audio_feat, ch_names, _, _, _, _, _, _ = load_data(config)
        print("Creating train/test datasets...")
        
        X_train, X_test, y_train, y_test = train_test_split(eeg_feat, audio_feat, test_size=0.30, shuffle=False, random_state=0)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, shuffle=False, random_state=0)
        
        # Convert numpy arrays to torch tensors and create the Dataset and DataLoader
        full_dataset = TimeSeriesDataset(torch.from_numpy(eeg_feat), torch.from_numpy(audio_feat), device, config['TR']["context_length"], config['TR']["context_offset"])
        train_set = TimeSeriesDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), device, config['TR']["context_length"], config['TR']["context_offset"])
        test_set = TimeSeriesDataset(torch.from_numpy(X_test), torch.from_numpy(y_test), device, config['TR']["context_length"], config['TR']["context_offset"])
        val_set = TimeSeriesDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), device, config['TR']["context_length"], config['TR']["context_offset"])

        full_loader = DataLoader(dataset=full_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
        train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, drop_last=False)

        print("Creating model...")

        print("#"*20)
        # Model Initialization
        autoencoder = EEG_net(input_shape=[config['TR']["context_length"], eeg_feat.shape[1]], latent_dim=config["latent_dim"])

        autoencoder = autoencoder.to(device)
        latent_seq_len = autoencoder.encoder_output[1]
        print(autoencoder.encoder_output)
        config['TR']["encoder_seq_len"] = latent_seq_len
        print(autoencoder)
        print("#"*20)

        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr = config["autoencoder_lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=0.001)

        # Calculate number of model parameters 
        model_parameters = filter(lambda p: p.requires_grad, autoencoder.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of model paramters: {params}")

        print("Training...")
        n_epochs = config["training_epochs_autoencoder"]
        epoch_loss = []
        epoch_loss_val = []
        epochs = []
        epochs_reg = []
        history_latent_mse = []
        history_latent_corr = []
        history_rec_mse = []
        history_rec_corr = []
        best_val_corr = 10000
        best_epoch = 0

        # Get baseline classification scores
        print("Creating classification baseline")
        X_train_lr, y_train_lr = get_baseline_data(train_loader)
        X_test_lr, y_test_lr = get_baseline_data(test_loader)

        # Flatten LR training data
        X_train_lr, y_train_lr = X_train_lr.reshape(-1, X_train_lr.shape[-1]), y_train_lr.reshape(-1, y_train_lr.shape[-1]),
        print(X_train_lr.shape)

        # Regression model
        est = LinearRegression().fit(X_train_lr, y_train_lr)
        # Evaluate regression model
        baseline_corr, baseline_mse, baseline_corr_std = evaluate_regression(X_test_lr, y_test_lr, est)

        early_stopper = EarlyStopperAE(patience=10, min_delta=0)
        start_time = datetime.now()
        for epoch in range(n_epochs):
            
            # TRAINING
            training_loss = train_model(autoencoder, train_loader, loss_function, optimizer, epoch, total_epochs=n_epochs)

            # VALIDATION
            valid_loss = validate_model(autoencoder, val_loader, loss_function, optimizer, scheduler, epoch, total_epochs=n_epochs)

            epoch_loss_val.append(round(valid_loss, 4))
            epoch_loss.append(round(training_loss, 4))
            epochs.append(epoch)

            if valid_loss < best_val_corr:
                best_val_corr = valid_loss
                best_epoch = epoch
                torch.save(autoencoder.state_dict(), f'{new_dir_path}/saved_models/autoencoder_{pt_id}.pt') 
            
            # if early_stopper.early_stop(valid_loss):
            #     print("Early stopping...")
            #     break

            # Regression VALIDATION
            if ((epoch+1) % 10 == 0) | (epoch == 0):
                
                # Encode the training and testing data
                X_train_latent_lr, y_train_latent_lr = encode_data(autoencoder, train_loader, 50, config['TR']["context_length"], 64, 80)
                X_val_latent_lr, y_val_latent_lr = encode_data(autoencoder, val_loader, 50, config['TR']["context_length"], 64, 80)

                # Reconstruct the training and testing data
                train_src_rec, train_tgt_rec = reconstruct_data(autoencoder, train_loader, eeg_feat.shape[1], config['TR']["context_length"],80) # [number of batches, bs, seq_len, number of features]
                X_val_rec_lr, y_val_rec_lr = reconstruct_data(autoencoder, val_loader, eeg_feat.shape[1], config['TR']["context_length"],80) # [number of batches, bs, seq_len, number of features]
                
                # Prepare encoded data for LR 
                X_train_latent_lr = np.repeat(X_train_latent_lr, repeats=2, axis=1) # from  [n_points x 50 x n_feat] -> [n_points x 100 x n_feat]
                X_val_latent_lr = np.repeat(X_val_latent_lr, repeats=2, axis=1)
                X_train_latent_lr, y_train_latent_lr = X_train_latent_lr.reshape(-1, X_train_latent_lr.shape[-1]), y_train_latent_lr.reshape(-1, y_train_latent_lr.shape[-1]),

                # Train LR on latent data
                est = LinearRegression().fit(X_train_latent_lr, y_train_latent_lr)
                latent_corr, latent_mse, latent_corr_std = evaluate_regression(X_val_latent_lr, y_val_latent_lr, est)
                
                # Prepare reconstructed data for LR 
                X_train_rec_lr, y_train_rec_lr = train_src_rec.reshape(-1, train_src_rec.shape[-1]), train_tgt_rec.reshape(-1, train_tgt_rec.shape[-1]),

                # Train LR on reconstructed data
                est = LinearRegression().fit(X_train_rec_lr, y_train_rec_lr)
                rec_corr, rec_mse, rec_corr_std = evaluate_regression(X_val_rec_lr, y_val_rec_lr, est)

                epochs_reg.append(epoch+1)
                history_latent_mse.append(latent_mse)
                history_latent_corr.append(latent_corr)
                history_rec_mse.append(rec_mse)
                history_rec_corr.append(rec_corr)

        print("="*20)
        print("Training done...")
        print(f"Training took: {datetime.now()-start_time}")
        print(history_rec_corr)
        print(history_latent_corr)
        print("Plotting training...")
        plot_training(epochs, epoch_loss, epoch_loss_val, title=f"Training for {pt_id}",save_fig=True, file_name=f"{new_dir_path}/{pt_id}_autoencoder_training_overview")

        print("Plotting classifier scores...")
        plot_regression_results(baseline_corr, history_latent_corr, history_rec_corr, epochs_reg, title=f"LR Speech Neuroprosthesis using different data for: {pt_id}", 
                                save_fig=True, file_name=f"{new_dir_path}/clf_performance/{pt_id}_latent_space_regression")

        # load best model
        autoencoder.load_state_dict(torch.load(f"{new_dir_path}/saved_models/autoencoder_{pt_id}.pt"))
        
        print(f"Saving patient {pt_id} data...")
        train_src_encoded, train_tgt_encoded = encode_data(autoencoder, train_loader, source_seq_len=latent_seq_len, target_seq_len=config["TR"]["context_length"], latent_dim=config["latent_dim"], target_num_feats=config["feature_extraction"]["num_feats"])
        val_src_encoded, val_tgt_encoded = encode_data(autoencoder, val_loader, source_seq_len=latent_seq_len, target_seq_len=config["TR"]["context_length"], latent_dim=config["latent_dim"], target_num_feats=config["feature_extraction"]["num_feats"])
        test_src_encoded, test_tgt_encoded = encode_data(autoencoder, test_loader, source_seq_len=latent_seq_len, target_seq_len=config["TR"]["context_length"], latent_dim=config["latent_dim"], target_num_feats=config["feature_extraction"]["num_feats"])

        np.save(f"{new_dir_path}/latent_data/train_source_encoded_{pt_id}", train_src_encoded)
        np.save(f"{new_dir_path}/latent_data/train_target_encoded_{pt_id}", train_tgt_encoded)
        np.save(f"{new_dir_path}/latent_data/val_source_encoded_{pt_id}", val_src_encoded)
        np.save(f"{new_dir_path}/latent_data/val_target_encoded_{pt_id}", val_tgt_encoded)
        np.save(f"{new_dir_path}/latent_data/test_source_encoded_{pt_id}", test_src_encoded)
        np.save(f"{new_dir_path}/latent_data/test_target_encoded_{pt_id}", test_tgt_encoded)

        src_encoded, tgt_encoded = encode_data(autoencoder, full_loader, source_seq_len=latent_seq_len, target_seq_len=config["TR"]["context_length"], latent_dim=config["latent_dim"], target_num_feats=config["feature_extraction"]["num_feats"])
        np.save(f"{new_dir_path}/latent_data/source_encoded_{pt_id}", src_encoded)
        np.save(f"{new_dir_path}/latent_data/target_encoded_{pt_id}", tgt_encoded)

        speech_labels = get_speech_labels(audio_feat)
        y_test_speech_labels = speech_labels[y_train.shape[0]:(y_train.shape[0]+y_test.shape[0])]
        speech_labels_sorted = sort_speech_labels(speech_labels, len(full_dataset), config["TR"]["context_length"], config["TR"]["context_offset"])
        y_test_speech_labels_sorted = sort_speech_labels(y_test_speech_labels, len(test_set), config["TR"]["context_length"], config["TR"]["context_offset"])

        print(f"y_test_speech_labels: {y_test_speech_labels.shape}")
        print(f"y_test_speech_labels_sorted: {y_test_speech_labels_sorted.shape}")

        spec_plot = np.empty((0,config["TR"]["context_length"],config["feature_extraction"]["num_feats"]))
        for _, tgt in test_loader:
            spec_plot = np.concatenate([spec_plot, tgt.cpu().detach().numpy()], axis=0)
        
        idx_to_plot = np.random.choice(range(y_test_speech_labels_sorted.shape[0]), 25, replace=False)
        for i in idx_to_plot:
            spec = spec_plot[i]
            labels = y_test_speech_labels_sorted[i]
            plot_spectrogram(spec, labels, file_name=f"{new_dir_path}/labeled_speech/mel_spec_labelled_{pt_id}_{i}.png")
        
        np.save(f"{new_dir_path}/latent_data/sorted_test_speech_labels_{pt_id}", y_test_speech_labels_sorted)
        np.save(f"{new_dir_path}/latent_data/sorted_speech_labels_{pt_id}", speech_labels_sorted)

        # Evaluate the model   
        autoencoder.eval()
        for _, (X_test, _) in enumerate(test_loader): 
            X_test = X_test[:,None,:,:]
            X_test = X_test.to(DEVICE)
            with torch.no_grad():
                y_pred = autoencoder(X_test)
                test_loss = loss_function(y_pred, X_test)
            test_loss_ppt[pt_id].append(test_loss.cpu().detach().numpy())
        test_loss_ppt[pt_id] = np.average(test_loss_ppt[pt_id])

        # Plotting reconstructions
        test, _ = next(iter(test_loader))
        test = test[0:1,None,:,:].to(device)
        
        print("Reconstructing signal...")
        autoencoder.eval()
        with torch.no_grad():
            test_rec = autoencoder(test)
        
        print("Plotting reconstructions...")
        plot_reconstruction_autoencoder(test_rec.cpu().detach().numpy(), test.cpu().detach().numpy(), ch_names, limit_display=False, fig_title="sEEG signal reconstruction", save_fig=True, file_name=f"{new_dir_path}/reconstructions_autoencoder/{pt_id}_reconstruction")
        plot_reconstruction_autoencoder(test_rec.cpu().detach().numpy(), test.cpu().detach().numpy(), ch_names, limit_display=True, fig_title=f"{pt_id} sEEG signal reconstruction", save_fig=True, file_name=f"{new_dir_path}/reconstructions_autoencoder/{pt_id}_reconstruction_short")

        # Get LR results 
        X_train_latent_lr = np.repeat(train_src_encoded, repeats=2, axis=1)
        X_test_latent_lr = np.repeat(test_src_encoded, repeats=2, axis=1)
        X_train_latent_lr, y_train_latent_lr = X_train_latent_lr.reshape(-1, X_train_latent_lr.shape[-1]), train_tgt_encoded.reshape(-1, train_tgt_encoded.shape[-1]),

        est = LinearRegression().fit(X_train_latent_lr, y_train_latent_lr)
        latent_corr, latent_mse, latent_corr_std = evaluate_regression(X_test_latent_lr, test_tgt_encoded, est)
  
    # Extracting keys and values from the dictionary
    patients = list(test_loss_ppt.keys())
    test_mean_arr = list(test_loss_ppt.values())

    print(patients)
    print(test_mean_arr)

    # Creating the bar plot
    plt.figure(figsize=(10,6))
    plt.bar(patients, test_mean_arr)
    plt.grid(axis="y")
    plt.ylim(0, 1)

    # Adding labels and title
    plt.xlabel('Patients')
    plt.ylabel('Test MSE')
    plt.title('Autoencoder reconstruction results')

    # Displaying the plot
    plt.savefig(f"{new_dir_path}/autoencoder_reconstruction_results.png")
    plt.close()


    print(f"Ending script at: {datetime.now()}")