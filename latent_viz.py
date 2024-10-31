import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import zscore

from matplotlib.colors import LinearSegmentedColormap

from datetime import datetime
from models.model_utils import get_signal_indicies

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import os 

simplefilter("ignore", category=ConvergenceWarning)



def main(latent_dim, new_dir_path, config, pt_arr, latent_data_filename, speech_labels_filename):
    print(f"Starting visualization at: {datetime.now()}")
    
    latent_cmb_avg = np.zeros((0,latent_dim), dtype=np.float32)
    latent_cmb_traj = np.zeros((0,latent_dim), dtype=np.float32)

    patient_datasets = {pt_id:{} for pt_id in pt_arr}
    pt_start_end_indicies = {pt_id:{} for pt_id in pt_arr}
    colors = []
    speech_ratios = []
    
    # extract and combine the latent spaces
    for pt_id in pt_arr:
        source_encoded = np.load(f"{new_dir_path}/latent_data/{latent_data_filename}_{pt_id}.npy")
        speech_labels = np.load(f"{new_dir_path}/latent_data/{speech_labels_filename}_{pt_id}.npy")

        # get speech and silence classes for calculating speech percentage within test examples
        speech_indicies, silence_indicies, speech_ratio_arr = get_signal_indicies(speech_labels, new_dir_path, pt_id)
        speech_ratios.extend(speech_ratio_arr)

        # 
        source_encoded_avg = np.mean(source_encoded, axis=1)
        source_encoded_avg_norm = zscore(source_encoded_avg, axis=0)

        source_encoded_flat = source_encoded.reshape(source_encoded.shape[0]*source_encoded.shape[1],source_encoded.shape[-1])
        source_encoded_flat_norm = zscore(source_encoded_flat, axis=0)

        # prepare data for scatter points tsne 
        latent_cmb_avg = np.concatenate((latent_cmb_avg, source_encoded_avg_norm), axis=0)
        latent_cmb_traj = np.concatenate((latent_cmb_traj, source_encoded_flat_norm), axis=0)

        patient_datasets[pt_id]["speech_labels"] = speech_labels
        patient_datasets[pt_id]["speech_indicies"] = speech_indicies
        patient_datasets[pt_id]["silence_indicies"] = silence_indicies

        pt_start_end_indicies[pt_id]["start"] = len(colors)
        pt_start_end_indicies[pt_id]["end"] = len(colors) + source_encoded.shape[0]
        colors.extend([config["pt_colors"][pt_id]] * source_encoded.shape[0])

    # red_methods = {"tsne":TSNE, "umap":UMAP, "pca":PCA}
    red_methods = {"tsne":TSNE}

    
    for red_met, fn_red in red_methods.items():
        
        if not os.path.exists(f"{new_dir_path}/{red_met}"):
            os.makedirs(f"{new_dir_path}/{red_met}")

        print("Reducing averages...")
        reducer_avg = fn_red(n_components=2, random_state=0)
        latent_red_cmb_avg = reducer_avg.fit_transform(latent_cmb_avg)
        np.save(f"{new_dir_path}/{red_met}/{red_met}_avg.npy", latent_red_cmb_avg)

        print("Reducing trajectories...")
        reducer_traj = fn_red(n_components=2, random_state=0)
        cmb_red_traj_flat = reducer_traj.fit_transform(latent_cmb_traj)
        np.save(f"{new_dir_path}/{red_met}/{red_met}_flat.npy", cmb_red_traj_flat)


        latent_cmb_tsne_avg = np.load("runs_latent_clustering/2024-08-29_18-05-58/tsne/tsne_avg.npy")
        cmb_tsne_traj_flat = np.load("runs_latent_clustering/2024-08-29_18-05-58/tsne/tsne_flat.npy")

        cmb_red_traj = cmb_red_traj_flat.reshape(latent_cmb_avg.shape[0], config["TR"]["encoder_seq_len"], 2)

        
        print("Plotting...")
        if red_met == "pca":
            avg_title = f"{red_met} visualization of Latent Space for Multiple Patients\nExplained Variance: {reducer_avg.explained_variance_ratio_}"
            traj_title = f"{red_met} Trajectory lines for multiple patients\nExplained Variance: {reducer_traj.explained_variance_ratio_}"
            ani_title = f"Distribution of points in the {red_met} space for each timepoint\nExplained Variance: {reducer_traj.explained_variance_ratio_}"
        else:
            avg_title = f"{red_met} visualization of Latent Space for Multiple Patients"
            traj_title = f"{red_met} Trajectory lines for multiple patients"
            ani_title = f"Distribution of points in the {red_met} space for each timepoint"

        plot_scatter_2d(latent_red_cmb_avg, speech_ratios, colors, pt_arr, config, new_dir_path, filename=f"{red_met}_avg", red_method=red_met, figure_title=avg_title)
        plot_traj_2d(cmb_red_traj, pt_start_end_indicies, patient_datasets, colors, pt_arr, config, new_dir_path,  filename=f"trajectory_{red_met}_avg", red_method=red_met, figure_title=traj_title)
        plot_animation(cmb_red_traj, pt_start_end_indicies, pt_arr, config, new_dir_path, red_method=red_met, figure_title=ani_title)


    print(f"Ending at: {datetime.now()}")

# Function to plot a 3D graph with specified elevation and azimuth
def plot_3d(data, data_avg, color_arr, color, ax, angle_elev, angle_azim, angle_roll=0,del_y_label=False,del_x_label=False,del_z_label=False, all_lines=False,sound_only=False, red_method="tsne"):
    z = np.arange(data.shape[1])  # Time dimension (0 to 50)
    
    if all_lines:
        for i in range(data.shape[0]):
            ax.scatter(data[i,:,0], data[i,:,1], z, color=color_arr[i], alpha=0.05, s=0.05, zorder=0)

    if sound_only:
        ax.plot(data_avg[0][:,0], data_avg[0][:,1], z, color=color, marker="o", markeredgecolor="black", markeredgewidth=0.5, label='Average Sound Line', linestyle="-", zorder=data.shape[0])
    else:
        ax.plot(data_avg[0][:,0], data_avg[0][:,1], z, color=color, marker="o", markeredgecolor="black", markeredgewidth=0.5, label='Average Sound Line', linestyle="-", zorder=data.shape[0])
        ax.plot(data_avg[1][:,0], data_avg[1][:,1], z, color=color, marker="X", markeredgecolor="black", markeredgewidth=0.5, label='Average Speech Line', linestyle="--", zorder=data.shape[0])

    # Set the viewing angle
    ax.view_init(elev=angle_elev, azim=angle_azim, roll=angle_roll)

    # Set labels
    ax.set_xlabel(f'{red_method} factor 1')
    ax.set_ylabel(f'{red_method} factor 2')
    ax.set_zlabel('Time')

    if del_y_label:
        ax.set_yticks([])
        ax.set_ylabel('')
    if del_x_label:
        ax.set_xticks([])
        ax.set_xlabel('')
    if del_z_label:
        ax.set_zticks([])
        ax.set_zlabel('')

# Function to create a colormap based on a single color
def create_colormap(color, name):
    # Define color transitions: start with white, transition to the color
    return LinearSegmentedColormap.from_list(name, ["white", color], N=256)

def plot_scatter_2d(data_2d, speech_ratios, colors, pt_arr, config, new_dir_path, filename="scatter_plot", red_method="tsne", figure_title="Visualization of Latent Space for Multiple Patients"):
        # Plot scatter tsne
        fig, ax = plt.subplots(1,1,figsize=(8, 6))

        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], s=(2**5), c=speech_ratios, cmap='flare', edgecolors=colors, linewidths=0.5)
        ax.set_title(f"{figure_title}")

        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Speech %')

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=config["pt_colors"][pt_id], markersize=10) for pt_id in pt_arr]
        labels = [f"{pt_id}" for pt_id in pt_arr]
        ax.legend(handles, labels, title="Patients")

        ax.set_xlabel(f"{red_method} factor 1")
        ax.set_ylabel(f"{red_method} factor 2")

        plt.close()
        fig.savefig(f"{new_dir_path}/{red_method}/{filename}")

def plot_traj_2d(traj_data_2d, pt_start_end_indicies, patient_datasets, colors, pt_arr, config, new_dir_path, filename="umap_avg", red_method="umap", figure_title="Trajectory lines for multiple patients"):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10), subplot_kw={'projection': '3d'}, tight_layout=True)
    fig.suptitle(figure_title, fontsize=15)

    for pt_id in pt_arr:
        start_idx = pt_start_end_indicies[pt_id]["start"]
        end_idx = pt_start_end_indicies[pt_id]["end"]

        pt_traj_tsne_data = traj_data_2d[start_idx:end_idx,:,:]
        pt_traj_color_arr = colors[start_idx:end_idx]

        silence_traj_pt, speech_traj_pt = pt_traj_tsne_data[patient_datasets[pt_id]["speech_indicies"]], pt_traj_tsne_data[patient_datasets[pt_id]["silence_indicies"]]
        traj_data_avg = [np.mean(speech_traj_pt, axis=0), np.mean(silence_traj_pt, axis=0)]

        patient_base_color = config["pt_colors"][pt_id]

        # Plot each view in a different subplot
        plot_3d(pt_traj_tsne_data, traj_data_avg, pt_traj_color_arr, patient_base_color, axs[0, 0], angle_elev=20, angle_azim=-60, all_lines=False, red_method=red_method)  # Default view
        axs[0, 0].set_title("Default View", fontsize=13)

        plot_3d(pt_traj_tsne_data, traj_data_avg, pt_traj_color_arr, patient_base_color, axs[0, 1], angle_elev=90, angle_azim=-90, del_z_label=True, all_lines=False, red_method=red_method)  # Top-down view
        axs[0, 1].set_title("Top-down View", fontsize=13)

        plot_3d(pt_traj_tsne_data, traj_data_avg, pt_traj_color_arr, patient_base_color,axs[1, 0], angle_elev=0, angle_azim=-90, angle_roll=-90, del_y_label=True, all_lines=False, red_method=red_method)     # Side view 1
        axs[1, 0].set_title("View along time 1", fontsize=13)

        plot_3d(pt_traj_tsne_data, traj_data_avg, pt_traj_color_arr, patient_base_color, axs[1, 1], angle_elev=0, angle_azim=0, angle_roll=-90,del_x_label=True, all_lines=False, red_method=red_method)    # Side view 2
        axs[1, 1].set_title("View along time 2", fontsize=13)

    # Create a single legend for the entire figure
    handles = []
    labels = []
    for pt_id in pt_arr:
        handle = plt.Line2D([0], [0], color=config["pt_colors"][pt_id], lw=2)
        handles.append(handle)
        labels.append(pt_id)

    handle = plt.Line2D([0], [0], marker='x', label='Averaged Silence Signal', color='gray')
    handles.append(handle)
    labels.append("Silence Signals")

    handle = plt.Line2D([0], [0], marker='o', label='Averaged Speech Signal', color='gray')
    handles.append(handle)
    labels.append("Speech Signals")

    # Add legend to the figure
    fig.legend(handles, labels, loc="center", title="Patients")

    plt.close()
    fig.savefig(f"{new_dir_path}/{red_method}/{filename}", bbox_inches='tight')


def plot_animation(traj_data_2d, pt_start_end_indicies, pt_arr, config, new_dir_path, red_method="tsne", figure_title="Distribution of points in the tsne space for each timepoint"):
    if not os.path.exists(f"{new_dir_path}/{red_method}/frames"):
        os.makedirs(f"{new_dir_path}/{red_method}/frames")

    # Set limits based on data
    x_min = np.min(traj_data_2d[:,:,0])
    x_max = np.max(traj_data_2d[:,:,0])
    y_min = np.min(traj_data_2d[:,:,1])
    y_max = np.max(traj_data_2d[:,:,1])

    # Prepare the figure and subplots
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(figure_title, fontsize=20, y=0.95)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.2, wspace=0.2)

    # Main joint plot
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.set_xlabel(f"{red_method} factor 1")
    ax_main.set_ylabel(f"{red_method} factor 2")

    # Marginal distributions
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Function to update the plot for each frame
    def update(frame):
        # Clear previous frame's plots
        ax_main.cla()
        ax_marg_x.cla()
        ax_marg_y.cla()

        for pt_id in pt_arr:
            start_idx = pt_start_end_indicies[pt_id]["start"]
            end_idx = pt_start_end_indicies[pt_id]["end"]

            pt_data = traj_data_2d[start_idx:end_idx,frame,:]
            pt_color = config["pt_colors"][pt_id]

            # Create a dictionary of colormaps based on each unique color
            pt_cmap = create_colormap(pt_color, f"cmap_{pt_id}")

            sns.kdeplot(
                x=pt_data[:,0], 
                y=pt_data[:,1], 
                cmap=pt_cmap, 
                ax=ax_main, 
                levels=10,  
                thresh=0.05, 
                alpha=0.8, 
                linewidths=1.75, 
                label=f"{pt_id}"
            )

            sns.scatterplot(
                x=pt_data[:,0], 
                y=pt_data[:,1],
                ax=ax_main,
                label=f"{pt_id}",
                color=pt_color,
                s=8
            )

            sns.kdeplot(x=pt_data[:,0], ax=ax_marg_x, color=pt_color, lw=1.5, alpha=0.6)
            sns.kdeplot(y=pt_data[:,1], ax=ax_marg_y, color=pt_color, lw=1.5, alpha=0.6)

        # Set the axis limits for consistent scaling
        ax_main.set_xlim(x_min, x_max)
        ax_main.set_ylim(y_min, y_max)
        ax_main.set_title(f'Timestep: {frame + 1}', fontsize=16)

        # Create a single legend for the entire figure
        handles = []
        labels = []
        for pt_id in pt_arr:
            handle = plt.Line2D([0], [0], color=config["pt_colors"][pt_id], lw=2)
            handles.append(handle)
            labels.append(pt_id)

        ax_main.legend(handles,labels, title="Patients")
        fig.canvas.draw() 
        fig.savefig(f'{new_dir_path}/{red_method}/frames/frame_{frame}.png')

    n_frames = traj_data_2d.shape[1]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=200, repeat=False)
    ani.save(f'{new_dir_path}/{red_method}/{red_method}_animation.gif', writer='pillow', fps=10)