from scipy.stats import spearmanr, pearsonr
import numpy as np


def compensate_eeg_audio(eeg, eeg_sr, audio, audio_sr, markers_time_stamps):
    # the sEEg and audio data might slightly differ in length with the markers. We should compensate by reducing to the smallest:
    print(
        "The ORIGINAL length audio {}, sEEG {}, markers {}".format(
            len(eeg) / eeg_sr, len(audio) / audio_sr, markers_time_stamps[-1]
        )
    )
    minimum = min(len(eeg) / eeg_sr, len(audio) / audio_sr, markers_time_stamps[-1])
    eeg = eeg[: int(minimum * eeg_sr)]
    audio = audio[: int(minimum * audio_sr)]
    print(
        "The REDUCED length audio {}, sEEG {}, markers {}".format(
            len(eeg) / eeg_sr, len(audio) / audio_sr, markers_time_stamps[-1]
        )
    )
    return eeg, audio

# TODO this function should be removed and substituted with all_electrodes() and the right parameters
def sel_best_elect(x, y, ch_index):
    """
    Feature selection using correlation and all the electrodes
    """

    if str(type(y)) == "<class 'torch.Tensor'>":
        y = np.array(y)

    y_mean = np.mean(y, axis=0)
    cs = np.zeros(x.shape[1])
    for f in range(x.shape[1]):
        # If the whole columns is zero then it should be ignored
        if np.isclose(np.sum(x[:, f]), 0):
            cs[f] = 0
            continue
        # get the spearman correlation coefficient
        cs[f], p = pearsonr(x[:, f], y_mean)
    # Indices that would sort the array of the absolute correlation values
    ord_indexes_abs_corr = np.argsort(np.abs(cs))
    # get the highest X values from the above array
    # being X, the maximum between -n_feats and -len(cs).
    select = ord_indexes_abs_corr[np.max([-1, -len(cs)]) :]

    # get the number of channels
    n_channels = len(ch_index)
    # get the name of the channel with the highest correlation
    index_of_channel_with_highest_corr = select[-1] % n_channels
    # get the shaft id of that channel
    channel_with_highest_corr = ch_index[index_of_channel_with_highest_corr]

    return select, cs[select[-1]], channel_with_highest_corr




def all_electrodes(x_train, y_train, **args):
    """
    Feature selection using correlation and all the electrodes
    """

    ch_names = args.get("ch_names")
    n_feats = args.get("n_feats")
    metric = args.get("metric")

    y_mean = np.mean(y_train, axis=1)
    print('y_mean.shape', y_mean.shape)
    cs = np.zeros(x_train.shape[1])
    for f in range(x_train.shape[1]):
        # If the whole columns is zero then it should be ignored
        if np.isclose(np.sum(x_train[:, f]), 0):
            cs[f] = 0
            continue
        # get the spearman correlation coefficient
        if metric == "spearmanr":
            cs[f], p = spearmanr(x_train[:, f], y_mean)
        elif metric == "pearsonr":
            cs[f], p = pearsonr(x_train[:, f], y_mean)
        else:
            return "Metric not recognized"
    # Indices that would sort the array of the absolute correlation values
    ord_indexes_abs_corr = np.argsort(np.abs(cs))
    # get the highest X values from the above array
    # being X, the maximum between -n_feats and -len(cs).
    select = ord_indexes_abs_corr[np.max([-n_feats, -len(cs)]) :]

    # np.nanmax(corrs_p),featName[np.nanargmax(corrs_p)]

    # get the number of channels
    n_channels = len(ch_names)
    print('n_channels', n_channels)
    # get the name of the channel with the highest correlation
    index_of_channel_with_highest_corr = select[-1] % n_channels
    print('index_of_channel_with_highest_corr', index_of_channel_with_highest_corr)
    # get the shaft id of that channel
    channel_with_highest_corr = ch_names[index_of_channel_with_highest_corr]

    return select, cs[select[-1]], channel_with_highest_corr


def get_shaft_indices(channels):
    """Get name of each shaft and its indices from a list of channel names

    Parameters
    ----------
    channels: array (electrodes, label)
        Channel names

    Returns
    ----------
    shafts: dict (shaft name, indices)
        Shaft names with corresponding indices
    """

    # get shaft information
    shafts = {}
    for i, chan in enumerate(channels):
        if chan.rstrip("0123456789") not in shafts:
            shafts[chan.rstrip("0123456789")] = [i]
        else:
            shafts[chan.rstrip("0123456789")].append(i)
    return shafts



def one_shaft(x_train, y_train, **args):
    """
    Feature selection using correlation and just the shaft with the electrode with the highest correlation
    """

    ch_names = args.get("ch_names")
    stack_step = args.get("stack_step")
    args["n_feats"] = 1
    # get the number of channels
    n_channels = len(ch_names)

    selected, highest_corr_value, id_of_channel_with_highest_corr = all_electrodes(
        x_train, y_train, **args
    )

    # Getting the entire shaft where this electrode is placed

    # get the ids of all the other electrodes in that shaft
    all_shafts_dict = get_shaft_indices(ch_names)

    indexes_of_electrodes_in_that_channel = all_shafts_dict.get(
        id_of_channel_with_highest_corr.rstrip("0123456789")
    )
    # get the indexes in the features array (n_channels * 5)
    selected_indexes = [
        [index + (n_channels * i) for index in indexes_of_electrodes_in_that_channel]
        for i in range(stack_step)
    ]
    # flatten the list
    selected_indexes = [
        index for index_list in selected_indexes for index in index_list
    ]

    # return the indexes
    return selected_indexes, highest_corr_value, id_of_channel_with_highest_corr


def align_features(eeg_features, audio_features, x_slice=slice(None,-10,None), y_slice=slice(20,None,None), print_stuff=True):
    if print_stuff: print('aligning')
    # align with the known slices
    if print_stuff: print('     initially eeg_feats: {}; audio_feats: {}'.format(eeg_features.shape, audio_features.shape))
    eeg_features = eeg_features[x_slice]
    audio_features = audio_features[y_slice]
    if print_stuff: print('     after slice eeg_feats: {}; audio_feats: {}'.format(eeg_features.shape, audio_features.shape))
    # just in case there's still a difference at the end, compensate it
    minimum = min(len(eeg_features), len(audio_features))
    eeg_features = eeg_features[0:minimum, :]
    audio_features = audio_features[0:minimum, :]
    if print_stuff: print('     after min eeg_feats: {}; audio_feats: {}'.format(eeg_features.shape, audio_features.shape))
    return eeg_features, audio_features
