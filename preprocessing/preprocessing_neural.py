import sys
import os
import numpy as np
from contextlib import contextmanager
import re
import mne
import scipy
import scipy.signal
from pathlib import Path
import loading_files


def _select_channels(ch_names, good_channels):
    """
    Selects all channels which match at least one specified pattern of the good channels.
    :param ch_names: List of all channel names
    :param good_channels: List of regex patterns to select the good channels
    :return: List of channel names which match the good channel pattern at least once
    """
    patterns = [re.compile(r"^{}$".format(gc)) for gc in good_channels]

    def matches(ch_name):
        for pattern in patterns:
            if pattern.match(ch_name):
                return True

        return False

    return [ch_name for ch_name in ch_names if matches(ch_name)]


def remove_bad_channels(eeg, ch_names, good_channels_regex="(?!E|el)(?![FCOTP])[A-Za-z0-9]*"): # old one (?!E)[A-Za-z0-9]*
    # some of the eeg channels contain useless information, so we will discard them
    regex_patterns = good_channels_regex.split(",")
    regex_patterns = list(map(lambda x: x.strip(), regex_patterns))
    sel_channels = _select_channels(ch_names, regex_patterns)

    # mark all non selected channels as bad channels
    bad_channels = [c for c in ch_names if c not in sel_channels]
    ch_types = ["eeg"] * len(ch_names)

    used_channel_names = [c for c in ch_names if c not in bad_channels]
    # Transform list of bad channels to their indices
    bad_channel_indices = [ch_names.index(bc) for bc in bad_channels]
    print(
        "Exclusion of the following bad channel indices: ["
        + " ".join(map(str, bad_channel_indices))
        + "]"
    )

    # exclude bad channels
    if len(bad_channel_indices) > 0:
        print("EEG original shape: {} x {}".format(*eeg.shape))
        mask = np.ones(eeg.shape[1], bool)
        mask[bad_channel_indices] = False
        eeg = eeg[:, mask]
        print("EEG truncated shape: {} x {}".format(*eeg.shape))
    else:
        print("No bad channels specified.")

    return eeg, bad_channel_indices, used_channel_names


@contextmanager
def suppress_stdout():
    """context manager for suppressing printing to stdout for a given block"""
    with open(os.devnull, "w") as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout


def herff2016_b(
    eeg, sr, window_length=0.05, window_shift=0.01, line_noise=50, skip_stacking=False
):
    """
    Offline computation of the feature extraction paradigm from:
    "Herff, C., et al. Towards direct speech synthesis from ECoG: A pilot study. EMBC 2026"
    This is the version which is compatible with the warm start from the node based system.
    :param eeg: sEEG data in the shape of samples x features
    :param sr: sampling frequency of sEEG data
    :param window_length: Window length
    :param window_shift: Frameshift
    :return: Neural features in the shape of samples x features
    """

    def create_filter(sr, l_freq, h_freq, method="fir", iir_params=None):
        with suppress_stdout():
            iir_params, method = mne.filter._check_method(method, iir_params)
            filt = mne.filter.create_filter(
                None,
                sr,
                l_freq,
                h_freq,
                "auto",
                "auto",
                "auto",
                method,
                iir_params,
                "zero",
                "hamming",
                "firwin",
            )
        return filt

    def extract_high_gamma_50Hz(data, sr, windowLength=0.05, frameshift=0.01):
        # Initialize filters and filter states
        iir_params = {"order": 8, "ftype": "butter"}
        hg_filter = create_filter(sr, 70, 170, method="iir", iir_params=iir_params)[
            "sos"
        ]
        fh_filter = create_filter(sr, 102, 98, method="iir", iir_params=iir_params)[
            "sos"
        ]
        sh_filter = create_filter(sr, 152, 148, method="iir", iir_params=iir_params)[
            "sos"
        ]

        hg_state = scipy.signal.sosfilt_zi(hg_filter)
        fh_state = scipy.signal.sosfilt_zi(fh_filter)
        sh_state = scipy.signal.sosfilt_zi(sh_filter)

        hg_state = np.repeat(hg_state, data.shape[1], axis=-1).reshape(
            [hg_state.shape[0], hg_state.shape[1], -1]
        )
        fh_state = np.repeat(fh_state, data.shape[1], axis=-1).reshape(
            [fh_state.shape[0], fh_state.shape[1], -1]
        )
        sh_state = np.repeat(sh_state, data.shape[1], axis=-1).reshape(
            [sh_state.shape[0], sh_state.shape[1], -1]
        )

        warm_start_filling = int(windowLength * sr) - int(frameshift * sr)
        zero_fill = np.zeros([warm_start_filling, data.shape[1]])

        # Initialize high gamma filter state (since online method uses a warm start)
        for i in range(data.shape[1]):
            hg_state[:, :, i] *= data[0, i]

        # Extract high gamma band
        data, hg_state = scipy.signal.sosfilt(hg_filter, data, axis=0, zi=hg_state)

        # Initialize first harmonic filter state (since online method uses a warm start)
        for i in range(data.shape[1]):
            fh_state[:, :, i] *= data[0, i]

        # Update the filter state of the second harmonic on the zero_fill data
        _, sh_state = scipy.signal.sosfilt(sh_filter, zero_fill, axis=0, zi=sh_state)

        # Attenuate first and second harmonic
        data, fh_state = scipy.signal.sosfilt(fh_filter, data, axis=0, zi=fh_state)
        data, sh_state = scipy.signal.sosfilt(sh_filter, data, axis=0, zi=sh_state)
        return data

    def extract_high_gamma_60Hz(data, sr, windowLength=0.05, frameshift=0.01):
        # Initialize filters and filter states
        iir_params = {"order": 8, "ftype": "butter"}
        hg_filter = create_filter(sr, 70, 170, method="iir", iir_params=iir_params)[
            "sos"
        ]
        fh_filter = create_filter(sr, 122, 118, method="iir", iir_params=iir_params)[
            "sos"
        ]

        hg_state = scipy.signal.sosfilt_zi(hg_filter)
        sh_state = scipy.signal.sosfilt_zi(fh_filter)

        hg_state = np.repeat(hg_state, data.shape[1], axis=-1).reshape(
            [hg_state.shape[0], hg_state.shape[1], -1]
        )
        sh_state = np.repeat(sh_state, data.shape[1], axis=-1).reshape(
            [sh_state.shape[0], sh_state.shape[1], -1]
        )

        warm_start_filling = int(windowLength * sr) - int(frameshift * sr)
        zero_fill = np.zeros([warm_start_filling, data.shape[1]])

        # Initialize high gamma filter state (since online method uses a warm start)
        for i in range(data.shape[1]):
            hg_state[:, :, i] *= data[0, i]

        # Extract high gamma band
        data, hg_state = scipy.signal.sosfilt(hg_filter, data, axis=0, zi=hg_state)

        # Update the filter state of the second harmonic on the zero_fill data
        _, sh_state = scipy.signal.sosfilt(fh_filter, zero_fill, axis=0, zi=sh_state)

        # Attenuate first and second harmonic
        data, sh_state = scipy.signal.sosfilt(fh_filter, data, axis=0, zi=sh_state)
        return data

    def compute_features(data, windowLength=0.05, frameshift=0.01):
        numWindows = (
            int(np.floor((data.shape[0] - windowLength * sr) / (frameshift * sr))) + 1
        )

        # Compute logarithmic high gamma broadband features
        eeg_features = np.zeros((numWindows, data.shape[1]))
    
        for win in range(numWindows):
            start_eeg = int(round((win * frameshift) * sr))
            stop_eeg = int(round(start_eeg + windowLength * sr))
            for c in range(data.shape[1]):
                eeg_features[win, c] = np.log(
                    np.sum(data[start_eeg:stop_eeg, c] ** 2) + 0.01
                )
        return eeg_features

    def stack_features(features, model_order=4, step_size=5):
        eeg_feat_stacked = np.zeros(
            [
                features.shape[0] - (model_order * step_size),
                (model_order + 1) * features.shape[1],
            ]
        )
        for f_num, i in enumerate(range(model_order * step_size, features.shape[0])):
            ef = features[i - model_order * step_size : i + 1 : step_size, :]
            eeg_feat_stacked[f_num, :] = ef.T.flatten()
        return eeg_feat_stacked

    # Extract HG features and add context information
    if line_noise == 50:
        eeg_feat = extract_high_gamma_50Hz(
            eeg, sr, windowLength=window_length, frameshift=window_shift
        )
    else:
        eeg_feat = extract_high_gamma_60Hz(
            eeg, sr, windowLength=window_length, frameshift=window_shift
        )

    eeg_feat = compute_features(
        eeg_feat, windowLength=window_length, frameshift=window_shift
    )

    if not skip_stacking:
        eeg_feat = stack_features(eeg_feat, model_order=4, step_size=5)
    return eeg_feat



def extract_eeg_features(eeg, eeg_sr, ch_names, window_size = 0.0464375, window_shift = 0.0115625, stacking=True):
    print('     removing bad channels')
    eeg, bad_channel_indices, used_channels_name = remove_bad_channels(eeg, ch_names if isinstance(ch_names, list) else ch_names.tolist())
    print('     extracting feats')
    eeg_features = herff2016_b(eeg, sr=eeg_sr, window_length=window_size, window_shift=window_shift, line_noise=50, skip_stacking = not stacking)
    print('     feats {}'.format(eeg_features.shape))
    return eeg, eeg_features, used_channels_name
