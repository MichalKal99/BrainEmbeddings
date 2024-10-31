import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy.signal import convolve
from scipy.fftpack import dct, idct
import scipy.io.wavfile as wav

# importing common_scripts
from pathlib import Path
from StreamingVocGan.streaming_voc_gan import StreamingVocGan
from scipy import signal
import torch




def extract_sound_segments_from_audio(audio_data, sample_rate, intervals):
    # Convert time intervals to sample indices and extract segments
    extracted_segments = []
    for start_time, end_time in intervals:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        extracted_segments.append(audio_data[start_sample:end_sample])

    # Concatenate the extracted segments into one array
    return np.concatenate(extracted_segments)



# Define a function to detect non-silence intervals
def detect_non_silence(audio, sample_rate, threshold=0.02, window_size=0.3):
    """
    Detects non-silence intervals in an audio signal.

    Parameters:
    - audio: The audio signal array.
    - threshold: The amplitude threshold to consider silence.

    Returns:
    - intervals: A list of tuples representing the start and end times of non-silence intervals.
    """
    is_silence = np.abs(audio) < threshold

    is_sound = 1 * (np.logical_not(is_silence))

    # Size of the Blackman window
    window_size = int(sample_rate*window_size)
    # Create the Blackman window
    blackman_window = np.blackman(window_size)
    # Perform convolution with 'same' mode to keep the original length
    is_sound_smoothed  = convolve(is_sound, blackman_window, mode='same') / np.sum(blackman_window)

    # Now, detect intervals of non-silence (sound) in the smoothed signal
    # Thresholding the smoothed signal to get binary (sound/silence) signal

    is_silence = np.abs(is_sound_smoothed) < 0.00001
    # it should always start with silence
    is_silence[0] = True

    # Find change points
    change_points = np.diff(1 * is_silence).nonzero()[0] + 1
    change_points = np.insert(change_points, 0, 0)
    change_points = np.append(change_points, len(audio))

    # Extract intervals
    intervals = []
    for i in range(0, len(change_points) - 1, 2):
        start = change_points[i]
        end = change_points[i + 1]
        if is_silence[start]:
            intervals.append((start / sample_rate, end / sample_rate))
        
    return intervals, is_sound_smoothed



def get_only_sound_intervals(org_audio, org_audio_sr, rec_audio, rec_audio_sr, plots=False):
    # ORIGINAL AUDIO
    # Normalize the audio data to the range [-1, 1] for easier thresholding
    org_audio = org_audio / np.max(np.abs(org_audio), axis=0)
    # Time axis for plotting
    time_axis = np.linspace(0, len(org_audio) / org_audio_sr, num=len(org_audio))

    # plot
    if plots:
        fig, ax = plt.subplots(3,2,figsize=(28,14))
        ax[0,0].plot(time_axis, org_audio)
        ax[0,0].set_ylabel('Original waveform')
        ax[0,0].set_title('ORIGINAL')


    # SMOOTHED NON-SILENCE INTERVALS
    silence_intervals, is_sound_smoothed = detect_non_silence(org_audio, org_audio_sr)

    # plot
    if plots:
        time_axis = np.linspace(0, len(is_sound_smoothed) / org_audio_sr, num=len(is_sound_smoothed))
        ax[0,1].plot(time_axis, org_audio)
        ax[0,1].set_title('RECONSTRUCTED')


    # ORIGINAL OVERLAPPED WITH INTERVALS
    non_silence_intervals = [(interval[1], silence_intervals[i+1][0]) for i, interval in enumerate(silence_intervals) if i!=len(silence_intervals)-1]

    # plot
    if plots:
        ax[1,0].plot(time_axis, org_audio)

        for start, end in non_silence_intervals:
            ax[1,0].axvline(x=start, color='red', linestyle='--')  # Start of word
            ax[1,0].axvline(x=end, color='red', linestyle='--')  # End of word
        ax[1,0].set_ylabel('Waveform with Non-Silence Intervals Highlighted')


    # RECONSTRUCTED OVERLAPPED WITH INTERVALS
    # Load the audio file
    # Normalize the audio data to the range [-1, 1] for easier thresholding
    rec_audio = rec_audio / np.max(np.abs(rec_audio), axis=0)

    # plot
    if plots:
        # Time axis for plotting
        time_axis = np.linspace(0, len(rec_audio) / rec_audio_sr, num=len(rec_audio))
        ax[1,1].plot(time_axis, rec_audio)
        for start, end in non_silence_intervals:
            ax[1,1].axvline(x=start, color='red', linestyle='--')  # Start of word
            ax[1,1].axvline(x=end, color='red', linestyle='--')  # End of word


    # SHORTED ORIGINAL
    org_audio_only_with_speech_segments = extract_sound_segments_from_audio(org_audio, org_audio_sr, non_silence_intervals)
    
    # Plot the concatenated segments
    if plots:
        ax[2,0].plot(np.linspace(0, len(org_audio_only_with_speech_segments) / org_audio_sr, num=len(org_audio_only_with_speech_segments)), org_audio_only_with_speech_segments)
        ax[2,0].set_xlabel('Time (seconds)')
        ax[2,0].set_ylabel('Shorted segments')


    # SHORTED RECONSTRUCTED
    rec_audio_only_with_speech_segments = extract_sound_segments_from_audio(rec_audio, rec_audio_sr, non_silence_intervals)
    # Plot the concatenated segments to visualize the result
    if plots:
        ax[2,1].plot(np.linspace(0, len(rec_audio_only_with_speech_segments) / rec_audio_sr, num=len(rec_audio_only_with_speech_segments)), rec_audio_only_with_speech_segments)
        ax[2,1].set_xlabel('Time (seconds)')

    return org_audio_only_with_speech_segments, rec_audio_only_with_speech_segments



def mel_spectrogram_to_mfcc(mel_spectrogram: np.ndarray, 
                           num_ceps: int = 40, 
                           log_already_applied: bool = False,
                           normalize: bool = False) -> np.ndarray:
    """
    Convert a Mel Spectrogram to MFCC.

    Parameters:
    mel_spectrogram (numpy.ndarray): The Mel Spectrogram to be converted.
    num_ceps (int): The number of cepstral coefficients to retain.
    log_already_applied (bool): Whether log transformation has already been applied.
    normalize (bool): Whether to apply mean normalization.

    Returns:
    numpy.ndarray: The MFCCs.
    """
    # Apply logarithmic scaling to the Mel Spectrogram.
    if log_already_applied:
        # NOTE VocGan features have already the log operation.
        log_mel_spectrogram = mel_spectrogram
    else:
        log_mel_spectrogram = np.log(mel_spectrogram + 1e-8)

    # Apply Discrete Cosine Transform to get MFCCs
    mfccs = dct(log_mel_spectrogram, type=2, axis=1, norm='ortho')[:, :num_ceps]

    # Normalize the MFCCs (mean normalization)
    if normalize:
        mfccs -= (np.mean(mfccs, axis=0) + 1e-8)

    return mfccs




def extract_audio_features(audio, audio_sr, method: str, num_logmels: int) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Extract audio features from a preloaded dataset.

    Parameters:
    - p_id (str): Participant identifier.
    - dataset (str): Name of the dataset.
    - method (str): Method used for audio feature extraction. Options include 'VocGan', 'MelSpec', or 'MFCC'.
    - num_logmels (int): Number of log-mel filters to use in feature extraction.

    Returns:
    tuple[np.ndarray, int, np.ndarray]: A tuple containing the processed audio array, the modified audio sampling rate,
    and the extracted audio features array.

    The function performs the following operations:
    1. Loads audio and its sampling rate from the specified dataset.
    2. If the audio file does not exist, saves it as a .wav file.
    3. Performs decimation if the audio sampling rate is not 48000 Hz.
    4. Extracts audio features based on the specified method ('VocGan', 'MelSpec', or 'MFCC').

    Raises:
    - ValueError: If the method parameter is not one of the expected values ('VocGan', 'MelSpec', 'MFCC').
    """
    if audio_sr==22050:
        scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
        audio_features = StreamingVocGan.waveform_to_mel_spectrogram(
            waveform=scaled, 
            original_sampling_rate=audio_sr, 
            mel_channel_count=num_logmels).T
        return audio_features

    # resample audio
    if audio_sr!=48000:
        # in case the sr is not the expected one, see what do I need to do, I'll leave it like this for now
        print('weird sampling rate: {}'.format(audio_sr))
        return
    
    print('\tbefore resampling: {}'.format(audio.shape))
    WAVE_SAMPLING_RATE = 22050

    audio = scipy.signal.resample(
        audio, int(audio.shape[0] / audio_sr * WAVE_SAMPLING_RATE)
    )
    audio_sr = WAVE_SAMPLING_RATE
    scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
    print('\tafter resampling: {}'.format(audio.shape))

    # extract the audio features
    print('\tlength: {}s'.format(audio.shape[0]/audio_sr))
    if method == 'VocGan':
        audio_features = StreamingVocGan.waveform_to_mel_spectrogram(
            waveform=scaled, 
            original_sampling_rate=audio_sr, 
            mel_channel_count=num_logmels).T
    else:
        ValueError(f'method not recognized: {method}')

    print('\tfeatures: {}'.format(audio_features.shape))

    return audio_features



def reconstruct_and_save_audio(output_path: str, 
                               file_name: str, 
                               data: np.ndarray,
                               feat_type: str = 'MFCC',
                               reconstruction_method: str = 'griffin_lim',
                               window_size: float = 0.0464375,
                               window_shift: float = 0.0115625,
                               sampling_rate: float = 16000) -> None:
    """
    Reconstruct audio from Mel Spectrogram and save it.

    Parameters:
    output_path (str): Path to save the audio file.
    file_name (str): Name of the audio file.
    data (numpy.ndarray): Mel Spectrogram data.
    """

    output_path_standard = output_path / str(file_name+'.wav')

    if feat_type == 'MFCC':
        data = idct(data, type=2, axis=1, norm='ortho')

    # Load model
    standard_model = StreamingVocGan(is_streaming=False,model_path=Path(".\\StreamingVocGan\\vctk_pretrained_model_3180.pt"))
    # Convert
    waveform_standard, processing_time = standard_model.mel_spectrogram_to_waveform(mel_spectrogram=torch.Tensor(data).to('cuda').T)
    # Save
    StreamingVocGan.save(waveform=waveform_standard.cpu(), file_path=output_path_standard)
    print(f'it took: {np.round(np.sum(processing_time),3)}s to reconstruct')
