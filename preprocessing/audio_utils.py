import numpy as np
import math
import scipy

import torch
from scipy.signal import get_window
import librosa.util as librosa_util


class MelFilterBank():
    def __init__(self, specSize, numCoefficients, sampleRate):
        numBands = int(numCoefficients)

        # Set up center frequencies
        minMel = 0
        maxMel = self.freqToMel(sampleRate / 2.0)
        melStep = (maxMel - minMel) / (numBands + 1)
        
        melFilterEdges = np.arange(0, numBands + 2) * melStep
        
         # Convert center frequencies to indices in spectrum
        centerIndices = list(map(lambda x: self.freqToBin(math.floor(self.melToFreq(x)), sampleRate, specSize), melFilterEdges))
        
        # Prepare matrix
        filterMatrix = np.zeros((numBands, specSize))
        
        # Construct matrix with triangular filters
        for i in range(numBands):
            start, center, end = centerIndices[i:i + 3]
            k1 = np.float64(center - start)
            k2 = np.float64(end - center)
            up = (np.array(range(start, center)) - start) / k1
            down = (end - np.array(range(center, end))) / k2

            filterMatrix[i][start:center] = up
            filterMatrix[i][center:end] = down

        # Save matrix and its best-effort inverse
        self.melMatrix = filterMatrix.transpose()
        self.melMatrix = self.makeNormal(self.melMatrix / self.normSum(self.melMatrix))
        
        self.melInvMatrix = self.melMatrix.transpose()
        self.melInvMatrix = self.makeNormal(self.melInvMatrix / self.normSum(self.melInvMatrix))
        
    def normSum(self, x):
        retSum = np.sum(x, axis = 0)
        retSum[np.where(retSum == 0)] = 1.0
        return retSum
    
    def fuzz(self, x):
        return x + 0.0000001
    
    def freqToBin(self, freq, sampleRate, specSize):
        return int(math.floor((freq / (sampleRate / 2.0)) * specSize))
        
    def freqToMel(self, freq):
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def melToFreq(self, mel):
        return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0)

    def toMelScale(self, spectrogram):
        return(np.dot(spectrogram, self.melMatrix))
    
    def fromMelScale(self, melSpectrogram):
        return(np.dot(melSpectrogram, self.melInvMatrix))
    
    
    def makeNormal(self, x):
        nanIdx = np.isnan(x)
        x[nanIdx] = 0
        
        infIdx = np.isinf(x)
        x[infIdx] = 0

        return(x)
    
    def toMels(self, spectrogram):
        return(self.toMelScale(spectrogram))
    
    def fromMels(self, melSpectrogram):
        return(self.fromMelScale(melSpectrogram))
    
    def toLogMels(self, spectrogram):
        return(self.makeNormal(np.log(self.fuzz(self.toMelScale(spectrogram)))))
    
    def fromLogMels(self, melSpectrogram):
        return(self.makeNormal(self.fromMelScale(np.exp(melSpectrogram))))
    
#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01, melBands=23):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = np.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+windowLength*sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = MelFilterBank(spectrogram.shape[1], melBands, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """
    Downsamples non-numerical data by using the mode
    
    Parameters
    ----------
    labels: array of str
        Label time series
    sr: intGratul
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows = int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        label, sent_counts = np.unique(labels[start:stop], return_counts=True)
        newLabels[w]=label[np.argmax(sent_counts)].encode("ascii", errors="ignore").decode()
    return newLabels

def windowEEG(data, sr, windowLength=0.05, frameshift=0.01, window=True):

    """
    Window data and extract frequency-band envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat: array (windows, channels)
        Frequency-band feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)

    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))

    # TODO: change to 8 from 4?
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)

    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)

    #Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data = scipy.signal.sosfiltfilt(sos,data,axis=0)

    #Create feature space
    data = np.abs(hilbert3(data))
    if window:
        return data
    
    # Length of the windows
    lenWindows = int(windowLength*sr)

    feat = np.zeros((numWindows, lenWindows, data.shape[1]))
    for win in range(numWindows):
        start = int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:,:] = data[start:stop,:]
 
    return feat

def find_class_regions(arr):
    class_regions = []
    start_index = 0
    current_class = arr[0]

    for i in range(1, len(arr)):
        if arr[i] != current_class:
            class_regions.append(([start_index, i - 1], "skyblue" if current_class==1 else "lightcoral"))
            start_index = i
            current_class = arr[i]

    # Add the last region
    class_regions.append(([start_index, len(arr) - 1], "skyblue" if current_class==1 else "lightcoral"))

    return class_regions

def improve_labels(mel_spec, threshold_value=0.45):
    spec_avg = np.mean(mel_spec, axis=1)
    threshold = (np.max(spec_avg)+np.min(spec_avg))*threshold_value
    labels = np.where(spec_avg>threshold, 1, 0)
    return labels


def get_speech_labels(melSpec, threshold_vlaue=1):
    melSpecAvg = np.mean(melSpec, axis=1)
    labels = np.where(melSpecAvg>np.mean(melSpecAvg)*threshold_vlaue, 1, 0) 
    return labels


def sort_speech_labels(speech_labels, n_win, seq_len, seq_offset):

    speech_labels_sorted = np.zeros((n_win, seq_len), dtype=int)

    for i in range(n_win):
        start_idx = i * seq_offset
        speech_labels_sorted[i] = speech_labels[start_idx:start_idx + seq_len]

    return speech_labels_sorted



def stackFeatures(features, modelOrder=4, stepSize=5):
    """
    Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() #Add 'F' if stacked the same as matlab
    return featStacked




## jonaskohler

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(data=win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C