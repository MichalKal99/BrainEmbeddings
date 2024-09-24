import numpy as np
import math


class MelFilterBank:
    def __init__(self, specSize, numCoefficients, sampleRate):
        numBands = int(numCoefficients)

        # Set up center frequencies
        minMel = 0
        maxMel = self.freqToMel(sampleRate / 2.0)
        melStep = (maxMel - minMel) / (numBands + 1)

        melFilterEdges = np.arange(0, numBands + 2) * melStep

        # Convert center frequencies to indices in spectrum
        centerIndices = list(
            map(lambda x: self.freqToBin(math.floor(self.melToFreq(x)), sampleRate, specSize), melFilterEdges))

        # Prepare matrix
        filterMatrix = np.zeros((numBands, specSize))

        # Construct matrix with triangular filters
        for i in range(numBands):
            start, center, end = centerIndices[i:i + 3]
            k1 = np.float(center - start)
            k2 = np.float(end - center)
            up = (np.array(range(start, center)) - start) / k1
            down = (end - np.array(range(center, end))) / k2

            filterMatrix[i][start:center] = up
            filterMatrix[i][center:end] = down

        # Save matrix and its best-effort inverse
        self.melMatrix = filterMatrix.transpose()
        self.melMatrix = self.makeNormal(
            self.melMatrix / self.normSum(self.melMatrix))

        self.melInvMatrix = self.melMatrix.transpose()
        self.melInvMatrix = self.makeNormal(
            self.melInvMatrix / self.normSum(self.melInvMatrix))

    def normSum(self, x):
        retSum = np.sum(x, axis=0)
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
        return (np.dot(spectrogram, self.melMatrix))

    def fromMelScale(self, melSpectrogram):
        return (np.dot(melSpectrogram, self.melInvMatrix))

    def makeNormal(self, x):
        nanIdx = np.isnan(x)
        x[nanIdx] = 0

        infIdx = np.isinf(x)
        x[infIdx] = 0

        return (x)

    def toMels(self, spectrogram):
        return (self.toMelScale(spectrogram))

    def fromMels(self, melSpectrogram):
        return (self.fromMelScale(melSpectrogram))

    def toLogMels(self, spectrogram):
        return (self.makeNormal(np.log(self.fuzz(self.toMelScale(spectrogram)))))

    def fromLogMels(self, melSpectrogram):
        return (self.makeNormal(self.fromMelScale(np.exp(melSpectrogram))))



def extractMelSpecs(audio: np.ndarray,
                    sr: int,
                    windowLength: float = 0.05,
                    frameshift: float = 0.01,
                    numFilter: int = 23) -> np.ndarray:
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms

    Parameters
    ----------
    audio: np.ndarray
        Audio time series as a NumPy array
    sr: int
        Sampling rate of the audio
    windowLength: float, optional
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float, optional
        Shift (in seconds) after which the next window will be extracted
    numFilter: int, optional
        Number of triangular filters in the mel filterbank
    
    Returns
    ----------
    spectrogram: np.ndarray, shape (numWindows, numFilter)
        Logarithmic mel-scaled spectrogram as a NumPy array
    """
    numWindows = int(np.floor((audio.shape[0] - windowLength * sr) / (frameshift * sr)))
    win = np.hanning(int(np.floor(windowLength * sr + 1)))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength * sr / 2 + 1))), dtype='complex')
    for w in range(numWindows):
        startAudio = int(np.floor((w * frameshift) * sr))
        stopAudio = int(np.floor(startAudio + windowLength * sr))
        a = audio[startAudio:stopAudio]
        spec = np.fft.rfft(win * a)
        spectrogram[w, :] = spec
    mfb = MelFilterBank(spectrogram.shape[1], numFilter, sr)  # You may need to import MelFilterBank or define it.
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

