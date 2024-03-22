import numpy as np
import math

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
    

def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01):
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
    mfb = MelFilterBank(spectrogram.shape[1], 23, sr)
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
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        label = np.unique(labels[start:stop])
        newLabels[w]=label[0].encode("ascii", errors="ignore").decode()
    return newLabels

def windowEEG(data, sr, windowLength=0.05, frameshift=0.01):

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

    #Number of windows
    numWindows = int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = data[start:stop,:]
    return feat

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