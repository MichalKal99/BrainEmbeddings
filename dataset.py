import torch
from config import modelArgs, dataArgs
from stft import STFT
import numpy as np
from audio_utils import windowEEG, dynamic_range_compression, dynamic_range_decompression
from librosa.core import resample
from librosa.filters import mel as librosa_mel_fn
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader, random_split

class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,  
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0): 
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length) # hop and window length are in samples.
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)    ### filter_length = number of FFT components

        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)
        
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)

        if (modelArgs.dense_model):
            return mel_output[:,:,3].unsqueeze(-1)
            #stft_fn.transform pads sequence with reflection to be twice the original size.
            #hence 5 MFCC framea are produced for the 50ms window. We take the middle one which should correspond best to the original frame.
        else:
            return mel_output
        
class EEGAudioDataset(torch.utils.data.Dataset):
    def __init__(self, eeg, audio, num_audio_classes,audio_eeg_sample_ratio, hop=None):
        super().__init__()

        self.audio_eeg_sample_ratio=audio_eeg_sample_ratio
        self.sampling_rate_audio=dataArgs.sampling_rate_eeg*self.audio_eeg_sample_ratio

        self.eeg = eeg
        self.audio = audio

        self.num_audio_classes = num_audio_classes #only meaningful if direct audio is synthesized

        window_size_eeg=dataArgs.window_size / 1000 * dataArgs.sampling_rate_eeg
        self.window_size_eeg = round(window_size_eeg)
        self.offset_eeg=round(window_size_eeg*dataArgs.offset_window)
        self.window_size_audio=round(window_size_eeg * self.audio_eeg_sample_ratio)

        self.hop =  self.window_size_eeg if hop is None else int(hop/1000*dataArgs.sampling_rate_eeg)
        self.tacotron_mel_transformer=TacotronSTFT() #all default values are used, i.e. ~ 50ms window size, 12.5 ms hop, 80 mel bins, 8000hz max frequency.

    def __len__(self):
        num_samples = len(self.eeg) - 2 * self.offset_eeg - self.window_size_eeg
        return num_samples // self.hop # integer division

    def __getitem__(self, idx):
        idx_eeg = idx*self.hop+self.offset_eeg
        idx_audio = round(idx_eeg*self.audio_eeg_sample_ratio) #(includes offset in audio)
        eeg = self.eeg[idx_eeg-self.offset_eeg:idx_eeg+self.window_size_eeg+self.offset_eeg]
        audio = self.audio[idx_audio:idx_audio+self.window_size_audio]
        return eeg, audio
    

def get_data(patient_id, session_id, split='train', hop=None):
    eeg_ts = np.load(f"data/{patient_id}_{session_id}_sentences_sEEG.npy")
    audio_ts = np.load(f"data/{patient_id}_{session_id}_sentences_audio.npy")

    winL = 0.05
    frameshift = 0.01
    modelOrder = 4
    stepSize = 5

    audio_sr = 48000
    target_sr = 22050
    eeg_sr = 1024

    audio = resample(audio_ts, orig_sr=audio_sr, target_sr=target_sr)

    #Extract HG features
    eeg = windowEEG(eeg_ts, eeg_sr, windowLength=winL,frameshift=frameshift, window=True)

    audio_eeg_sample_ratio = len(audio) / len(eeg)

    num_train_samples = round(len(eeg) * dataArgs.train_test_split)
    num_train_samples_audio = round(len(audio) * dataArgs.train_test_split)

    num_test_samples = len(eeg)-num_train_samples
    num_test_samples_audio = len(audio)-num_train_samples_audio

    test_set_beginning=False
    if test_set_beginning:
        if split == 'train':
            eeg = eeg[num_test_samples:]
            audio = audio[num_test_samples_audio:]
        elif split == 'test':
            eeg = eeg[:num_test_samples]
            audio = audio[:num_test_samples_audio]
    else:
        if split == 'train':
            eeg = eeg[:num_train_samples]
            audio = audio[:num_train_samples_audio]

        elif split == 'test':
            eeg = eeg[num_train_samples:]
            audio = audio[num_train_samples_audio:]

    eeg_tensor = torch.from_numpy(eeg).float()
    audio = torch.from_numpy(audio).float()

    return EEGAudioDataset(eeg_tensor, audio, dataArgs.num_audio_classes, audio_eeg_sample_ratio, hop=hop)



def mu_law(x): #for audio conversion from regression to classification
    return np.sign(x)*np.log(1+255*np.abs(x))/np.log(1+255)

def mu_law_inverse(x):
    return np.sign(x)*(1./255)*(np.power(1.+255, np.abs(x)) - 1.)

def create_MFCC_plot(MFCCs, targets):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True)
    ax[0].imshow(targets.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    
    ax[1].imshow(MFCCs.detach().cpu().numpy(), cmap='viridis',aspect='auto') 
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return torch.from_numpy(data.transpose(1,0,2)).float() / 255

def audio_classes_to_signal_th(audio):
    audio = audio.detach().cpu().numpy()
    audio = audio.astype(np.float) / (dataArgs.num_audio_classes - 1)
    audio = audio * 2. - 1.
    audio = audio * 128.
    audio = mu_law_inverse(audio / 128.)
    audio = torch.from_numpy(audio)
    return audio

def create_audio_plot(audios_with_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for audio, label in audios_with_labels:
        ax.plot(audio.detach().cpu().numpy(), label=label)
    ax.legend(loc='best')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return torch.from_numpy(data).float() / 255

class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, device:torch.device, sequence_length: int = 5, offset: int = 1) -> None:
        self.data = data.to(device).float()
        self.targets = targets.to(device).float()
        self.sequence_length = sequence_length
        self.offset = offset

    def __len__(self):
        return (len(self.data) - self.sequence_length) // self.offset

    def __getitem__(self, idx):
        # Calculate the starting index of each chunk
        start_idx = idx * self.offset
        return (self.data[start_idx:start_idx + self.sequence_length],
                self.targets[start_idx:start_idx + self.sequence_length])
    

class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, device:torch.device, sequence_length: int = 5, offset: int = 1) -> None:
        self.data = data.to(device).float()
        self.targets = targets.to(device)
        self.sequence_length = sequence_length
        self.offset = offset

    def __len__(self):
        return (len(self.data) - self.sequence_length) // self.offset

    def __getitem__(self, idx):
        # Calculate the starting index of each chunk
        start_idx = idx * self.offset
        return (self.data[start_idx:start_idx + self.sequence_length],
                self.targets[start_idx:start_idx + self.sequence_length]
                )
    

class BatchDataset(Dataset):
    def __init__(self, source: torch.Tensor, targets: torch.Tensor, device:torch.device) -> None:
        self.source = source.to(device).float()
        self.targets = targets.to(device).float()

    def __len__(self):
        # Return the total number of samples
        return len(self.source)

    def __getitem__(self, idx):
        
        # Calculate batch index and within-batch index
        return (self.source[idx], self.targets[idx])