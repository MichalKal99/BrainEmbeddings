import torch

from preprocessing.stft import STFT
from preprocessing.audio_utils import dynamic_range_compression, dynamic_range_decompression
from torch.utils.data import Dataset

from librosa.filters import mel as librosa_mel_fn

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
                self.targets[start_idx:start_idx + self.sequence_length])
    

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