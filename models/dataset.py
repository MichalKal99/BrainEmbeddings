# Import required libraries
import torch
from torch.utils.data import Dataset


# Define a custom dataset for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, device: torch.device, sequence_length: int = 5, offset: int = 1) -> None:
        """
        Initializes the TimeSeriesDataset.

        Args:
            data (torch.Tensor): Input data tensor.
            targets (torch.Tensor): Target data tensor.
            device (torch.device): Device to store the data on (e.g., 'cpu' or 'cuda').
            sequence_length (int): Length of each sequence chunk. Default is 5.
            offset (int): Offset between sequences. Default is 1.
        """
        # Move data and targets to the specified device and set the data type
        self.data = data.to(device).float()
        self.targets = targets.to(device)
        self.sequence_length = sequence_length
        self.offset = offset

    def __len__(self):
        return (len(self.data) - self.sequence_length) // self.offset

    def __getitem__(self, idx):
        # Calculate the starting index of the sequence chunk
        start_idx = idx * self.offset
        # Return the data and target sequence chunk
        return (self.data[start_idx:start_idx + self.sequence_length],
                self.targets[start_idx:start_idx + self.sequence_length])
    

# Define a custom dataset for general batch data
class BatchDataset(Dataset):
    def __init__(self, source: torch.Tensor, targets: torch.Tensor, device: torch.device) -> None:
        """
        Initializes the BatchDataset.

        Args:
            source (torch.Tensor): Source data tensor.
            targets (torch.Tensor): Target data tensor.
            device (torch.device): Device to store the data on (e.g., 'cpu' or 'cuda').
        """
        # Move source and targets to the specified device and set the data type
        self.source = source.to(device).float()
        self.targets = targets.to(device).float()

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        # Retrieve and return the source data and target at the given index
        return (self.source[idx], self.targets[idx])
