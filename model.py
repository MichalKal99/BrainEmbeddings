import torch
import torch.nn as nn
import torch.optim as optim

class baselineAutoencoder(nn.Module):
    def __init__(self, input_size, seq_length):
        super(baselineAutoencoder, self).__init__()
        encoder_input_size = (2*input_size)-1
        self.input_size = input_size
        self.seq_length = seq_length

        self.input_layer = nn.Sequential(
            nn.Conv1d(in_channels=seq_length, out_channels=80, kernel_size=input_size, padding="valid", stride=1),
            nn.ReLU()
        )
        

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=80, out_channels=50, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=2, stride=5),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=50, out_channels=50, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=50, out_channels=80, kernel_size=2, stride=2),
            nn.ReLU()
            )

        self.output_layer = nn.Sequential(
            nn.ConvTranspose1d(in_channels=80, out_channels=seq_length, kernel_size=input_size, stride=1),
            nn.ReLU()
        ) 
        
    def forward(self, x):

        X_ext = torch.concat((x[:,:,:], x[:,:,:-1]), 2)
        
        X_in = self.input_layer(X_ext)

        X_enc = self.encoder(X_in)

        X_dec = self.decoder(X_enc)

        X_out = self.output_layer(X_dec)
        
        return X_out[:,:,:self.input_size]


    
