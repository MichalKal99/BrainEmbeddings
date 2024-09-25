import torch
import torch.nn as nn
import numpy as np 
from math import floor

class EEG_net(nn.Module):
  def __init__(self, input_shape, latent_dim = 2):
    super(EEG_net, self).__init__()
    self.model_type = "ae"

    self.in_shape = input_shape


    self.encoder = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3,1), stride=1, padding="valid"),
              nn.ReLU(),
              nn.BatchNorm2d(5),
              nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(5,1), stride=1, padding="valid"),
              nn.ReLU(),
              nn.BatchNorm2d(10),
              nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(11,1), stride=1, padding="valid"),
              nn.ReLU(),
              nn.BatchNorm2d(15),
              nn.Conv2d(in_channels=15, out_channels=30, kernel_size=(15,input_shape[1]), stride=1, padding="valid"),
              nn.ReLU(),
              nn.BatchNorm2d(30),
              nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(21,1), stride=1, padding="valid"),
              nn.ReLU(),
              nn.BatchNorm2d(30),
          )

    self.decoder = nn.Sequential(
              nn.ConvTranspose2d(in_channels=30, out_channels=30, kernel_size=(21,1), stride=1),
              nn.ReLU(),
              nn.BatchNorm2d(30),
              nn.ConvTranspose2d(in_channels=30, out_channels=15, kernel_size=(15,input_shape[1]), stride=1),
              nn.ReLU(),
              nn.BatchNorm2d(15),
              nn.ConvTranspose2d(in_channels=15, out_channels=10, kernel_size=(11,1), stride=1),
              nn.ReLU(),
              nn.BatchNorm2d(10),
              nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=(5,1), stride=1),
              nn.ReLU(),
              nn.BatchNorm2d(5),
              nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=(3,1), stride=1),
              nn.ReLU(),
              nn.BatchNorm2d(1),
              )
    self.encoder_flat_dim = self.get_flat_dim()

    self.encoder_fc = nn.Sequential(
      nn.Linear(30, latent_dim),
      nn.ReLU()
    )

    self.decoder_fc = nn.Sequential(
      nn.Linear(latent_dim, 30),
      nn.ReLU()
        )

  def get_conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    
    if type(kernel_size) is not tuple:
      kernel_size = (kernel_size, kernel_size)
    if type(pad) is tuple:
      h = floor( ((h_w[0] + (2 * pad[0]) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
      w = floor( ((h_w[1] + (2 * pad[1]) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    else:
      h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
      w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w
  
  def get_flat_dim(self):
    in_shape = self.in_shape
    final_conv_layer = None
    for lay in self.encoder:
      if isinstance(lay, nn.Conv2d):
        lay_out = self.get_conv_output_shape(in_shape, lay.kernel_size)
        in_shape = lay_out
        final_conv_layer = lay
    self.encoder_output = (final_conv_layer.out_channels, *in_shape)
    flat_dim = np.prod(np.array([*in_shape, final_conv_layer.out_channels]))
    return flat_dim

  def encode(self, x):
    x = self.encoder(x)
    x = x.permute(0,3,2,1)
    x = self.encoder_fc(x)
    return x
  
  def decode(self, x):
    x = self.decoder_fc(x)
    x = x.reshape(x.size(0), **self.encoder_output)
    x = self.decoder(x)
    return x

  def forward(self, x):
    x = self.encoder(x)
    # print(f"end encoder: {x.shape}")
    x = x.permute(0,3,2,1)
    x = self.encoder_fc(x)
    # print(f"latent: {x.shape}")
    x = self.decoder_fc(x)
    x = x.permute(0,3,2,1)
    x = self.decoder(x)
    # print(f"end decoder: {x.shape}")
    return x

class VAE(nn.Module):
  def __init__(self, input_shape, latent_dim=64):
    super(VAE, self).__init__()

    self.model_type = "vae"

    self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3,1), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Conv2d(in_channels=5, out_channels=15, kernel_size=(5,1), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.Conv2d(in_channels=15, out_channels=15, kernel_size=(11,1), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.Conv2d(in_channels=15, out_channels=30, kernel_size=(15, input_shape[1]), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(21,1), stride=1, padding="valid"),
            nn.ReLU(),
            nn.BatchNorm2d(30),
        )

    self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=30, out_channels=30, kernel_size=(21,1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(30),
            nn.ConvTranspose2d(in_channels=30, out_channels=15, kernel_size=(15, input_shape[1]), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.ConvTranspose2d(in_channels=15, out_channels=15, kernel_size=(11,1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.ConvTranspose2d(in_channels=15, out_channels=5, kernel_size=(5,1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=(3,1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            )
    self.dropout_1 = nn.Dropout(p=0.1)
    self.dropout_2 = nn.Dropout(p=0.1)
    
    self.in_shape = input_shape
    self.linear_dim = self.get_flat_dim()
    self.flatten = nn.Flatten()
    self.linear_dec = nn.Linear(latent_dim, self.linear_dim)
    
    self.fc_mu = nn.Linear(self.linear_dim, latent_dim)
    self.fc_var = nn.Linear(self.linear_dim, latent_dim)

  def get_conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    if type(kernel_size) is not tuple:
      kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w
  
  def get_flat_dim(self):
    in_shape = self.in_shape
    final_conv_layer = None
    for lay in self.encoder:
      if isinstance(lay, nn.Conv2d):
        lay_out = self.get_conv_output_shape(in_shape, lay.kernel_size)
        in_shape = lay_out
        final_conv_layer = lay
    self.encoder_output = (final_conv_layer.out_channels, *in_shape)
    flat_dim = np.prod(np.array([*in_shape, final_conv_layer.out_channels]))
    return flat_dim
  
  def encode(self, x):
    x = self.dropout_1(self.encoder(x))
    x_flat = self.flatten(x)
    mu = self.fc_mu(x_flat)
    log_var = self.fc_var(x_flat)
    z = self.reparameterize(mu, log_var)
    return mu, log_var, z
  

  def decode(self, z):
    z = self.linear_dec(z)
    z = z.reshape(z.shape[0], *self.encoder_output)
    x = self.dropout_2(self.decoder(z))
    return x
  
  def reparameterize(self, mu, logvar):
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = eps.mul(std).add_(mu)
    
    return z
  
  def forward(self, x):
    mu, log_var, z = self.encode(x)
    x = self.decode(z)
    return x, mu, log_var