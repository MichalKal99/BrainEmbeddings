import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class ConvNorm(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                padding=None, dilation=1, bias=True, w_init_gain='linear'):
      super(ConvNorm, self).__init__()
      if padding is None:
          assert(kernel_size % 2 == 1)
          padding = int(dilation * (kernel_size - 1) / 2)

      self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  bias=bias)

      torch.nn.init.xavier_uniform_(
          self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

  def forward(self, signal):
      conv_signal = self.conv(signal)
      return conv_signal


class MelLinear(torch.nn.Module):
    def __init__(self, dec_out_dim, n_mel_channels):
        super(MelLinear, self).__init__()

        self.fc = nn.Linear(dec_out_dim, n_mel_channels)

    def forward(self, x):
        return self.fc(x)

class Postnet(torch.nn.Module):
  def __init__(self, hidden_dim, dropout=0.1):
      super(Postnet, self).__init__()
      self.conv_list = torch.nn.ModuleList()
      self.postnet_kernel_size=3
      self.postnet_n_convolutions=3
      self.n_mel_channels=80 # in channels
      
      self.conv_list.append(
          torch.nn.Sequential(
              ConvNorm(self.n_mel_channels, hidden_dim,
                        kernel_size=self.postnet_kernel_size, stride=1,
                        padding=int((self.postnet_kernel_size - 1) / 2),
                        # padding=None,
                        dilation=1, w_init_gain="tanh"))
      )

      for _ in range(1, self.postnet_n_convolutions - 1):
          self.conv_list.append(
              torch.nn.Sequential(
                  ConvNorm(hidden_dim,
                            hidden_dim,
                            kernel_size=self.postnet_kernel_size, stride=1,
                            padding=int((self.postnet_kernel_size - 1) / 2),
                            # padding=None,
                            dilation=1, w_init_gain="tanh"))
                                      )

      self.final_conv = torch.nn.Sequential(
          ConvNorm(hidden_dim, self.n_mel_channels,
                   kernel_size=self.postnet_kernel_size, stride=1,
                   padding=int((self.postnet_kernel_size - 1) / 2),
                # padding=None,
                   dilation=1))
          
      self.dropout_list = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(self.postnet_n_convolutions)])
      self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_dim) for _ in range(self.postnet_n_convolutions)])

  def forward(self, x):
    x = x.permute(0,2,1)
    for dropout, norm, conv in zip(self.dropout_list, self.batch_norm_list, self.conv_list):
        x = dropout(torch.tanh(norm(conv(x))))
    x = self.final_conv(x)
    x = x.permute(0,2,1)
    return x        
  
class EncoderPrenet(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
    super(EncoderPrenet, self).__init__()
      
    self.conv_list = torch.nn.ModuleList()
    self.postnet_n_convolutions=3
    self.postnet_kernel_size=21
    self.kernel_arr = [25,11,5]
    self.hidden_arr = [50,75,100]
    self.stride_arr = [3,3,3]

    self.conv_list.append(
        torch.nn.Sequential(
            ConvNorm(input_dim, self.hidden_arr[0],
                    kernel_size=self.kernel_arr[0], stride=self.stride_arr[0],
                    padding=int((self.kernel_arr[0] - 1) / 2),
                    dilation=1, w_init_gain='relu'))
    )

    for i in range(1, self.postnet_n_convolutions):
        self.conv_list.append(
            torch.nn.Sequential(
                ConvNorm(self.hidden_arr[i-1],
                        self.hidden_arr[i],
                        kernel_size=self.kernel_arr[i], stride=self.stride_arr[i],
                        padding=int((self.kernel_arr[i] - 1) / 2),
                        dilation=1, w_init_gain='relu'))
        )

    self.dropout_list = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(self.postnet_n_convolutions)])
    self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(num_features=self.hidden_arr[i]) for i in range(self.postnet_n_convolutions)])
    
    self.projection = nn.Linear(self.hidden_arr[-1], hidden_dim) 

  def forward(self, x):
    x=x.permute(0,2,1)
    for dropout, norm, conv in zip(self.dropout_list, self.batch_norm_list, self.conv_list):
        x = dropout(F.relu(norm(conv(x))))
    x=x.permute(0,2,1)
    x = self.projection(x)
    return x    

class DecoderPrenet(torch.nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.1):
    super(DecoderPrenet, self).__init__()

    self.layer = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(input_dim, hidden_dim)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(dropout)),
             ('fc2', nn.Linear(hidden_dim, output_dim)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(dropout))
        ]))

  def forward(self, x):
    x = self.layer(x)
    return x       

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.alpha = nn.Parameter(torch.ones(1))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        pos = self.pe[:x.shape[1]]
        pos = torch.stack([pos]*x.shape[0], 0) # [bs x seq_len(x) x n_pos]
        x = (pos * self.alpha) + x
        return self.dropout(x)

class SimpleEncoder(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, embedding_size, hidden_dim, dropout):
        super(SimpleEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dropout=dropout, dim_feedforward=hidden_dim,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class SimpleDecoder(nn.Module):
    def __init__(self, num_heads, num_layers, embedding_size, hidden_dim, dropout):
        super(SimpleDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads, dropout=dropout, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, target, enc_output, tgt_mask=None, tgt_is_causal=None):
        x = self.transformer_decoder(target, enc_output, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        # Prenets
        # self.encoder_prenet = EncoderPrenet(config['TR']["num_features"], 
        #                                     hidden_dim=config['TR']["embedding_size"], 
        #                                     dropout=config["TR"]["dropout"])
        
        # self.decoder_prenet = DecoderPrenet(input_dim=config["feature_extraction"]["num_feats"], 
        #                                     output_dim=config['TR']["embedding_size"],
        #                                     hidden_dim=config['TR']["embedding_size"],
        #                                     dropout=config["TR"]["dropout"])
        
        # self.norm = nn.Linear(config['TR']["embedding_size"], config['TR']["embedding_size"])
        
        # Embedding layers
        self.enc_embedding = nn.Linear(config['TR']["num_features"], config['TR']["embedding_size"])
        self.dec_embedding = nn.Linear(config["feature_extraction"]["num_feats"], config['TR']["embedding_size"])

        # Positional encoding layers
        self.enc_pe = PositionalEncoding(config['TR']["embedding_size"], dropout=config["TR"]["dropout"], max_len=config["TR"]["encoder_seq_len"])
        self.dec_pe = PositionalEncoding(config['TR']["embedding_size"], dropout=config["TR"]["dropout"], max_len=config["TR"]["context_length"])

        # Transformer
        self.encoder = SimpleEncoder(
            config['TR']["num_features"], config['TR']["num_heads"], config['TR']["encoder_layers"], 
            config['TR']["embedding_size"], config['TR']["hidden_size"], config['TR']["dropout"])
    
        self.decoder = SimpleDecoder(
            config['TR']["num_heads"], config['TR']["decoder_layers"], 
            config['TR']["embedding_size"], config['TR']["hidden_size"], config['TR']["dropout"])

        # Mel Linear
        self.mel_linear = MelLinear(config['TR']["embedding_size"], 80)

        # Postnet
        # self.decoder_postnet = Postnet(config['TR']["embedding_size"], config['TR']["dropout"])


    def encode(self, src):
        src = self.enc_embedding(src)
        # src = self.encoder_prenet(src)
        src = self.enc_pe(src)
        memory = self.encoder(src)
        return memory
    
    def decode(self, tgt, memory, tgt_mask, tgt_is_causal):
        tgt = self.dec_embedding(tgt)
        # tgt = self.decoder_prenet(tgt)
        # tgt = self.norm(tgt)
        tgt = self.dec_pe(tgt)
        dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        
        x_mel = self.mel_linear(dec_out)
        # x_post = self.decoder_postnet(x_mel)
        # mel_spec = x_mel + x_post

        return x_mel
    
    def forward(self, src, tgt, tgt_mask=None, tgt_is_causal=None):
        src, tgt = self.enc_embedding(src), self.dec_embedding(tgt)
        # src, tgt = self.encoder_prenet(src), self.decoder_prenet(tgt)
        src, tgt = self.enc_pe(src), self.dec_pe(tgt)
        # tgt = self.norm(tgt)
        memory = self.encoder(src)
        dec_out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)

        x_mel = self.mel_linear(dec_out)
        # x_post = self.decoder_postnet(x_mel)
        # mel_spec = x_mel + x_post

        return x_mel
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embedding_size, hidden_dim, output_dim, n_heads, n_layers, seq_len, dropout=0.5):
        super(Transformer, self).__init__()

        self.emb = nn.Linear(input_dim, embedding_size)
        self.pos = PositionalEncoding(embedding_size, dropout=dropout, max_len=seq_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=n_heads, dropout=dropout, dim_feedforward=hidden_dim,batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.mel_lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        
        src_emb = self.emb(src)
        src_emb = self.pos(src_emb)
        src_enc = self.encoder(src_emb)
        mel = self.mel_lin(src_enc)

        return mel

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 