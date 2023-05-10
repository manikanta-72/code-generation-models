import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class TransformerGPT(nn.Module()):
    def __init__(self, input_dim, n_heads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.d_model = input_dim
        self.n_heads = n_heads
        self.num_encoders = num_encoder_layers
        self.num_decoders = num_decoder_layers
        self.encoder = Encoder(d_model=self.d_model,
                               n_heads=self.n_heads,
                               num_layers=self.num_encoders)
        self.decoder = Decoder(d_model=self.d_model,
                               n_heads=self.n_heads,
                               num_layers=self.num_decoders)
    def forward(self, X):
        encode_x = self.encode(X)
        output = self.decode(encode_x)
        return output
