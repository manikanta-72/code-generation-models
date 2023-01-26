import torch
import torch.nn as nn

class TransformerGPT(nn.Module()):
    def __init__(self, input_dim, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_encoders = num_encoder_layers
        self.num_decoders = num_decoder_layers

    def encoder(self):
        pass

    def decoder(self):
        pass

    def forward(self):
        pass
