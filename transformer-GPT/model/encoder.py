import torch
import torch.nn as nn
from blocks.encoder_block import EncoderBlock

class Encoder(nn.Module):

    def __init__(self, num_layers, d_model = 512, n_heads = 8):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        encoders = [EncoderBlock(d_model=self.d_model, n_heads=self.n_head) for _ in range(num_layers)]

        self.layers = nn.ModuleList(encoders)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        return x