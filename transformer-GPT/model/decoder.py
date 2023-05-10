import torch
import torch.nn as nn
from blocks.decoder_block import DecoderBlock

class Decoder(nn.Module):

    def __init__(self, output_size, num_layers, d_model=512, n_heads=8):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        encoders = [DecoderBlock(d_model=self.d_model, n_heads=self.n_head) for _ in range(num_layers)]

        self.layers = nn.ModuleList(encoders)   
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        output = self.linear(x)
        return output