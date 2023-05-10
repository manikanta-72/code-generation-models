import torch
import torch.nn as nn
from layers.multi_attention import MultiAttention
from layers.layer_norm import LayerNorm
from layers.position_wise_feedforward import PositionwiseFeedforward

class EncoderBlock(nn.Module()):

    def __init__(self, d_model=512, n_heads=8):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.multi_attention = MultiAttention(self.d_model, self.n_heads)
        self.layer_norm = LayerNorm(self.d_model)
        self.position_wise_feedfoward = PositionwiseFeedforward()

    def forward(self, X):
        mul_attn = self.multi_attention.forward(X)
        norm_attn = X + self.layer_norm.forward(mul_attn)
        ff_output = self.position_wise_feedfoward.forward(norm_attn)
        norm_output = norm_attn + self.layer_norm.forward(ff_output)

        return norm_output