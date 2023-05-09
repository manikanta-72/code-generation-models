import torch
import torch.nn as nn
import torch.nn.functional as F
from self_attention import SelfAttention

class MultiAttention(nn.Module):

    def __init__(self, d_model = 512, n_heads = 8):
        super(MultiAttention).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.w_o = nn.Linear(d_model,d_model, bias=False)
        self.attention_heads = [SelfAttention(self.d_model, self.d_k, self.d_v) for i in range(self.n_heads)]

    def forward(self, X):
        head_outputs = [self_attention_i.forward(X) for self_attention_i in self.attention_heads]

        output_concat = torch.concat(head_outputs)

        outputs = self.w_o(output_concat)

        return outputs