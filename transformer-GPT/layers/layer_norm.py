import torch
from torch import nn

class LayerNorm(nn.Module):

    def __init__(self, d_model=512, eps=1e-12):
        self.gamma = nn.Parameter(torch.zeros(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self,x):

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbaised=False, keepdim=True)

        x = (x - mean) / torch.sqrt( var+self.eps)
        x = self.gamma * x + self.beta

        return x