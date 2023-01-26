import torch
from torch import nn

# TODO device specific either "cuda" or "cpu"
class PositionalEncoding(nn.Module):

    def init(self, d_model, max_length):
        
        super(PositionalEncoding,self).__init__()
        
        self.encoding = torch.zeros(max_length, d_model)
        self.encoding.require_grad = False

        pos = torch.arange(0, max_length)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos/1000**(_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos/1000**(_2i / d_model))

    def forward(self, x):
        _ , m = x.size()

        return self.encoding[:m, :]