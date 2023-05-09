import torch
import torch.nn as nn

class DecoderBlock(nn.Module):

    def __init__(self, d_model):
        super(DecoderBlock,self).__init__()
        