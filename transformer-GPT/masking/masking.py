import torch
import torch.nn as nn

class PadMask(nn.Module()):

    def __init__(self, src_pad_token, trg_pad_token):
        super().__init__()
        self.src_pad_token = src_pad_token
        self.trg_pad_token = trg_pad_token

    def create_pad_mask(self, x):
        mask = (x == self.src_pad_token).unsqueeze(-2)
        return mask

    def create_tri_mask(self, l):
        ones = torch.ones(l, l, dtype=torch.uint8)
        mask = torch.triu(ones, diagonal=1).unsqueeze(0)

        return mask


