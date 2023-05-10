import torch
import torch.nn as nn
import torch.nn.functional as F
from masking.masking import PadMask
class SelfAttention(nn.Module()):

    def __init__(self, d_model, d_k=None, d_v=None, enable_mask = False):
        super(SelfAttention).__init__()
        self.d_model = d_model
        if(d_k and d_v):
            self.d_k = self.d_q = d_k
            self.d_v = d_v
        else:
            self.d_k = self.d_v = self.d_q = d_model
        self.masking = enable_mask

        # w's are initialized from uniform distribution
        self.q = nn.Linear(self.d_model, self.d_q, bias=False)
        self.k = nn.Linear(self.d_model, self.d_k, bias=False)
        self.v = nn.Linear(self.d_model, self.d_v, bias=False)
        self.softmax = nn.Softmax(dims=-1)
        self.mask = PadMask

    def forward(self, X, mask, dropout):
        ''' * represents size of x.
            dim of Q = (*,d_q)
            dim of K_T = (d_k,*)
            dim of V = (*,d_v)
            dim of A = (*,d_v)
            d_v == d_model
        '''
        # A = torch.softmax(torch.matmul((torch.matmul(Q,K_T)/torch.sqrt(self.d_k)),V))
        Q, K_T, V = self.q(X), self.k(X).T, self.v(X)
        
        # Compute logits and mask to allow the passage of past knowledge
        logits = (Q @ K_T)
        if self.masking:
            tri_mask = self.mask.create_tri_mask(self.d_k)
            logits[tri_mask] = float("Inf")

        # apply softmax to obtain the attention weightage of previous tokens to current token
        scores = self.softmax(logits/torch.sqrt(self.d_k))
        
        outputs = scores @ V 
        
        return outputs