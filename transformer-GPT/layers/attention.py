import torch
import torch.nn as nn
import torch.nn.functional as F

class selfAttention(nn.Module()):

    def __init__(self, em_dim, d_k=None, d_v=None):
        super().__init__()
        self.em_dim = em_dim
        if(d_k and d_v):
            self.d_k = self.d_q = d_k
            self.d_v = d_v
        else:
            self.d_k = self.d_v = self.d_q = em_dim

        # w's are initialized from uniform distribution
        self.q = nn.Linear(self.em_dim, self.d_q, bias=False)
        self.k = nn.Linear(self.em_dim, self.d_k, bias=False)
        self.v = nn.Linear(self.em_dim, self.d_v, bias=False)
        

    def scalar_dot_product_attention(self, Q, V, K_T):
        ''' * represents size of x.
            dim of Q = (*,d_q)
            dim of K_T = (d_k,*)
            dim of V = (*,d_v)
            dim of A = (*,d_v)
            d_v == d_model
        '''
        # A = torch.softmax(torch.matmul((torch.matmul(Q,K_T)/torch.sqrt(self.d_k)),V))
        A = F.softmax( ( (Q @ K_T)/torch.sqrt(self.d_k) ) @ V , dim=0)
        
        return A

    def forward(self, Q, K, V, mask, dropout):
        A = self.scalar_dot_product_attention(self.q(Q), self.k(K).T, self.v(V))

        return
