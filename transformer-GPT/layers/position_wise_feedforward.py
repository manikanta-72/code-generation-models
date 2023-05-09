import torch
import torch.nn as nn

#ToDo dropout

class PositionwiseFeedforward(nn.Module()):

    def __inti__(self,d_model,d_hidden):

        super(PositionwiseFeedforward,self).__init__()

        self.L1 = nn.Linear(d_model, d_hidden)
        self.L2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)

        return x
