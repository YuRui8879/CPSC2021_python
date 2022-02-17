import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.gru1 = nn.GRU(512, 128,batch_first=True,bidirectional=True)
        self.relu = nn.ReLU(False)
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(256,128)
        self.linear2 = nn.Linear(128,2)
        self.softmax = nn.Softmax(-1)

    def forward(self,x):
        x,h = self.gru1(x)
        x = self.relu(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.softmax(self.linear2(x))
        return x