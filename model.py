import torch
import torch.nn as nn

class CNNBlock(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(CNNBlock,self).__init__()
        self.conv1 = nn.Conv1d(in_channel,out_channel,3,padding = 1)
        self.conv2 = nn.Conv1d(out_channel,out_channel,3,padding = 1)
        self.conv3 = nn.Conv1d(out_channel,out_channel,25,2,padding = 12)
        self.bn = nn.BatchNorm1d(in_channel)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class RCNN(nn.Module):

    def __init__(self):
        super(RCNN,self).__init__()
        self.conv1 = CNNBlock(1, 64)
        self.conv2 = CNNBlock(64, 128)
        self.conv3 = CNNBlock(128, 128)
        self.conv4 = CNNBlock(128, 128)
        self.conv5 = CNNBlock(128, 128)
        self.gru = nn.GRU(128, 128,batch_first=True,bidirectional=True)
        self.relu = nn.ReLU(True)
        self.linear_unit = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16,2),
            nn.Softmax(-1)
        )

    def forward(self,x):
        batch_size = x.size(0)
        x = x.view(-1,1,2000)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.permute([0,2,1])
        _,h = self.gru(x)
        h = self.relu(h.permute([1,0,2]))
        x = torch.cat((h[:,0,:],h[:,1,:]),dim = 1)
        x = x.view(x.size(0),-1)
        x = self.linear_unit(x)
        return x


if __name__ == '__main__':
    model = RCNN()
    x = torch.rand(128,2000)
    y = model(x)
    print(y.size())