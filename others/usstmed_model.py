import torch
import torch.nn as nn

# Team Name: usstmed

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv1d(2,64,7,1,3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool1d(2)
        self.conv2 = nn.Sequential(
            ResBlock(64,64,7,1,3),
            ResBlock(64,64,7,1,3),
            ResBlock(64,64,7,1,3),
            ResBlock(64,128,7,2,3),
            ResBlock(128,128,7,1,3),
            ResBlock(128,128,7,1,3),
            ResBlock(128,128,7,1,3),
            ResBlock(128,256,7,2,3),
            ResBlock(256,256,7,1,3),
            ResBlock(256,256,7,1,3),
            ResBlock(256,256,7,1,3),
            ResBlock(256,512,7,2,3),
            ResBlock(512,512,7,1,3),
            ResBlock(512,512,7,1,3),
            ResBlock(512,512,7,1,3),
            ResBlock(512,512,7,1,3),
        )
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = x.view(x.size(0),1,x.size(1))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = x.view(x.size(0),-1)
        return x


class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride = 1,padding = 0):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv1d(in_ch,in_ch,kernel_size,padding = kernel_size//2,bias = False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv2 = nn.Conv1d(in_ch,out_ch,kernel_size,stride,padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.1)
        self.se = SEBlock(out_ch)
        if in_ch!=out_ch:
            self.sideway = nn.Conv1d(in_ch,out_ch,1,stride)
        else:
            self.sideway = None


    def forward(self,x):
        shortcut = x
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        if self.sideway:
            x += self.sideway(shortcut)
        else:
            x += shortcut
        x = self.relu(x)
        return x

class SEBlock(nn.Module):

    def __init__(self,planes):
        super(SEBlock,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(planes, planes//8, kernel_size=1)
        self.fc2 = nn.Conv1d(planes//8, planes, kernel_size=1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgtmp = self.avgpool(x)
        maxtmp = self.maxpool(x)
        tmp = avgtmp + maxtmp
        tmp = self.fc1(tmp)
        tmp = self.relu(tmp)
        tmp = self.fc2(tmp)
        tmp = self.sigmoid(tmp)
        return x * tmp