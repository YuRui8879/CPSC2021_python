import torch
import torch.nn as nn

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

class CAMBlock(nn.Module):

    def __init__(self):
        super(CAMBlock,self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(2, 1, 7, padding = 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x_tmp = x.permute([0,2,1])
        maxpool_output = self.maxpool(x_tmp)
        avgpool_output = self.avgpool(x_tmp)
        x_tmp = torch.cat([maxpool_output,avgpool_output],dim = -1)
        x_tmp = x_tmp.permute([0,2,1])
        x_tmp = self.sigmoid(self.conv(x_tmp))
        return x * x_tmp


class CNNBlock(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(CNNBlock,self).__init__()
        self.convt = nn.Conv1d(in_channel,out_channel,1,stride=2)
        self.conv1 = nn.Conv1d(in_channel,out_channel,3,padding = 1)
        self.conv2 = nn.Conv1d(out_channel,out_channel,3,padding = 1)
        self.conv3 = nn.Conv1d(out_channel,out_channel,25,2,padding = 12)
        self.bn = nn.BatchNorm1d(in_channel)
        self.se = SEBlock(out_channel)
        self.cam = CAMBlock()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        shortcut = self.convt(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x += shortcut
        x = self.relu(x)
        x = self.se(x)
        x = self.cam(x)
        x = self.dropout(x)
        
        return x

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
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
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
            nn.Softmax(-1)
        )

    def forward(self,x):
        batch_size = x.size(0)
        x = x.view(-1,1,1000)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.permute([0,2,1])
        _,h = self.gru(x)
        h = self.relu(h.permute([1,0,2]))
        x = h.reshape(h.size(0),-1)
        fea = x
        x = self.linear_unit(x)
        return x,fea

class RNN(nn.Module):

    def __init__(self):
        super(RNN,self).__init__()
        self.gru1 = nn.GRU(256, 128,batch_first=True,bidirectional=True)
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



if __name__ == '__main__':
    model = CNN()
    x = torch.rand(128,1000)
    y,fea = model(x)
    print(y.size())
    model = RNN()
    # x = torch.rand(24,30,256)
    # y = model(x)
    # print(y.size())