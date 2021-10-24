import torch
import torch.nn as nn

class SEBlock(nn.Module):

    def __init__(self,planes):
        super(SEBlock,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(planes, planes//8, kernel_size=(1,1))
        self.fc2 = nn.Conv2d(planes//8, planes, kernel_size=(1,1))
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
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(2, 1, (7,7),padding=(3,3))
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x_tmp = x.permute([0,2,3,1])
        maxpool_output = self.maxpool(x_tmp)
        avgpool_output = self.avgpool(x_tmp)
        x_tmp = torch.cat([maxpool_output,avgpool_output],dim = -1)
        x_tmp = x_tmp.permute([0,3,1,2])
        x_tmp = self.sigmoid(self.conv(x_tmp))
        return x * x_tmp


class CNNBlock(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(CNNBlock,self).__init__()
        self.convt = nn.Conv2d(in_channel,out_channel,(1,1),stride=(2,2))
        self.conv1 = nn.Conv2d(in_channel,out_channel,(1,1))
        self.conv2 = nn.Conv2d(out_channel,out_channel,(3,3),padding = (1,1))
        self.conv3 = nn.Conv2d(out_channel,out_channel,(1,1),(2,2))
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.se = SEBlock(out_channel)
        self.cam = CAMBlock()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):
        shortcut = self.convt(x)
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn3(x)
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
        x = x.view(-1,1,160,1000)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0),x.size(1),-1).permute([0,2,1])
        _,h = self.gru(x)
        h = self.relu(h.permute([1,0,2]))
        x = h.reshape(h.size(0),-1)
        x = self.linear_unit(x)
        return x


if __name__ == '__main__':
    model = CNN()
    device = torch.device("cuda:1")
    model.to(device)
    x = torch.rand(8,160,1000).to(device)
    y = model(x)
    print(y.size())