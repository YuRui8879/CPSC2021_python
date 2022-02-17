import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv1d(2,32,5,2)
        self.conv2 = nn.Conv1d(32,32,5,2)
        self.conv3 = nn.Conv1d(32,32,5,2)
        self.conv4 = nn.Conv1d(32,32,5,2)
        self.conv5 = nn.Conv1d(32,32,5,2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(32)
        self.gru1 = nn.GRU(32,32)
        self.gru2 = nn.GRU(32,32)
        self.gru3 = nn.GRU(32,32)
        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0),x.size(2),x.size(1))
        x,h = self.gru1(x)
        x,h = self.gru2(x)
        x,h = self.gru3(x)
        x = self.sigmoid(self.fc(x))
        x = x.view(x.size(0),-1)
        return x

if __name__ == '__main__':
    model = Model()
    x = torch.rand(2,2,12000)
    y = model(x)
    print(y.size())