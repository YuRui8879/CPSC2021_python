import torch.utils.data as Data
import torch
import random
from read_code import load_wavelet_data
import os
import torch.utils.data as Data

class DataAdapter(Data.Dataset):

    def __init__(self,X,Y):
        super(DataAdapter,self).__init__()
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)

    def __getitem__(self,index):
        return self.X[index,:],self.Y[index]

    def __len__(self):
        return len(self.X)

class DataAdapter2(Data.Dataset):

    def __init__(self,X1,X2):
        super(DataAdapter2,self).__init__()
        self.X1 = torch.FloatTensor(X1)
        self.X2 = torch.FloatTensor(X2)

    def __getitem__(self, index):
        return self.X1[index,:],self.X2[index,:]

    def __len__(self):
        return len(self.X1)

class WaveletDataAdapter(Data.Dataset):

    def __init__(self,path,seed = 0,type = 0,is_valid = False):
        super(WaveletDataAdapter,self).__init__()
        self.dirs = os.listdir(path)
        random.seed(seed)
        random.shuffle(self.dirs)
        self.path = path
        self.type = type
        if is_valid:
            train_size = int(len(self.dirs) * 0.7)
            valid_size = int(len(self.dirs) * 0.1)
            self.train_set = self.dirs[:train_size]
            self.valid_set = self.dirs[train_size:train_size+valid_size]
            self.test_set = self.dirs[train_size+valid_size:]
            if type == 0:
                self.length = len(self.train_set)
            elif type == 1:
                self.length = len(self.valid_set)
            elif type == 2:
                self.length = len(self.test_set)
            else:
                raise Exception('DataAdapter初始化错误')
        else:
            train_size = int(len(self.dirs) * 0.8)
            self.train_set = self.dirs[:train_size]
            self.test_set = self.dirs[train_size:]
            if type == 0:
                self.length = len(self.train_set)
            elif type == 1:
                self.length = len(self.test_set)
            else:
                raise Exception('DataAdapter初始化错误')

    def __getitem__(self, index):
        if self.type == 0:
            data,label = load_wavelet_data(self.train_set[index],self.path)
        elif self.type == 1:
            data,label = load_wavelet_data(self.valid_set[index],self.path)
        elif self.type == 2:
            data,label = load_wavelet_data(self.test_set[index],self.path)
        else:
            raise Exception('数据集类型错误')
        if label == 0:
            res_label = torch.zeros(1).long()
        else:
            res_label = torch.ones(1).long()
        return torch.FloatTensor(data),res_label

    def __len__(self):
        return self.length


if __name__ == '__main__':
    path = r'C:\Users\yurui\Desktop\item\cpsc\data\wavelet\pretrain_lead0'
    da = WaveletDataAdapter(path)
    loader = Data.DataLoader(da,batch_size = 2,shuffle = False,num_workers = 0)
    for i,data in enumerate(loader,0):
        print(data[0].size())
        print(data[1])