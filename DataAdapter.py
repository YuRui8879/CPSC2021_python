import torch.utils.data as Data
import torch
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence

class DataAdapter(Data.Dataset):

    def __init__(self,X,Y):
        super(DataAdapter,self).__init__()
        self.X = torch.FloatTensor(X)
        self.Y = torch.LongTensor(Y)

    def __getitem__(self,index):
        return self.X[index,:],self.Y[index]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    X = [[1,2,3],[1,3,4],[4,5,6]]
    Y = [[1,0,1],[1,1,1],[1,0,0]]
    da = DataAdapter(X,Y)