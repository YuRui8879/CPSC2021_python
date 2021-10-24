import numpy as np
import random

class Dataset:

    def __init__(self,X,Y,fold):
        self.idx = 0
        self.fold = fold
        self.X = np.array(X)
        self.Y = np.array(Y)
        if self.X.shape[0] != self.Y.shape[0]:
            raise Exception('数据与标签维度不一致')
        self.length = self.Y.shape[0]
        self.step_len = self.length//self.fold
        
    def shuffle(self,seed = 0):
        random.seed(seed)
        index = list(range(self.length))
        random.shuffle(index)
        index = np.array(index)
        self.X = self.X[index,:]
        self.Y = self.Y[index]

    def get_fold_data(self,n = -1):
        if n >= self.fold:
            print('游标超出范围，请设置在(0,{})范围内'.format(self.fold))
            res_X = []
            res_Y = []
        else:
            if n != -1:
                if n == self.fold - 1:
                    lidx = n * self.step_len
                    res_X = self.X[lidx:,:]
                    res_Y = self.Y[lidx:]
                else:
                    lidx = n * self.step_len
                    ridx = (n + 1) * self.step_len
                    res_X = self.X[lidx:ridx,:]
                    res_Y = self.Y[lidx:ridx]
            else:
                if self.idx == self.fold - 1:
                    lidx = self.idx * self.step_len
                    res_X = self.X[lidx:,:]
                    res_Y = self.Y[lidx:]
                else:
                    lidx = self.idx * self.step_len
                    ridx = (self.idx + 1) * self.step_len
                    res_X = self.X[lidx:ridx,:]
                    res_Y = self.Y[lidx:ridx]

        return res_X,res_Y

    def get_res_data(self):
        if self.idx == self.fold - 1:
            lidx = self.idx * self.step_len
            res_X = self.X[:lidx,:]
            res_Y = self.Y[:lidx]
        elif self.idx == 0:
            ridx = (self.idx + 1) * self.step_len
            res_X = self.X[ridx:,:]
            res_Y = self.Y[ridx:]
        else:
            lidx = self.idx * self.step_len
            ridx = (self.idx + 1) * self.step_len
            res_X1 = self.X[:lidx,:]
            res_X2 = self.X[ridx:,:]
            res_Y1 = self.Y[:lidx]
            res_Y2 = self.Y[ridx:]
            res_X = np.vstack((res_X1,res_X2))
            res_Y = np.hstack((res_Y1,res_Y2))
        return res_X,res_Y

    def step(self,n = 1):
        self.idx += n
        self.idx = self.idx % self.fold
        print('向前搜索{}步，现在输出第{}折数据'.format(n,self.idx))

    def reset(self,n = 0):
        if n > self.fold:
            print('游标超出范围，请设置在(0,{})范围内'.format(self.fold))
        else:
            self.idx = n
            print('游标设置为{}，现在输出第{}折数据'.format(self.idx,self.idx))

class AssembleDataset:

    def __init__(self,X,Y,fold,train_rate = 0.8,seed = -1):
        X = np.array(X)
        Y = np.array(Y)
        if X.shape[0] != Y.shape[0]:
            raise Exception('数据与标签维度不一致')
        train_size = int(len(Y) * train_rate)
        if seed != -1:
            random.seed(seed)
            index = list(range(len(Y)))
            random.shuffle(index)
            index = np.array(index)
            X = X[index,:]
            Y = Y[index]
        self.train_X = X[:train_size,:]
        self.train_Y = Y[:train_size]
        self.test_X = X[train_size:,:]
        self.test_Y = Y[train_size:]
        self.idx = 0
        self.fold = fold
        self.length = self.train_Y.shape[0]
        self.step_len = self.length//self.fold

    def get_test_set(self):
        return self.test_X,self.test_Y

    def get_train_set(self):
        return self.train_X,self.train_Y

    def get_fold_data(self,n = -1):
        if n >= self.fold:
            print('游标超出范围，请设置在(0,{})范围内'.format(self.fold))
            res_X = []
            res_Y = []
        else:
            if n != -1:
                if n == self.fold - 1:
                    lidx = n * self.step_len
                    res_X = self.train_X[lidx:,:]
                    res_Y = self.train_Y[lidx:]
                else:
                    lidx = n * self.step_len
                    ridx = (n + 1) * self.step_len
                    res_X = self.train_X[lidx:ridx,:]
                    res_Y = self.train_Y[lidx:ridx]
            else:
                if self.idx == self.fold - 1:
                    lidx = self.idx * self.step_len
                    res_X = self.train_X[lidx:,:]
                    res_Y = self.train_Y[lidx:]
                else:
                    lidx = self.idx * self.step_len
                    ridx = (self.idx + 1) * self.step_len
                    res_X = self.train_X[lidx:ridx,:]
                    res_Y = self.train_Y[lidx:ridx]

        return res_X,res_Y

    def get_res_data(self):
        if self.idx == self.fold - 1:
            lidx = self.idx * self.step_len
            res_X = self.train_X[:lidx,:]
            res_Y = self.train_Y[:lidx]
        elif self.idx == 0:
            ridx = (self.idx + 1) * self.step_len
            res_X = self.train_X[ridx:,:]
            res_Y = self.train_Y[ridx:]
        else:
            lidx = self.idx * self.step_len
            ridx = (self.idx + 1) * self.step_len
            res_X1 = self.train_X[:lidx,:]
            res_X2 = self.train_X[ridx:,:]
            res_Y1 = self.train_Y[:lidx]
            res_Y2 = self.train_Y[ridx:]
            res_X = np.vstack((res_X1,res_X2))
            res_Y = np.hstack((res_Y1,res_Y2))
        return res_X,res_Y

    def step(self,n = 1):
        self.idx += n
        self.idx = self.idx % self.fold
        print('向前搜索{}步，现在输出第{}折数据'.format(n,self.idx))

    def reset(self,n = 0):
        if n > self.fold:
            print('游标超出范围，请设置在(0,{})范围内'.format(self.fold))
        else:
            self.idx = n
            print('游标设置为{}，现在输出第{}折数据'.format(self.idx,self.idx))


if __name__ == '__main__':
    X = np.empty((10,5))
    Y = np.linspace(0,9,10)
    d = AssembleDataset(X,Y,5,seed = 0)
    a,b = d.get_fold_data()
    print(a,b)
    a,b = d.get_res_data()
    print(a,b)
