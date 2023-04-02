import torch.utils.data as Data
import torch
import random
import os
import wfdb
import numpy as np
from score_2021 import RefInfo
from scipy.signal import butter,filtfilt

class BaseAdapter:

    def __init__(self,data_path,n_lead = 1):
        sample = self._get_signal(data_path,n_lead)
        train_sample,valid_sample,test_sample = self._gen_sample(sample)
        self.train_X,self.train_Y = self._gen_cnn_X_Y(train_sample)
        self.valid_X,self.valid_Y = self._gen_cnn_X_Y(valid_sample)
        self.test_X,self.test_Y = self._gen_cnn_X_Y(test_sample)
        self.save_RECORD(test_sample)

    def get_train_set(self):
        return self.train_X,self.train_Y

    def get_valid_set(self):
        return self.valid_X,self.valid_Y

    def get_test_set(self):
        return self.test_X,self.test_Y

    def _get_signal(self,path,channel = 1):
        '''
        Description:
            读取信号并获取采样频率，信号长度，心拍位置，房颤开始位置，房颤结束位置，类别等
        Params:
            path 读取数据的路径
        Return:
            res 该结构是一个列表，每个元素代表一个样本，样本的结构为一个字典，字典中包含名称，
                信号，采样频率，信号长度，心拍位置，房颤开始位置，房颤结束位置，类别等信息
        '''
        res = []

        for file in os.listdir(path):
            sample = {'name':'','sig':0,'fs':0,'len_sig':0,'beat_loc':0,'af_starts':0,'af_ends':0,'class_true':0}
            if file.endswith('.hea'):
                name = file.split('.')[0]
                sample['name'] = name
                sig,_,_ = self._load_data(os.path.join(path,name))
                if sig.shape[1] < sig.shape[0]:
                    sig = np.transpose(sig)
                sample['sig'] = sig[channel,:]
                ref = RefInfo(os.path.join(path,name))
                sample['fs'] = ref.fs
                sample['len_sig'] = ref.len_sig
                sample['beat_loc'] = ref.beat_loc
                sample['af_starts'] = ref.af_starts
                sample['af_ends'] = ref.af_ends
                sample['class_true'] = ref.class_true
                res.append(sample)
        print('文件读取完成')
        return res

    def _gen_sample(self,res,seed = 0,train_rate = 0.7,valid_rate = 0.1):
        random.seed(seed)
        index = list(range(105))
        random.shuffle(index)
        train_size = int(len(index) * train_rate)
        valid_size = int(len(index) * valid_rate)
        train_index = index[:train_size]
        valid_index = index[train_size:train_size + valid_size]
        test_index = index[train_size + valid_size:]
        train_samp = []
        valid_samp = []
        test_samp = []
        for samp in res:
            name = int(samp['name'].split('_')[1])
            if name in train_index:
                train_samp.append(samp)
            elif name in valid_index:
                valid_samp.append(samp)
            else:
                test_samp.append(samp)
        print('训练集样本：',len(train_index))
        print('验证集样本：',len(valid_index))
        print('测试集样本：',len(test_index))
        return train_samp,valid_samp,test_samp

    def _norm(self,x):
        return (x - np.mean(x))/np.std(x)

    def _gen_cnn_X_Y(self,res,n_samp = 10,n_rate = 1,af_rate = 1):
        '''
        Description:
            生成模型训练所需要的数据，将数据切割为5s一段的ECG片段
        Params:
            res: _get_signal返回的字典，包含了样本的信息
            n_samp: 从每段数据中采样的ECG片段数量
            n_rate: 正常ECG片段的比率
            af_rate: 房颤ECG片段的比率
        Return:
            res_X: 模型训练所需的数据
            res_Y: 模型训练所需的标签
        '''
        res_X = []
        res_Y = []
        [b,a] = butter(3,[0.5/100,40/100],'bandpass') # 巴特沃斯带通滤波器，通频带是0.5-40Hz，阶数是3阶
        for samp in res:
            class_true = int(samp['class_true']) # 获取当前受试者的分类标签
            sig = self._norm(filtfilt(b,a,samp['sig'])) # 对信号滤波后标准化
            fs = samp['fs']
            sig_len = len(sig)
            if sig_len < 2*n_samp*fs:
                continue # 如果信号长度太短，该受试者的数据舍去
            
            if class_true == 0: # 正常受试者
                square_wave = np.zeros(sig_len) # 因为最终的标签是正常，所以整段数据都认为是正常的
                nEN = sig_len//int(n_rate * n_samp) # 确定切割的步长，默认是采样10个ECG片段
                tmp_X,tmp_Y = self._data_enhance(sig,square_wave,5*fs,nEN) # 切割数据，理论上最终会返回10个ECG片段
                res_X.extend(tmp_X)
                res_Y.extend(tmp_Y)
            elif class_true == 1: # 持续性房颤受试者
                square_wave = np.ones(sig_len) # 因为最终的标签是持续性房颤，所以整段数据都认为是房颤的
                nEN = sig_len//int(af_rate * n_samp) # 确定切割的步长，默认是采样10个ECG片段
                tmp_X,tmp_Y = self._data_enhance(sig,square_wave,5*fs,nEN)
                res_X.extend(tmp_X)
                res_Y.extend(tmp_Y)
            else: # 阵发性房颤受试者
                af_start = samp['af_starts'] # 阵发性房颤的起始点，是一个QRS波的索引
                af_end = samp['af_ends'] # 阵发性房颤的结束点，是一个QRS波的索引
                beat_loc = samp['beat_loc'] # QRS波的位置
                square_wave = np.zeros(sig_len) # 先生成一个全是0的array，就是都是正常的
                for j in range(len(af_start)):
                    square_wave[int(beat_loc[int(af_start[j])]):int(beat_loc[int(af_end[j])])] = 1 # 然后在阵发性房颤的位置上标注1，认为该段是房颤
                tmp_X,tmp_Y = self._data_enhance(sig,square_wave,5*fs,fs) # 滑动窗口是5s，重叠4s的方式去采样
                AF_index = np.where(np.array(tmp_Y) == 1,True,False) # 找到返回的房颤ECG片段的索引
                Normal_index = np.where(np.array(tmp_Y) == 0,True,False)
                AF_X = np.array(tmp_X)[AF_index,:] # 从tmp_X中提取出相应的数据
                AF_Y = np.array(tmp_Y)[AF_index]
                N_X = np.array(tmp_X)[Normal_index,:] # 正常的ECG片段使用相同的放法提取
                N_Y = np.array(tmp_Y)[Normal_index]
                nAF = len(AF_Y)//n_samp # 这一步是为了得到n_samp个AF和N的ECG片段，从得到的AF_X, AF_Y, N_X, N_Y二次采样
                nN = len(N_Y)//n_samp
                if nAF == 0 or nN ==0:
                    continue
                for n in range(0,len(AF_Y),nAF): # 根据步长从AF_X和AF_Y得到所需的数据
                    res_X.append(AF_X[n,:])
                    res_Y.append(AF_Y[n])
                for n in range(0,len(N_Y),nN):
                    res_X.append(N_X[n,:])
                    res_Y.append(N_Y[n])

        return torch.FloatTensor(res_X),torch.LongTensor(res_Y)

    def stat(self):
        print('=======================')
        print('训练集标签1数量:',np.sum(np.where(np.array(self.train_Y) == 1,1,0)))
        print('训练集标签0数量:',np.sum(np.where(np.array(self.train_Y) == 0,1,0)))
        print('-----------------------')
        print('验证集标签1数量:',np.sum(np.where(np.array(self.valid_Y) == 1,1,0)))
        print('验证集标签0数量:',np.sum(np.where(np.array(self.valid_Y) == 0,1,0)))
        print('-----------------------')
        print('测试集标签1数量:',np.sum(np.where(np.array(self.test_Y) == 1,1,0)))
        print('测试集标签0数量:',np.sum(np.where(np.array(self.test_Y) == 0,1,0)))
        print('=======================')

    def _load_data(self,sample_path):
        sig, fields = wfdb.rdsamp(sample_path)
        length = len(sig)
        fs = fields['fs']

        return sig, length, fs

    def _data_enhance(self,sig,label,win_len,step):
        '''
        Description:
            从受试者的ECG数据中采样，切割成所需的信号长度
        Params:
            sig: 受试者的整段ECG信号
            label: 生成的与sig长度相同的标签
            win_len: 窗口长度，这里切割成了5s的数据
            step: 切割的步长
        Return:
            res_X: 切割好的数据
            res_Y: 对应的标签
        '''
        sig_len = len(sig)
        res_sig = []
        res_label = []
        for ii in range(0,sig_len-win_len,step):
            tmp = label[ii:ii+win_len] # label是一个长度与sig相同的array，生成方式看_gen_cnn_X_Y
            if np.sum(tmp) > len(tmp)//2: # 如果tmp中有超过一半是房颤
                res_label.append(1) # 那么这个ECG片段的标签就是房颤，否则就是正常
                res_sig.append(sig[ii:ii+win_len])
            else:
                res_label.append(0)
                res_sig.append(sig[ii:ii+win_len])
            
        return res_sig,res_label

    def save_RECORD(self,sample):
        test_record_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\test_record'
        with open(os.path.join(test_record_path,'RECORDs'),'w+') as f:
            for samp in sample:
                name = samp['name']
                f.write(name + '\n')
        print('测试索引文件记录完成')


class TrainAdapter(Data.Dataset):

    def __init__(self,base_adapter):
        super(TrainAdapter,self).__init__()
        self.train_X,self.train_Y = base_adapter.get_train_set()

    def __getitem__(self,index):
        return self.train_X[index],self.train_Y[index]

    def __len__(self):
        return len(self.train_Y)

class ValidAdapter(Data.Dataset):

    def __init__(self,base_adapter):
        super(ValidAdapter,self).__init__()
        self.valid_X,self.valid_Y = base_adapter.get_valid_set()

    def __getitem__(self,index):
        return self.valid_X[index],self.valid_Y[index]

    def __len__(self):
        return len(self.valid_Y)

class TestAdapter(Data.Dataset):

    def __init__(self,base_adapter):
        super(TestAdapter,self).__init__()
        self.test_X,self.test_Y = base_adapter.get_test_set()

    def __getitem__(self,index):
        return self.test_X[index],self.test_Y[index]

    def __len__(self):
        return len(self.test_Y)
    

class DataAdapter2(Data.Dataset):

    def __init__(self,X1,X2):
        super(DataAdapter2,self).__init__()
        self.X1 = torch.FloatTensor(X1)
        self.X2 = torch.FloatTensor(X2)

    def __getitem__(self, index):
        return self.X1[index,:],self.X2[index,:]

    def __len__(self):
        return len(self.X1)


if __name__ == '__main__':
    train_x = [torch.tensor([1, 2, 3, 4, 5, 6, 7]),
           torch.tensor([2, 3, 4, 5, 6, 7]),
           torch.tensor([3, 4, 5, 6, 7]),
           torch.tensor([4, 5, 6, 7]),
           torch.tensor([5, 6, 7]),
           torch.tensor([6, 7]),
           torch.tensor([7])]
    class Adapter(Data.Dataset):
        def __init__(self,x):
            super(Adapter,self).__init__()
            self.x = x
        def __getitem__(self, index):
            return self.x[index],self.x[index]
        def __len__(self):
            return len(self.x)
    adapter = Adapter(train_x)
    train_dataloader = Data.DataLoader(adapter, batch_size=2, collate_fn=collate_fn)
    for data, length, label in train_dataloader:
        print(data)
        print(length)
        print(label)