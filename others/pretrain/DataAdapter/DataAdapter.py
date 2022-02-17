import torch.utils.data as Data
import torch
import random
import os
from scipy.signal import butter,filtfilt
from scipy.io import loadmat


class DataAdapter(Data.Dataset):

    def __init__(self,data_path,n_lead = 1,seed = 0,mode = 0):
        # data_path说明
        # 该文件夹需要包含WFDB_CPSC2018，WFDB_CPSC2018_2，WFDB_Ga，WFDB_PTBXL，WFDB_ChapmanShaoxing中的
        # 一个或多个子文件夹，子文件夹中需要包含.mat数据文件及.hea头文件
        super(DataAdapter,self).__init__()
        sample = self._read_pretrain_data(data_path,n_lead)
        self.mode = mode # 用于控制生成训练集，验证集还是测试集
        self.seed = seed
        self.train_sample,self.test_sample = self._split_data(sample,0.8,seed)


    # 读取physionet数据
    def _read_pretrain_data(self,data_path,n_lead = 1):
        sample = []
        for path in os.listdir(data_path):
            if path == 'WFDB_CPSC2018' or path == 'WFDB_CPSC2018_2' or path == 'WFDB_Ga' or path == 'WFDB_PTBXL' or path == 'WFDB_ChapmanShaoxing':
                new_path = os.path.join(data_path,path)
                for p in os.listdir(new_path):
                    if p.endswith('.hea'):
                        tmp = {}
                        data_name,fs,adc,sig_len,label,num_leads,baseline = self._read_physionet_header(os.path.join(new_path,p))
                        if label == -1:
                            continue
                        data = loadmat(os.path.join(new_path,data_name))['val']
                        tmp['data'] = (data[n_lead,:] - baseline)/adc
                        tmp['label'] = label
                        tmp['fs'] = fs
                        sample.append(tmp)           

        return sample

    # 读取physionet数据库头文件
    def _read_physionet_header(self,file_path):
        with open(file_path) as f:
            context = f.readlines()
        label = -1
        for idx in range(len(context)):
            line = context[idx].strip().split(' ')
            if idx == 0:
                num_leads = line[1]
                fs = line[2]
                sig_len = line[3]
            elif idx == 1:
                data_name = line[0]
                adc = line[2].split('/')[0]
                baseline = line[4]
            elif idx == 15:
                tag = line[-1].split(',')
                if '164889003' in tag:
                    label = 1
                elif '426783006' in tag:
                    label = 0
                else:
                    label = -1
        return data_name,float(fs),float(adc),int(sig_len),label,num_leads,float(baseline)

    # 生成成对的可训练的数据X与标签Y
    def _gen_pretrain_X_Y(self,res,seed = 0):
        random.seed(seed)
        [b,a] = butter(3,[0.5/100,40/100],'bandpass') # 0.5~40Hz的3阶巴特沃斯滤波器
        win_len = 1000
        res_X = []
        res_Y = []
        af_count = 0
        n_count = 0
        for samp in res:
            fs = samp['fs']
            data = samp['data']
            label = samp['label']
            if fs != 200:
                data = self._resample(data,fs,200) # 对于采样频率不是200Hz的数据进行重采样
            data = self._norm(filtfilt(b, a, data))
            if len(data) < win_len:
                continue
            if label == 0: # 对于标签为N的数据，随机抽取5s的数据段
                rd = random.randint(0,len(data)-win_len)
                res_X.append(data[rd:rd+win_len])
                res_Y.append(0)
                n_count += 1
            else: # 如果标签为AF，则循环分割数据段，以解决数据不平衡的问题
                cursor = 0
                step = win_len
                while cursor + win_len < len(data):
                    res_X.append(data[cursor:cursor + win_len])
                    res_Y.append(1)
                    cursor += step
                    af_count += 1

        print('af_count:',af_count)
        print('n_count:',n_count)
        return res_X,res_Y

    # 重采样函数
    def _resample(self,x,ori_fs,dis_fs):
        f = interp1d(np.arange(len(x)),x,kind='cubic')
        xnew = np.arange(0,len(x)-1,ori_fs/dis_fs)
        ynew = f(xnew)
        return ynew

    # 归一化函数
    def _norm(self,x):
        min_ = np.min(x)
        max_ = np.max(x)
        x = (x - min_)/(max_ - min_)
        return x
        
    # 用于分割训练集和测试集
    def _split_data(self,sample,train_rate = 0.8,seed = 0):
        random.seed(seed)
        random.shuffle(sample)
        train_size = int(len(sample) * train_rate)
        train_sample = sample[:train_size]
        test_sample = sample[train_size:]
        return train_sample,test_sample

    def __getitem__(self,index):
        if self.mode == 0:
            X,Y = self._gen_pretrain_X_Y(self.train_sample,self.seed)
        else:
            X,Y = self._gen_pretrain_X_Y(self.test_sample,self.seed)
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

        return X[index,:],Y[index]

    def __len__(self):
        if self.mode == 0:
            length = len(self.train_sample)
        else:
            length = len(self.test_sample)
        return length