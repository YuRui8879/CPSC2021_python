import torch.utils.data as Data
import torch
import random
import os

class RNNAdapter(Data.Dataset):

    def __init__(self,res):
        super(RNNAdapter,self).__init__()
        self.res = res
        b,a = butter(3,[0.5/100,40/100],'bandpass')
        self.b = b
        self.a = a

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
                sig,_,_ = load_data(os.path.join(path,name))
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
    
    def __getitem__(self,index):
        samp = self.res[index]
        sig = filtfilt(b,a,samp['sig'])
        class_true = int(samp['class_true'])
        if class_true == 0:
            label = np.zeros(step)
        elif class_true == 1:
            label = np.ones(step)
        else:
            af_start = samp['af_starts']
            af_end = samp['af_ends']
            beat_loc = samp['beat_loc']
            label = np.zeros(sig_len)
            for j in range(len(af_start)):
                label[int(beat_loc[int(af_start[j])]):int(beat_loc[int(af_end[j])])] = 1
        return torch.FloatTensor(sig),torch.FloatTensor(label)

    def __len__(self):
        return len(self.res)

def collate_fn(train_data):
    data = []
    label = []
    for items in train_data:
        data.append(items[0])
        label.append(items[1])
    data.sort(key=lambda da: len(da), reverse=True)
    label.sort(key=lambda da: len(da), reverse=True)
    length = [len(da) for da in data]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    label = torch.nn.utils.rnn.pad_sequence(label, batch_first=True, padding_value=0)
    return data, length, label