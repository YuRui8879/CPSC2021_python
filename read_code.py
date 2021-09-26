import os
import numpy as np
from score_2021 import RefInfo
from scipy.signal import filtfilt,butter
import wfdb
import random

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def get_signal(path):
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
            sample['sig'] = sig[1,:]
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

def select_data(res,record_path):
    tmp_set = open(record_path, 'r').read().splitlines()
    dataset = []
    for samp in res:
        name = samp['name']
        if name in tmp_set:
            dataset.append(samp)
    return dataset

def gen_sample(res,seed = 0,train_rate = 0.7,valid_rate = 0.1):
    random.seed(seed)
    index = list(range(len(res)))
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

def data_enhance(sig,label,win_len,step):
    sig_len = len(sig)
    res_sig = []
    res_label = []
    last_std = np.std(sig[:win_len])
    for ii in range(0,sig_len-win_len,step):
        # if np.std(sig[ii:ii+win_len])>3*last_std:
        #     continue
        tmp = label[ii:ii+win_len]
        if np.sum(tmp) > len(tmp)//2:
            res_label.append(1)
            res_sig.append(sig[ii:ii+win_len])
            last_std = np.std(sig[ii:ii+win_len])
        else:
            res_label.append(0)
            res_sig.append(sig[ii:ii+win_len])
            last_std = np.std(sig[ii:ii+win_len])
        
    return res_sig,res_label

def gen_X_Y(res,n_samp = 10,n_rate = 1,af_rate = 1):
    res_X = []
    res_Y = []
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
    for samp in res:
        class_true = int(samp['class_true'])
        sig = filtfilt(b,a,samp['sig'])
        fs = samp['fs']
        sig_len = len(sig)
        if sig_len < 2*n_samp*fs:
            continue
        
        if class_true == 0:
            square_wave = np.zeros(sig_len)
            nEN = sig_len//int(n_rate * n_samp)
            tmp_X,tmp_Y = data_enhance(sig,square_wave,5*fs,nEN)
            res_X.extend(tmp_X)
            res_Y.extend(tmp_Y)
        elif class_true == 1:
            square_wave = np.ones(sig_len)
            nEN = sig_len//int(af_rate * n_samp)
            tmp_X,tmp_Y = data_enhance(sig,square_wave,5*fs,nEN)
            res_X.extend(tmp_X)
            res_Y.extend(tmp_Y)
        else:
            af_start = samp['af_starts']
            af_end = samp['af_ends']
            beat_loc = samp['beat_loc']
            square_wave = np.zeros(sig_len)
            for j in range(len(af_start)):
                square_wave[int(beat_loc[int(af_start[j])]):int(beat_loc[int(af_end[j])])] = 1
            tmp_X,tmp_Y = data_enhance(sig,square_wave,5*fs,fs)
            AF_index = np.where(np.array(tmp_Y) == 1,True,False)
            Normal_index = np.where(np.array(tmp_Y) == 0,True,False)
            AF_X = np.array(tmp_X)[AF_index,:]
            AF_Y = np.array(tmp_Y)[AF_index]
            N_X = np.array(tmp_X)[Normal_index,:]
            N_Y = np.array(tmp_Y)[Normal_index]
            nAF = len(AF_Y)//n_samp
            nN = len(N_Y)//n_samp
            if nAF == 0 or nN ==0:
                continue
            for n in range(0,len(AF_Y),nAF):
                res_X.append(AF_X[n,:])
                res_Y.append(AF_Y[n])
            for n in range(0,len(N_Y),nN):
                res_X.append(N_X[n,:])
                res_Y.append(N_Y[n])

    return res_X,res_Y

if __name__ == '__main__':
    data_path = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data'
    res = get_signal(data_path)
    train_samp,valid_samp,test_samp = gen_sample(res)
    train_X,train_Y = gen_X_Y(train_samp,af_rate = 1.5)
    valid_X,valid_Y = gen_X_Y(valid_samp,n_rate = 2)
    test_X,test_Y = gen_X_Y(test_samp,af_rate = 2)
    print(np.sum(np.where(np.array(train_Y)==1,1,0)))
    print(np.sum(np.where(np.array(train_Y)==0,1,0)))
    print(np.sum(np.where(np.array(valid_Y)==1,1,0)))
    print(np.sum(np.where(np.array(valid_Y)==0,1,0)))
    print(np.sum(np.where(np.array(test_Y)==1,1,0)))
    print(np.sum(np.where(np.array(test_Y)==0,1,0)))