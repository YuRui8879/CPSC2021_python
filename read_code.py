import os
import numpy as np
from score_2021 import RefInfo
from scipy.signal import filtfilt,butter
import wfdb
import random
from model import RCNN
import torch
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d

def norm(x):
    min_ = np.min(x)
    max_ = np.max(x)
    x = (x - min_)/(max_ - min_)
    return x

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

def gen_ensemble_samp(res,n_part = 5,fold = 0,seed = 0, test_rate = 0.2):
    random.seed(seed)
    index = list(range(105))
    random.shuffle(index)
    train_size = int(len(index) * (1 - test_rate))
    test_size = train_size // n_part
    train_index = index[:train_size]
    test_index = index[train_size:]
    valid_index = train_index[fold * test_size:(fold + 1) * test_size]
    train_index = list(set(train_index) - set(valid_index))
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
    return train_samp,valid_samp,test_samp

def data_enhance(sig,label,win_len,step):
    sig_len = len(sig)
    res_sig = []
    res_label = []
    for ii in range(0,sig_len-win_len,step):
        tmp = label[ii:ii+win_len]
        if np.sum(tmp) > len(tmp)//2:
            res_label.append(1)
            res_sig.append(sig[ii:ii+win_len])
        else:
            res_label.append(0)
            res_sig.append(sig[ii:ii+win_len])
        
    return res_sig,res_label

def gen_rnn_data(res,time_step = 64):
    res_X = []
    res_Y = []
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
    n_count = 0
    af_count = 0
    for samp in res:
        class_true = int(samp['class_true'])
        sig = norm(filtfilt(b,a,samp['sig']))
        fs = samp['fs']
        sig_len = len(sig)
        step = (time_step + 5)*fs
        if sig_len < step:
            continue
        if class_true == 0 or class_true == 1:
            if class_true == 0:
                label = np.zeros(step)
            else:
                label = np.ones(step)
            cursor = 0
            while cursor + step < sig_len:
                tmp = sig[cursor:cursor + step]
                X,Y = data_enhance(tmp,label,5*fs,fs)
                res_X.append(X)
                res_Y.append(Y)
                if np.sum(Y) == 0:
                    n_count += time_step
                else:
                    af_count += time_step
                if cursor % step > 4:
                    break
                cursor += step
        else:
            af_start = samp['af_starts']
            af_end = samp['af_ends']
            beat_loc = samp['beat_loc']
            label = np.zeros(sig_len)
            for j in range(len(af_start)):
                label[int(beat_loc[int(af_start[j])]):int(beat_loc[int(af_end[j])])] = 1
            cursor = 0
            while cursor + step < sig_len:
                tmp = sig[cursor:cursor + step]
                X,Y = data_enhance(tmp,label,5*fs,fs)
                res_X.append(X)
                res_Y.append(Y)
                tmp_count = np.sum(np.where(np.array(Y) == 1,1,0))
                n_count += (len(Y) - tmp_count)
                af_count += tmp_count
                cursor += step
    print('房颤标签数量：',af_count)
    print('正常标签数量：',n_count)
    return res_X,res_Y

def get_cnn_featrue(X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RCNN()
    model.load_state_dict(torch.load(r'.\model\RCNN_best_model.pt',map_location='cuda:0'))
    model.eval()
    model.to(device)
    res_X = []
    for x in X:
        with torch.no_grad():
            x = torch.FloatTensor(x).to(device)
            _,y = model(x)
            res_X.append(y.cpu().numpy())
    return res_X


def gen_cnn_X_Y(res,n_samp = 10,n_rate = 1,af_rate = 1):
    res_X = []
    res_Y = []
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
    for samp in res:
        class_true = int(samp['class_true'])
        sig = norm(filtfilt(b,a,samp['sig']))
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

def read_pretrain_data(data_path):
    sample = []
    for path in os.listdir(data_path):
        if path == 'WFDB_CPSC2018' or path == 'WFDB_CPSC2018_2' or path == 'WFDB_Ga' or path == 'WFDB_PTBXL' or path == 'WFDB_ChapmanShaoxing':
            new_path = os.path.join(data_path,path)
            for p in os.listdir(new_path):
                if p.endswith('.hea'):
                    tmp = {}
                    data_name,fs,adc,sig_len,label,num_leads,baseline = read_physionet_header(os.path.join(new_path,p))
                    if label == -1:
                        continue
                    data = loadmat(os.path.join(new_path,data_name))['val']
                    tmp['data'] = (data[1,:] - baseline)/adc
                    tmp['label'] = label
                    tmp['fs'] = fs
                    sample.append(tmp)           

    return sample

def read_physionet_header(file_path):
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
    return data_name,float(fs),float(adc),int(sig_len),label,num_leads,baseline

def gen_pretrain_X_Y(res,seed = 0):
    random.seed(seed)
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
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
            data = resample(data,fs,200)
        data = norm(filtfilt(b, a, data))
        if len(data) < win_len:
            continue
        if label == 0:
            rd = random.randint(0,len(data)-win_len)
            res_X.append(data[rd:rd+win_len])
            res_Y.append(0)
            n_count += 1
        else:
            cursor = 0
            while cursor + win_len < len(data):
                res_X.append(data[cursor:cursor + win_len])
                res_Y.append(1)
                cursor += win_len
                af_count += 1

    print('af_count:',af_count)
    print('n_count:',n_count)
    return res_X,res_Y

def resample(x,ori_fs,dis_fs):
    f = interp1d(np.arange(len(x)),x,kind='cubic')
    xnew = np.arange(0,len(x)-1,ori_fs/dis_fs)
    ynew = f(xnew)
    return ynew

def load_pretrained_mdoel(model_path):
    model = RCNN()
    pretrained_dict = torch.load(model_path)
    pretrained_dict.pop('linear_unit.0.weight')
    pretrained_dict.pop('linear_unit.0.bias')
    pretrained_dict.pop('linear_unit.3.weight')
    pretrained_dict.pop('linear_unit.3.bias')
    model.load_state_dict(pretrained_dict, strict=False)
    return model
    

if __name__ == '__main__':
    model_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\pretrain_model.pt'
    load_pretrained_mdoel(model_path)