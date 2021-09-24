#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from DataAdapter import DataAdapter
import torch.utils.data as Data
from model import RCNN
import torch
from read_code import load_data
from scipy.signal import butter,filtfilt

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn

Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""
def move_windows(X,fs,win_n = 10):
    step = fs
    windows_length = win_n * fs
    res_X = []
    res_std = []
    if len(X) % windows_length > windows_length // 2:
        X = X.reshape(len(X),1)
        X = np.vstack((X,np.zeros((len(X) % windows_length,1))))
        X = np.squeeze(X)
    else:
        X = X[:-(len(X) % windows_length)]

    for i in range(0,len(X) - windows_length,step):
        tmp = X[i:i+windows_length]
        res_X.append(tmp)
        res_std.append(np.std(tmp))
        
    return res_X,res_std

def cal_cross_th(X):
    cross_th = 0
    for i in range(len(X)-1):
        if X[i] == 0 and X[i+1] == 1:
            cross_th += 1
        elif X[i] == 1 and X[i+1] == 0:
            cross_th += 1
    return cross_th

def plus_inhibition(X):
    patience = 5
    flag = 0
    count = 0
    for i in range(len(X)-1):
        if X[i] == 0 and X[i+1] == 1:
            count = 0
            flag = 1
        if flag == 1:
            count += 1
        if X[i] == 1 and X[i+1] == 0:
            if count < patience:
                X[i-count:i] = 0
            flag = 0
    return X

def find_start_end(X,fs,sig_len):
    res = []
    tmp = []
    for i in range(len(X)-1):
        if X[i] == 0 and X[i+1] == 1:
            tmp.append((i+5)*fs)
        elif X[i] == 1 and X[i+1] == 0:
            if (i+5)*fs > sig_len:
                tmp.append(sig_len-1)
            else:
                tmp.append((i+5)*fs)
            if len(tmp) == 2:
                res.append(tmp)
            tmp = []
    return res

def challenge_entry(sample_path):
    """
    This is a baseline method.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = r'.\RCNN_best_model.pt'
    model = RCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')

    sig, _, fs = load_data(sample_path)
    sig = filtfilt(b,a,sig[:, 1])
    end_points = []

    batch_size = 256
    res_X,res_std = move_windows(sig,fs)
    test_set = DataAdapter(res_X, np.zeros(len(res_X)))
    test_loader = Data.DataLoader(test_set,batch_size = batch_size,shuffle = False,num_workers = 0)

    res = np.zeros(len(res_X))
    idx = 0
    for i,data in enumerate(test_loader,0):
        inputs,labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        _,pred = outputs.max(1)

        with torch.no_grad():
            p = pred.cpu().numpy()
            res[idx:idx + batch_size] = p
            idx += batch_size

    cross_th =  cal_cross_th(res)
    if cross_th > 30:
        n_count = np.sum(np.where(res == 0,1,0))
        af_count = len(res) - n_count
        if n_count > af_count:
            end_points = []
        else:
            tmp = []
            tmp.append(0)
            tmp.append(len(sig)-1)
            end_points.append(tmp)
    
    res = plus_inhibition(res)
    if np.sum(np.where(res == 0,1,0)) > len(res) * 0.9:
        end_points = []
    elif np.sum(np.where(res == 1,1,0)) > len(res) * 0.9:
        tmp = []
        tmp.append(0)
        tmp.append(len(sig)-1)
        end_points.append(tmp)
    else:
        end_points = find_start_end(res,fs,len(sig))
    
    pred_dcit = {'predict_endpoints': end_points}
    
    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    # DATA_PATH = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data'
    # RESULT_PATH = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\out'
    # RECORDS_PATH = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\test_record'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    # test_set = open(os.path.join(RECORDS_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

