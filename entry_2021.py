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
from score_2021 import RefInfo
import matplotlib.pyplot as plt

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn

Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""
def move_windows(X,fs,win_n = 5):
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
            if (i+5)*fs >= sig_len:
                break
            else:
                tmp.append((i+5)*fs)
        elif X[i] == 1 and X[i+1] == 0:
            if (i+5)*fs >= sig_len:
                tmp.append(sig_len-1)
                res.append(tmp)
                break
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
    debug = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = []
    for fold in range(5):
        model_path = r'.\model\RCNN_best_model'+ str(fold) + '.pt'
        model.append(RCNN())
        model[fold].load_state_dict(torch.load(model_path,map_location='cuda:0'))
        model[fold].eval()
        model[fold].to(device)
    [b,a] = butter(3,[0.5/100,40/100],'bandpass')

    sig, _, fs = load_data(sample_path)
    sig = filtfilt(b,a,sig[:, 1])
    end_points = []

    batch_size = 512
    res_X,res_std = move_windows(sig,fs)
    test_set = DataAdapter(res_X, np.zeros(len(res_X)))
    test_loader = Data.DataLoader(test_set,batch_size = batch_size,shuffle = False,num_workers = 0)

    res = np.zeros((len(model),len(res_X)))
    idx = 0
    for i,data in enumerate(test_loader,0):
        inputs,labels = data[0].to(device),data[1].to(device)
        for fold in range(len(model)):
            outputs,_ = model[fold](inputs)
            _,pred = outputs.max(1)

            with torch.no_grad():
                p = pred.cpu().numpy()
                res[fold,idx:idx + batch_size] = p.reshape(1,len(p))

        idx += batch_size
    
    res = np.squeeze(np.where(np.sum(res,axis=0)>len(model)/2,1,0))

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
        predict_label = 0
    elif np.sum(np.where(res == 1,1,0)) > len(res) * 0.9:
        tmp = []
        tmp.append(0)
        tmp.append(len(sig)-1)
        end_points.append(tmp)
        predict_label = 1
    else:
        end_points = find_start_end(res,fs,len(sig))
        predict_label = 2
    
    if debug:
        pic_path = r'.\pic'
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
        ref = RefInfo(sample_path)
        beat_loc = ref.beat_loc
        af_start = ref.af_starts
        af_end = ref.af_ends
        class_true = int(ref.class_true)
        sig_len = ref.len_sig
        if class_true == 2:
            real_wave = np.zeros(sig_len)
            for i in range(len(af_start)):
                real_wave[int(beat_loc[int(af_start[i])]):int(beat_loc[int(af_end[i])])] = 1
            predict_wave = np.zeros(sig_len)
            for (starts,ends) in end_points:
                predict_wave[starts:ends] = 1
            plt.plot(real_wave,label = 'real')
            plt.plot(predict_wave,label = 'predict')
            plt.legend()
            name = sample_path.split('\\')[-1]
            plt.title(name)
            plt.savefig(os.path.join(pic_path,name) + '.png')
            plt.clf()
        file_path = os.path.join(pic_path,'confusion_matrix.npy')
        if os.path.exists(file_path):
            confusion_matrix = np.load(file_path)
            confusion_matrix[class_true,predict_label] += 1
            np.save(file_path,confusion_matrix)
        else:
            confusion_matrix = np.zeros((3,3))
            confusion_matrix[class_true,predict_label] += 1
            np.save(file_path,confusion_matrix)
        
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

