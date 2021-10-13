#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from DataAdapter import DataAdapter2
import torch.utils.data as Data
from model import CNN,RNN
import torch
from read_code import load_data
from scipy.signal import butter,filtfilt
from score_2021 import RefInfo
import matplotlib.pyplot as plt
from read_code import norm
from scipy import interpolate
from utils import p_t_qrs

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
    patience = 10
    flag = 0
    count = 0
    for i in range(len(X)-1):
        if X[i] == 1 and X[i+1] == 0:
            count = 0
            flag = 1
        if flag == 1:
            count += 1
        if X[i] == 0 and X[i+1] == 1:
            if count < patience:
                X[i-count:i] = 1
            flag = 0

    return X

def forward_backward_search(X,end_points,qrs_pos):
    segment = []
    res_end_points = []
    th = 10
    for i in range(len(qrs_pos)-1):
        segment.append(X[int(qrs_pos[i]):int(qrs_pos[i+1])])
    for (starts,ends) in end_points:
        tmp = []
        loc = find_segment_loc(starts, qrs_pos)
        corr_list = []
        for i in range(-th,th + 1):
            st = min(max(0,loc + i - 1),len(segment)-1)
            en = max(0,min(len(segment)-1,loc + i))
            corr_list.append(cal_corr(segment[en],segment[st]))
        idx = check_index(corr_list)
        if idx == -1:
            continue
        tmp.append(qrs_pos[loc + idx])

        loc = find_segment_loc(ends, qrs_pos)
        corr_list = []
        for i in range(-th,th + 1):
            st = min(max(0,loc + i - 1),len(segment)-1)
            en = max(0,min(len(segment)-1,loc + i))
            corr_list.append(cal_corr(segment[en],segment[st]))
        idx = check_index(corr_list)
        if idx != -1:
            tmp.append(qrs_pos[loc + idx])
            res_end_points.append(tmp)

    return res_end_points

def check_index(corr_index):
    corr_th = 0.5
    middle = len(corr_index)//2
    if corr_index[middle] < corr_th:
        return 0
    for i in range(1,len(corr_index)//2 + 1):
        if corr_index[middle - i] < corr_th:
            return -i
        if corr_index[middle + i] < corr_th:
            return i
    return -1

def cal_corr(seg,lseg):
    if len(seg) != len(lseg):
        if len(seg) > len(lseg):
            lseg = seg # 保证lseg的长度大于seg的长度
        xline = np.linspace(0,len(seg)-1,len(seg))
        new_xline = np.linspace(0,len(seg)-1,len(lseg))
        f = interpolate.interp1d(xline, seg, kind = 'cubic')
        y_new = f(new_xline) # 插值之后的seg
        if len(y_new) > len(lseg):
            y_new = y_new[:len(lseg)]
        elif len(y_new) < len(lseg):
            lseg = lseg[:len(y_new)]
        corr = np.corrcoef(y_new,lseg)[0,1]
    else:
        corr = np.corrcoef(seg,lseg)[0,1]
    return corr

def find_segment_loc(l,qrs_pos):
    for i in range(len(qrs_pos)):
        if l < qrs_pos[i]:
            return i - 1
    return 0

def find_start_end(X,fs,sig_len):
    res = []
    tmp = []
    for i in range(len(X)-1):
        if X[i] == 0 and X[i+1] == 1:
            if (i+2)*fs >= sig_len:
                break
            else:
                tmp.append((i+2)*fs)
        elif X[i] == 1 and X[i+1] == 0:
            if (i+2)*fs >= sig_len:
                tmp.append(sig_len-1)
                res.append(tmp)
                break
            else:
                tmp.append((i+2)*fs)
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

    cnn_path = r'.\model\CNN_best_model0.pt'
    cnn = CNN()
    cnn.load_state_dict(torch.load(cnn_path,map_location='cuda:0'))
    cnn.eval()
    cnn.to(device)

    cnn_path = r'.\model\CNN_best_model1.pt'
    cnn2 = CNN()
    cnn2.load_state_dict(torch.load(cnn_path,map_location='cuda:0'))
    cnn2.eval()
    cnn2.to(device)

    # rnn_path = r'.\model\RNN_best_model.pt'
    # rnn = RNN()
    # rnn.load_state_dict(torch.load(rnn_path,map_location='cuda:0'))
    # rnn.eval()
    # rnn.to(device)

    [b,a] = butter(3,[0.5/100,40/100],'bandpass')
    sig, _, fs = load_data(sample_path)
    qrs_pos = p_t_qrs(sig[:,1],fs)
    sig1 = norm(filtfilt(b,a,sig[:, 0]))
    sig2 = norm(filtfilt(b,a,sig[:, 1]))
    end_points = []

    batch_size = 64
    res_X,_ = move_windows(sig1,fs)
    res_X2,_ = move_windows(sig2,fs)
    test_set = DataAdapter2(res_X, res_X2)
    test_loader = Data.DataLoader(test_set,batch_size = batch_size,shuffle = False,num_workers = 0)

    res = np.zeros(len(res_X))
    idx = 0
    for i,data in enumerate(test_loader,0):
        inputs,inputs2 = data[0].to(device),data[1].to(device)
        cnn_outputs,fea1 = cnn(inputs)
        cnn2_outputs,fea2 = cnn2(inputs2)

        max_num1,pred1 = cnn_outputs.max(1)
        max_num2,pred2 = cnn2_outputs.max(1)
        
        # if len(pred1) < batch_size:
        pred = np.zeros(len(pred1))
        for j in range(len(pred1)):
            if pred1[j] == 0 and pred2[j] == 0:
                pred[j] = 0
            elif pred1[j] == 1 and pred2[j] == 1:
                pred[j] = 1
            else:
                if max_num1[j] > max_num2[j]:
                    pred[j] = pred1[j].cpu().numpy()
                else:
                    pred[j] = pred2[j].cpu().numpy()
        # else:
        #     rnn_fea = torch.cat((fea1,fea2),-1)
        #     rnn_fea = rnn_fea.view(1,rnn_fea.size(0),rnn_fea.size(1))
        #     pred = rnn(rnn_fea).squeeze()
        #     _,pred = pred.max(1)
        #     with torch.no_grad():
        #         pred = pred.cpu().numpy()

        with torch.no_grad():
            res[idx:idx + batch_size] = pred
        idx += batch_size
    
    res = np.squeeze(res)
    
    if np.sum(np.where(res == 0,1,0)) > len(res) * 0.95:
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
        # end_points = forward_backward_search(sig,end_points,qrs_pos)
        predict_label = 2
    
    cross_th =  cal_cross_th(res)
    if cross_th > 30:
        n_count = np.sum(np.where(res == 0,1,0))
        af_count = len(res) - n_count
        if n_count > af_count:
            end_points = []
            predict_label = 0
        else:
            tmp = []
            end_points = []
            tmp.append(0)
            tmp.append(len(sig)-1)
            end_points.append(tmp)
            predict_label = 1

    if end_points != []:
        res_points = []
        for (starts,ends) in end_points:
            if ends - starts > 5 * fs:
                res_points.append([starts,ends])
        end_points = res_points
        if end_points == []:
            n_count = np.sum(np.where(res == 0,1,0))
            af_count = len(res) - n_count
            if n_count > af_count:
                end_points = []
                predict_label = 0
            else:
                tmp = []
                end_points = []
                tmp.append(0)
                tmp.append(len(sig)-1)
                end_points.append(tmp)
                predict_label = 1
        

    if debug:
        pic_path = r'.\pic'
        norm_path = os.path.join(pic_path,'norm')
        if not os.path.exists(norm_path):
            os.makedirs(norm_path)
        af_path = os.path.join(pic_path,'af')
        if not os.path.exists(af_path):
            os.makedirs(af_path)
        paf_path = os.path.join(pic_path,'paf')
        if not os.path.exists(paf_path):
            os.makedirs(paf_path)

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
            plt.savefig(os.path.join(paf_path,name) + '.png')
            plt.clf()
        elif class_true == 1:
            if end_points != []:
                print(end_points)
                real_wave = np.ones(sig_len)
                predict_wave = np.zeros(sig_len)
                for (starts,ends) in end_points:
                    predict_wave[starts:ends] = 1
                plt.plot(real_wave,label = 'real')
                plt.plot(predict_wave,label = 'predict')
                plt.legend()
                name = sample_path.split('\\')[-1]
                plt.title(name)
                plt.savefig(os.path.join(af_path,name) + '.png')
                plt.clf()
        elif class_true == 0:
            if end_points != []:
                print(end_points)
                real_wave = np.zeros(sig_len)
                predict_wave = np.zeros(sig_len)
                for (starts,ends) in end_points:
                    predict_wave[starts:ends] = 1
                plt.plot(real_wave,label = 'real')
                plt.plot(predict_wave,label = 'predict')
                plt.legend()
                name = sample_path.split('\\')[-1]
                plt.title(name)
                plt.savefig(os.path.join(norm_path,name) + '.png')
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
