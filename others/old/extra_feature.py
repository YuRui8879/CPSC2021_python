from utils import p_t_qrs,comp_cosEn
import numpy as np

# 该函数用于计算RR间期相关的特征
def extract_featrue(sig,fs,step):
    rr_seg = split_sig(sig,fs,step)
    res_fea = np.zeros((len(rr_seg),15))
    flag = 0
    for idx in range(len(rr_seg)):
        if rr_seg[idx] != []:
            rr_interval = np.diff(rr_seg[idx])
            rr_interval = remove_outline(rr_interval)
            mean_rr_ = mean_rr(rr_interval)
            max_rr_ = max_rr(rr_interval)
            min_rr_ = min_rr(rr_interval)
            std_rr_ = std_rr(rr_interval)
            diff_std_rr_ = diff_std_rr(rr_interval)
            rmssd_rr_ = rmssd_rr(rr_interval)
            diff_median_rr_ = diff_median_rr(rr_interval)
            slow_rr_ = slow_rr(rr_interval,fs)
            fast_rr_ = fast_rr(rr_interval,fs)
            cvcd_rr_ = cvcd_rr(rr_interval)
            cvnni_rr_ = cvnni_rr(rr_interval)
            mean_hr_ = mean_hr(rr_interval,fs)
            max_hr_ = max_hr(rr_interval,fs)
            min_hr_ = min_hr(rr_interval,fs)
            std_hr_ = std_hr(rr_interval,fs)
            feature = np.array([mean_rr_, max_rr_, min_rr_, std_rr_, diff_std_rr_, rmssd_rr_, diff_median_rr_, slow_rr_, fast_rr_,
                                cvcd_rr_, cvnni_rr_, mean_hr_, max_hr_, min_hr_, std_hr_])
        else:
            feature = np.zeros((1,15))
            flag = 1

        res_fea[idx,:] = feature
        
    return res_fea,flag

def remove_outline(rr_interval):
    th = 600
    res_rr_interval = []
    for rr in rr_interval:
        if rr <= th:
            res_rr_interval.append(rr)
    return res_rr_interval


def split_sig(sig,fs,step,n = 30):
    res_rr = []
    qrs_pos = p_t_qrs(sig,fs)
    sig_len = len(sig)
    step = step
    win_len = 5*fs
    for ii in range(0,sig_len-win_len,step):
        left_cursor = max(0,ii - n * fs)
        right_cursor = min(sig_len, ii + n * fs)
        left_th = find_th(qrs_pos, left_cursor)
        right_th = find_th(qrs_pos, right_cursor)
        res_rr.append(qrs_pos[left_th:right_th])
    return res_rr

def find_th(pos,cursor):
    index = np.where(np.array(pos) > cursor,1,0)
    res = len(index)
    for i in range(len(index)):
        if index[i] == 1:
            res = i
            break
    return res

def mean_rr(rr_interval):
    return np.mean(rr_interval)

def max_rr(rr_interval):
    return np.max(rr_interval)

def min_rr(rr_interval):
    return np.min(rr_interval)

def std_rr(rr_interval):
    return np.std(rr_interval)

def diff_std_rr(rr_interval):
    rr = np.diff(rr_interval)
    return np.std(rr)

def rmssd_rr(rr_interval):
    rr = np.diff(rr_interval)
    rr_2 = np.power(rr,2)
    return np.sqrt(np.mean(rr_2))

def diff_median_rr(rr_interval):
    rr = np.diff(rr_interval)
    return np.abs(np.median(rr))

def slow_rr(rr_interval,fs):
    th = fs * 0.6
    lr = np.sum(np.where(np.array(rr_interval) < th,1,0))
    return lr/len(rr_interval)

def fast_rr(rr_interval,fs):
    th = fs
    fr = np.sum(np.where(np.array(rr_interval) > th,1,0))
    return fr/len(rr_interval)

def cvcd_rr(rr_interval):
    rmssd = rmssd_rr(rr_interval)
    mean_rr_ = mean_rr(rr_interval)
    return rmssd/mean_rr_

def cvnni_rr(rr_interval):
    sdnn = std_rr(rr_interval)
    mean_rr_ = mean_rr(rr_interval)
    return sdnn/mean_rr_

def mean_hr(rr_interval,fs):
    hr = [60*fs/i for i in rr_interval]
    return np.mean(hr)

def max_hr(rr_interval,fs):
    hr = [60*fs/i for i in rr_interval]
    return np.max(hr)

def min_hr(rr_interval,fs):
    hr = [60*fs/i for i in rr_interval]
    return np.min(hr)

def std_hr(rr_interval,fs):
    hr = [60*fs/i for i in rr_interval]
    return np.std(hr)

if __name__ == '__main__':
    from read_code import load_data,norm
    path = r'C:\Users\yurui\Desktop\item\cpsc\data\training_I\data_10_10'
    sig, length, fs = load_data(path)
    qrs_pos = p_t_qrs(norm(sig[:,1]),fs)
    print(len(sig[:,1]))
    print(len(qrs_pos))
    # extract_featrue(sig[:,1],200,57000)