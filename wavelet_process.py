from scipy.signal import butter,filtfilt
from read_code import norm,load_data
from tools.ssqueezepy.synsq_cwt import synsq_cwt_fwd
import matplotlib.pyplot as plt
import numpy as np

sample_path = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data\data_0_1'
[b,a] = butter(3,[0.5/100,40/100],'bandpass')
sig, _, fs = load_data(sample_path)
sig1 = norm(filtfilt(b,a,sig[:, 0]))

OPTS = {'type': 'morlet',
        'difftype': 'direct',  # 'phase',
        'mu': np.pi / 1,
        # 'gamma': 1.4901e-08
        # , 'minfrequency': 0.1
        # ,'freqscale': 'linear'
        # , 'freqscale': 'log'
        }

wsst, fs, *_ = synsq_cwt_fwd(sig1[:1000],fs=200, nv=64, opts=OPTS)
print(wsst.shape)
im = plt.imshow(np.abs(wsst), cmap=plt.cm.jet)
plt.colorbar(im)
plt.show()

