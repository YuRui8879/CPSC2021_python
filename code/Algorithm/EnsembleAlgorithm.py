import sys
sys.path.append(r'..\DataAdapter')
sys.path.append(r'..\Model')

import torch
import torch.nn as nn
import torch.optim as optim
from Model.Model import Model
from DataAdapter.EnsembleAdapter import DataAdapter,EnsembleDataAdapter
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
import time
import numpy as np
np.set_printoptions(suppress = True)
import os
import wfdb
from score_2021 import RefInfo
from scipy.signal import butter,filtfilt

class Algorithm():

    def __init__(self,parallel = True):
        super(Algorithm,self).__init__()
        self.save_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\CNN' # 存放训练好的模型的路径
        lead = 1
        if lead == 1:
            self.pretrain_model_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\pretrain\pretrain_model1.pt' # 预训练模型存放路径
        else:
            self.pretrain_model_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\pretrain\pretrain_model0.pt'
        data_path = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data' # cpsc数据库路径
        self.parallel = parallel
        self.batch_size = 512
        self.epochs = 1
        self.learning_rate = 0.0001
        self.patience = 10
        self.folds = 5

        res = self._get_signal(data_path,lead)
        X,Y = self._gen_cnn_X_Y(res,50,af_rate = 2)
        self.dataset = EnsembleDataAdapter(X,Y,self.folds,seed = 0)
        


    def train(self):
        for fold in range(self.folds):

            valid_X,valid_Y = self.dataset.get_fold_data()
            train_X,train_Y = self.dataset.get_res_data()

            train_set = DataAdapter(train_X, train_Y)
            valid_set = DataAdapter(valid_X, valid_Y)

            train_loader = Data.DataLoader(train_set,batch_size = self.batch_size,shuffle = True,num_workers = 0)
            valid_loader = Data.DataLoader(valid_set,batch_size = self.batch_size,shuffle = False,num_workers = 0)

            model = self._load_pretrained_mdoel(self.pretrain_model_path)
            if self.parallel:
                model = nn.DataParallel(model)
            model.cuda()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            early_stopping = EarlyStopping(self.save_path,self.patience, verbose=False)
            # clr = CosineAnnealingLR(optimizer,T_max = 32)
            clr = MultiStepLR(optimizer,[20,50],gamma=0.1)

            reg_loss = Regularization(model, 0.001)
            best_loss = 100

            for epoch in range(1,self.epochs + 1):
                start_time = time.time()
                train_res = self._cal_batch(train_loader,model,criterion,optimizer,reg_loss,'train')
                test_res = self._cal_batch(valid_loader,model,criterion,optimizer,reg_loss,'test')
                end_time = time.time()
                print('- Epoch: %d - Train_loss: %.5f - Train_mean_acc: %.5f - Val_loss: %.5f - Val_mean_acc: %5f - T_Time: %.3f' \
                    %(epoch,train_res['loss'],train_res['acc'],test_res['loss'],test_res['acc'],end_time - start_time))
                print('当前学习率：%f' %optimizer.state_dict()['param_groups'][0]['lr'])
                clr.step()
                early_stopping(test_res['loss'], model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 结束模型训练
                    break

            print('CNN Training Finished')
            self.dataset.step()


    def _cal_batch(self,loader,model,criterion,optimizer,reg_loss,types = 'train'):
        
        loss_list = []
        acc_list = []
        confusion_matrix = np.zeros((2,2))

        if types == 'train':
            model.train()
        else:
            model.eval()

        for i,data in enumerate(loader,0):

            inputs,labels = data[0].cuda(),data[1].cuda()
            outputs = model(inputs)
            _,pred = outputs.max(1)

            loss = criterion(outputs,labels) # 计算loss

            with torch.no_grad():
                C,acc = self._cal_acc(pred.cpu().numpy(),labels.cpu().numpy()) #计算各类指标
                confusion_matrix += C

            if reg_loss:
                loss += reg_loss(model)

            if types == 'train':
                optimizer.zero_grad() # 梯度清零
                loss.backward() #反向传播
                optimizer.step() #更新参数
        
            loss_list.append(loss.item())
            acc_list.append(acc)

        return {
            'loss':np.mean(loss_list),
            'acc':np.mean(acc_list),
            'confusion_matrix':confusion_matrix,
            'F1':self._cal_F1(confusion_matrix)
        }
    
    # 计算ACC和混淆矩阵
    def _cal_acc(self,pred,real):
        C = np.zeros((2,2))
        for i in range(len(pred)):
            C[pred[i],real[i]] += 1
        acc = np.sum(np.diag(C))/np.sum(C)
        return C,acc

    # 计算F1值
    def _cal_F1(self,C):
        pre = C[1,1] / (C[0,1] + C[1,1])
        rec = C[1,1] / (C[1,0] + C[1,1])
        if pre + rec == 0:
            F1 = 0
        else:
            F1 = 2 * pre * rec / (pre + rec)
        return F1

    # 测试函数
    def test(self):
        criterion = nn.CrossEntropyLoss()
        test_X,test_Y = self.dataset.get_test_set()
        test_set = DataAdapter(test_X, test_Y)
        test_loader = Data.DataLoader(test_set,batch_size = self.batch_size,shuffle = False,num_workers = 0)
        model = Model()
        if self.parallel:
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(self.save_path,'EnsembleCNN_model.pt')))
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        reg_loss = Regularization(model, 0.001)
        start_time = time.time()
        res = self._cal_batch(test_loader,model,criterion,optimizer,reg_loss,'test')
        end_time = time.time()
        print('T_T Time:',end_time - start_time)
        print('Test Loss:',res['loss'])
        print('Test acc:',res['acc'])
        print('Test F1:',res['F1'])
        print('Test confusion matrix')
        print(res['confusion_matrix'])

    def _load_pretrained_mdoel(self,model_path):
        model = Model()
        pretrained_dict = torch.load(model_path)
        pretrained_dict.pop('linear_unit.0.weight')
        pretrained_dict.pop('linear_unit.0.bias')
        pretrained_dict.pop('linear_unit.3.weight')
        pretrained_dict.pop('linear_unit.3.bias')
        pretrained_dict.pop('linear_unit.6.weight')
        pretrained_dict.pop('linear_unit.6.bias')
        model.load_state_dict(pretrained_dict, strict=False)
        return model

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

    def _load_data(self,sample_path):
        sig, fields = wfdb.rdsamp(sample_path)
        length = len(sig)
        fs = fields['fs']

        return sig, length, fs

    def _norm(self,x):
        return (x - np.mean(x))/np.std(x)

    def _gen_cnn_X_Y(self,res,n_samp = 10,n_rate = 1,af_rate = 1):
        res_X = []
        res_Y = []
        [b,a] = butter(3,[0.5/100,40/100],'bandpass')
        for samp in res:
            class_true = int(samp['class_true'])
            sig = self._norm(filtfilt(b,a,samp['sig']))
            fs = samp['fs']
            sig_len = len(sig)
            if sig_len < 2*n_samp*fs:
                continue
            
            if class_true == 0:
                square_wave = np.zeros(sig_len)
                nEN = sig_len//int(n_rate * n_samp)
                tmp_X,tmp_Y = self._data_enhance(sig,square_wave,5*fs,nEN)
                res_X.extend(tmp_X)
                res_Y.extend(tmp_Y)
            elif class_true == 1:
                square_wave = np.ones(sig_len)
                nEN = sig_len//int(af_rate * n_samp)
                tmp_X,tmp_Y = self._data_enhance(sig,square_wave,5*fs,nEN)
                res_X.extend(tmp_X)
                res_Y.extend(tmp_Y)
            else:
                af_start = samp['af_starts']
                af_end = samp['af_ends']
                beat_loc = samp['beat_loc']
                square_wave = np.zeros(sig_len)
                for j in range(len(af_start)):
                    square_wave[int(beat_loc[int(af_start[j])]):int(beat_loc[int(af_end[j])])] = 1
                tmp_X,tmp_Y = self._data_enhance(sig,square_wave,5*fs,fs)
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

    def _data_enhance(self,sig,label,win_len,step):
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

class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        print('save best model..')
        torch.save(model.state_dict(), os.path.join(self.save_path,'EnsembleCNN_model.pt'))
        self.val_loss_min = val_loss


class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            log.log("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)

 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss=0

        for name, w in weight_list:
            if p == 2 or p == 0:
                reg_loss += torch.sum(torch.pow(w, 2))
            else:
                reg_loss += torch.sum(torch.abs(w))
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")
