import sys
sys.path.append(r'..\DataAdapter')
sys.path.append(r'..\Model')

import torch
import torch.nn as nn
import torch.optim as optim
from Model.Model import Model
from DataAdapter.DataAdapter import BaseAdapter,TrainAdapter,ValidAdapter,TestAdapter
import torch.utils.data as Data
from torch.optim.lr_scheduler import MultiStepLR
import time
import numpy as np
np.set_printoptions(suppress = True)
import os

class Algorithm():

    def __init__(self,lead,parallel = True):
        super(Algorithm,self).__init__()
        self.save_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\CNN' # 保存训练好的模型文件的路径
        if lead == 0:
            pretrain_model_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\pretrain\pretrain_model0.pt' # 加载预训练模型文件的路径
        else:
            pretrain_model_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\pretrain\pretrain_model1.pt'
        data_path = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data' # 存放cpsc文件的路径
        self.parallel = parallel
        batch_size = 512 # 批大小
        learning_rate = 0.0001 # 学习率大小
        patience = 10 # 提前停止参数
        self.epochs = 80
        
        base_adapter = BaseAdapter(data_path,lead) # 训练集的数据生成器
        train_adapter = TrainAdapter(base_adapter)
        valid_adapter = ValidAdapter(base_adapter)
        test_adapter = TestAdapter(base_adapter)
        self.train_loader = Data.DataLoader(train_adapter,batch_size=batch_size,shuffle=True,num_workers=0)
        self.valid_loader = Data.DataLoader(valid_adapter,batch_size=batch_size,shuffle=False,num_workers=0)
        self.test_loader = Data.DataLoader(test_adapter,batch_size=batch_size,shuffle=False,num_workers=0)
        
        self.model = self._load_pretrained_mdoel(pretrain_model_path)
        if parallel:
            self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.criterion = nn.CrossEntropyLoss() # 使用交叉熵作为损失函数
        self.optimizer = optim.Adam(self.model.parameters(),lr = learning_rate) # Adam算法作为优化器
        self.early_stopping = EarlyStopping(self.save_path,patience,verbose = False)

        self.clr = MultiStepLR(self.optimizer,[20,50],gamma=0.1) # 动态学习率
        self.reg_loss = Regularization(self.model,0.001) # L2正则化

    def train(self):
        for epoch in range(1,self.epochs + 1):
            start_time = time.time()
            train_res = self._cal_batch(self.train_loader,self.model,self.criterion,self.optimizer,self.reg_loss,'train')
            test_res = self._cal_batch(self.valid_loader,self.model,self.criterion,self.optimizer,self.reg_loss,'test')
            end_time = time.time()
            print('- Epoch: %d - Train_loss: %.5f - Train_mean_acc: %.5f - Val_loss: %.5f - Val_mean_acc: %5f - T_Time: %.3f' \
                %(epoch,train_res['loss'],train_res['acc'],test_res['loss'],test_res['acc'],end_time - start_time))
            print('当前学习率：%f' %self.optimizer.state_dict()['param_groups'][0]['lr'])
            self.clr.step()
            self.early_stopping(test_res['loss'], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break
        print('Train finished...')

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
        model = Model()
        if self.parallel:
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(self.save_path,'CNN_model.pt')))
        model.cuda()
        start_time = time.time()
        res = self._cal_batch(self.test_loader,model,self.criterion,self.optimizer,self.reg_loss,'test')
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
        torch.save(model.state_dict(), os.path.join(self.save_path,'CNN_model.pt'))
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
