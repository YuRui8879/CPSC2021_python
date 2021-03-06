import torch
import torch.nn as nn
from read_code import *
from DataAdapter import RNNAdapter
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
from model import RNN
import time
from batch import cal_rnn_batch
import torch.optim as optim
from EarlyStopping import EarlyStopping
from Regularization import Regularization

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data'
batch_size = 64
epochs = 80
learning_rate = 0.001
patience = 10

res = get_signal(data_path,0)
train_samp,valid_samp,test_samp = gen_sample(res)

train_set = RNNAdapter(train_samp)
valid_set = RNNAdapter(valid_samp)

train_loader = Data.DataLoader(train_set,batch_size = batch_size,shuffle = True,num_workers = 0)
valid_loader = Data.DataLoader(valid_set,batch_size = batch_size,shuffle = False,num_workers = 0)

model = RNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience, verbose=False)
# clr = CosineAnnealingLR(optimizer,T_max = 32)
clr = MultiStepLR(optimizer,[20,50],gamma=0.1)

reg_loss = Regularization(model, 0.001)
best_loss = 100

for epoch in range(1,epochs + 1):
    time_all=0
    start_time = time.time()
    # 训练模型
    train_res = cal_rnn_batch(train_loader, model, criterion, device, optimizer, reg_loss, True)
    # 验证集测试模型
    clr.step()
    valid_res = cal_rnn_batch(valid_loader, model, criterion, device, optimizer, reg_loss, False)
    time_all = time.time()-start_time
    # 打印训练及测试结果
    print('- Epoch: %d - Train_loss: %.5f - Train_mean_acc: %.5f - Train_F1: %.5f - Val_loss: %.5f - Val_mean_acc: %5f - Val_F1: %.5f - T_Time: %.3f' \
        %(epoch,train_res['loss'],train_res['acc'],train_res['F1'],valid_res['loss'],valid_res['acc'],valid_res['F1'],time_all))
    print('当前学习率：%f' %optimizer.state_dict()['param_groups'][0]['lr'])

    # 保存最优模型
    if valid_res['loss'] < best_loss:
        best_loss = valid_res['loss']
        print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), r'.\model\RNN_best_model.pt')

    early_stopping(valid_res['loss'], model)
        # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break

print('RNN Training Finished')

result = cal_rnn_batch(test_loader, model, criterion, device, optimizer, reg_loss, False)
print('confusion_matrix:',result['confusion_matrix'])
print('acc:',result['acc'])