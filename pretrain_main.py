from read_code import read_pretrain_data,gen_pretrain_X_Y
import torch
import torch.nn as nn
from DataAdapter import DataAdapter
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
from model import CNN
import time
from batch import cal_cnn_batch
import torch.optim as optim
from EarlyStopping import EarlyStopping
from Regularization import Regularization

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = r'C:\Users\Public\Database\PhysioNet2021\physionet_database'
res = read_pretrain_data(data_path,1)
X,Y = gen_pretrain_X_Y(res)

batch_size = 128
epochs = 80
learning_rate = 0.0001
patience = 10
train_rate = 0.8

train_size = int(len(X) * train_rate)
train_X = X[:train_size]
train_Y = Y[:train_size]
test_X = X[train_size:]
test_Y = Y[train_size:]

train_set = DataAdapter(train_X, train_Y)
test_set = DataAdapter(test_X, test_Y)

train_loader = Data.DataLoader(train_set,batch_size = batch_size,shuffle = True,num_workers = 0)
test_loader = Data.DataLoader(test_set,batch_size = batch_size,shuffle = False,num_workers = 0)

model = CNN()
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
    train_res = cal_cnn_batch(train_loader, model, criterion, device, optimizer, reg_loss, True)
    # 验证集测试模型
    clr.step()
    valid_res = cal_cnn_batch(test_loader, model, criterion, device, optimizer, reg_loss, False)
    time_all = time.time()-start_time
    # 打印训练及测试结果
    print('- Epoch: %d - Train_loss: %.5f - Train_mean_acc: %.5f - Train_F1: %.5f - Val_loss: %.5f - Val_mean_acc: %5f - Val_F1: %.5f - T_Time: %.3f' \
        %(epoch,train_res['loss'],train_res['acc'],train_res['F1'],valid_res['loss'],valid_res['acc'],valid_res['F1'],time_all))
    print('当前学习率：%f' %optimizer.state_dict()['param_groups'][0]['lr'])

    # 保存最优模型
    if valid_res['loss'] < best_loss:
        best_loss = valid_res['loss']
        print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), r'.\model\pretrain_model2.pt')

    early_stopping(valid_res['loss'], model)
        # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break

print('CNN Training Finished')