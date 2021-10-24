import torch
import torch.nn as nn
from read_code import *
from DataAdapter import DataAdapter
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
from model import CNN
import time
from batch import cal_cnn_batch
import torch.optim as optim
from EarlyStopping import EarlyStopping
from Regularization import Regularization
from Dataset import AssembleDataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = r'C:\Users\yurui\Desktop\item\cpsc\data\all_data'
pretrain_model_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\model\pretrain_model1.pt'
batch_size = 512
epochs = 80
learning_rate = 0.0001
patience = 10
folds = 5
lead = 1

res = get_signal(data_path,lead)
X,Y = gen_cnn_X_Y(res,50,af_rate = 2)
dataset = AssembleDataset(X,Y,folds,seed = 0)

for fold in range(folds):

    valid_X,valid_Y = dataset.get_fold_data()
    train_X,train_Y = dataset.get_res_data()

    train_set = DataAdapter(train_X, train_Y)
    valid_set = DataAdapter(valid_X, valid_Y)

    train_loader = Data.DataLoader(train_set,batch_size = batch_size,shuffle = True,num_workers = 0)
    valid_loader = Data.DataLoader(valid_set,batch_size = batch_size,shuffle = False,num_workers = 0)

    model = load_pretrained_mdoel(pretrain_model_path)
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
        valid_res = cal_cnn_batch(valid_loader, model, criterion, device, optimizer, reg_loss, False)
        time_all = time.time()-start_time
        # 打印训练及测试结果
        print('- Epoch: %d - Train_loss: %.5f - Train_mean_acc: %.5f - Train_F1: %.5f - Val_loss: %.5f - Val_mean_acc: %5f - Val_F1: %.5f - T_Time: %.3f' \
            %(epoch,train_res['loss'],train_res['acc'],train_res['F1'],valid_res['loss'],valid_res['acc'],valid_res['F1'],time_all))
        print('当前学习率：%f' %optimizer.state_dict()['param_groups'][0]['lr'])

        # 保存最优模型
        if valid_res['loss'] < best_loss:
            best_loss = valid_res['loss']
            print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), r'.\model\CNN_best_model' + str(lead) + '_' + str(fold) +'.pt')

        early_stopping(valid_res['loss'], model)
            # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break

    print('CNN Training Finished')
    dataset.step()

test_X,test_Y = dataset.get_test_set()
test_set = DataAdapter(test_X, test_Y)
test_loader = Data.DataLoader(test_set,batch_size = batch_size,shuffle = False,num_workers = 0)

result = cal_cnn_batch(test_loader, model, criterion, device, optimizer, reg_loss, False)
print('confusion_matrix:',result['confusion_matrix'])
print('acc:',result['acc'])

test_record_path = r'C:\Users\yurui\Desktop\item\cpsc\code\pretrain\test_record'
with open(os.path.join(test_record_path,'RECORDs'),'w') as f:
    for samp in test_samp:
        name = samp['name']
        f.write(name + '\n')
print('测试索引文件记录完成')