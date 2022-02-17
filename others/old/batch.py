import numpy as np
import torch
import torch.nn as nn

def cal_cnn_batch(train_loader,model,criterion,device,optimizer = None,reg_loss = None,is_train = True):

    model.train()
    loss_list = []
    acc_list = []
    confusion_matrix = np.zeros((2,2))

    for i,data in enumerate(train_loader,0):

        inputs,labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        _,pred = outputs.max(1)

        loss = criterion(outputs,labels)

        with torch.no_grad():
            num_correct = (pred == labels).sum().item()
            acc = num_correct/len(labels)
            confusion_matrix += cal_confusion_matrix(pred.cpu().numpy(),labels.cpu().numpy())

        if reg_loss:
            loss += reg_loss(model)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        loss_list.append(loss.item())
        acc_list.append(acc)

    return {
        'loss':np.mean(loss_list),
        'acc':np.mean(acc_list),
        'confusion_matrix':confusion_matrix,
        'F1':cal_F1(confusion_matrix)
    }

def cal_rnn_batch(train_loader,model,criterion,device,optimizer = None,reg_loss = None,is_train = True):

    model.train()
    loss_list = []
    acc_list = []
    confusion_matrix = np.zeros((2,2))

    for i,data in enumerate(train_loader,0):

        inputs,length,labels = data[0],data[1],data[2]

        outputs = model(inputs)
        outputs = outputs.view(-1,2)
        _,pred = outputs.max(1)
        labels = labels.view(-1)
        loss = criterion(outputs,labels)

        with torch.no_grad():
            num_correct = (pred == labels).sum().item()
            acc = num_correct/len(labels)
            confusion_matrix += cal_confusion_matrix(pred.cpu().numpy(),labels.cpu().numpy())

        if reg_loss:
            loss += reg_loss(model)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        loss_list.append(loss.item())
        acc_list.append(acc)

    return {
        'loss':np.mean(loss_list),
        'acc':np.mean(acc_list),
        'confusion_matrix':confusion_matrix,
        'F1':cal_F1(confusion_matrix)
    }

def cal_confusion_matrix(pred,label):
    confusion_matrix = np.zeros((2,2))
    for ii in range(len(pred)):
        if pred[ii] == 0 and label[ii] == 0:
            confusion_matrix[0,0] += 1
        elif pred[ii] == 0 and label[ii] == 1:
            confusion_matrix[1,0] += 1
        elif pred[ii] == 1 and label[ii] == 0:
            confusion_matrix[0,1] += 1
        elif pred[ii] == 1 and label[ii] == 1:
            confusion_matrix[1,1] += 1
    return confusion_matrix

def cal_F1(C):
    pre = C[1,1] / (C[0,1] + C[1,1])
    rec = C[1,1] / (C[1,0] + C[1,1])
    if pre + rec == 0:
        F1 = 0
    else:
        F1 = 2 * pre * rec / (pre + rec)
    return F1