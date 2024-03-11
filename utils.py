import torch
import numpy as np
from collections import defaultdict
torch.set_default_tensor_type(torch.FloatTensor)
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.optim as optim
import torch.nn as nn


def train(number_1, train_loader, train_loader2, train_loader3, model, optimizer, criterion):
    model.train()
    correct = 0
    loss_a = 0
    correct5 = 0


    for i in range(len(train_loader)):
        data1 = train_loader[i]
        data2 = train_loader2[i]
        data3 = train_loader3[i]
        out = model(data1.x,data2.x,data3.x, data1.edge_index,data2.edge_index,data3.edge_index, data1.batch,data2.batch,data3.batch)  # Perform a single forward pass.
        loss = criterion(out, data1.y)  # Compute the loss.
        pred = out.argmax(dim=1)
        correct += int((pred == data1.y).sum())
        loss_a += loss.item()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        top_k = 5
        out_np = out.cpu().detach().numpy()
        for index, o in enumerate(out_np):
            top5 = o.argsort()[::-1][:top_k]
            if int(data1.y[index]) in top5:
                correct5 = correct5 + 1

    return loss_a,correct / number_1,correct5/number_1

def test(number_2, test_loader, test_loader2, test_loader3, model, n_class):
    model.eval()
    ll1 = []
    ll2 = []
    correct = 0
    correct5 = 0
    dic = {}
    for i in range(n_class):
        dic[i] = [0,0,0,0]
    dicx = {}
    dicy = {}
    for i in range(len(test_loader)):
        data1 = test_loader[i]
        data2 = test_loader2[i]
        data3 = test_loader3[i]
        lst_x = []
        data_list = data1.to_data_list()
        for batch in data_list:
            x = batch.x.shape[0]
            lst_x.append(x)
            if x not in dicx:
                dicx[x] = 1
            else:
                dicx[x] += 1

        out = model(data1.x,data2.x,data3.x, data1.edge_index,data2.edge_index,data3.edge_index, data1.batch,data2.batch,data3.batch)

        pred = out.argmax(dim=1)  # Use the class with highest probability.

        bool_lst = pred == data1.y
        int_list = [int(value) for value in bool_lst.tolist()]
        for mn in lst_x:
            if mn not in dicy:
                dicy[mn] = 0
        for index, result in zip(lst_x, int_list):
            if result == 1:
                dicy[index] += 1

        correct += int((pred == data1.y).sum())  # Check against ground-truth labels.
        top_k = 5
        out_np = out.cpu().detach().numpy()
        l = list(pred)
        l2 = list(data1.y)
        ll1 += l
        ll2 += l2
        for i in range(len(l)):
            pre = int(l[i])
            real = int(l2[i])

            dic[real][2] += 1
            dic[pre][1] += 1
            if pre == real:
                dic[pre][0] += 1
            if pre != real:
                dic[pre][3] += 1
        for index, o in enumerate(out_np):
            top5 = o.argsort()[::-1][:top_k]
            if int(data1.y[index]) in top5:
                correct5 = correct5 + 1
    return correct /number_2, correct5/number_2 ,dic,ll1,ll2 # Derive ratio of correct predictions.
