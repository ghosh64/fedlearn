import pandas as pd
from preprocessing import get_data
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.dataloader_federated import dataset
from models import model
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from temporal_averaging import temporal_avg,diff_input

def train_model(data_load,test_data_loader,epochs,optimizer,criterion,model,tmp=False,diff=False, device=torch.device("cuda:1")):
    #training loss is loss per batch per epoch
    #predictions_cf
    training_loss,training_accuracy,testing_loss,testing_accuracy=[],[],[],[]
    tloss_previous=10
    model.train()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        pred_cf=[]
        y_train=[]
        csamp,closs=0,0
        for i,(data,labels) in enumerate(tqdm(data_load)):
            if tmp:
                data, labels=temporal_avg(data, labels)
            if diff:
                data, labels=diff_input(data,labels)
            data=data.to(device=device, dtype=torch.float)
            y_train.extend(labels.cpu().numpy())
            labels=labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            data=torch.squeeze(data, dim=1)
            predictions=model(data)
            _,pred=torch.max(predictions,dim=1)
            pred_cf.extend(pred.cpu().numpy())
            csamp+=pred.eq(labels).sum().item()
            loss=criterion(predictions,labels)
            closs+=loss.item()
            loss.backward()
            optimizer.step()
        tloss, tacc,predictions_cf,y_test=test(test_data_loader,criterion,model,device,tmp,diff)
        if tloss<tloss_previous:
            cf_test=confusion_matrix(y_test,predictions_cf)
            cf_train=confusion_matrix(y_train,pred_cf)
        testing_accuracy.append(tacc)
        training_accuracy.append(csamp/len(data_load.dataset))
        testing_loss.append(tloss)
        training_loss.append(closs/len(data_load))
    history={}
    history['training_loss']=training_loss
    history['training_accuracy']=training_accuracy
    history['validation_loss']=testing_loss
    history['validation_accuracy']=testing_accuracy
    history['cf_test']=cf_test
    history['cf_train']=cf_train
    
    return history

def test(data_load, criterion, model,device,tmp=False,diff=False):
    predictions_cf,y_test=[],[]
    model.eval()
    csamp,closs=0,0
    with torch.no_grad():
        for i,(data,labels) in enumerate(tqdm(data_load)):
            y_test.extend(labels.cpu().numpy())
            #to disable temporal averaging during testing, do not pass tmp
            #to disable differential inputs during testing, do not pass diff
            #no temporal averaging for testing data, changed for tmp_avg_avg code
            if tmp:
                data,labels=temporal_avg(data,labels)
            if diff:
                data,labels=diff_input(data, labels)
            data=data.to(device=device, dtype=torch.float)
            labels=labels.to(device=device, dtype=torch.long)
            data=torch.squeeze(data, dim=1)
            predictions=model(data)
            _,pred=torch.max(predictions,dim=1)
            predictions_cf.extend(pred.cpu().numpy())
            csamp+=pred.eq(labels).sum().item()
            loss=criterion(predictions,labels)
            closs+=loss.item()
    return closs/len(data_load),csamp/len(data_load.dataset),predictions_cf, y_test 

def get_figures(loss, acc,rounds, save=True, root_path='figures/'):
    x=[i for i in range(len(loss))]
    figure, axis=plt.subplots(1,2)
    
    axis[0].plot(x, loss)
    axis[0].set_title('Loss vs rounds')
    
    axis[1].plot(x,acc)
    axis[1].set_title('Acc vs rounds')
    
    if save:
        path=root_path+f'round_{rounds}'
        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(path+'/Loss_accuracy_plot.png')
        plt.grid()        
        plt.show()
    
    return plt
