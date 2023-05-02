import pandas as pd
from preprocessing import get_data
import numpy as np
import torch
import torch.optim as optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.dataloader_nsskdd import dataset_whole
from dataloader.dataloader_federated import dataset,dataset_transfer
from models import second, resnet, reduced_resnet
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from train_func import train_model, test, get_figures
from aggregation_algorithms import FedAvg
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--model',type=str,default='resnet')
parser.add_argument('--n_classes', type=int, default=5)
parser.add_argument('--algo', type=str, default='FedAvg')
parser.add_argument('--device',type=int, default=0)
parser.add_argument('--silobn',type=str, default='True')
parser.add_argument('--weight_path',type=str,default='weights/')
parser.add_argument('--tb_path', type=str, default='tensorboard/')
parser.add_argument('--transferability', type=str, default='False')
parser.add_argument('--cf',type=str,default='conf_matrix_figures/fed_trans/')
parser.add_argument('--n_rounds', type=int, default=20)
parser.add_argument('--tmp', type=str, default='False')
parser.add_argument('--diff', type=str, default='False')
parser.add_argument('--epochs',type=int, default=20)
args=parser.parse_args()

device=torch.device(f"cuda:{args.device}")
args.model=args.model+'()'

tmp=True if args.tmp=='True' else False
diff=True if args.diff=='True' else False

if not os.path.exists(args.cf):
    os.makedirs(args.cf)

if not os.path.exists(args.tb_path):
    os.makedirs(args.tb_path)
if not os.path.exists(args.weight_path):
    os.makedirs(args.weight_path)

#silobn/nosilobn
if args.silobn=='True': args.silobn=True
else: args.silobn=False
    
#transferability/multi-class
if args.transferability=='True': 
    args.transferability=True
    args.n_classes=2
else: args.transferability=False

tb=SummaryWriter(f'{args.tb_path}')

n_rounds=args.n_rounds
n_devices=5

#check what kind of model is chosen
global_model=eval(args.model)
two_class=False
#change n_classes here for transferability
if args.n_classes==2 or args.transferability: 
    two_class=True
    if args.model=='resnet()':
        global_model.second.output_layer=nn.Linear(in_features=100, out_features=args.n_classes)
    else: global_model.block2=second(layer1_in=4800, layer1_out=1000, layer2_in=1000, layer2_out=500, layer3_in=500, n_classes=args.n_classes)
contains_bn_layers=False

#Horizontal FL, all models same
#check for batchnorm here
mean_variance=None
for name,layer in global_model.named_modules():
    if (isinstance(layer,nn.BatchNorm1d) or isinstance(layer,nn.BatchNorm2d)) and args.silobn:
        contains_bn_layers=True
        mean_variance=[[] for _ in range(n_devices)]
        break

state_dict={}

#Get length of dataset of each device for the fedavg algorithm
dataset_len=np.zeros(n_devices)

for i in range(n_devices):
    if args.transferability:
        training_data=dataset_transfer(device=i)
    else: training_data=dataset(device=i)
    dataset_len[i]=len(training_data.labels)

path=args.weight_path

torch.save(global_model.state_dict(),path+'global_model_weights.pt')
test_loss, test_acc=[],[]
for comm_round in range(n_rounds):
    accuracies=np.zeros(n_devices)
    losses=np.zeros(n_devices)
    print(f"********** Communication Round {comm_round+1}******************")
    for dev in range(n_devices):
        print(f'.............Device {dev + 1}.................')
        #get local training data
        #get local testing/validation data
        if not args.transferability:
            training_data=dataset(device=dev, two_class=two_class)
            val_data=dataset(test=True, device=dev, two_class=two_class)
        else:
            training_data=dataset_transfer(device=dev)
            val_data=dataset_transfer(test=True, device=dev)
        train_data_loader=DataLoader(training_data, batch_size=64, shuffle=True)
        val_data_loader=DataLoader(val_data, batch_size=64, shuffle=True)
        
        #initialize models with global model
        #check what kind of model chosen
        device_model=eval(args.model)
        #set n_classes here for transferability
        if two_class:
            #check which model, this is for resnet
            #reduced resnet, resnet
            if args.model=='resnet()':
                device_model.second.output_layer=nn.Linear(in_features=100, out_features=args.n_classes)
            else:
                device_model.block2=second(layer1_in=4800, layer1_out=1000, layer2_in=1000, layer2_out=500, layer3_in=500, n_classes=args.n_classes)
        sd=torch.load(path+'global_model_weights.pt')
        
        #SiloBN:if BN, load local mean/variance statistics
        if contains_bn_layers and len(mean_variance[dev]):
            for layer in mean_variance[dev]:
                for key in layer.keys():
                    sd[key]=layer[key]
            mean_variance[dev]=[]
                    
        device_model.load_state_dict(sd)
        #optimizer, loss
        #train resnet backbone at a lower rate than the normal backbone if pretrained resnet
        if args.model=='resnet()':
            param_first=device_model.first.parameters()
            param_second=device_model.second.parameters()
            param_bb=device_model.resnet_backbone.parameters()
        
            optimizer_parameters=[{'params': param_first, 'lr': 1e-4},{'params': param_bb,'lr': 1e-5},{'params':param_second,'lr':1e-4}]
        elif args.model=='reduced_resnet()':
            optimizer_parameters=[{'params':device_model.parameters(),'lr':1e-4}]
        
        #optimizer_parameters=device_model.parameters()
        optim=optimizer.AdamW(optimizer_parameters,lr=0.0001, weight_decay=1e-2)
        criterion=nn.CrossEntropyLoss()
        epochs=args.epochs
        
        #train model on local data
        #history: training_accuracy, training_loss, validation_accuracy, validation_loss, cf_training, cf_testing
        history=train_model(train_data_loader, val_data_loader, epochs, optim, criterion, device_model, tmp,diff,device=device)
        
        #add device validation accuracy and validation loss to tensorboard
        #validation accuracy and validation loss in tb is the average of the accuracies/losses over 20 training epochs
        tb.add_scalar(f'Device {dev}: Validation accuracy', sum(history['validation_accuracy'])/len(history['validation_accuracy']),comm_round)
        tb.add_scalar(f'Device {dev}: Validation loss', sum(history['validation_loss'])/len(history['validation_loss']),comm_round)
        
        #store the trained weights in a dict
        state_dict[dev]=device_model.state_dict()
        
        #SiloBN:if it has batch norm layers, save the running_mean, variance and num_batches_tracked
        if contains_bn_layers:
            for name,layer in device_model.named_modules():
                if isinstance(layer,nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
                    a=dict()
                    a[name+'.running_mean']=device_model.state_dict()[name+'.running_mean']
                    a[name+'.running_var']=device_model.state_dict()[name+'.running_var']
                    a[name+'.num_batches_tracked']=device_model.state_dict()[name+'.num_batches_tracked']
                    mean_variance[dev].append(a)
        
        #test device on any testing set other than it's because it is being validated on it
        testing_set=np.arange(0,n_devices,1)
        testing_set=np.delete(testing_set, dev)
        #for transferability, device 1 and device 0 are being trained on the same attack(attack 1)
        #we still test on each other but only to observe

        random_choice=np.random.randint(0,len(testing_set))
        testing_device=testing_set[random_choice]
        print("Testing with data of dev", testing_device+1)
        
        if args.transferability:
            test_data=dataset_transfer(test=True, device=testing_device)
        else:test_data=dataset(test=True, device=testing_device)
        test_data_loader=DataLoader(test_data, batch_size=64, shuffle=True)
        
        #if you want to test on original data, remove tmp, diff from test()
        testing_loss, testing_accuracy,_,_=test(test_data_loader,criterion, device_model,device,tmp,diff)
        accuracies[dev]=testing_accuracy
        losses[dev]=testing_loss
        
        #log tensorboard accuracies and loss for each device
        tb.add_scalar(f'Device {dev}:Testing accuracy', testing_accuracy,comm_round)
        tb.add_scalar(f'Device {dev}:Testing loss', testing_loss,comm_round)
        
        #generate confusion matrices for the last communication round, each device is tested on validation data of all other
        #devices,other than its own.
        #Each device has confusion matrices corresponding to the validation data it is tested on. device_n/cf_{device},
        #{device} is the device corr to the val data on which the model has been tested.
        
        if args.transferability and (comm_round+1)%5==0:
            for testing_device in testing_set:
                dataset_test=dataset_transfer(test=True, device=testing_device)
                data_load=DataLoader(dataset_test,batch_size=64,shuffle=True)
                #if you want to test on original data, remove tmp, diff from test()
                loss,accuracy,y_pred,y_test=test(data_load,criterion,device_model,device,tmp,diff)
                cf_test=confusion_matrix(y_test,y_pred)
                disp=ConfusionMatrixDisplay(confusion_matrix=cf_test)
                disp.plot()
                cf_path=args.cf+f'comm_round_{comm_round+1}/device_{dev}/'
                
                if not os.path.exists(cf_path):
                    os.makedirs(cf_path)
                
                disp.figure_.savefig(cf_path+f'cf_{testing_device}.png',dpi=300)
    
    print(accuracies, accuracies.mean())
    #choose aggregation algorithm: FedAvg
    updated_weights=FedAvg(state_dict,n_devices,dataset_len,device)
    
    #update global weights
    torch.save(updated_weights,path+'global_model_weights.pt')
    
    tb.add_scalar('Average Testing loss', losses.mean(), comm_round)
    tb.add_scalar('Average Testing accuracy', accuracies.mean(), comm_round)
    
    test_loss.append(losses.mean())
    test_acc.append(accuracies.mean())
    
    #figs=get_figures(test_loss, test_acc, comm_round, save=True, root_path='FederatedResnet/')
tb.close()
