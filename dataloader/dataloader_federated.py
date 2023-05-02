from torch.utils.data import Dataset
from preprocessing import get_data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import json

class data_splitter(object):
    def __init__(self,test, n_devices, path, device):
        self.n_devices=n_devices
        self.device=device
        self.test=test
        datasets=dict()
        datasets[0]=pd.read_csv('/media/ext_storage/Datasets/nsl-kdd/KDDTrain+.txt',header=None)
        datasets[1]=pd.read_csv('/media/ext_storage/Datasets/nsl-kdd/KDDTest+.txt',header=None)
        #datasets[0]=datasets[0].sample(frac=1)
        #datasets[1]=datasets[1].sample(frac=1)
        data_train, labels_train, data_test, labels_test=get_data(datasets)
        self.data=data_train if not test else data_test
        self.label=labels_train if not test else labels_test
        self.classes=np.unique(self.label)
        self.path=path
        if not os.path.exists(self.path+f'NSL_KDD_dataset_{self.n_devices}.txt'):
            self.create_federated_datasets()
        
    def create_federated_datasets(self):
        dataset=[[] for _ in range(self.n_devices)]
        labels=[[] for _ in range(self.n_devices)]
        for cls in self.classes:
            indices=np.where(self.label==cls)
            indices=indices[0]
            len_each=int(len(indices)/self.n_devices)
            indices=indices[:len_each*self.n_devices]
            for i in range(0,len(indices), len_each):
                ind=indices[i:i+len_each]
                data=[list(self.data[j]) for j in ind]
                label=[int(self.label[j]) for j in ind]
                dev=int(i/len_each)
                dataset[dev].extend(data)
                labels[dev].extend(label)
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        with open(self.path+f'NSL_KDD_dataset_{self.n_devices}.txt','w') as f: f.write(json.dumps(dataset))
        with open(self.path+f'NSL_KDD_label_{self.n_devices}.txt','w') as f:f.write(json.dumps(labels))
            
    def iid_data(self):
        dataset=json.load(open(self.path+f'NSL_KDD_dataset_{self.n_devices}.txt','r'))
        labels=json.load(open(self.path+f'NSL_KDD_label_{self.n_devices}.txt','r'))
        dataset, labels=dataset[self.device], labels[self.device]
        return dataset, labels
    
    def get_trans_data(self):
        #Check if trans datasets path exists
        if not self.test:
            path=self.path+'../transferability_unique_classes/train/'
        else:
            path=self.path+'../transferability_unique_classes/test/'
        if not os.path.exists(path):
            os.makedirs(path)
            #This file will exist because in __init__ if it does not, it creates federated datasets
            dataset=json.load(open(self.path+f'NSL_KDD_dataset_{self.n_devices}.txt','r'))
            labels=json.load(open(self.path+f'NSL_KDD_label_{self.n_devices}.txt','r'))
            dataset_list=[j for sub in dataset for j in sub]
            labels_list=[j for sub in labels for j in sub]

            #for now, each device will be trained with a different class of data. There will be one device that will be trained 
            #with only benign data. And 4 other devices will be trained with 4 different types of attacks. This is only possible
            #if there are 5 devices so this code checks for that.
            #dataset_trans,labels_trans=[[]],[[]]
            if self.n_devices!=5:
                print('This kind of transferability is only observable for 5 devices')
            else:
                indices=[[] for _ in range(len(np.unique(labels_list)))]
                for cls in np.unique(labels_list):
                    indices[cls]=[j for j in range(len(labels_list)) if labels_list[j]==cls]
                len_each=(len(indices[0])/self.n_devices)
                print(len_each)
                dataset_trans, labels_trans=[[] for _ in range(self.n_devices)],[[] for _ in range(self.n_devices)]
                for device in range(self.n_devices):
                    #each device will have benign/5 and device 1 and 2 will have attack data corr to attack1 & attack1/2 each
                    dataset_trans[device].extend([dataset_list[ind] for ind in indices[0][int(device*len_each):int((device+1)*len_each)]])
                    labels_trans[device].extend([labels_list[ind] for ind in indices[0][int(device*len_each):int((device+1)*len_each)]])
                    
                    if device!=0 and device!=1:
                        dataset_trans[device].extend([dataset_list[ind]for ind in indices[device]])
                        labels_trans[device].extend([labels_list[ind] for ind in indices[device]])
                    else:
                        if device==0:
                            dataset_trans[device].extend([dataset_list[ind] for ind in indices[1][:int(len(indices[1])/2)]])
                            labels_trans[device].extend([labels_list[ind] for ind in indices[1][:int(len(indices[1])/2)]])
                        if device==1:
                            dataset_trans[device].extend([dataset_list[ind] for ind in indices[1][int(len(indices[1])/2):]])
                            labels_trans[device].extend([labels_list[ind] for ind in indices[1][int(len(indices[1])/2):]])       
                            
                with open(path+f'NSL_KDD_dataset_{self.n_devices}.txt','w') as f: f.write(json.dumps(dataset_trans))
                with open(path+f'NSL_KDD_label_{self.n_devices}.txt','w') as f: f.write(json.dumps(labels_trans))
       
        dataset=json.load(open(path+f'NSL_KDD_dataset_{self.n_devices}.txt','r'))
        labels=json.load(open(path+f'NSL_KDD_label_{self.n_devices}.txt','r'))
        
        return dataset[self.device], labels[self.device]
        
    def __call__(self):
        return self.iid_data()
    
class dataset_transfer(Dataset):
    def __init__(self,test=False,n_devices=5,device=0,transforms=transforms.ToTensor()):
        super().__init__()
        self.n_devices=n_devices
        self.path='datasets/train/' if not test else 'datasets/test/'    
        ds=data_splitter(test, self.n_devices, self.path, device)
        self.dataset, self.labels=ds.get_trans_data()
        self.classes=np.unique(self.labels)
        self.transforms=transforms

    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        if self.labels[idx]>0: self.labels[idx]=1
        return self.transforms(np.array(self.dataset[idx]).reshape(1,-1)), self.labels[idx]

class dataset(Dataset):
    def __init__(self,test=False,n_devices=5,device=0,transforms=transforms.ToTensor(),two_class=False):
        super().__init__()
        self.n_devices=n_devices
        self.path='datasets/train/' if not test else 'datasets/test/'    
        get_iid_data=data_splitter(test, self.n_devices, self.path, device)
        self.dataset, self.labels=get_iid_data()
        self.classes=np.unique(self.labels)
        self.transforms=transforms
        self.two_class=two_class
        
    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx):
        if self.two_class:
            if self.labels[idx]>0:self.labels[idx]=1
        return self.transforms(np.array(self.dataset[idx]).reshape(1,-1)), self.labels[idx]
       