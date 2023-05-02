from torch.utils.data import Dataset
from preprocessing import get_data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class dataset_whole(Dataset):
    def __init__(self,test=False,transforms=transforms.ToTensor()):
        super().__init__()
        datasets=dict()
        datasets[0]=pd.read_csv('/media/ext_storage/Datasets/nsl-kdd/KDDTrain+.txt',header=None)
        datasets[1]=pd.read_csv('/media/ext_storage/Datasets/nsl-kdd/KDDTest+.txt',header=None)
        data_train, labels_train, data_test, labels_test=get_data(datasets)
        self.data=data_train if not test else data_test
        self.labels=labels_train if not test else labels_test
        self.transforms=transforms
        self.classes=np.unique(self.labels)
        
    def __len__(self): return len(self.labels)
    
    def __getitem__(self, idx):
        return self.transforms(self.data[idx].reshape(1,-1)), self.labels[idx]