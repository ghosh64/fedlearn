import torch.nn as nn
import torch
from torch.nn import functional as F

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(in_channels=1, out_channels=16,kernel_size=2)
        #self.conv2=nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.maxpooling=nn.MaxPool1d(kernel_size=2)
        self.flatten=nn.Flatten()
        #self.bnorm=nn.BatchNorm1d(20)
        self.dropout=nn.Dropout(0.2)
        self.fc1=nn.Linear(in_features=320,out_features=50)
        self.relu=nn.ReLU()
        #self.bnorm2=nn.BatchNorm1d(10)
        self.fc2=nn.Linear(in_features=50,out_features=5)
        self.softmax=nn.Softmax()
        
    def forward(self, x):
        x=self.conv(x)
        #print("After layer1 conv", x.shape)
        #x=self.conv2(x)
        #print("After layer2 conv", x.shape)
        x=self.maxpooling(x)
        #print("After maxpooling", x.shape)
        x=self.flatten(x)
        x=self.dropout(x)
        #print("After flatten", x.shape)
        #x=self.bnorm(x)
        x=self.fc1(x)
        #print("After fc1", x.shape)
        x=self.relu(x)
        #print("After fc2", x.shape)
        #x=self.bnorm2(x)
        x=self.fc2(x)
        #print("After fc2", x.shape)
        x=self.softmax(x)
        
        return x

class first(nn.Module):
    def __init__(self, output_3dim=False):
        super().__init__()
        self.flatten=nn.Flatten()
        self.input_layer=nn.Linear(in_features=41,out_features=100)
        self.relu=nn.ReLU()
        #self.dim_inc=nn.Upsample(scale_factor=3)
        self.dim_inc=nn.Linear(in_features=100, out_features=300) 
        self.output_3dim=output_3dim

    def forward(self,x):
        #print(x.shape)
        x=self.flatten(x)
        #print("Flatten", x.shape)
        x=self.input_layer(x)
        #print("input layer", x.shape)
        x=self.relu(x)
        x=self.dim_inc(x)
        x=self.relu(x)
        #print("dim_inc", x.shape)
        if self.output_3dim:
            x=torch.reshape(x,(-1,1,300))
        else:
            x=torch.reshape(x,(-1,3,10,10))
        #print("reshape", x.shape)
        
        return x

class second(nn.Module):
    def __init__(self, layer1_in=1000, layer1_out=500, layer2_in=500, layer2_out=100, layer3_in=100, n_classes=5):
        super().__init__()
        self.layer_1=nn.Linear(in_features=layer1_in, out_features=layer1_out)
        self.layer_2=nn.Linear(in_features=layer2_in, out_features=layer2_out)
        self.dropout=nn.Dropout(0.2)
        self.output_layer=nn.Linear(in_features=layer3_in, out_features=n_classes)
        self.softmax=nn.Softmax()
        self.relu=nn.ReLU()
        
    def forward(self, x):
        x=self.layer_1(x)
        #print('layer1', x.shape)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.layer_2(x)
        x=self.relu(x)
        #print('layer2',x.shape)
        #add another dropout layer here
        x=self.dropout(x)
        x=self.output_layer(x)
        x=self.softmax(x)
        #print('output layer', x.shape)
        
        return x


class resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=first()
        self.resnet_backbone=torch.hub.load('pytorch/vision:v0.8.0','resnet18',pretrained=True)
        self.second=second()
        
    def forward(self, x):
        x=self.first(x)
        x=self.resnet_backbone(x)
        #print("Resnet backbone", x.shape)
        x=self.second(x)
        
        return x
    
class model_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=first()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=16,kernel_size=3)
        self.maxpool=nn.MaxPool2d(kernel_size=2)
        self.lstm=nn.LSTM(16,20,2)
        self.flatten=nn.Flatten()
        self.dropout=nn.Dropout(0.2)
        self.fc1=nn.Linear(in_features=320,out_features=50)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(in_features=50,out_features=5)
        self.softmax=nn.Softmax()
        
    def forward(self,x):
        x=self.first(x)
        x=self.conv1(x)
        x=self.maxpool(x)
        #print('shape after maxpool',x.shape)
        x=torch.reshape(x,(-1,16,16))
        x,_=self.lstm(x)
        x=self.flatten(x)
        x=self.dropout(x)
        #print("shaoe after dropout", x.shape)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.softmax(x)
        
        return x
    
class model_gru(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=first()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=16,kernel_size=3)
        self.maxpool=nn.MaxPool2d(kernel_size=2)
        self.lstm=nn.GRU(16,20,2)
        self.flatten=nn.Flatten()
        self.dropout=nn.Dropout(0.2)
        self.fc1=nn.Linear(in_features=320,out_features=50)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(in_features=50,out_features=5)
        self.softmax=nn.Softmax()
        
    def forward(self,x):
        x=self.first(x)
        x=self.conv1(x)
        x=self.maxpool(x)
        #print('shape after maxpool',x.shape)
        x=torch.reshape(x,(-1,16,16))
        x,_=self.lstm(x)
        x=self.flatten(x)
        x=self.dropout(x)
        #print("shaoe after dropout", x.shape)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.softmax(x)
        
        return x

class resnet_module(nn.Module):
    def __init__(self, num_channels=1, use_1x1conv=False,padding=1, stride=1):
        super().__init__()
        self.conv1=nn.Conv1d(in_channels=1,out_channels=64, kernel_size=3, padding=padding, stride=stride)
        self.bn1=nn.BatchNorm1d(num_features=64)
        self.conv2=nn.Conv1d(in_channels=64,out_channels=16, kernel_size=3,padding=padding, stride=stride)
        self.bn2=nn.BatchNorm1d(num_features=16)
        if use_1x1conv:
            self.conv3=nn.Conv1d(in_channels=128,out_channels=256, kernel_size=3, padding=padding,stride=stride)
        else: self.conv3=None
            
    def forward(self, x):
        y=F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        y+=x
        return y

#reduced resnet model does not use a pretrained resnet module.
#resnet-18 backbone is 18 deep layers deep but this is I guess 4 layers deep with 2 conv layers and two bn layers
class reduced_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=first(output_3dim=True)
        self.resnet=resnet_module()
        self.flatten=nn.Flatten()
        self.block2=second(layer1_in=4800, layer1_out=1000, layer2_in=1000, layer2_out=500, layer3_in=500, n_classes=5)
    
    def forward(self,x):
        #print("inside first")
        x=self.block1(x)
        #print("shape", x.shape)
        x=self.resnet(x)
        #print("outside resnet", x.shape)
        x=self.flatten(x)
        #print("fl", x.shape)
        x=self.block2(x)
        
        return x
    
class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=41, out_features=10)
        self.bottleneck=nn.Linear(in_features=10, out_features=5)
        self.layer2=nn.Linear(in_features=5, out_features=10)
        self.layer3=nn.Linear(in_features=10, out_features=41)
        self.relu=nn.ReLU()
        self.float()
    def forward(self,x,test=False):
        x=self.layer1(x)
        bn=self.bottleneck(x)
        x=self.layer2(bn)
        x=self.layer3(x)
        #x=self.relu(x)
        
        if test: return bn
        
        return x

class model_autoenc_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=first()
        self.first.input_layer=nn.Linear(5,100)
        self.resnet_backbone=torch.hub.load('pytorch/vision:v0.8.0','resnet18',pretrained=True)
        self.second=second()
        self.second.output_layer=nn.Linear(in_features=100, out_features=2)
        
    def forward(self, x):
        x=self.first(x)
        x=self.resnet_backbone(x)
        #print("Resnet backbone", x.shape)
        x=self.second(x)
        
        return x