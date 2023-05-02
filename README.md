# An Analysis of Transferability in Network Intrusion Detection using Distributed Deep Learning
Code base of the paper An Analysis of Transferability in Network Intrusion Detection using Distributed Deep Learning @ ICLR 2023 Tiny Papers Track. This paper can be found at https://openreview.net/pdf?id=FPzByCI0yz1

Run train_resnet.py to start federated training with NSLKDD dataset. Models available-resnet and reduced resnet. For resnet, a pretrained resnet18 backbone is used and during training it is finetuned to our dataset using a lower learning rate. For reduced_resnet, a single resent module is used, not pretrained. The entire model is trained at the same learning rate. This can also be run as a two class training(benign and attack). In the five class scenario, each device has an equal distribution of attack data. In the two class scenario(by default, this trains with 5 classes), each device has data corresponding to each class of attack in the same proportion and the split is about 50% benign and 50% attack. By default, the training script trains on 5 devices, however this can be used for any number of devices. The dataset for each device is regenerated to have the same distribution of data. If the model has Batch Norm layers, they are trained using Siloed Federated Learning by default as outlined in : https://arxiv.org/pdf/2008.07424.pdf. The current aggregation algorithm supported is Federated Averaging.

This script can also be used for transferability studies. Transferability is only for 5 devices. There are 4 attacks and one benign class. Device 0 and 1 are trained with benign and attack 1 data equally split. Benign data is split equally among all devices. Device 2,3,4 are treated with benign and 3 remaining attacks respectively. During validation, each device has a validation set(from NSL KDD Testing dataset) and tested on the validation dataset of other devices. At the end of 5 communication rounds, each device is tested on the validation data of every other device.  

# Environment

To setup the environment: 

conda env -f environment.yml

# Federated Training and Testing

Arguments include:
    --model        resnet, reduced_resnet
    
    --n_classes    2,5(default)
    
    --algo         FedAvg(default)
    
    --device       0(default) GPU
    
    --silobn       True(default),False
    
    --weight_path  weights/
    
    --tb_path      tensorboard/
    
    --transferability True/False(default)
    
    --cf           path of the saved confusion matrix
    
    --n_rounds     number of communication rounds
    
    --tmp          enable temporal averaging of data as a preprocessing step
    
    --diff         enable differential inputs as a preprocessing step
    
    --epochs       no of epochs of training for each device
    
To start federated training:

python train_resnet.py --device 0 --model resnet

To start federated training transferability:

python train_resnet.py --device 0 --model resnet --transferability True 

# Results

Results from the paper are availabe at conf_matrix_figures and tensorboard which contain the confusion matrices and tensorboard plots of federated training for normal transferability, temporal averaging and differential inputs. Training weights of global model available at weights.
