#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import numpy as np
import os

import torch.nn as nn
from torchvision import transforms
from _code.color_lib import RGBmean,RGBstdv
from _code.Resnet import resnet18, resnet50
from visualization.Reader3 import ImageReader
from torch.utils.data.sampler import SequentialSampler
from random import sample 



# In[ ]:


def recall(Fvec, imgLab,rank=None):
    N = len(imgLab)
    imgLab = torch.LongTensor([imgLab[i] for i in range(len(imgLab))])
    
    D = Fvec.mm(torch.t(Fvec))
    D[torch.eye(len(imgLab)).byte()] = -1
    
    if rank==None:
        _,idx = D.max(1)
        imgPre = imgLab[idx]
        A = (imgPre==imgLab).float()
        return (torch.sum(A)/N).item()
    else:
        _,idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab[idx[:,i]]
                A += (imgPre==imgLab).float()
            acc_list.append((torch.sum((A>0).float())/N).item())
        return torch.Tensor(acc_list), idx

def eva(dsets, model):
    Fvecs = []
    Fmap4s = []
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=400, sampler=SequentialSampler(dsets), num_workers=48)
    torch.set_grad_enabled(False)
    model.eval()
    for data in dataLoader:
        _, inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
        fvec = model(inputs_bt.cuda())
        Fmap4s.extend(fvec[-1].cpu())
        fvec = norml2(fvec[0])
        fvec = fvec.cpu()
        Fvecs.append(fvec)
    print('Fmap4s size : '+str(len(Fmap4s)))
            
    return torch.cat(Fvecs,0), Fmap4s

def norml2(vec):# input N by F
    F = vec.size(1)
    w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F,1))))
    return vec.div(w)


# In[ ]:


src_1 = '/pless_nfs/home/jiayin19/EPHN-OpenSet/_result/EPSHN/CUB_R18/G8/0.0_1/'
src_2 = '/pless_nfs/home/jiayin19/EPHN-OpenSet/_result/EPSHN/CUB_R18/G8/0.1_3/'
k = 1
rank = [k]
phase = 'val'

# Fvec = torch.load(src +'79'+ phase + 'Fvecs.pth')
# dsets = torch.load(src + phase + 'dsets.pth')

#load model
out_dim = 64
avg= 8

model_1 = resnet18(pretrained=False)
num_ftrs_1 = model_1.fc.in_features
model_1.fc = nn.Linear(num_ftrs_1, out_dim)
model_1.avgpool = nn.AvgPool2d(avg)
model_1.load_state_dict(torch.load(src_1+'model_params.pth')) 
model_1.train(False) 
model_1 = model_1.cuda()

model_2 = resnet18(pretrained=False)
num_ftrs_2 = model_2.fc.in_features                           
model_2.fc = nn.Linear(num_ftrs_2, out_dim)                   
model_2.avgpool = nn.AvgPool2d(avg)                           
model_2.load_state_dict(torch.load(src_2+'model_params.pth')) 
model_2.train(False)                                          
model_2 = model_2.cuda()

 
#calculate fvec
imgsize = 256
data_dict = torch.load('/pless_nfs/home/datasets/CUB/data_dict_emb.pth')
val_transforms = transforms.Compose([transforms.Resize(imgsize),
                                                  transforms.CenterCrop(imgsize),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(RGBmean['CUB'], RGBstdv['CUB'])])
dsets_val = ImageReader(data_dict['val'], transform = val_transforms) 
Fvec_val_1, Fmap4s_1 = eva(dsets_val, model_1)
Fvec_val_2, Fmap4s_2 = eva(dsets_val, model_2) 


_, idx_1= recall(Fvec_val_1, dsets_val.idx_to_class,rank=rank)
_, idx_2= recall(Fvec_val_2, dsets_val.idx_to_class,rank=rank)

num = len(dsets_val.intervals)
print('Number of classes: {}'.format(num))

Norm_1 = [Fmap4s_1[i].mean() for i in range(len(Fmap4s_1))]
Norm_2 = [Fmap4s_2[i].mean() for i in range(len(Fmap4s_2))]


print('Norm_1 min:{}, max:{}.'.format(min(Norm_1), max(Norm_1)))
print('Norm_2 min:{}, max:{}.'.format(min(Norm_2), max(Norm_2)))    

Intervals = [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0),(1.0,1.2),(1.2,1.4),(1.4,1.6),(1.6,1.8),(1.8,2.0),(2.0,2.2),(2.2,2.4),(2.4,2.6),(2.6,2.8),(2.8,3.0),(3.0,3.2),(3.2,3.4)]
Intervals_2 = [(0,0.5),(1.0,2.0),(2.5,3.0)]

dict_sum_1 = {}
dict_sum_2 = {}

dict_right_1 = {}
dict_right_2 = {}

acc_1=[]
acc_2=[]
sum_1=[]
sum_2=[]

dict_sample={}

for i in range(0, len(Intervals)):
    dict_sum_1[Intervals[i]]=0
    dict_sum_2[Intervals[i]]=0
    dict_right_1[Intervals[i]]=0
    dict_right_2[Intervals[i]]=0

for i in range(0, len(Intervals_2)):
    dict_sample[Intervals_2[i]]=[]

for i in range(0,dsets_val.intervals[-1][1]):
    for j in range(0, len(Intervals)):
        if(Norm_1[i] >= Intervals[j][0] and Norm_1[i] < Intervals[j][1]):
            dict_sum_1[Intervals[j]]= dict_sum_1[Intervals[j]]+1
            if(dsets_val.idx_to_class[i] == dsets_val.idx_to_class[idx_1[i][0].item()]):
                dict_right_1[Intervals[j]] = dict_right_1[Intervals[j]]+1

        if(Norm_2[i] >= Intervals[j][0] and Norm_2[i] < Intervals[j][1]):
            dict_sum_2[Intervals[j]]= dict_sum_2[Intervals[j]]+1
            if(dsets_val.idx_to_class[i] == dsets_val.idx_to_class[idx_2[i][0].item()]):
                dict_right_2[Intervals[j]] = dict_right_2[Intervals[j]]+1
        
    for j in range(0, len(Intervals_2)):
        if(Norm_2[i] >= Intervals_2[j][0] and Norm_2[i] < Intervals_2[j][1]):
            dict_sample[Intervals_2[j]].append(i)
            
for i in range(0, len(Intervals)):
    if(dict_sum_1[Intervals[i]]!=0):
        acc_1.append(dict_right_1[Intervals[i]]/dict_sum_1[Intervals[i]])
    else:
        acc_1.append(0)
    if(dict_sum_2[Intervals[i]]!=0):
        acc_2.append(dict_right_2[Intervals[i]]/dict_sum_2[Intervals[i]])
    else:
        acc_2.append(0)
    sum_1.append(dict_sum_1[Intervals[i]])
    sum_2.append(dict_sum_2[Intervals[i]])

a = np.arange(0.1, 3.5, 0.2)
plt.xlabel("Norm")
plt.ylabel("Accuracy")
plt.plot(a, acc_1, label='old')
plt.plot(a, acc_2, label='new')
plt.legend()
plt.title('Norm - Accuracy')
plt.savefig('./ACC.png')
plt.close()

plt.xlabel("Norm")
plt.ylabel("Number of Images")
plt.plot(a, sum_1, label='old')
plt.plot(a, sum_2, label='new')
plt.legend()
plt.title('Norm - Number of Images')
plt.savefig('./SUM.png')
plt.close()

for i in range(0, len(Intervals_2)):
    print(len(dict_sample[Intervals_2[i]]))
    lst = sample(dict_sample[Intervals_2[i]],10)
    plt.figure(figsize=(25,10))
    for s,j in enumerate(lst):
        plt.subplot(2,5,s+1)
        plt.imshow(dsets_val.__getitem__(j)[0].permute(1,2,0))
        plt.axis('off')
    # plt.title('Norm Range: {} - {}'.format(Intervals_2[i][0], Intervals_2[i][1]))
    plt.savefig('./Sample{}.png'.format(i))
    plt.close()
# for i in range(0,dsets_val.intervals[-1][1]):

#     classidx_1 = dsets_val.idx_to_class[idx_1[j][0].item()]
#     classidx_2 = dsets_val.idx_to_class[idx_2[j][0].item()]
#     originidx  = dsets_val.idx_to_class[j]
        
       
        
    

        
    

