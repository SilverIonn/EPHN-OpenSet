#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
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
k = 5
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

width = 5
norm = matplotlib.colors.Normalize(vmin=0, vmax=5)
norm2 = matplotlib.colors.Normalize(vmin=0, vmax=3.5)
for i in range(num):
    count = 0
    dir = './topkresult-refine-2/'+str(i).zfill(3)+'/'
    if not os.path.exists(dir): os.makedirs(dir)
    for j in range(dsets_val.intervals[i][0],dsets_val.intervals[i][1]):
        top1_1 = dsets_val.idx_to_class[idx_1[j][0].item()]
        top1_2 = dsets_val.idx_to_class[idx_2[j][0].item()]
        origin = dsets_val.idx_to_class[j]
        if(top1_1 != top1_2 and (top1_1 == origin or top1_2 == origin)):
        
            plt.figure(figsize=(20,20))
            gs = GridSpec(4, 6)
            plt.subplot(gs[1:3,0])
            plt.imshow(dsets_val.__getitem__(j)[0].permute(1,2,0))
            plt.axis('off')

            plt.subplot(4,6,1)
            plt.imshow(Fmap4s_1[j],cmap='rainbow',norm=norm2)
            plt.axis('off')

            plt.subplot(4,6,19)
            plt.imshow(Fmap4s_2[j],cmap='rainbow',norm=norm)
            plt.axis('off')

            for s in range(k):   
                plt.subplot(4,6,2+s)
                plt.imshow(Fmap4s_1[idx_1[j][s].item()],cmap='rainbow',norm=norm2)
                plt.axis('off')

            for s in range(k):   
                plt.subplot(4,6,8+s)
                plt.imshow(dsets_val.__getitem__(idx_1[j][s])[0].permute(1,2,0))
                #plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                if (dsets_val.idx_to_class[j]==dsets_val.idx_to_class[idx_1[j][s].item()]):
                    color = 'green'
                else:
                    color = 'red'
                ax = plt.gca()
                ax.spines['left'].set_color(color)
                ax.spines['right'].set_color(color)
                ax.spines['top'].set_color(color)
                ax.spines['bottom'].set_color(color)
                ax.spines['bottom'].set_linewidth(width)
                ax.spines['left'].set_linewidth(width)
                ax.spines['top'].set_linewidth(width)
                ax.spines['right'].set_linewidth(width)
            
            for s in range(k):   
                plt.subplot(4,6,14+s)
                plt.imshow(dsets_val.__getitem__(idx_2[j][s])[0].permute(1,2,0))
                #plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                if (dsets_val.idx_to_class[j]==dsets_val.idx_to_class[idx_2[j][s].item()]):
                    color = 'green'
                else:
                    color = 'red'
                ax = plt.gca()
                ax.spines['left'].set_color(color)
                ax.spines['right'].set_color(color)
                ax.spines['top'].set_color(color)
                ax.spines['bottom'].set_color(color)
                ax.spines['bottom'].set_linewidth(width)
                ax.spines['left'].set_linewidth(width)
                ax.spines['top'].set_linewidth(width)
                ax.spines['right'].set_linewidth(width)

            for s in range(k):   
                plt.subplot(4,6,20+s)
                plt.imshow(Fmap4s_2[idx_2[j][s].item()],cmap='rainbow',norm=norm)
                plt.axis('off')

            plt.tight_layout()
            plt.savefig(dir+str(count).zfill(3)+'.png')
            plt.close()
            
            count = count + 1
    print('class {} finished! Number: {}'.format(i, count))

