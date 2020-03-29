from torch.utils.data.sampler import Sampler
import numpy as np
import random
import torch

class BalanceSampler(Sampler):
    def __init__(self, intervals, GSize=2):
        
        class_len = len(intervals)
        list_sp = []
        
        # find the max interval
        interval_list = [np.arange(b[0],b[1]) for b in intervals]
        len_max = max([b[1]-b[0] for b in intervals])
        
        if len_max>1000:
            len_max = 100
        
        # exact division
        if len_max%GSize != 0:
            len_max = len_max+(GSize-len_max%GSize)
        
        for l in interval_list:
            if l.shape[0]<len_max:
                l_ext = np.random.choice(l,len_max-l.shape[0])
                l_ext = np.concatenate((l, l_ext), axis=0)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0]>len_max:
                l_ext = np.random.choice(l,len_max,replace=False)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0]==len_max:
                l_ext = np.random.permutation(l)
            
            list_sp.append(l_ext)
            
        random.shuffle(list_sp)
        self.idx = np.vstack(list_sp).reshape((GSize*class_len,-1)).T.reshape((1,-1)).flatten().tolist()

    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)


class BalanceSampler2(Sampler):
    def __init__(self, intervals, n_img=5):
        # generate interval list
        interval_list = []
        for b in intervals:
            index_list = torch.arange(b[0],b[1]).tolist()
            if b[1]-b[0]>n_img:
                interval_list.append(random.sample(index_list,n_img))
            else:
                interval_list.append(index_list)
                
        random.shuffle(interval_list)
        
        self.idx = []
        for l in interval_list:
            self.idx += l

    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)

    
class BalanceSampler3(Sampler):
    def __init__(self, intervals, n_class= 8, n_img=16, n_noise=32):    
        
        class_len = len(intervals[:-1])    #exclude background
        list_sp = []
        origin_batchsize = n_class*n_img
    
        
        # find the max interval
        interval_list = [np.arange(b[0],b[1]) for b in intervals]
        len_max = max([b[1]-b[0] for b in intervals[:-1]])   
        
        if len_max>1000:
            len_max = 100
        
        # exact division
        if len_max%n_img != 0:
            len_max = len_max+(n_img-len_max%n_img)
        
        for l in interval_list[:-1]:  #exclude background
            if l.shape[0]<len_max:
                l_ext = np.random.choice(l,len_max-l.shape[0])
                l_ext = np.concatenate((l, l_ext), axis=0)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0]>len_max:
                l_ext = np.random.choice(l,len_max,replace=False)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0]==len_max:
                l_ext = np.random.permutation(l)
            
            list_sp.append(l_ext)
            
        random.shuffle(list_sp)

        #Each batch should contain a) n_class different classes and each class should contain n_img different images 
        #                          b) n_noise background images
        origin_idx = np.vstack(list_sp).reshape((n_img*class_len,-1)).T.reshape((1,-1)).flatten() 
        temp_idx = np.pad(origin_idx, (0,origin_batchsize-origin_idx.shape[0]%origin_batchsize), 'edge')
        temp_idx = temp_idx.reshape(-1,origin_batchsize)   
        background_ext = np.random.choice(interval_list[-1], temp_idx.shape[0]*n_noise).reshape(temp_idx.shape[0],n_noise)
        self.idx = np.concatenate((temp_idx, background_ext), axis=1).reshape((1,-1)).flatten().tolist()



    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)
    