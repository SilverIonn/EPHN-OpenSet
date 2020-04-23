from _code.Train import learn
import os, torch
import argparse

parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP, ICR or LMK')
parser.add_argument('--model', type=str, help='backbone model: R18 or R50')
parser.add_argument('--dim', type=int, help='embedding dimension')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--method', type=str, help='method')
parser.add_argument('--g', type=int, help='gsize')
parser.add_argument('--n', type=int, help='noise size per batch')
parser.add_argument('--c', type=int, help='number of classes per batch')
parser.add_argument('--imgsize', type=int, help='size of images')
parser.add_argument('--ep', type=int, help='epochs')
parser.add_argument('--w', type=float, help='weight of loss_norm')
args = parser.parse_args()


data_dir = '/SEAS/groups/plessgrp/Landmark_V1/Background'          
data_dict = torch.load('/SEAS/groups/plessgrp/Landmark_V1/data_dict_LMK.pth')


dst = '_result/{}/{}_{}/G{}/{}_0/'.format(args.method, args.Data, args.model, args.g, args.w)
print(dst)  
    
x = learn(dst, args.Data, data_dict, data_dir)
x.n_class = args.c          #number of classes per batch
x.n_img = args.g            #size of each class in a batch
x.n_noise = args.n          #number of background images per batch

x.w = args.w
x.init_lr = args.lr
x.imgsize = args.imgsize

if args.method=='EPSHN':
    x.criterion.semi = True
x.run(args.dim, args.model, num_epochs=args.ep)

# SOP_EPHN = ['SOP','R50',512,0.0005,'EPHN',0.1]
# ICR_EPHN = ['ICR','R50',512,0.0005,'EPHN',0.1]
# CUB_EPHN = ['CUB','R18', 64,0.0001,'EPHN',0.1]
# CAR_EPHN = ['CAR','R18', 64,0.0005,'EPHN',0.1]

# SOP_EPSHN = ['SOP','R50',512,0.0005,'EPSHN',0.1]
# ICR_EPSHN = ['ICR','R50',512,0.0005,'EPSHN',0.1]
# CUB_EPSHN = ['CUB','R18', 64,0.0001,'EPSHN',0.1]
# CAR_EPSHN = ['CAR','R18', 64,0.0005,'EPSHN',0.1]
