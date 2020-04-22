from _code.Train import learn
import os, torch
import argparse

parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP or ICR')
parser.add_argument('--model', type=str, help='backbone model: R18 or R50')
parser.add_argument('--dim', type=int, help='embedding dimension')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--method', type=str, help='method')
parser.add_argument('--g', type=int, help='gsize')
parser.add_argument('--ep', type=int, help='epochs')
parser.add_argument('--w', type=float, help='weight of loss_norm')
args = parser.parse_args()


data_dir = '/pless_nfs/home/datasets/Background_dst/Background/'          
data_dict = torch.load('/pless_nfs/home/datasets/CUB/data_dict_emb.pth')
dst = '_result/{}/{}_{}/G{}/{}_12/'.format(args.method, args.Data, args.model, args.g, args.w)
print(dst)  
    
x = learn(dst, args.Data, data_dict, data_dir)
x.n_class = 16          #number of classes per batch
x.n_img = args.g        #size of each class in a batch
x.n_noise = 128          #number of background images per batch

x.w = args.w
x.init_lr = args.lr
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
