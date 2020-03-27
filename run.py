from _code.Train import learn
import os, torch

SOP_EPHN = ['SOP','R50',512,0.0005,'EPHN',0.1]
ICR_EPHN = ['ICR','R50',512,0.0005,'EPHN',0.1]
CUB_EPHN = ['CUB','R18', 64,0.0001,'EPHN',0.1]
CAR_EPHN = ['CAR','R18', 64,0.0005,'EPHN',0.1]

SOP_EPSHN = ['SOP','R50',512,0.0005,'EPSHN',0.1]
ICR_EPSHN = ['ICR','R50',512,0.0005,'EPSHN',0.1]
CUB_EPSHN = ['CUB','R18', 64,0.0001,'EPSHN',0.1]
CAR_EPSHN = ['CAR','R18', 64,0.0005,'EPSHN',0.1]

for Data,model,dim,LR,method,sigma in [CUB_EPSHN]:# select dataset and model to test
    data_dir = '/pless_nfs/home/datasets/Background_images/03branch'          
    data_dict = torch.load('/pless_nfs/home/datasets/CUB/data_dict_emb.pth')
    for i in [0]:# run multiple times
        for g in [16]:# select the class size
            dst = '_result/{}/{}_{}/G{}/{}/'.format(method,Data,model,g,i)
            print(dst)
            
            x = learn(dst, Data, data_dict, data_dir)

            x.n_class = 8          #number of classes per batch
            x.n_img = g            #size of each class in a batch
            x.n_noise = 32          #number of background images per batch

            x.init_lr = LR
            x.criterion.sigma = sigma
            if method=='EPSHN':
                x.criterion.semi = True
            x.run(dim, model, num_epochs=60)
