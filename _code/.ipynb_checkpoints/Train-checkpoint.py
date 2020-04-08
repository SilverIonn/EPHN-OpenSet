import os, time

from torchvision import transforms, datasets
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import torch.nn as nn
import torch

from .Sampler import BalanceSampler, BalanceSampler2, BalanceSampler3
from .Reader import ImageReader
from .Loss import EPHNLoss
from .Utils import recall, recall2, recall2_batch, eva
from .color_lib import RGBmean, RGBstdv
from .Resnet import resnet18, resnet50

PHASE = ['tra','val']

class learn():
    def __init__(self, dst, Data, data_dict, data_dir):
        self.dst = dst
        self.gpuid = [0]
            
        self.imgsize = 256
        # self.batch_size = 128
        self.num_workers = 32
        
        self.decay_time = [False,False]
        self.init_lr = 0.001
        self.decay_rate = 0.1
        self.avg = 8
        
        self.Data = Data
        self.data_dir = data_dir
        self.data_dict = data_dict
        
        self.RGBmean = RGBmean[Data]
        self.RGBstdv = RGBstdv[Data]
        
        self.criterion = EPHNLoss() 
        self.n_class = 8         ##n_class different classes
        self.n_img = 16          ##each class should contain n_img different images
        self.n_noise = 32        ##n_noise background images
        
        self.w = 1               ##weight of loss_norm 
        if not self.setsys(): print('system error'); return
        
    def run(self, emb_dim, model_name, num_epochs=20):
        self.out_dim = emb_dim
        self.num_epochs = num_epochs
        self.loadData()
        self.setModel(model_name)
        print('output dimension: {}'.format(emb_dim))
        self.opt()

    ##################################################
    # step 0: System check
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        return True
    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        self.tra_transforms = transforms.Compose([transforms.Resize(int(self.imgsize*1.1)),
                                                  transforms.RandomCrop(self.imgsize),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])
        
        self.val_transforms = transforms.Compose([transforms.Resize(self.imgsize),
                                                  transforms.CenterCrop(self.imgsize),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.RGBmean, self.RGBstdv)])

        self.dsets = ImageReader(self.data_dict['tra'], self.data_dir, self.tra_transforms) 
        self.intervals = self.dsets.intervals
        self.classSize = len(self.intervals)
        print('number of classes: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self, model_name):
        self.model_name = model_name
        if model_name == 'R18':
            self.model = resnet18(pretrained=True)
            print('Setting model: resnet18')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.out_dim)
            self.model.avgpool = nn.AvgPool2d(self.avg)
        elif model_name == 'R50':
            self.model = resnet50(pretrained=True)
            print('Setting model: resnet50')
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.out_dim)
            self.model.avgpool = nn.AvgPool2d(self.avg)
        elif model_name == 'GBN':
            self.model = models.googlenet(pretrained=True, transform_input=False)
            self.model.aux_logits=False
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.classSize)
            print('Setting model: GoogleNet')
        else:
            print('model is not exited!')

        print('Training on Single-GPU')
        print('LR is set to {}'.format(self.init_lr))
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>=0.5*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.75*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def opt(self):
        # recording time and epoch info
        since = time.time()
        self.record_norm = []
        self.record_acc = []

#         # calculate the retrieval accuracy
#         if self.Data in ['SOP','CUB','CAR']:
#             acc = self.recall_val2val(-1)
#         elif self.Data=='ICR':
#             acc = self.recall_val2gal(-1)
#         elif self.Data=='HOTEL':
#             acc = self.recall_val2tra(-1)
#         else:
#             acc = self.recall_val2tra(-1)
        
#         self.record.append([-1, 0]+acc)
    
        for epoch in range(self.num_epochs): 
            # adjust the learning rate
            print('Epoch {}/{} \n '.format(epoch+1, self.num_epochs) + '-' * 40)
            self.lr_scheduler(epoch+1)
            
            # train 
            tra_loss, dst_norm, bkgd_norm = self.tra()
            
            # calculate the retrieval accuracy
            if epoch>0 and (epoch+1)%5==0:
                if self.Data in ['SOP','CUB','CAR']:
                    acc = self.recall_val2val(epoch)
                elif self.Data=='ICR':
                    acc = self.recall_val2gal(epoch)
                elif self.Data=='HOTEL':
                    acc = self.recall_val2tra(epoch)
                else:
                    acc = self.recall_val2tra(epoch)
                self.record_acc.append([epoch+1]+acc)

            self.record_norm.append([epoch+1, dst_norm, bkgd_norm])

        # save model
        torch.save(self.model.cpu().state_dict(), self.dst + 'model_params.pth')
        torch.save(torch.Tensor(self.record_acc), self.dst + 'record_acc.pth')
        torch.save(torch.Tensor(self.record_norm), self.dst + 'record_norm.pth')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        return
    
    def tra(self):
        if self.model_name == 'GBN':
            self.model.eval()  # Fix batch norm of model
        else:
            self.model.train()  # Fix batch norm of model
            
        if self.Data in ['CUB','CAR']:
            dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.n_class*self.n_img+self.n_noise, sampler=BalanceSampler3(self.intervals, n_class=self.n_class, n_img=self.n_img, n_noise=self.n_noise), num_workers=self.num_workers)
        # else: 
        #     dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.batch_size, sampler=BalanceSampler2(self.intervals, n_img=self.n_img), num_workers=self.num_workers)
        
        L_data, N_data = 0.0, 0
        DN_data, LN_data, NN_data = 0.0, 0.0, 0
        # iterate batch
        for data in dataLoader:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                if len(labels_bt)<self.n_class*self.n_img+self.n_noise:
                    break
                fvec,_,_,_,fmap4 = self.model(inputs_bt.cuda())
                loss_ephn = self.criterion(fvec[:-self.n_noise,:], labels_bt[:-self.n_noise].cuda())
                loss_norm = fmap4[-self.n_noise:,:,:].mean()
                data_norm = fmap4[:-self.n_noise,:,:].mean()
                loss = loss_ephn+self.w*loss_norm
                #print(loss_ephn.item(),loss_norm.item())
                loss.backward()
                self.optimizer.step()  
            
            L_data += loss.item()
            N_data += len(labels_bt)
            DN_data += data_norm.item()
            LN_data += loss_norm.item()
            NN_data += 1
        return L_data/N_data, DN_data/NN_data, LN_data/NN_data
        
    def recall_val2val(self, epoch):
        self.model.train(False)  # Set model to testing mode
        dsets_tra = ImageReader(self.data_dict['tra'], transform = self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], transform = self.val_transforms) 
        Fvec_tra = eva(dsets_tra, self.model)
        Fvec_val = eva(dsets_val, self.model)
        
        if epoch>=self.num_epochs-5:
            torch.save(Fvec_tra, self.dst + str(epoch) + 'traFvecs.pth')
            torch.save(Fvec_val, self.dst + str(epoch) + 'valFvecs.pth')
            torch.save(dsets_tra, self.dst + 'tradsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')
            
        acc_tra = recall(Fvec_tra, dsets_tra.idx_to_class)
        acc_val = recall(Fvec_val, dsets_val.idx_to_class)
        print('R@1_tra:{:.1f}  R@1_val:{:.1f}'.format(acc_tra*100, acc_val*100)) 
        
        return [acc_tra, acc_val]
    
    def recall_val2tra(self, epoch):
        self.model.train(False)  # Set model to testing mode
        dsets_tra = ImageReader(self.data_dict['tra'], transform =self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], transform =self.val_transforms) 
        Fvec_tra = eva(dsets_tra, self.model)
        Fvec_val = eva(dsets_val, self.model)

        if epoch==self.num_epochs-1:
            torch.save(Fvec_tra, self.dst + 'traFvecs.pth')
            torch.save(Fvec_val, self.dst + 'valFvecs.pth')
            torch.save(dsets_tra, self.dst + 'tradsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')

        acc = recall2(Fvec_val, Fvec_tra, dsets_val.idx_to_class, dsets_tra.idx_to_class)
        print('R@1:{:.2f}'.format(acc)) 
        
        return [acc]
    
    def recall_val2gal(self, epoch):
        self.model.train(False)  # Set model to testing mode
        dsets_gal = ImageReader(self.data_dict['gal'], transform =self.val_transforms) 
        dsets_val = ImageReader(self.data_dict['val'], transform =self.val_transforms) 
        Fvec_gal = eva(dsets_gal, self.model)
        Fvec_val = eva(dsets_val, self.model)
        
        if epoch==self.num_epochs-1:
            torch.save(Fvec_gal, self.dst + 'galFvecs.pth')
            torch.save(Fvec_val, self.dst + 'valFvecs.pth')
            torch.save(dsets_gal, self.dst + 'galdsets.pth')
            torch.save(dsets_val, self.dst + 'valdsets.pth')
            
        acc = recall2(Fvec_val, Fvec_gal, dsets_val.idx_to_class, dsets_gal.idx_to_class)
        print('R@1:{:.2f}'.format(acc)) 
        
        return [acc]
    
