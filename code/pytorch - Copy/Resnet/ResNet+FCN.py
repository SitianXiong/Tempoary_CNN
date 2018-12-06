
# coding: utf-8

# # epoch = 50

# In[7]:


import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
# from torchvision.datasets import CIFAR10
from datetime import datetime
import os
import gdal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)
    
class resnet(nn.Module):
    def __init__(self, in_channel, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512)
        )
        
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        return x
    
aresnet= resnet(7,True)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn,self).__init__()
        
        self.stage1=nn.Sequential(*list(aresnet.children())[:-2]) #128
        self.stage2=nn.Sequential(*list(aresnet.children())[-2])  #256
        self.stage3=nn.Sequential(*list(aresnet.children())[-1])  #512
        
        self.scores1=nn.Conv2d(512,num_classes,1)
        self.scores2=nn.Conv2d(256,num_classes,1)
        self.scores3=nn.Conv2d(128,num_classes,1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 0, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 3)
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 
    
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3) 
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s)
        return s
    
    
net=fcn(2)
net=net.cuda()
criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,weight_decay=1e-4)


band_fns= [im for im in os.listdir('./data/2014_121_64/croppedrst/') if (im.split('.')[-1]=='rst')]
bands_dict={}
for i in range(8):
    in_ds=gdal.Open('./data/2014_121_64/croppedrst/'+band_fns[i])
    in_band=in_ds.GetRasterBand(1)
    bands_dict['b{}'.format(i+1)]=in_band.ReadAsArray().astype('float')
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor
    

xlist=[3308,3300,3642,3814,4000,4166,4235,4380,4569,4363,4664,4286,4750,4166,3600,4200,3763,4114,3805,4226,5500,4827,5850,6114,6337,6509]  
ylist=[330,760,760,1265,1650,1935,2338,2600,2492,2784,2827,3042,2960,3136,3060,3522,3617,3908,4183,4277,5550,4800,5400,5487,5616,5625]


for epoch in range(50):#train1
    for i in range(len(xlist)):
        abatch=[]
        x=xlist[i]
        y=ylist[i]
        for j in range(8):
            img=bands_dict['b{}'.format(j+1)]
            img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
            if j<7:
                img=min_max_normalization(img,0,1)
            abatch.append(img)
        abatch=tuple(abatch) 
        
        acc=0
        traindata=Variable(torch.cat(abatch[:-1],0).unsqueeze(0).float().cuda())
        label=Variable(abatch[-1].long().cuda())

        net=net.train()
        out=net(traindata)
        out=F.log_softmax(out,dim=1)
        loss=criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels=out.max(dim=1)[1]
        acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))

        if (epoch+1)%25==0 or epoch==0:
            print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
            _,figs=plt.subplots(1,3,figsize=(10,10))
            band4=traindata[:,3,:,:].cpu().view(1,512,512)
            band6=traindata[:,5,:,:].cpu().view(1,512,512)
            band5=traindata[:,4,:,:].cpu().view(1,512,512)
            aimg=torch.cat((band4,band6,band5),0).permute(1,2,0)
            bimg=label.cpu().view(512,512)
            cimg=pred_labels.view(512,512)
            figs[0].imshow(aimg)
            figs[1].imshow(bimg)
            figs[2].imshow(cimg)
        
        
for epoch in range(30):#test
    acc=0
    batch_1=[]
    x=int(torch.randint(3700,4500,(1,)).item())
    y=int(torch.randint(3400,4700,(1,)).item())
    for j in range(8):
        img=bands_dict['b{}'.format(j+1)]
        img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
        if j<7:
            img=min_max_normalization(img,0,1)
        batch_1.append(img)
    batch_1=tuple(batch_1)
    traindata=torch.cat(batch_1[:-1],0).unsqueeze(0).float().cuda()
    label=batch_1[-1].long().cuda()
    
    net=net.eval()
    out=net(traindata)
    out=F.log_softmax(out,dim=1)
    loss=criterion(out,label)
    
    pred_labels=out.max(dim=1)[1]
    acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))
    
    if (epoch+1)%1==0 or epoch==0:
        print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
        _,figs=plt.subplots(1,3,figsize=(10,10))
        band4=traindata[:,3,:,:].cpu().view(1,512,512)
        band6=traindata[:,5,:,:].cpu().view(1,512,512)
        band5=traindata[:,4,:,:].cpu().view(1,512,512)
        ximg=torch.cat((band4,band6,band5),0).permute(1,2,0)
        yimg=label.cpu().view(512,512)
        zimg=pred_labels.view(512,512)
        figs[0].imshow(ximg)
        figs[1].imshow(yimg)
        figs[2].imshow(zimg)
#torch.save(net.state_dict(), './netdict_resnetANDfcn.pth')


# # epoch = 100

# In[6]:


import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
# from torchvision.datasets import CIFAR10
from datetime import datetime
import os
import gdal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)
    
class resnet(nn.Module):
    def __init__(self, in_channel, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512)
        )
        
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        return x
    
aresnet= resnet(7,True)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn,self).__init__()
        
        self.stage1=nn.Sequential(*list(aresnet.children())[:-2]) #128
        self.stage2=nn.Sequential(*list(aresnet.children())[-2])  #256
        self.stage3=nn.Sequential(*list(aresnet.children())[-1])  #512
        
        self.scores1=nn.Conv2d(512,num_classes,1)
        self.scores2=nn.Conv2d(256,num_classes,1)
        self.scores3=nn.Conv2d(128,num_classes,1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 0, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 3)
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 
    
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3) 
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s)
        return s
    
    
net=fcn(2)
net=net.cuda()
criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,weight_decay=1e-4)


band_fns= [im for im in os.listdir('./data/2014_121_64/croppedrst/') if (im.split('.')[-1]=='rst')]
bands_dict={}
for i in range(8):
    in_ds=gdal.Open('./data/2014_121_64/croppedrst/'+band_fns[i])
    in_band=in_ds.GetRasterBand(1)
    bands_dict['b{}'.format(i+1)]=in_band.ReadAsArray().astype('float')
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor
    

xlist=[3308,3300,3642,3814,4000,4166,4235,4380,4569,4363,4664,4286,4750,4166,3600,4200,3763,4114,3805,4226,5500,4827,5850,6114,6337,6509]  
ylist=[330,760,760,1265,1650,1935,2338,2600,2492,2784,2827,3042,2960,3136,3060,3522,3617,3908,4183,4277,5550,4800,5400,5487,5616,5625]


for epoch in range(100):#train1
    for i in range(len(xlist)):
        abatch=[]
        x=xlist[i]
        y=ylist[i]
        for j in range(8):
            img=bands_dict['b{}'.format(j+1)]
            img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
            if j<7:
                img=min_max_normalization(img,0,1)
            abatch.append(img)
        abatch=tuple(abatch) 
        
        acc=0
        traindata=Variable(torch.cat(abatch[:-1],0).unsqueeze(0).float().cuda())
        label=Variable(abatch[-1].long().cuda())

        net=net.train()
        out=net(traindata)
        out=F.log_softmax(out,dim=1)
        loss=criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels=out.max(dim=1)[1]
        acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))

        if (epoch+1)%33==0 or epoch==0:
            print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
            _,figs=plt.subplots(1,3,figsize=(10,10))
            band4=traindata[:,3,:,:].cpu().view(1,512,512)
            band6=traindata[:,5,:,:].cpu().view(1,512,512)
            band5=traindata[:,4,:,:].cpu().view(1,512,512)
            aimg=torch.cat((band4,band6,band5),0).permute(1,2,0)
            bimg=label.cpu().view(512,512)
            cimg=pred_labels.view(512,512)
            figs[0].imshow(aimg)
            figs[1].imshow(bimg)
            figs[2].imshow(cimg)
        
        
for epoch in range(30):#test
    acc=0
    batch_1=[]
    x=int(torch.randint(3700,4500,(1,)).item())
    y=int(torch.randint(3400,4700,(1,)).item())
    for j in range(8):
        img=bands_dict['b{}'.format(j+1)]
        img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
        if j<7:
            img=min_max_normalization(img,0,1)
        batch_1.append(img)
    batch_1=tuple(batch_1)
    traindata=torch.cat(batch_1[:-1],0).unsqueeze(0).float().cuda()
    label=batch_1[-1].long().cuda()
    
    net=net.eval()
    out=net(traindata)
    out=F.log_softmax(out,dim=1)
    loss=criterion(out,label)
    
    pred_labels=out.max(dim=1)[1]
    acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))
    
    if (epoch+1)%1==0 or epoch==0:
        print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
        _,figs=plt.subplots(1,3,figsize=(10,10))
        band4=traindata[:,3,:,:].cpu().view(1,512,512)
        band6=traindata[:,5,:,:].cpu().view(1,512,512)
        band5=traindata[:,4,:,:].cpu().view(1,512,512)
        ximg=torch.cat((band4,band6,band5),0).permute(1,2,0)
        yimg=label.cpu().view(512,512)
        zimg=pred_labels.view(512,512)
        figs[0].imshow(ximg)
        figs[1].imshow(yimg)
        figs[2].imshow(zimg)
#torch.save(net.state_dict(), './netdict_resnetANDfcn.pth')


# # epoch =200

# In[8]:


import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
# from torchvision.datasets import CIFAR10
from datetime import datetime
import os
import gdal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)
    
class resnet(nn.Module):
    def __init__(self, in_channel, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512)
        )
        
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        return x
    
aresnet= resnet(7,True)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn,self).__init__()
        
        self.stage1=nn.Sequential(*list(aresnet.children())[:-2]) #128
        self.stage2=nn.Sequential(*list(aresnet.children())[-2])  #256
        self.stage3=nn.Sequential(*list(aresnet.children())[-1])  #512
        
        self.scores1=nn.Conv2d(512,num_classes,1)
        self.scores2=nn.Conv2d(256,num_classes,1)
        self.scores3=nn.Conv2d(128,num_classes,1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 0, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 3)
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 
    
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3) 
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s)
        return s
    
    
net=fcn(2)
net=net.cuda()
criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,weight_decay=1e-4)


band_fns= [im for im in os.listdir('./data/2014_121_64/croppedrst/') if (im.split('.')[-1]=='rst')]
bands_dict={}
for i in range(8):
    in_ds=gdal.Open('./data/2014_121_64/croppedrst/'+band_fns[i])
    in_band=in_ds.GetRasterBand(1)
    bands_dict['b{}'.format(i+1)]=in_band.ReadAsArray().astype('float')
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor
    

xlist=[3308,3300,3642,3814,4000,4166,4235,4380,4569,4363,4664,4286,4750,4166,3600,4200,3763,4114,3805,4226,5500,4827,5850,6114,6337,6509]  
ylist=[330,760,760,1265,1650,1935,2338,2600,2492,2784,2827,3042,2960,3136,3060,3522,3617,3908,4183,4277,5550,4800,5400,5487,5616,5625]


for epoch in range(200):#train1
    for i in range(len(xlist)):
        abatch=[]
        x=xlist[i]
        y=ylist[i]
        for j in range(8):
            img=bands_dict['b{}'.format(j+1)]
            img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
            if j<7:
                img=min_max_normalization(img,0,1)
            abatch.append(img)
        abatch=tuple(abatch) 
        
        acc=0
        traindata=Variable(torch.cat(abatch[:-1],0).unsqueeze(0).float().cuda())
        label=Variable(abatch[-1].long().cuda())

        net=net.train()
        out=net(traindata)
        out=F.log_softmax(out,dim=1)
        loss=criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels=out.max(dim=1)[1]
        acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))

        if (epoch+1)%100==0 or epoch==0:
            print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
            _,figs=plt.subplots(1,3,figsize=(10,10))
            band4=traindata[:,3,:,:].cpu().view(1,512,512)
            band6=traindata[:,5,:,:].cpu().view(1,512,512)
            band5=traindata[:,4,:,:].cpu().view(1,512,512)
            aimg=torch.cat((band4,band6,band5),0).permute(1,2,0)
            bimg=label.cpu().view(512,512)
            cimg=pred_labels.view(512,512)
            figs[0].imshow(aimg)
            figs[1].imshow(bimg)
            figs[2].imshow(cimg)
        
        
for epoch in range(30):#test
    acc=0
    batch_1=[]
    x=int(torch.randint(3700,4500,(1,)).item())
    y=int(torch.randint(3400,4700,(1,)).item())
    for j in range(8):
        img=bands_dict['b{}'.format(j+1)]
        img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
        if j<7:
            img=min_max_normalization(img,0,1)
        batch_1.append(img)
    batch_1=tuple(batch_1)
    traindata=torch.cat(batch_1[:-1],0).unsqueeze(0).float().cuda()
    label=batch_1[-1].long().cuda()
    
    net=net.eval()
    out=net(traindata)
    out=F.log_softmax(out,dim=1)
    loss=criterion(out,label)
    
    pred_labels=out.max(dim=1)[1]
    acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))
    
    if (epoch+1)%1==0 or epoch==0:
        print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
        _,figs=plt.subplots(1,3,figsize=(10,10))
        band4=traindata[:,3,:,:].cpu().view(1,512,512)
        band6=traindata[:,5,:,:].cpu().view(1,512,512)
        band5=traindata[:,4,:,:].cpu().view(1,512,512)
        ximg=torch.cat((band4,band6,band5),0).permute(1,2,0)
        yimg=label.cpu().view(512,512)
        zimg=pred_labels.view(512,512)
        figs[0].imshow(ximg)
        figs[1].imshow(yimg)
        figs[2].imshow(zimg)
#torch.save(net.state_dict(), './netdict_resnetANDfcn.pth')


# # epoch = 1000

# In[5]:


import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
# from torchvision.datasets import CIFAR10
from datetime import datetime
import os
import gdal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)
    
class resnet(nn.Module):
    def __init__(self, in_channel, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512)
        )
        
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        return x
    
aresnet= resnet(7,True)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn,self).__init__()
        
        self.stage1=nn.Sequential(*list(aresnet.children())[:-2]) #128
        self.stage2=nn.Sequential(*list(aresnet.children())[-2])  #256
        self.stage3=nn.Sequential(*list(aresnet.children())[-1])  #512
        
        self.scores1=nn.Conv2d(512,num_classes,1)
        self.scores2=nn.Conv2d(256,num_classes,1)
        self.scores3=nn.Conv2d(128,num_classes,1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 0, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 3)
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 
    
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3) 
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s)
        return s
    
    
net=fcn(2)
net=net.cuda()
criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,weight_decay=1e-4)


band_fns= [im for im in os.listdir('./data/2014_121_64/croppedrst/') if (im.split('.')[-1]=='rst')]
bands_dict={}
for i in range(8):
    in_ds=gdal.Open('./data/2014_121_64/croppedrst/'+band_fns[i])
    in_band=in_ds.GetRasterBand(1)
    bands_dict['b{}'.format(i+1)]=in_band.ReadAsArray().astype('float')
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor
    

xlist=[3308,3300,3642,3814,4000,4166,4235,4380,4569,4363,4664,4286,4750,4166,3600,4200,3763,4114,3805,4226,5500,4827,5850,6114,6337,6509]  
ylist=[330,760,760,1265,1650,1935,2338,2600,2492,2784,2827,3042,2960,3136,3060,3522,3617,3908,4183,4277,5550,4800,5400,5487,5616,5625]


for epoch in range(1000):#train1
    for i in range(len(xlist)):
        abatch=[]
        x=xlist[i]
        y=ylist[i]
        for j in range(8):
            img=bands_dict['b{}'.format(j+1)]
            img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
            if j<7:
                img=min_max_normalization(img,0,1)
            abatch.append(img)
        abatch=tuple(abatch) 
        
        acc=0
        traindata=Variable(torch.cat(abatch[:-1],0).unsqueeze(0).float().cuda())
        label=Variable(abatch[-1].long().cuda())

        net=net.train()
        out=net(traindata)
        out=F.log_softmax(out,dim=1)
        loss=criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels=out.max(dim=1)[1]
        acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))

        if (epoch+1)%250==0 or epoch==0:
            print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
            _,figs=plt.subplots(1,3,figsize=(10,10))
            band4=traindata[:,3,:,:].cpu().view(1,512,512)
            band6=traindata[:,5,:,:].cpu().view(1,512,512)
            band5=traindata[:,4,:,:].cpu().view(1,512,512)
            aimg=torch.cat((band4,band6,band5),0).permute(1,2,0)
            bimg=label.cpu().view(512,512)
            cimg=pred_labels.view(512,512)
            figs[0].imshow(aimg)
            figs[1].imshow(bimg)
            figs[2].imshow(cimg)
        
        
for epoch in range(30):#test
    acc=0
    batch_1=[]
    x=int(torch.randint(3700,4500,(1,)).item())
    y=int(torch.randint(3400,4700,(1,)).item())
    for j in range(8):
        img=bands_dict['b{}'.format(j+1)]
        img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
        if j<7:
            img=min_max_normalization(img,0,1)
        batch_1.append(img)
    batch_1=tuple(batch_1)
    traindata=torch.cat(batch_1[:-1],0).unsqueeze(0).float().cuda()
    label=batch_1[-1].long().cuda()
    
    net=net.eval()
    out=net(traindata)
    out=F.log_softmax(out,dim=1)
    loss=criterion(out,label)
    
    pred_labels=out.max(dim=1)[1]
    acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))
    
    if (epoch+1)%1==0 or epoch==0:
        print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
        _,figs=plt.subplots(1,3,figsize=(10,10))
        band4=traindata[:,3,:,:].cpu().view(1,512,512)
        band6=traindata[:,5,:,:].cpu().view(1,512,512)
        band5=traindata[:,4,:,:].cpu().view(1,512,512)
        ximg=torch.cat((band4,band6,band5),0).permute(1,2,0)
        yimg=label.cpu().view(512,512)
        zimg=pred_labels.view(512,512)
        figs[0].imshow(ximg)
        figs[1].imshow(yimg)
        figs[2].imshow(zimg)
#torch.save(net.state_dict(), './netdict_resnetANDfcn.pth')


# # epoch = 2000

# In[9]:


import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
# from torchvision.datasets import CIFAR10
from datetime import datetime
import os
import gdal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)
    
class resnet(nn.Module):
    def __init__(self, in_channel, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512)
        )
        
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        return x
    
aresnet= resnet(7,True)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn,self).__init__()
        
        self.stage1=nn.Sequential(*list(aresnet.children())[:-2]) #128
        self.stage2=nn.Sequential(*list(aresnet.children())[-2])  #256
        self.stage3=nn.Sequential(*list(aresnet.children())[-1])  #512
        
        self.scores1=nn.Conv2d(512,num_classes,1)
        self.scores2=nn.Conv2d(256,num_classes,1)
        self.scores3=nn.Conv2d(128,num_classes,1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 0, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 3)
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 
    
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3) 
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s)
        return s
    
    
net=fcn(2)
net=net.cuda()
criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,weight_decay=1e-4)


band_fns= [im for im in os.listdir('./data/2014_121_64/croppedrst/') if (im.split('.')[-1]=='rst')]
bands_dict={}
for i in range(8):
    in_ds=gdal.Open('./data/2014_121_64/croppedrst/'+band_fns[i])
    in_band=in_ds.GetRasterBand(1)
    bands_dict['b{}'.format(i+1)]=in_band.ReadAsArray().astype('float')
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor
    

xlist=[3308,3300,3642,3814,4000,4166,4235,4380,4569,4363,4664,4286,4750,4166,3600,4200,3763,4114,3805,4226,5500,4827,5850,6114,6337,6509]  
ylist=[330,760,760,1265,1650,1935,2338,2600,2492,2784,2827,3042,2960,3136,3060,3522,3617,3908,4183,4277,5550,4800,5400,5487,5616,5625]


for epoch in range(2000):#train1
    for i in range(len(xlist)):
        abatch=[]
        x=xlist[i]
        y=ylist[i]
        for j in range(8):
            img=bands_dict['b{}'.format(j+1)]
            img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
            if j<7:
                img=min_max_normalization(img,0,1)
            abatch.append(img)
        abatch=tuple(abatch) 
        
        acc=0
        traindata=Variable(torch.cat(abatch[:-1],0).unsqueeze(0).float().cuda())
        label=Variable(abatch[-1].long().cuda())

        net=net.train()
        out=net(traindata)
        out=F.log_softmax(out,dim=1)
        loss=criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels=out.max(dim=1)[1]
        acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))

        if (epoch+1)%500==0 or epoch==0:
            print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
            _,figs=plt.subplots(1,3,figsize=(10,10))
            band4=traindata[:,3,:,:].cpu().view(1,512,512)
            band6=traindata[:,5,:,:].cpu().view(1,512,512)
            band5=traindata[:,4,:,:].cpu().view(1,512,512)
            aimg=torch.cat((band4,band6,band5),0).permute(1,2,0)
            bimg=label.cpu().view(512,512)
            cimg=pred_labels.view(512,512)
            figs[0].imshow(aimg)
            figs[1].imshow(bimg)
            figs[2].imshow(cimg)
        
        
for epoch in range(30):#test
    acc=0
    batch_1=[]
    x=int(torch.randint(3700,4500,(1,)).item())
    y=int(torch.randint(3400,4700,(1,)).item())
    for j in range(8):
        img=bands_dict['b{}'.format(j+1)]
        img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
        if j<7:
            img=min_max_normalization(img,0,1)
        batch_1.append(img)
    batch_1=tuple(batch_1)
    traindata=torch.cat(batch_1[:-1],0).unsqueeze(0).float().cuda()
    label=batch_1[-1].long().cuda()
    
    net=net.eval()
    out=net(traindata)
    out=F.log_softmax(out,dim=1)
    loss=criterion(out,label)
    
    pred_labels=out.max(dim=1)[1]
    acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))
    
    if (epoch+1)%1==0 or epoch==0:
        print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
        _,figs=plt.subplots(1,3,figsize=(10,10))
        band4=traindata[:,3,:,:].cpu().view(1,512,512)
        band6=traindata[:,5,:,:].cpu().view(1,512,512)
        band5=traindata[:,4,:,:].cpu().view(1,512,512)
        ximg=torch.cat((band4,band6,band5),0).permute(1,2,0)
        yimg=label.cpu().view(512,512)
        zimg=pred_labels.view(512,512)
        figs[0].imshow(ximg)
        figs[1].imshow(yimg)
        figs[2].imshow(zimg)
#torch.save(net.state_dict(), './netdict_resnetANDfcn.pth')


# # more training images

# In[25]:


import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
# from torchvision.datasets import CIFAR10
from datetime import datetime
import os
import gdal
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)
    
class resnet(nn.Module):
    def __init__(self, in_channel):
        super(resnet, self).__init__()
        
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512)
        )
        
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x
    
aresnet= resnet(7,True)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn,self).__init__()
        
        self.stage1=nn.Sequential(*list(aresnet.children())[:-2]) #128
        self.stage2=nn.Sequential(*list(aresnet.children())[-2])  #256
        self.stage3=nn.Sequential(*list(aresnet.children())[-1])  #512
        
        self.scores1=nn.Conv2d(512,num_classes,1)
        self.scores2=nn.Conv2d(256,num_classes,1)
        self.scores3=nn.Conv2d(128,num_classes,1)
        
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 0, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) 
        
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 3, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 3)
        
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) 
    
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        
        x = self.stage2(x)
        s2 = x # 1/16
        
        x = self.stage3(x)
        s3 = x # 1/32
        
        s3 = self.scores1(s3) 
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2
        s = self.upsample_8x(s)
        return s
    
    
net=fcn(2)
net=net.cuda()
criterion=nn.NLLLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,weight_decay=1e-4)


band_fns= [im for im in os.listdir('./data/2014_121_64/croppedrst/') if (im.split('.')[-1]=='rst')]
bands_dict={}
for i in range(8):
    in_ds=gdal.Open('./data/2014_121_64/croppedrst/'+band_fns[i])
    in_band=in_ds.GetRasterBand(1)
    bands_dict['b{}'.format(i+1)]=in_band.ReadAsArray().astype('float')
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor
    

xlist=[]  
ylist=[]
x_sublist=[i for i in range(3500,4700) if i%50==0 or i==3500]
y_sublist=[int(2.25*i-7375) for i in range(3500,4700) if i%50==0 or i==3500]
for sublen in range(len(x_sublist)):
    xlist.append(x_sublist[sublen])
    ylist.append(y_sublist[sublen])
x_sublist=[i for i in range(3650,4650) if i%50==0 or i==3650]
y_sublist=[int(1.4*i-1910) for i in range(3650,4650) if i%50==0 or i==3650]
for sublen in range(len(x_sublist)):
    xlist.append(x_sublist[sublen])
    ylist.append(y_sublist[sublen])
x_sublist=[i for i in range(3800,4600) if i%50==0 or i==3800]
y_sublist=[int(-2.375*i+13425) for i in range(3800,4600) if i%50==0 or i==3800]
for sublen in range(len(x_sublist)):
    xlist.append(x_sublist[sublen])
    ylist.append(y_sublist[sublen])
x_sublist=[i for i in range(5450,6600) if i%50==0 or i==5450]
y_sublist=[int(0.174*i+4601.7) for i in range(5450,6600) if i%50==0 or i==5450]
for sublen in range(len(x_sublist)):
    xlist.append(x_sublist[sublen])
    ylist.append(y_sublist[sublen])
    
for epoch in range(3000):#train1
    losslist=[]
    acclist=[]
    for i in range(len(xlist)):
        
        abatch=[]
        x=xlist[i]
        y=ylist[i]
        for j in range(8):
            img=bands_dict['b{}'.format(j+1)]
            img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
            if j<7:
                img=min_max_normalization(img,0,1)
            abatch.append(img)
        abatch=tuple(abatch) 
        
        acc=0
        traindata=Variable(torch.cat(abatch[:-1],0).unsqueeze(0).float().cuda())
        label=Variable(abatch[-1].long().cuda())

        net=net.train()
        out=net(traindata)
        out=F.log_softmax(out,dim=1)
        loss=criterion(out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels=out.max(dim=1)[1]
        acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))
        
        losslist.append(loss.data)
        acclist.append(acc*100)
        
            
        if (epoch+1)%3000==0 :
            _,figs=plt.subplots(1,3,figsize=(10,10))
            band4=traindata[:,3,:,:].cpu().view(1,512,512)
            band6=traindata[:,5,:,:].cpu().view(1,512,512)
            band5=traindata[:,4,:,:].cpu().view(1,512,512)
            aimg=torch.cat((band4,band6,band5),0).permute(1,2,0)
            bimg=label.cpu().view(512,512)
            cimg=pred_labels.view(512,512)
            figs[0].imshow(aimg)
            figs[1].imshow(bimg)
            figs[2].imshow(cimg)
    if (epoch+1)%300==0 or epoch==0:
        print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,sum(losslist)/len(losslist),sum(acclist)/len(acclist)))
        
for epoch in range(30):#test
    acc=0
    batch_1=[]
    x=int(torch.randint(3700,4500,(1,)).item())
    y=int(torch.randint(3400,4700,(1,)).item())
    for j in range(8):
        img=bands_dict['b{}'.format(j+1)]
        img=torch.from_numpy(img[x:x+512,y:y+512]).unsqueeze(0)
        if j<7:
            img=min_max_normalization(img,0,1)
        batch_1.append(img)
    batch_1=tuple(batch_1)
    traindata=torch.cat(batch_1[:-1],0).unsqueeze(0).float().cuda()
    label=batch_1[-1].long().cuda()
    
    net=net.eval()
    out=net(traindata)
    out=F.log_softmax(out,dim=1)
    loss=criterion(out,label)
    
    pred_labels=out.max(dim=1)[1]
    acc=(pred_labels==label).sum().item()/(label.size(1)*label.size(2))
    
    if (epoch+1)%1==0 or epoch==0:
        print('epoch:{} ,Loss: {}, Accuracy: {:.4f}%'.format(epoch+1,loss.data,acc*100.0))
        _,figs=plt.subplots(1,3,figsize=(10,10))
        band4=traindata[:,3,:,:].cpu().view(1,512,512)
        band6=traindata[:,5,:,:].cpu().view(1,512,512)
        band5=traindata[:,4,:,:].cpu().view(1,512,512)
        ximg=torch.cat((band4,band6,band5),0).permute(1,2,0)
        yimg=label.cpu().view(512,512)
        zimg=pred_labels.view(512,512)
        figs[0].imshow(ximg)
        figs[1].imshow(yimg)
        figs[2].imshow(zimg)
#torch.save(net.state_dict(), './netdict_resnetANDfcn.pth')


# In[6]:


from torchvision.models import resnet34
net = resnet34(pretrained=True)
list(net.children())[-3]

