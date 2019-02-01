'''
Author: Boka Luo
Date: Dec 24, 2018
Description:
This file is to create dataset of remote sensing for pytorch, including data loading, channel construction, small patches cropping and data normalization.
It is also appliable to large scale raster data with crop needs
'''

#module import
import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import time
import gc

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


#function to get parameters for pytorch normalization
def normPara(filepath,dtype=np.float32):
    print('--------------getting parameters for normalization--------------')
    files=[m for m in os.listdir(filepath) if(m.endswith('.rst'))]
    mean=[]
    std=[]
    for file in files[:-1]:
        mean.append(file.mean())
        std.append(file.std())
    print('mean sequence: {}\nstd sequence: {}'.format(mean,std))
    return mean,std


#function to operate min max normalization
def mmNorm (inArray):
    nom=inArray-inArray.min()
    denom=inArray.max()-inArray.min()
    return nom/denom


#load data as tensor
def load_data(root,file,mNorm=True,dtype=np.float32):
    print('processing file:{}'.format(file))
    x=gdal.Open(os.path.join(root,file))
    x=x.ReadAsArray().astype(dtype)
    if mNorm:
        x=mmNorm(x)
    return x

#transformation functions
##center rotation for images with any number of channels
def centerRotate(data,degree):
    (h,w)=data.shape[:2]
    center=(w//2,h//2)
    rotMtrx=cv2.getRotationMatrix2D(center,degree,1.0)
    rotated=cv2.warpAffine(data,rotMtrx,(w,h))
    return rotated

##flip for images with any number of channels
def flip(data, ftype='vflip'):
    if ftype=='vflip':
        return cv2.flip(data,0)
    if ftype=='hflip':
        return cv2.flip(data,1)
    if ftype=='dflip':
        return cv2.flip(data,-1)

#get index for cropping
def batchIndex(cropRef,dsize,samp_num,trainIn,test,randImg=False):
    x=np.argwhere(cropRef>0)
    if randImg==False:
        x_min=min(x[:,0])
        x_max=max(x[:,0])
        y_min=min(x[:,1])
        y_max=max(x[:,1])
    else:
        x_min=min(x[:,0])-random.randint(10,50)
        x_max=max(x[:,0])+random.randint(10,50)
        y_min=min(x[:,1])-random.randint(10,50)
        y_max=max(x[:,1])+random.randint(10,50)
    
    index=[]
    j=y_min
    for j in range(y_min,y_max+1,dsize):
        for i in range(x_min,x_max+1,dsize):
            ref=cropRef[i-dsize//2:i+dsize//2,j-dsize//2:j+dsize//2]
            if ref.any()==1:
                index.append([i,j])
                
    print('#index:{}'.format(len(index)))
    
    if samp_num>len(index)//2:   
        if test:
            test_index=[m for m in index if (m not in trainIn)]
            print('testIn:\n{}'.format(test_index))
            return test_index
        else:
            train_index=random.sample(index,len(index)//2)
            print('trainIn:\n{}'.format(train_index))
            return train_index
    else:
        if test:
            test_index=random.sample([m for m in index if (m not in traindIn)],samp_num)
            print('testIn:\n{}'.format(test_index))
            return test_index
        else:
            train_index=random.sample(index,samp_num)
            print('trainIn:\n{}'.format(train_index))
            return train_index


#extract labels from loaded data
def getImageLabel(data):
     #construct image
    img=np.concatenate(data[:-1],-1)
    #check  if image is successfully constructed
    _,_,img_c=img.shape  #extract channel counts
    for i in range(img_c):
        if img[:,:,i].all()==data[i][:,:,0].all():
            print ("Constructed {} channels".format(i+1))
        else:
            print ("Failed to construct image")
    #construct label
    label=np.squeeze(data[-1])
    return img, label


#dataset of remote sensing for pytorch
class rstData(Dataset):
      
    def __init__(self,datapath,dsize,samp_num=20,randImg=False,trainIn=None,cropIn=None,test=False,mNorm=True,flip=None,deRotate=None,transform=None):
        
        self.deRotate=deRotate
        #extract data from files as narray
        files=[m for m in os.listdir(datapath) if(m.endswith('.rst'))]
        self.file=files
        data=[]
        print('--------------loading data--------------')
        for i in files:
            dataLoad=load_data(datapath,i,mNorm)  ##expand dataset
            data.append(np.expand_dims(dataLoad,-1))
    
    
        #construct image
        print('--------------constructing image channels--------------')
        img,label=getImageLabel(data)
        
        #create index for patches cropping
        print('---------------cropping {}x{} batches--------------'.format(dsize,dsize))
        if cropIn==None:
            index=batchIndex(label,dsize,samp_num,trainIn,test,randImg)
            gc.collect() #release memory
            self.crop_x=[]
            self.crop_y=[]
            for i in range(len(index)):
                self.crop_x.append(index[i][0])
                self.crop_y.append(index[i][1]) 
        else:
            index=cropIn
            self.crop_x=[]
            self.crop_y=[]
            for i in range(len(index)):
                self.crop_x.append(index[i][0])
                self.crop_y.append(index[i][1])
                  
                  
        #create_dataset
        self.img=[]
        self.label=[]
        self.coor=[]
        for i in range(len(self.crop_x)):
            x=self.crop_x[i]
            y=self.crop_y[i]
            self.img.append(img[x-(dsize//2):x+(dsize//2),y-(dsize//2):y+(dsize//2),:])
            self.label.append(label[x-(dsize//2):x+(dsize//2),y-dsize//2:y+(dsize//2)])
            self.coor.append([x,y])
            #data augmentation-flip
            if flip!=None:
                  for ftype in flip:
                        self.img.append(flip(img[x-(dsize//2):x+(dsize//2),y-dsize//2:y+(dsize//2),:],ftype))
                        self.label.append(flip(label[x-(dsize//2):x+(dsize//2),y-dsize//2:y+(dsize//2)],ftype))
                        self.coor.append([x,y])
            
        print('-------------{} bathces cropped-----------'.format(len(self.img)))
    
                  
    def __getitem__(self,index):
        img=self.img[index]
        label=self.label[index]
        coor=self.coor[index]
        deRotate=self.deRotate
        if deRotate!=None:
            degree=random.uniform(deRotate)
            label=centerRotate(label,degree)
            img=centerRotate(img,degree)
        #transform numpy array to tensor
        label=torch.from_numpy(label).long()
        img=torch.from_numpy(img.transpose((2, 0, 1)))
        return img, label,index, coor
       
        
    def __len__(self):
        return len(self.img)
