'''
Author: Boka Luo
Date: Dec 24, 2018
Description:
This file is to create dataset of remote sensing for pytorch, including data loading, channel construction, small patches cropping and data normalization.
It is also appliable to large scale raster data with crop needs
'''

import gdal
import numpy as np
import random
import gc

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


#function to get parameters for pytorch normalization
def normPara(filepath,dtype=np.float32):
    print('--------------getting parameters for normalization--------------')
    files=[m for m in os.listdir(filepath) if(m.endswith('.rst'))]
    img_data=[]
    mean=[]
    std=[]
    for i in range(len(files[:-1])):
        img_data.append(load_data(filepath,files[i]))
        mean.append(torch.mean(img_data[i]))
        std.append(torch.std(img_data[i]))
    print('mean sequence: {}\nstd sequence: {}'.format(mean,std))
    return mean,std


#function to operate min max normalization
def mmNorm (inTensor):
    nom=inTensor-inTensor.min()
    denom=inTensor.max()-inTensor.min()
    return nom/denom


#load data as tensor
def load_data(root,file,mNorm=False,dtype=np.float32):   
    x=gdal.Open(os.path.join(root,file))
    x=x.ReadAsArray().astype(dtype)
    x=torch.from_numpy(x)
    if mNorm:
        x=mmNorm(x)
    return x   


#get index for cropping
def batchIndex(cropRef,dsize,samp_num,trainIn,test,randImg=False):
    cropRef=cropRef.numpy()
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
            if [i,j] in x.tolist():
                index.append([i,j])
                
    print(index)#
    print('#index:{}'.format(len(index)))
    
    if samp_num>len(index)//2:   
        if test:
            test_index=random.sample([m for m in index if m not in trainIn],len(index)//2) #half of cropped images not for train are for test
            print('testIn:\n{}'.format(test_index))
            return test_index
        else:
            train_index=random.sample(index,len(index)//2)
            print('trainIn:\n{}'.format(train_index))
            return train_index
    else:
        if test:
            test_index=random.sample([m for m in index if m not in traindIn],samp_num)
            print('testIn:\n{}'.format(test_index))
            return test_index
        else:
            train_index=random.sample(index,samp_num)
            print('trainIn:\n{}'.format(train_index))
            return train_index


#extract labels from loaded data
def getImageLabel(data):
     #construct image
    img=torch.cat(data[:-1],0)
    #check  if image is successfully constructed
    img_c,_,_=img.size()  #extract channel counts
    for i in range(img_c):
        if torch.all(torch.eq(img[i,:,:], data[i][0,:,:])):
            print ("Constructed {} channels".format(i+1))
        else:
            print ("Failed to construct image")
    #construct label
    label=data[-1].squeeze().long()
    return img, label


#dataset of remote sensing for pytorch
class rstData(Dataset):

    def __init__(self,datapath,dsize,samp_num=20,randImg=False,trainIn=None,cropIn=None,test=False,mNorm=False,transform=None):
        
        self.transform=transform
        #extract data from files as tensor
        files=[m for m in os.listdir(datapath) if(m.endswith('.rst'))]
        self.file=files
        data=[]
        print('--------------loading data--------------')
        for i in range(len(files)):
            print('processing file:{}'.format(files[i]))
            
            data.append(load_data(datapath,files[i],mNorm))
            data[i]=data[i].unsqueeze(0)
            
        #construct image
        print('--------------constructing image channels--------------')
        img, label= getImageLabel(data)
       
       
        #create index for batch cropping
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

        
        #construct img, label batches by cropping image and label using created index
        self.img=[]
        self.label=[]
        for i in range(len(self.crop_x)):
            x=self.crop_x[i]
            y=self.crop_y[i]
            self.img.append(img[:,x-(dsize//2):x+(dsize//2),y-(dsize//2):y+(dsize//2)])
            self.label.append(label[x-(dsize//2):x+(dsize//2),y-(dsize//2):y+(dsize//2)])
        print('-------------{} bathces cropped-----------'.format(len(self.img)))
   
            
    def __getitem__(self,index,transform=None):
        img=self.img[index]
        label=self.label[index]
        if transform!= None:
            img=self.transform(img)
        return img, label, index
       
        
    def __len__(self):
        return len(self.img)
