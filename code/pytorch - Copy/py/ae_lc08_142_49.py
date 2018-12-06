import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import numpy
from PIL import Image
import torch
from osgeo import gdal


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 3),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(True)
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




os.chdir('E:/pyScripts/pytorch/data/142_49/LC08_L1TP_142049_20170120_20170311_01_T1/')

band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B1.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band1=torch.from_numpy(in_band.ReadAsArray().astype('float'))[4000:4700,3000:3500]
band1=min_max_normalization(band1,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B2.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band2=torch.from_numpy(in_band.ReadAsArray().astype('float'))[4000:4700,3000:3500]
band2=min_max_normalization(band2,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B3.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band3=torch.from_numpy(in_band.ReadAsArray().astype('float'))[4000:4700,3000:3500]
band3=min_max_normalization(band3,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B4.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band4=torch.from_numpy(in_band.ReadAsArray().astype('float'))[4000:4700,3000:3500]
band4=min_max_normalization(band4,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B5.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band5=torch.from_numpy(in_band.ReadAsArray().astype('float'))[4000:4700,3000:3500]
band5=min_max_normalization(band5,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B6.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band6=torch.from_numpy(in_band.ReadAsArray().astype('float'))[4000:4700,3000:3500]
band6=min_max_normalization(band6,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B7.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band7=torch.from_numpy(in_band.ReadAsArray().astype('float'))[4000:4700,3000:3500]
band7=min_max_normalization(band7,0.0,1.0)


num_epochs=300

learning_rate = 1e-4
#batch_size=55

model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)



for epoch in range(num_epochs):
    for i in range(0,band1.size(0)):
        b1_input=band1[i].view(-1,1)
        b2_input=band2[i].view(-1,1)
        b3_input=band3[i].view(-1,1)
        b4_input=band4[i].view(-1,1)
        b5_input=band5[i].view(-1,1)
        b6_input=band6[i].view(-1,1)
        b7_input=band7[i].view(-1,1)
        bands_input = torch.cat((
            b1_input,b2_input,b3_input,b4_input,b5_input,b6_input,b7_input),1).float()
        bands_input=bands_input.cuda()
        output=model(bands_input)
        loss = criterion(output, bands_input)
        MSE_loss = nn.MSELoss()(output, bands_input)
        # ===================backward====================
        optimizer.zero_grad()
#         MSE_loss.backward()
        loss.backward()
        optimizer.step()
        # ===================log========================
    if (epoch+1)%1==0:
        print('epoch [{}/{}], loss:{:.7f}, MSE_loss:{:.7f}'.format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
    
torch.save(model.state_dict(), './sim_autoencoder.pth')



# calculate the whole IMG outputs

band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B1.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band1=torch.from_numpy(in_band.ReadAsArray().astype('float'))[2000:6000,2000:6000]
band1=min_max_normalization(band1,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B2.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band2=torch.from_numpy(in_band.ReadAsArray().astype('float'))[2000:6000,2000:6000]
band2=min_max_normalization(band2,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B3.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band3=torch.from_numpy(in_band.ReadAsArray().astype('float'))[2000:6000,2000:6000]
band3=min_max_normalization(band3,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B4.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band4=torch.from_numpy(in_band.ReadAsArray().astype('float'))[2000:6000,2000:6000]
band4=min_max_normalization(band4,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B5.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band5=torch.from_numpy(in_band.ReadAsArray().astype('float'))[2000:6000,2000:6000]
band5=min_max_normalization(band5,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B6.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band6=torch.from_numpy(in_band.ReadAsArray().astype('float'))[2000:6000,2000:6000]
band6=min_max_normalization(band6,0.0,1.0)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B7.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band7=torch.from_numpy(in_band.ReadAsArray().astype('float'))[2000:6000,2000:6000]
band7=min_max_normalization(band7,0.0,1.0)


outlayers_tupple = []
hiddenlayer_tupple=[]
for i in range(0,band1.size(0)):
    b1_input=band1[i].view(-1,1)
    b2_input=band2[i].view(-1,1)
    b3_input=band3[i].view(-1,1)
    b4_input=band4[i].view(-1,1)
    b5_input=band5[i].view(-1,1)
    b6_input=band6[i].view(-1,1)
    b7_input=band7[i].view(-1,1)
    bands_input = torch.cat((
        b1_input,b2_input,b3_input,b4_input,b5_input,b6_input,b7_input),1).float()
    bands_input=bands_input.cuda()
    
    outputlayers=model(bands_input)
    outputlayers=outputlayers.cpu()
    output_hidden=model.encoder(bands_input).cpu()
    
    outlayers_tupple.append(outputlayers)
    hiddenlayer_tupple.append(output_hidden)

outlayers_tupple=tuple(outlayers_tupple)
hiddenlayer_tupple=tuple(hiddenlayer_tupple)

outlayer_data = torch.cat(outlayers_tupple,0)
hiddenlayer_data = torch.cat(hiddenlayer_tupple,0)


gtiff_driver = gdal.GetDriverByName('GTiff')
out_ds = gtiff_driver.Create(
    'out_wholeIMG_times10000.tif',band1.size(1), band1.size(0), 7, in_band.DataType)
out_ds.SetProjection(in_ds.GetProjection())
out_ds.SetGeoTransform((386385.0+30*2000, 30.0, 0.0, 1873815.0-30.0*2000, 0.0, -30.0),)
for j in range(7):
    out_band = out_ds.GetRasterBand((j+1))
    readyData=(outlayer_data[:,j].view(band1.size(0),band1.size(1))).cpu().detach().numpy()
    readyData=readyData*10000
    print(readyData.shape)
    out_band.WriteArray(readyData)
out_ds.FlushCache()
for k in range(1, 7):
    out_ds.GetRasterBand(k).ComputeStatistics(False)
del out_ds
readyData_blue=(outlayer_data[:,1].view(1,band1.size(0),band1.size(1)))
readyData_green=(outlayer_data[:,2].view(1,band1.size(0),band1.size(1)))
readyData_red=(outlayer_data[:,3].view(1,band1.size(0),band1.size(1)))
readyData_BGR=torch.cat((readyData_blue,readyData_green,readyData_red),0)
save_image(readyData_BGR,'out_wholeIMG_BRG.png')



gtiff_driver = gdal.GetDriverByName('GTiff')
out_ds = gtiff_driver.Create(
    'out_hiddenwholeIMG_times10000.tif',band1.size(1), band1.size(0), 3, in_band.DataType)
out_ds.SetProjection(in_ds.GetProjection())
out_ds.SetGeoTransform((386385.0+30*2000, 30.0, 0.0, 1873815.0-30.0*2000, 0.0, -30.0),)
for j in range(3):
    out_band = out_ds.GetRasterBand((j+1))
    readyData=(hiddenlayer_data[:,j].view(band1.size(0),band1.size(1))).cpu().detach().numpy()
    readyData=readyData*10000
    print(readyData.shape)
    out_band.WriteArray(readyData)
out_ds.FlushCache()
for k in range(1, 3):
    out_ds.GetRasterBand(k).ComputeStatistics(False)
del out_ds
readyData_blue=(hiddenlayer_data[:,0].view(1,band1.size(0),band1.size(1)))
readyData_green=(hiddenlayer_data[:,1].view(1,band1.size(0),band1.size(1)))
readyData_red=(hiddenlayer_data[:,2].view(1,band1.size(0),band1.size(1)))
readyData_BGR=torch.cat((readyData_blue,readyData_green,readyData_red),0)
save_image(readyData_BGR,'out_hiddenwholeIMG_BRG.png')



print('all set!')