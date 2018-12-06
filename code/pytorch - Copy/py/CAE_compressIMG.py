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
            nn.Conv2d(7,32,3,stride=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32,16,3,stride=2,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,32,3,stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,16,5,stride=3,padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,7,2,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



os.chdir('E:/Moore/pyScripts/pytorch/data/142_49/LC08_L1TP_142049_20170120_20170311_01_T1/')

band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B1.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band1=torch.from_numpy(in_band.ReadAsArray(2000,2000,4000,4000).astype('float'))
band1=min_max_normalization(band1,0.0,1.0).view(1,1,4000,4000)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B2.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band2=torch.from_numpy(in_band.ReadAsArray(2000,2000,4000,4000).astype('float'))
band2=min_max_normalization(band2,0.0,1.0).view(1,1,4000,4000)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B3.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band3=torch.from_numpy(in_band.ReadAsArray(2000,2000,4000,4000).astype('float'))
band3=min_max_normalization(band3,0.0,1.0).view(1,1,4000,4000)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B4.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band4=torch.from_numpy(in_band.ReadAsArray(2000,2000,4000,4000).astype('float'))
band4=min_max_normalization(band4,0.0,1.0).view(1,1,4000,4000)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B5.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band5=torch.from_numpy(in_band.ReadAsArray(2000,2000,4000,4000).astype('float'))
band5=min_max_normalization(band5,0.0,1.0).view(1,1,4000,4000)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B6.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band6=torch.from_numpy(in_band.ReadAsArray(2000,2000,4000,4000).astype('float'))
band6=min_max_normalization(band6,0.0,1.0).view(1,1,4000,4000)


band_fn='LC08_L1TP_142049_20170120_20170311_01_T1_B7.TIF'
in_ds = gdal.Open(band_fn)
in_band=in_ds.GetRasterBand(1)
band7=torch.from_numpy(in_band.ReadAsArray(2000,2000,4000,4000).astype('float'))
band7=min_max_normalization(band7,0.0,1.0).view(1,1,4000,4000)

bands_input = torch.cat((band1,band2,band3,band4,band5,band6,band7),1).float().cuda()


num_epochs=20000

learning_rate = 1e-3
#batch_size=55

model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)



for epoch in range(num_epochs):
    output=model(bands_input)
    loss = criterion(output, bands_input)
    MSE_loss = nn.MSELoss(reduction='sum')(output, bands_input)
    # ===================backward====================
    optimizer.zero_grad()
#         MSE_loss.backward()
    MSE_loss.backward()
    optimizer.step()
    # ===================log========================
    if (epoch+1)%100==0:
        print('epoch [{}/{}], loss:{:.7f}, MSE_loss:{:.7f}'.format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
    
# torch.save(model.state_dict(), './sim_autoencoder.pth')



# calculate the whole IMG outputs



outputlayers=model(bands_input)
outputlayers=outputlayers.cpu().view(7,4000,4000)
output_hidden=model.encoder(bands_input).cpu().permute(1,0,2,3)
output_hidden=min_max_normalization(output_hidden,0.0,1.0)


gtiff_driver = gdal.GetDriverByName('GTiff')
out_ds = gtiff_driver.Create(
    r'E:\Moore\pyScripts\pytorch\data\142_49\CAE_outputs\out_wholeIMG_times10000.tif',band1.size(3), band1.size(2), 7, in_band.DataType)
out_ds.SetProjection(in_ds.GetProjection())
out_ds.SetGeoTransform((386385.0+30*2000, 30.0, 0.0, 1873815.0-30.0*2000, 0.0, -30.0),)
for j in range(7):
    out_band = out_ds.GetRasterBand((j+1))
    readyData=(outputlayers[j,:,:].view(band1.size(2),band1.size(3))).cpu().detach().numpy()
    readyData=readyData*10000
    out_band.WriteArray(readyData)
out_ds.FlushCache()
for k in range(1, 7):
    out_ds.GetRasterBand(k).ComputeStatistics(False)
del out_ds

save_image(output_hidden, r'E:\Moore\pyScripts\pytorch\data\142_49\CAE_outputs\out_hidden.tif')




print('all set!')
print(nn.MSELoss(reduction='sum')(outputlayers[0,:,:].view(-1).float(), band1.view(-1).float()))
print(nn.MSELoss(reduction='sum')(outputlayers[1,:,:].view(-1).float(), band2.view(-1).float()))
print(nn.MSELoss(reduction='sum')(outputlayers[2,:,:].view(-1).float(), band3.view(-1).float()))
print(nn.MSELoss(reduction='sum')(outputlayers[3,:,:].view(-1).float(), band4.view(-1).float()))
print(nn.MSELoss(reduction='sum')(outputlayers[4,:,:].view(-1).float(), band5.view(-1).float()))
print(nn.MSELoss(reduction='sum')(outputlayers[5,:,:].view(-1).float(), band6.view(-1).float()))
print(nn.MSELoss(reduction='sum')(outputlayers[6,:,:].view(-1).float(), band7.view(-1).float()))