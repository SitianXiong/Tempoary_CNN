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

if not os.path.exists('E:/pyScripts/pytorch/clouds_remove'):
    os.mkdir('E:/pyScripts/pytorch/clouds_remove')

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


img_transform = transforms.Compose([
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
])
        

class LandsatDataset():

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
#         self.batch_size=batch_size
    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
#         samples=[]
#         for i in range(self.batch_size):
        img_name = os.listdir(self.root_dir)[idx]#[idx*self.batch_size+i]
        img = Image.open(self.root_dir+img_name)
        image = numpy.array(img)[4000:4700,3000:3500]
        image_tensor=torch.from_numpy(image)
        image_tensor=image_tensor.cuda()
        #samples.append(image_tensor)
        #tensor=torch.cat((*samples),(0))
#         if self.transform:
#             sample = self.transform(sample)
        return image_tensor

num_epochs=1500
learning_rate = 1e-3
batch_size=55

dataset = LandsatDataset('E:/pyScripts/pytorch/landsat8_142_49_truecolor/',transform=None)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(55, 33),
            nn.ReLU(True),
            nn.Linear(33, 11),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(11, 33),
            nn.ReLU(True),
            nn.Linear(33, 55),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



model = autoencoder().cuda()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for data in dataloader:
    ini = data.float().permute(0,3,1,2)
    save_image(ini, 'E:/pyScripts/pytorch/clouds_remove/x_inputlayer.png')
    img_B = min_max_normalization(data[0:,0:,0:,0].float(),0.0,1.0)
    img_G = min_max_normalization(data[0:,0:,0:,1].float(),0.0,1.0)
    img_R = min_max_normalization(data[0:,0:,0:,2].float(),0.0,1.0)
    img_B=img_B.view(img_B.size(0),-1).float()
    img_B = torch.transpose(img_B, 0, 1)
    img_G=img_G.view(img_G.size(0),-1).float()
    img_G = torch.transpose(img_G, 0, 1)
    img_R=img_R.view(img_R.size(0),-1).float()
    img_R = torch.transpose(img_R, 0, 1)
    for epoch in range(num_epochs):
        img_B = Variable(img_B).cuda()
        output_B = model(img_B)
        loss = criterion(output_B, img_B)
        MSE_loss = nn.MSELoss()(output_B, img_B)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        if (epoch+1)%500==0:
            print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
    
    for epoch in range(num_epochs):
        img_G = Variable(img_G).cuda()
        output_G = model(img_G)
        loss = criterion(output_G, img_G)
        MSE_loss = nn.MSELoss()(output_G, img_G)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        if (epoch+1)%500==0:
            print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
    
    for epoch in range(num_epochs):
        img_R = Variable(img_R).cuda()
        output_R = model(img_R)
#         loss = criterion(output_R.long(), img_R.long())
        loss = nn.MSELoss()(output_R, img_R)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        if (epoch+1)%500==0:
            print('epoch [{}/{}], loss:{:.4f}, MSE_loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0], MSE_loss.data[0]))
    
    
#     img_fb= torch.transpose(img_B,0,1)
#     img_fg= torch.transpose(img_G,0,1)
#     img_fr= torch.transpose(img_R,0,1)
    output_fb=torch.transpose(output_B,0,1).view(55,1,700,500)
    output_fg=torch.transpose(output_G,0,1).view(55,1,700,500)
    output_fr=torch.transpose(output_R,0,1).view(55,1,700,500)
    output=torch.cat((output_fb,output_fg,output_fr),1).cpu().data
    # MIDIMG= model.encoder(img).view(img.size(0),1,56,56).cpu().data
    # save_image(x, './mlp_pansharpened/x_{}.png'.format(epoch))
    save_image(output, 'E:/pyScripts/pytorch/clouds_remove/x_output.png')
    # save_image(MIDIMG, './mlp_pansharpened/x_midlayer{}.png'.format(epoch))

    

