
# coding: utf-8

# In[ ]:


import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import glob
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from segmentation_dataset import SegmentationDataset
from diceloss import BinaryDiceLoss

train_dir = './suim/train_val/images'
mask_dir = './suim/train_val/masks' 

paths1 = train_dir + "/*.jpg"    
paths2 = mask_dir + "/*.bmp"    

tfiles = glob.glob(paths1)  
mfiles = glob.glob(paths2)  

tfiles.sort()
mfiles.sort()

transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((480,640)), transforms.ToTensor()])

sds = SegmentationDataset(tfiles, mfiles, transform) 


# In[ ]:


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)         
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)         
        self.relu = nn.ReLU()     
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# In[ ]:


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))     
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


# In[ ]:


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)     
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


# In[ ]:


class unet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)         
        
        """ Bottleneck """
        self.b = conv_block(128, 256)         
        
        """ Decoder """
        self.d1 = decoder_block(256, 128)
        self.d2 = decoder_block(128, 64)
        self.d3 = decoder_block(64, 32)
        self.d4 = decoder_block(32, 16)         
        
        """ Classifier """
        self.outputs = nn.Conv2d(16, 1, kernel_size=1, padding=0)  
        self.sig = nn.Sigmoid()
        
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)         
        
        """ Bottleneck """
        b = self.b(p4)         
        
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)         
        
        """ Classifier """
        outputs = self.sig(self.outputs(d4))
        return outputs
    


# In[ ]:


net = unet()

#criterion = BinaryDiceLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



# In[ ]:


valid_size = 0.2  # proportion de "train_data" utilisée pour la validation
batch_size = 8   # taille de "batch"

# mélanger aléatoirement train_data et séparer en un jeu d'apprentissage et de validation
num_train = len(tfiles)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]


# In[ ]:


# Pour charger les données en GPU automatiquement si disponible (sinon en CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
print("Device: " + device)

# pour échantilloner le jeu de données d'apprentissage (+ validation) de manière uniforme lors de l'apprentissage
train_sampler = SubsetRandomSampler(train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(sds, batch_size=batch_size, sampler=train_sampler, **kwargs)
valid_loader = torch.utils.data.DataLoader(sds, batch_size=batch_size, sampler=valid_sampler, **kwargs)


# In[17]:


import cv2
import matplotlib.pyplot as plt


it = iter(train_loader)
images = next(it)
i = torch.permute(images[0][0],(1,2,0))
plt.imshow(i)
plt.show()



# In[18]:


i = images[1][0].squeeze()
plt.imshow(i)
plt.show()


# In[ ]:


# Training of classifier

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, masks = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')


# In[ ]:


torch.save(net, 'seg_model.pth')


# In[ ]:


net = torch.load('seg_model.pth')


# In[23]:


import matplotlib.pyplot as plt

it = iter(train_loader)
images = next(it)
outputs = net(images[0])
i = torch.permute(images[0][0],(1,2,0))
plt.imshow(i)
plt.show()



# In[7]:


plt.imshow(outputs[0].detach().numpy().squeeze())
plt.show()


# In[6]:




