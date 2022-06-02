
# coding: utf-8

# In[4]:


import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

# fonction pour la normalisation des données
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5,),std=(0.5,))])  

