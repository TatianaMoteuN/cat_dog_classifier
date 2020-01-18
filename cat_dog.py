#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('python3 main.py')


# # In[2]:


# get_ipython().system('python3 model.py')


# # In[3]:


# get_ipython().system('python3 dataset.py')


# In[4]:

from dataset import CatDogDataset
from model import CNN
from main import train,test


import numpy as np # Matrix Operations (Matlab of Python)
import pandas as pd # Work with Datasources
import matplotlib.pyplot as plt # Drawing Library

from PIL import Image

import torch # Like a numpy but we could work with GPU by pytorch library
import torch.nn as nn # Nural Network Implimented with pytorch
import torchvision # A library for work with pretrained model and datasets

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import glob
import os


# In[5]:


image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]


# In[6]:


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean, std)])


#Create Dataset


path    = '/home/aims/aims-courses/Deep learning/asign/data/train/'
dataset = CatDogDataset(path, transform=transform)

path1 = '/home/aims/aims-courses/Deep learning/asign/data/val/'
test1 = CatDogDataset(path1, transform=transform)


# In[7]:



#Create DataLoader
shuffle     = True
batch_size  = 32
num_workers = 0
dataloader  = DataLoader(dataset=dataset, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)

#create test_loader
shuffle     = True
batch_size  = 16
num_workers = 0
test_loader = DataLoader(dataset=test1, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)
# In[8]:


# In[8]:


train_loader = dataloader


# In[9]:


import torch.optim as optim
# function to count number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np


n_hidden = 8 # number of hidden units
input_size  = 100*100  # images are 28x28 pixels
output_size = 2  

model_fnn = CNN(input_size, n_hidden, output_size)
optimizer = optim.SGD(model_fnn.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_fnn)))

n_hidden = 4 # number of hidden units
input_size  = 100*100  # images are 28x28 pixels
output_size = 2  

model_fnn1 = CNN2(input_size, n_hidden, output_size)
optimizer = optim.SGD(model_fnn1.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_fnn1)))

for epoch in range(0, 1):
    train(epoch, model_fnn,train_loader,optimizer)
    test(model_fnn,test_loader)

for epoch in range(0,1):
    train(epoch, model_fnn1, train_loader, optimizer)
    test(model_fnn1, test_loader)
# In[ ]:




