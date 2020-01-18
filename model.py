#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#get_ipython().magic('matplotlib inline')

image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]



# In[2]:


class CNN(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature*22*22, 50)
        self.fc2 = nn.Linear(50, 2)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature*22*22)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class CNN2(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(CNN2, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=5)
        self.conv2 = nn.Conv2d(30, n_feature, kernel_size=5)
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size = 5)
        self.fc1 = nn.Linear(n_feature*9*9, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature*9*9)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
# In[ ]:




