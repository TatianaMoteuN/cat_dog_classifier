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

get_ipython().magic('matplotlib inline')

image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]



# In[ ]:





# In[2]:



class CatDogDataset(Dataset):
    def __init__(self, path, transform=None):
        self.classes   = os.listdir(path)
        self.path      = [f"{path}/{className}" for className in self.classes]
        self.file_list = [glob.glob(f"{x}/*") for x in self.path]
        self.transform = transform
       
        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.file_list[i]:
                files.append([i, className, fileName])
        self.file_list = files
        files = None
       
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im, classCategory


# In[3]:


get_ipython().system('pwd')


# In[4]:


#Define Transoformation And Augmentation


mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Normalize(mean, std)])


#Create Dataset


path    = '/home/aims/aims-courses/Deep learning/cat-and-dog/training_set/training_set/'
dataset = CatDogDataset(path, transform=transform)



# In[5]:


dataset[0][0]


# In[6]:


#Display Sample Image
def imshow(source):
    plt.figure(figsize=(10,10))
    imt = (source.view(-1, image_size[0], image_size[0]))
    imt = imt.numpy().transpose([1,2,0])
    imt = (std * imt + mean).clip(0,1)
    plt.subplot(1,2,2)
    plt.imshow(imt)


imshow(dataset[0][0])
imshow(dataset[2][0])
imshow(dataset[6000][0])



# In[7]:


#Create DataLoader
shuffle     = True
batch_size  = 64
num_workers = 0
dataloader  = DataLoader(dataset=dataset, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)


# In[8]:


i = 0
for i, (X,Y) in enumerate(dataloader):
    print(X.size())
    print(Y.size())
    print(i)
    break
    


# ## Create the model classes

# In[9]:


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



# ## Running on a GPU: device string

# In[10]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[11]:


accuracy_list = []

def train(epoch, model, perm=torch.arange(0, 784).long()):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
#         # permute pixels
#         data = data.view(1,-1, 100*100)
#         data = data[:, perm]
        data = data.view(-1,3, 100, 100)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def test(model, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
#         # permute pixels
#         data = data.view(1,-1, 100*100)
#         data = data[:, perm]
        data = data.view(-1,3, 100, 100)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss                                                               
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))


# In[12]:


train_loader = dataloader


# ## Train a small fully-connected network

# In[13]:


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

for epoch in range(0, 1):
    train(epoch, model_fnn)
    test(model_fnn)


# In[ ]:





# In[14]:


# Training settings import torch.optim as optim
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

for epoch in range(0, 1):
    train(epoch, model_fnn)
    test(model_fnn)
n_features = 6 # number of feature maps

model_cnn = CNN(input_size, n_features, output_size)
optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)
print('Number of parameters: {}'.format(get_n_params(model_cnn)))

for epoch in range(0, 1):
    train(epoch, model_cnn)
    test(model_cnn)


# In[ ]:




