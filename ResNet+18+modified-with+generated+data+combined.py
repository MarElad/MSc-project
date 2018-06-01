
# coding: utf-8

# # ResNet 18 modified with generated data

# ## Import packages

# In[24]:

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as nm
import bisect


# ## Device configuration

# In[25]:

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.device_count())


# ## Data generation

# In[26]:

import random

#frames generator class
class Frame_generator(object):

    def __init__(self, array, dim, train_samples=4):
        self.array = array
        self.dim = dim
        self.train_samples = train_samples
        self.zeros = nm.zeros((dim, dim))
        self.itemindex = self.itemindex_fun
        self.indices = self.datasets_indices
#         self.frame = self.frame(x=array)
        
    def itemindex_fun(self,value):
        itemindex = nm.where(self.array==value)
        return itemindex
    
    def datasets_indices(self, train_samples):
        indices = nm.arange(len(self.itemindex(1)[0]))
        one_indices =  random.sample(list(indices), train_samples)
        return(one_indices)

    #using self.definitions for the functions didn't work out
    #forward part doesn't work
    def frame(self):
        y = self.itemindex(1)
        z = self.indices(4)
        res = self.zeros
        res = res*1
        for i in z:
            res[y[0][i]][y[1][i]]=1
            if (y[0][i])==0 and (y[1][i])==0:
                res[y[0][i]][y[1][i]+1]=1
                res[y[0][i]+1][y[1][i]]=1
                res[y[0][i]+1][y[1][i]+1]=1
            elif (y[0][i])==0 and (y[1][i])==self.dim-1:
                res[y[0][i]][y[1][i]-1]=1
                res[y[0][i]+1][y[1][i]-1]=1
                res[y[0][i]+1][y[1][i]]=1
            elif (y[0][i])==self.dim-1 and (y[1][i])==0:
                res[y[0][i]-1][y[1][i]]=1
                res[y[0][i]-1][y[1][i]+1]=1
                res[y[0][i]][y[1][i]+1]=1
            elif (y[0][i])==self.dim-1 and (y[1][i])==self.dim-1:
                res[y[0][i]-1][y[1][i]]=1
                res[y[0][i]-1][y[1][i]-1]=1
                res[y[0][i]][y[1][i]-1]=1
            elif (y[0][i])==0:
                res[y[0][i]][y[1][i]-1]=1
                res[y[0][i]][y[1][i]+1]=1
                res[y[0][i]+1][y[1][i]-1]=1
                res[y[0][i]+1][y[1][i]]=1
                res[y[0][i]+1][y[1][i]+1]=1
            elif (y[0][i])==self.dim-1:
                res[y[0][i]][y[1][i]-1]=1
                res[y[0][i]][y[1][i]+1]=1
                res[y[0][i]-1][y[1][i]-1]=1
                res[y[0][i]-1][y[1][i]]=1
                res[y[0][i]-1][y[1][i]+1]=1
            elif (y[1][i])==0:
                res[y[0][i]-1][y[1][i]]=1
                res[y[0][i]+1][y[1][i]]=1
                res[y[0][i]-1][y[1][i]+1]=1
                res[y[0][i]][y[1][i]+1]=1
                res[y[0][i]+1][y[1][i]+1]=1
            elif (y[1][i])==self.dim-1:
                res[y[0][i]-1][y[1][i]]=1
                res[y[0][i]+1][y[1][i]]=1
                res[y[0][i]-1][y[1][i]-1]=1
                res[y[0][i]][y[1][i]-1]=1
                res[y[0][i]+1][y[1][i]-1]=1
            else:
                res[y[0][i]-1][y[1][i]-1]=1
                res[y[0][i]-1][y[1][i]]=1
                res[y[0][i]-1][y[1][i]+1]=1
                res[y[0][i]][y[1][i]-1]=1
                res[y[0][i]][y[1][i]+1]=1
                res[y[0][i]+1][y[1][i]-1]=1
                res[y[0][i]+1][y[1][i]]=1
                res[y[0][i]+1][y[1][i]+1]=1

        return(res)
    
    #produce dataset
    def frames(self, n_frames):
        frames = nm.empty((n_frames, self.dim, self.dim))
        for i in range(n_frames):
            frames[i][:][:]=self.frame()
        return frames

# final image generator
def final_image(dim):
    n = nm.random.poisson(lam=40, size=1)
    a = nm.random.rand(n[0],2)
    intervals = nm.arange(0,1,1/dim)
    loc1 = []
    loc2 = []
    for n in range(len(a)):
        loc1.append(bisect.bisect_left(intervals, a[n][0])-1)
        loc2.append(bisect.bisect_left(intervals, a[n][1])-1)
    x = nm.zeros((dim,dim))
    for i,j in zip(loc1,loc2):
        x[i][j]=1
    return(x)


# images generator function
def frames_data_fun(n_images, n_frames, dim):
    labels = nm.empty((n_images, dim, dim))
    data = nm.empty((n_images, n_frames, dim, dim))
    for n in range(n_images):
        x = final_image(dim)
        tmp = Frame_generator(x, dim)
        frames = tmp.frames(n_frames=n_frames)
        labels[n][:][:]=x
        data[n][:][:][:]=frames
    return(labels, data)

# call images generator function
# 1000 images, 10000 frames, 100X100 dimension
res = frames_data_fun(n_images=1000, n_frames=10000, dim=100)
labels = res[0]
data = res[1]

print(data.shape)


# ## Hyper parameters

# In[ ]:

# Hyper-parameters
num_epochs = 60
learning_rate = 0.001
channel_dim = data.shape[1]
input_dim=data.shape[2]
data_samples = len(data)
train_samples = round(0.8*data_samples)

# # Image preprocessing modules
# transform = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
# transforms.ToTensor()])


# ## Train and test datasets

# In[ ]:

import random

# Data loader
def datasets_indices(data_samples, train_samples):    
    indices = nm.arange(data_samples)
    #print(indices)
    train_indices =  random.sample(list(indices), train_samples)
    #print(train_indices)
    test_indices = list(set(list(indices))-set(train_indices))
    #print(test_indices)
    return(train_indices, test_indices)

indices = datasets_indices(data_samples, train_samples)
train_loader = data[indices[0],:,:,:]
train_labels = labels[indices[0],:,:]
test_loader = data[indices[1],:,:,:]
test_labels = labels[indices[1],:,:]

# define mini-batch
train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           batch_size=32,shuffle=False)
train_labels = torch.utils.data.DataLoader(dataset=train_labels,
                                           batch_size=32,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_loader,
                                           batch_size=32,shuffle=False)
test_labels = torch.utils.data.DataLoader(dataset=test_labels,
                                           batch_size=32,shuffle=False)

# datasets shapes
#print(train_loader.shape)
#print(test_loader.shape)


# ## Define a model
# ResNet 18 modified version (terminal fully connected layer is replaced with another convolution layer).

# In[ ]:

# 3x3 convolution

def conv3x3(in_channels, out_channels, stride=1, input_dim=input_dim):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=nm.ceil((input_dim*stride-input_dim-stride+3)/2), bias=True)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #print("block conv1 size:", out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #print("block conv2 size:", out.shape)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        #print("block size:", out.shape)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(channel_dim, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1, 
                     stride=1, padding=0, bias=True)
        self.out_act = nn.Sigmoid()
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.conv_final(out)
        out = self.out_act(out)
        #print("net size:", out.shape)
        return out


# ## Run the model
#  - Loss function: absolute deviation
#  - Train the model on the train data.
#  - Test the model on the test data and report model accuracy.

# In[ ]:

# Check if multiple GPUs are available, and if so parallelize the computations
model = ResNet(ResidualBlock, [2, 2, 2, 2]).double().to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# Loss and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
print(input_dim)
print(total_step)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(zip(train_loader, train_labels)):
        images = images.to(device)
        labels = labels.to(device)
        print(images.shape)
        print(labels.shape)
        # Forward pass
        outputs = model(images.view(-1,channel_dim,input_dim,input_dim))
        outputs = outputs.view(-1,input_dim,input_dim)
        print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print train run results
        if (i+1) % 10 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images,labels) in enumerate(zip(test_loader, test_labels)):
        images = images.to(device)
        labels = labels.to(device)
        #print(images.shape)
        outputs = model(images.view(-1,channel_dim,input_dim,input_dim))
        outputs = outputs.view(-1,input_dim,input_dim)
        predicted = outputs.data.round()
        print('predicted shape:', predicted.shape)
        print('labels shape:', labels.shape)
        total += labels.shape[0]*input_dim*input_dim
        correct += (predicted == labels).sum().item()
        
    with open('out.txt', 'a') as f:
       print('images:', data_samples, 'channels:', data.shape[1], 'dim:', data.shape[2], 'epoch:', num_epochs, ':', 'Accuracy of the model on the test images: {} %'.format(100 * correct / total), file=f)
        
# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')

