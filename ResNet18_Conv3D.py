
# coding: utf-8

# # ResNet 18 modified with generated data

# ## Import packages

# In[8]:

import sys
sys.path.append('/home/eym16/anaconda3/lib/python3.6/site-packages')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import bisect
import pandas as pd


# ## Device configuration

# In[9]:

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.device_count())


# ## Data generation

# In[10]:

epsilon = 10**(-4)

import random

#frames generator class
class Frame_generator(object):

    def __init__(self, dim, train_samples=4):
#         self.array = array
        self.dim = dim
        self.train_samples = train_samples
        self.zeros = np.zeros((dim, dim))
#         self.itemindex = self.itemindex_fun
#         self.indices = self.datasets_indices
#         self.frame = self.frame(x=array)
        
#     def itemindex_fun(self,value):
#         itemindex = np.where(self.array==value)
#         return itemindex
    
#     def datasets_indices(self, train_samples):
#         indices = np.arange(len(self.itemindex(1)[0]))
#         one_indices =  random.sample(list(indices), train_samples)
#         return(one_indices)

    #using self.definitions for the functions didn't work out
    #forward part doesn't work
    def frame(self, loc1, loc2):
        res = np.copy(self.zeros)
        for i,j in zip(loc1,loc2):
#             print('i:', i)
#             print('j:', j)
            res[i][j]+=1
            if i==0 and j==0:
                res[i][j+1]+=1
                res[i+1][j]+=1
                res[i+1][j+1]+=1
            elif i==0 and j==self.dim-1:
                res[i][j-1]+=1
                res[i+1][j-1]+=1
                res[i+1][j]+=1
            elif i==self.dim-1 and j==0:
                res[i-1][j]+=1
                res[i-1][j+1]+=1
                res[i][j+1]+=1
            elif i==self.dim-1 and j==self.dim-1:
                res[i-1][j]+=1
                res[i-1][j-1]+=1
                res[i][j-1]+=1
            elif i==0:
                res[i][j-1]+=1
                res[i][j+1]+=1
                res[i+1][j-1]+=1
                res[i+1][j]+=1
                res[i+1][j+1]+=1
            elif i==self.dim-1:
                res[i][j-1]+=1
                res[i][j+1]+=1
                res[i-1][j-1]+=1
                res[i-1][j]+=1
                res[i-1][j+1]+=1
            elif j==0:
                res[i-1][j]+=1
                res[i+1][j]+=1
                res[i-1][j+1]+=1
                res[i][j+1]+=1
                res[i+1][j+1]+=1
            elif j==self.dim-1:
                res[i-1][j]+=1
                res[i+1][j]+=1
                res[i-1][j-1]+=1
                res[i][j-1]+=1
                res[i+1][j-1]+=1
            else:
                res[i-1][j-1]+=1
                res[i-1][j]+=1
                res[i-1][j+1]+=1
                res[i][j-1]+=1
                res[i][j+1]+=1
                res[i+1][j-1]+=1
                res[i+1][j]+=1
                res[i+1][j+1]+=1

        return(res)
    
    #produce dataset
    def frames(self, n_frames, loc1, loc2):
        frames = np.empty((n_frames, self.dim, self.dim))
        for i in range(n_frames):
            loc1_tmp = loc1[loc1[:,i].nonzero()[0],i]
            loc2_tmp = loc2[loc2[:,i].nonzero()[0],i]
            intervals = np.arange(0,1,1/self.dim)
            loc1_index = []
            loc2_index = []
#             print(loc1_tmp.shape)
            for n in range(loc1_tmp.shape[0]):
#                 print(loc1_tmp)
                loc1_index.append(bisect.bisect_left(intervals, loc1_tmp[n])-1)
                loc2_index.append(bisect.bisect_left(intervals, loc2_tmp[n])-1)
            frames[i][:][:]=self.frame(loc1_index, loc2_index)
        return frames

# final image generator
def final_image(dim, n_frames, p_1_1=0.05):
    n = np.random.poisson(lam=10, size=1)
    #if n==0, generate one point
    n = max(n,1)
    loc_mat = np.random.rand(n[0],2)
    intervals = np.arange(0,1,1/dim)
    loc1 = []
    loc2 = []
    for n in range(loc_mat.shape[0]):
        loc1.append(bisect.bisect_left(intervals, loc_mat[n,0])-1)
        loc2.append(bisect.bisect_left(intervals, loc_mat[n,1])-1)
    x = np.zeros((dim,dim))
    for i,j in zip(loc1,loc2):
        x[i][j] += 1
    #generate location matrices
#     print(loc_mat[:,0].shape)
    
    #correlation adjustment
    #define correlation parameters
    a_1 = 4/loc_mat.shape[0]
    a_1 = np.clip(a_1, a_min=epsilon, a_max=1-epsilon)
    a_2 = 1-a_1
    p_1_1=p_1_1
    p_2_2=(a_2-a_1*(1-p_1_1))/a_2
    #define correlation matrix
    binom = np.zeros((loc_mat.shape[0],n_frames))
    binom[:,0] = np.random.binomial(n=1,p=min(1,4/loc_mat.shape[0]),size=loc_mat.shape[0])
    for j in range(1,n_frames):
        for i in range(loc_mat.shape[0]):
            if binom[i,j-1]==1:
                binom[i,j]= np.random.binomial(n=1,p=min(1,p_1_1),size=1)
            else:
                binom[i,j]= np.random.binomial(n=1,p=min(1,1-p_2_2),size=1)
#     binom = np.random.binomial(n=1,p=min(1,4/loc_mat.shape[0]),size=(loc_mat.shape[0],n_frames))
#     print('binom shape: ', binom.shape)
#     print(binom)
    loc1 = np.multiply(np.reshape(loc_mat[:,0],(loc_mat.shape[0],-1)),binom)
    loc2 = np.multiply(np.reshape(loc_mat[:,1],(loc_mat.shape[0],-1)),binom)
    loc1 = loc1[loc1[:,:].nonzero()[0],:]
    loc2 = loc2[loc2[:,:].nonzero()[0],:]
    return(x, loc1, loc2)


# images generator function
def frames_data_fun(n_images, n_frames, dim):
    labels = np.zeros((n_images, 2, dim, dim))
    data = np.empty((n_images, n_frames, dim, dim))
    for n in range(n_images):
        x, loc1, loc2 = final_image(dim, n_frames)
        tmp = Frame_generator(dim)
        #30 is burn-in samples
        frames = tmp.frames(n_frames=n_frames, loc1=loc1, loc2=loc2)
#         print(frames.shape)
        labels[n][1][:][:]=x
        m = (x!=0)
        labels[n][0][:][:]=1*m
        data[n][:][:][:]=frames
    return(labels, data)

# call images generator function
# 1000 images, 10000 frames, 100X100 dimension
#30 burn-in samples
n_images = 500
n_frames = 1000
input_dim = 10
res = frames_data_fun(n_images=n_images, n_frames=n_frames+30, dim=input_dim)
labels = res[0]
print(labels.shape)
data = res[1][:,-n_frames:,:,:]

print(data.shape)
# print('data frame:', data[2,3,:,:])
# print('lable:', labels[2,:,:])


# ## Hyper parameters

# In[ ]:

# Hyper-parameters
num_epochs = 200
learning_rate = 0.001
channel_dim = data.shape[1]
input_dim=data.shape[2]
data_samples = len(data)
train_samples = round(0.8*data_samples)
conv_layers0 = 16

# # Image preprocessing modules
# transform = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
# transforms.ToTensor()])


# ## Train and test datasets

# In[12]:

import random

# Data loader
def datasets_indices(data_samples, train_samples):    
    indices = np.arange(data_samples)
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

epsilon = 10**(-4)

depth_input = 21
kernel_size = [depth_input,3,3]

# 3x3 convolution

def conv3x3(in_channels, out_channels, stride=1, input_dim=input_dim):
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=[np.ceil((input_dim*stride-input_dim-stride+kernel_size[0])/2),
                                             np.ceil((input_dim*stride-input_dim-stride+kernel_size[1])/2), 
                                            np.ceil((input_dim*stride-input_dim-stride+kernel_size[2])/2)], bias=True)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
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
#         print('out shape', out.shape)
#         print('residual shape', residual.shape)
        out += residual
        out = self.relu(out)
        #print("block size:", out.shape)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.layer0 = conv_layers0
        self.in_channels = conv_layers0
        self.conv = conv3x3(1, conv_layers0)
        self.bn = nn.BatchNorm3d(conv_layers0)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, conv_layers0, layers[0])
        self.layer2 = self.make_layer(block, conv_layers0*2, layers[0], 2)
        self.layer3 = self.make_layer(block, conv_layers0*4, layers[1], 2)
#         self.conv_final = nn.Conv3d(conv_layers0*4, 2, kernel_size=[15,1,1], 
#                      stride=1, padding=0, bias=True)
        self.out_act = nn.Sigmoid()
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm3d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
#         print('input', x.shape)
        out = self.conv(x)
#         print('out1', out.shape)
        out = self.bn(out)
#         print('out2', out.shape)
        out = self.relu(out)
#         print('out3', out.shape)
        out = self.layer1(out)
#         print('out4', out.shape)
        out = self.layer2(out)
#         print('out5', out.shape)
        out = self.layer3(out)
#         print('out6', out.shape)
#         out = self.conv_final(out)
        m = nn.Conv3d(self.layer0*4, 2, kernel_size=[out.shape[2],1,1], 
                     stride=1, padding=0, bias=True)
        m = m.double()
        out = m(out)
#         print('out7', out.shape)
        out0 = torch.unsqueeze(out[:,0,:,:],1)
#         print('out8', out0.shape)
        out1 = torch.unsqueeze(out[:,1,:,:],1)
#         print('out9', out1.shape)
        #replace sigmoid by ReLu
        out = self.out_act(out)
        out0 = self.out_act(out0)
#         print('out10', out0.shape)
        out0 = out0.clamp(min=epsilon, max=1-epsilon)
#         print('out11', out0.shape)
        out1 = self.relu(out1)
#         print('out12', out1.shape)
        out = torch.cat((out0,out1),1)
#         print('out13', out.shape)
#         print("net size:", out.shape)
        return out


# ## Run the model
#  - Loss function: absolute deviation
#  - Train the model on the train data.
#  - Test the model on the test data and report model accuracy.

# In[ ]:

# epsilon = 10**(-6)

# Check if multiple GPUs are available, and if so parallelize the computations
model = ResNet(ResidualBlock, [2, 2, 2, 2]).double().to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

#loss function
class Loss(torch.nn.Module):

    def __init__(self):
        super(Loss,self).__init__()

    def forward(self,x,y):
#         print('x shape:', x.shape)
#         print(x[0,:,:])
        loss_layer0 = -(y[:,0,:,:]*torch.log(x[:,0,:,:])*10+(1-y[:,0,:,:])*torch.log(1-x[:,0,:,:]))
#         print(loss_layer0)
        loss_layer0 = torch.sum(loss_layer0)
        loss_layer1 = 0
        for image in range(x.shape[0]):
            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    if y[image,0,i,j]==1:
                        loss_layer1+=abs(y[image,1,i,j]-x[image,1,i,j])/(x.shape[2]^2)
        totloss = loss_layer0+loss_layer1
        return totloss

# Loss and optimizer
criterion = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

# Train the model
total_step = len(train_loader)
# print(input_dim)
# print(total_step)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(zip(train_loader, train_labels)):
        images = images.to(device)
        labels = labels.to(device)
#         print(images.shape)
#         print('labels shape', labels.shape)
        # Forward pass
        outputs = model(images.view(-1,1,channel_dim,input_dim,input_dim))
        outputs = outputs.view(-1,2,input_dim,input_dim)
#         print('labels shape', labels.shape)
#         print('outputs dim:', outputs.shape)
#         print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print train run results
        if (i+1) % 5 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
        torch.save(model.state_dict(), 'resnet.ckpt')
        print('model:', epoch)

# Test the model (train data)
print('################')
print('convolution layers:({},{},{})'.format(conv_layers0, conv_layers0*2, conv_layers0*4))
print('################')
print('train data results')
model.eval()
with torch.no_grad():
    correct = 0
    semi_correct = 0
    total = 0
    total_ones = 0
    total_zeros = 0
    correct_ones = 0
    correct_zeros = 0
    for i, (images,labels) in enumerate(zip(train_loader, train_labels)):
        images = images.to(device)
        labels = labels.to(device)
        #print(images.shape)
        outputs = model(images.view(-1,1,channel_dim,input_dim,input_dim))
        outputs = outputs.view(-1,2,input_dim,input_dim)
        predicted = outputs.data.round()
#         print('predicted shape:', predicted.shape)
#         print('labels shape:', labels.shape)
#         print(labels.shape[0])
        total += labels.shape[0]*input_dim*input_dim
        m = (labels[:,0,:,:]>0)*(predicted[:,0,:,:] == labels[:,0,:,:])
        correct += (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()-(m*(predicted[:,1,:,:] != labels[:,1,:,:])).sum().item()
        semi_correct += (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()
        total_ones +=  (labels[:,0,:,:]==1).sum().item()
        total_zeros += (labels[:,0,:,:]==0).sum().item()
        correct_ones += ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==1)).sum().item()
        correct_zeros += ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==0)).sum().item()
print(correct_ones, total_ones, correct_zeros, total_zeros)
print('accuracy:', correct/total*100, '%')
print('semi_correct:', semi_correct/total*100, '%')
print('correct ones:', correct_ones/total_ones*100, '%')
print('correct zeros:', correct_zeros/total_zeros*100, '%')

with open('out.txt', 'a') as f:
    print('\n####\n','convolution layers:({},{},{}), '.format(conv_layers0, conv_layers0*2, conv_layers0*4),'\n','images:', data_samples, ', channels:', data.shape[1], ', dim:', data.shape[2], ', epoch:', num_epochs, '\n', 'Accuracy of the model on the train images: {} %'.format(100 * correct / total), file=f, sep='')

# Test the model (test data)
print('################')
print('test data results')
def test_results(threshold):
    print(threshold)
    model.eval()
    res = []
    for j in threshold:
        print('j:', j)
        with torch.no_grad():
            correct = 0
            semi_correct = 0
            total = 0
            total_ones = 0
            total_zeros = 0
            correct_ones = 0
            correct_zeros = 0
            for i, (images,labels) in enumerate(zip(test_loader, test_labels)):
                images = images.to(device)
                labels = labels.to(device)
                #print(images.shape)
                outputs = model(images.view(-1,1,channel_dim,input_dim,input_dim))
                outputs = outputs.view(-1,2,input_dim,input_dim)
#                 print('outputs', (outputs>j))
                predicted = ((outputs>j)*1).double()
        #         print('predicted shape:', predicted.shape)
        #         print('labels shape:', labels.shape)
        #         print(labels.shape[0])
                total += labels.shape[0]*input_dim*input_dim
#                 print('predicted:', predicted[:,0,:,:])
#                 print('labels:', labels[:,0,:,:])
                m = (labels[:,0,:,:]>0)*(predicted[:,0,:,:] == labels[:,0,:,:])
                correct += (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()-(m*(predicted[:,1,:,:] != labels[:,1,:,:])).sum().item()
                semi_correct += (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()
                total_ones +=  (labels[:,0,:,:]==1).sum().item()
                total_zeros += (labels[:,0,:,:]==0).sum().item()
                correct_ones += ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==1)).sum().item()
                correct_zeros += ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==0)).sum().item()
            res.append([correct_ones,total_ones,correct_zeros,total_zeros])
#         print('shape:', len(res))
        res_out = pd.DataFrame(data=res, columns=['correct_ones','total_ones','correct_zeros','total_zeros'])
#         print(correct_ones, total_ones, correct_zeros, total_zeros)
        print('accuracy:', correct/total*100, '%')
        print('semi_correct:', semi_correct/total*100, '%')
        print('correct ones:', correct_ones/total_ones*100, '%')
        print('correct zeros:', correct_zeros/total_zeros*100, '%')

        with open('out.txt', 'a') as f:
            print('\n', 'threshold: ', j,', ', 'Accuracy of the model on the test images: {} %'.format(100 * correct / total), file=f, sep='')
    return(res_out)

threshold = [0.1,0.3,0.5,0.7,0.9]

res = test_results(threshold)
print('res:\n', res, type(res))
# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')

