
# coding: utf-8

# ## Import packages

# In[23]:

import sys
sys.path.append('/home/eym16/anaconda3/lib/python3.6/site-packages')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import bisect
import pandas as pd
import bisect
import os
import matplotlib.pyplot as plt


# ## Device configuration

# In[ ]:

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.device_count())


# ## Data loading and preparation

# In[ ]:

max_intensity = 1000
sd = 80
dim = 10
Path = '/home/eym16/Simulated_Data/Scattered_photons/Localisation_UC/Data1000/'
filelist = os.listdir(Path)
print(filelist)
# filelist.remove('.ipynb_checkpoints')
filelist.sort()
data_tmp = pd.read_csv(Path + filelist[0+int(len(filelist)/2)], delimiter=',')

# print('filelist', filelist)
# print(data_tmp)
# print('len',int(len(filelist)/2))
# print(len(filelist))

n_images = int(len(filelist)/2)
n_frames = data_tmp['frame'].max()

labels = np.zeros((n_images,2,dim,dim))
frames = np.zeros((n_images,n_frames,dim,dim))

for i in range(int(len(filelist)/2)):
#     print('i=', i)
    #read labels data
    labels_file = pd.read_csv(Path + filelist[i], delimiter=',')
    names = list(labels_file)

    labels_file[names[0]] = labels_file[names[0]]/1234 #1,234 is the size in nm of the original x locations
    labels_file[names[1]] = labels_file[names[1]]/1234 #1,234 is the size in nm of the original y locations
    intervals = np.arange(0,1,1/dim)
    loc1 = []
    loc2 = []
    for n in range(labels_file.shape[0]):
        loc1.append(bisect.bisect_left(intervals, labels_file.iat[n,0])-1)
        loc2.append(bisect.bisect_left(intervals, labels_file.iat[n,1])-1)
    for h,j in zip(loc1,loc2):
        labels[i][1][h][j] += 1
    m = (labels[i][1][:][:]!=0)
    labels[i][0][:][:]=1*m
    
    #read frames data
    data_file = pd.read_csv(Path + filelist[i+int(len(filelist)/2)], delimiter=',')
    #normalise columns
    data_file['xnano'] = data_file['xnano']/1234
    data_file['ynano'] = data_file['ynano']/1234
#     data_file['intensity'] = data_file['intensity']/1000

#generate new data file
    df_photons = pd.DataFrame(columns = ['frame', 'xnano', 'ynano'])
    for n in range(len(data_file)):
#         print('col names: ', list(data_file))
        tmp = np.random.multivariate_normal(mean=(data_file.at[n,'xnano'],data_file.at[n,'ynano']), cov=np.array([[sd**2/1234**2, 0], [0, sd**2/1234**2]]), size=int(data_file.at[n,'intensity'])) #change size
        tmp = np.clip(tmp, 1e-4, 0.9999)
#         print(tmp.shape)
#         print(count)
        d = {'frame': [n]*len(tmp), 'xnano': tmp[:,0], 'ynano': tmp[:,1]}
        df_photons_tmp = pd.DataFrame(data=d)
        df_photons = df_photons.append(df_photons_tmp)
#         print('df_photons_tmp', df_photons_tmp)
#         print('df_photons', df_photons)
        
#generate frames       
    if i==0:
        max_frame = data_file['frame'].max()
#         print(max_frame)
    frames_data = np.zeros((max_frame,dim,dim))
#     print(frames_data.shape)
    for frame in range(max_frame):
        df_tmp = df_photons[df_photons.frame==frame]
        loc1 = []
        loc2 = []
        intervals = np.arange(0,1,1/dim)
#     print(labels_file)
        for n in range(df_tmp.shape[0]):
            loc1.append(bisect.bisect_left(intervals, df_tmp.iat[n,1])-1)
            loc2.append(bisect.bisect_left(intervals, df_tmp.iat[n,2])-1)
        for h,j in zip(loc1,loc2):
            frames[i][frame-1][h][j] += 1
#         print('frames', frames[i][frame-1][:][:])

####normalise intensities
frames = frames/max_intensity
print('frames', frames[i][frame-1][:][:])

data = frames


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

# In[ ]:

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


# ## Data analysis

# In[ ]:

# import matplotlib.pyplot as plt
# %matplotlib inline
# # train_loader
# print(type(train_loader))
# print(train_loader[2,3,:,:])
# probability = train_loader.reshape(-1)
# print(probability.shape)
# print('max train', probability.max())
# plt.hist(probability, weights=np.zeros_like(probability) + 1. / probability.size)

# # train_labels
# labels = train_labels.reshape(-1)
# # plt.hist(labels, weights=np.zeros_like(labels) + 1. / labels.size)


# ## Define a model
# ResNet 18 modified version (terminal fully connected layer is replaced with another convolution layer).

# In[ ]:

epsilon = 10**(-4)
kernel_size = 3

# 3x3 convolution

def conv3x3(in_channels, out_channels, stride=1, input_dim=input_dim):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=np.ceil((input_dim*stride-input_dim-stride+kernel_size)/2), bias=True)

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
        self.in_channels = conv_layers0
        self.conv = conv3x3(channel_dim, conv_layers0)
        self.bn = nn.BatchNorm2d(conv_layers0)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, conv_layers0, layers[0])
        self.layer2 = self.make_layer(block, conv_layers0*2, layers[0], 2)
        self.layer3 = self.make_layer(block, conv_layers0*4, layers[1], 2)
        self.conv_final = nn.Conv2d(conv_layers0*4, 2, kernel_size=1, 
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
        out0 = torch.unsqueeze(out[:,0,:,:],1)
        out1 = torch.unsqueeze(out[:,1,:,:],1)
        #replace sigmoid by ReLu
#         out = self.out_act(out)
        out0 = self.out_act(out0)
        out0 = out0.clamp(min=epsilon, max=1-epsilon)
        out1 = self.relu(out1)
        out = torch.cat((out0,out1),1)
        #print("net size:", out.shape)
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
#         print(labels.shape)
        # Forward pass
        outputs = model(images.view(-1,channel_dim,input_dim,input_dim))
        outputs = outputs.view(-1,2,input_dim,input_dim)
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
        outputs = model(images.view(-1,channel_dim,input_dim,input_dim))
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
                outputs = model(images.view(-1,channel_dim,input_dim,input_dim))
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

threshold = np.arange(0,1.05,0.05)

res = test_results(threshold)
print('res:\n', res, type(res))
# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')


# ## plot historgram

# In[ ]:

import matplotlib.pyplot as plt
# %matplotlib inline

torch.no_grad()
probability = torch.empty(0,1,10,10)
for i, images in enumerate(test_loader):
                images = images.to(device)
                print(images.shape)
                outputs = model(images.view(-1,channel_dim,input_dim,input_dim)).float()
                probability = torch.cat((probability, outputs[:,0,:,:]),0)
                print(type(outputs))
print(probability.shape)
probability = probability.view(-1).detach().numpy()
print(probability.shape)
plt.hist(probability, weights=np.zeros_like(probability) + 1. / probability.size)
plt.title('Test probabilities')
plt.xlabel('probabilities')
plt.ylabel('frequency')
plt.savefig('pred_prob_80uc.jpg', bbox_inches="tight")
# plt.show()


# ## plot ROC curve

# In[ ]:

res['true_positives_rate']=res.correct_ones/res.total_ones
res['false_positives_rate']=(res.total_zeros-res.correct_zeros)/res.total_zeros
print(res)

true_positives_rate_1 = [pd.Series([0]),res.true_positives_rate[::-1],pd.Series([1])]
res_a = pd.concat(true_positives_rate_1)
print(res_a)
false_positives_rate_1 = [pd.Series([0]),res.false_positives_rate[::-1],pd.Series([1])]
res_b = pd.concat(false_positives_rate_1)
print(res_b)
plt.plot(res_b, res_a,  marker='o', color='b')
# plt.annotate('0.9', xy=(false_positives_rate_1[1], true_positives_rate_1[1]))
plt.plot([0,1],[0,1], label='identity line', linestyle='dashed')
plt.xlabel('false positives rate')
plt.ylabel('true positives rate')
plt.suptitle('ROC curve')
plt.savefig('ROC_80uc.jpg', bbox_inches="tight")
# plt.show()


# In[ ]:



