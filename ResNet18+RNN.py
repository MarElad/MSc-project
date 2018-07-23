
# coding: utf-8

# # ResNet 18 modified with generated data

# ## Import packages

# In[15]:

import sys
sys.path.append('/home/eym16/anaconda3/lib/python3.6/site-packages')

import numpy as np
import bisect
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling2D, Conv2D
from keras.layers import TimeDistributed
from keras.datasets import imdb
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.core import Reshape
from keras.layers import InputLayer
import numpy as np


# ## Data generation

# In[16]:

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
def final_image(dim, n_frames):
    n = np.random.poisson(lam=10, size=1)
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
    binom = np.random.binomial(n=1,p=min(1,4/loc_mat.shape[0]),size=(loc_mat.shape[0],n_frames))
#     print(binom.shape)
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
        frames = tmp.frames(n_frames=n_frames, loc1=loc1, loc2=loc2)
        labels[n][1][:][:]=x
        m = (x!=0)
        labels[n][0][:][:]=1*m
        data[n][:][:][:]=frames
    return(labels, data)

# call images generator function
# 1000 images, 10000 frames, 100X100 dimension
n_images = 88
n_frames = 50
dim = 10
res = frames_data_fun(n_images=n_images, n_frames=n_frames, dim=dim)
labels = res[0]
data = res[1]

print(data.shape)
# print('data frame:', data[2,3,:,:])
# print('lable:', labels[2,:,:])


# ## Train and test datasets

# In[17]:

import random

data_samples = len(data)
train_samples = round(0.8*data_samples)

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

print(train_loader.shape)
print(test_loader.shape)


# ## Defind the model

# In[18]:

# Convolution
kernel_size = 3
filters = 20
pool_size = 4

# LSTM
lstm_output_size = 1

# Training
batch_size = 32
epochs = 5

print('Build model...')

####define a model

model = Sequential()
model.add(InputLayer(input_shape=(n_frames, dim, dim, 1)))
print(model.output_shape)
model.add(TimeDistributed(
    Conv2D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1,
                data_format="channels_last",
                input_shape=(n_frames, dim, dim, 1))))
print('conv2D shape', model.output_shape)

model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='same')))
print (model.output_shape)

model.add(TimeDistributed(Conv2D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1,
                data_format="channels_last",
                input_shape=(n_frames, dim, dim, 1))))
print('conv2D shape', model.output_shape)

model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='same')))
print (model.output_shape)

model.add(TimeDistributed(Conv2D(filters,
                 kernel_size,
                 padding='same',
                 activation='relu',
                 strides=1,
                data_format="channels_last",
                input_shape=(n_frames, dim, dim, 1))))
print('conv2D shape', model.output_shape)

model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='same')))
print (model.output_shape)

model.add(Reshape((n_frames,dim*dim*filters)))
print('reshape', model.output_shape)
model.add(LSTM(2*dim*dim, return_sequences=False, input_shape=(n_frames, dim*dim)))
print('LSTM', model.output_shape)
model.add(Activation('sigmoid'))
print('sigmoid', model.output_shape)
model.add(Reshape((2,dim,dim)))
print('reshape2', model.output_shape)

####

####define a loss function

def customloss(y_true, y_pred):
    loss_layer0 = -(y_true[:,0,:,:]*K.log(y_pred[:,0,:,:])*10+(1-y_true[:,0,:,:])*K.log(1-y_pred[:,0,:,:]))
#         print(loss_layer0)
    loss_layer0 = K.sum(loss_layer0)
    loss_layer1 = 0
    print('y_pred shape', y_pred.shape)
    loss_layer1 = K.abs(y_true[:,1,:,:]-y_pred[:,1,:,:])/K.cast(K.pow(y_pred.shape[2],2), 'float32')*(y_true[:,0,:,:]==1)
    totloss = loss_layer0+loss_layer1
    return totloss
    
####
def accuracy_met(y_true, y_pred):
    correct = 0
    total = 0
    input_dim = y_pred.shape[2]
    total = input_dim*input_dim
#     print('total', total)
    m = (y_true[:,0,:,:]>0)&K.equal(K.round(y_pred[:,0,:,:]),y_true[:,0,:,:])
    correct = tf.count_nonzero(K.equal(K.round(y_pred[:,0,:,:]), y_true[:,0,:,:]))-tf.count_nonzero(m&(K.not_equal(K.round(y_pred[:,1,:,:]), y_true[:,1,:,:])))
#     print('correct', correct)
    return correct/total*100

####compile the model

####
def count_met(y_true, y_pred):
    correct = 0
    total = 0
    input_dim = y_pred.shape[2]
    total = input_dim*input_dim
#     print('total', total)
    m = (y_true[:,0,:,:]>0)&K.equal(K.round(y_pred[:,0,:,:]), y_true[:,0,:,:])
    correct = tf.count_nonzero(m&(K.equal(K.round(y_pred[:,1,:,:]), y_true[:,1,:,:])))
#     print('correct', correct)
    return correct

####compile the model

model.compile(loss=customloss,
              optimizer='adam',
              metrics=[accuracy_met]
             )

####
def accuracy_met(y_true, y_pred):
    correct = 0
    total = 0
    input_dim = y_pred.shape[2]
    total = y_true.shape[0]*input_dim*input_dim
#     print('total', total)
    m = (y_true[:,0,:,:]>0)&K.equal(K.round(y_pred[:,0,:,:]),y_true[:,0,:,:])
    correct = tf.count_nonzero(K.equal(K.round(y_pred[:,0,:,:]), y_true[:,0,:,:]))-tf.count_nonzero(m&(K.not_equal(K.round(y_pred[:,1,:,:]), y_true[:,1,:,:])))
#     print('correct', correct)
    return correct/total*100

####
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Train...')

    model.fit(np.expand_dims(train_loader,4), train_labels[:,:,:,:],
              batch_size=batch_size,
              epochs=epochs)

    ## train accuracy
    print('#########')
    print('train results')
    train_pred = model.predict(np.expand_dims(train_loader,4))
    predicted = train_pred.round()
    labels = train_labels
    print('labels shape', labels.shape)
    total = labels.shape[0]*labels.shape[2]*labels.shape[3]
    m = (labels[:,0,:,:]>0)*(predicted[:,0,:,:] == labels[:,0,:,:])
    correct = (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()-(m*(predicted[:,1,:,:] != labels[:,1,:,:])).sum().item()
    semi_correct = (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()
    total_ones =  (labels[:,0,:,:]==1).sum().item()
    total_zeros = (labels[:,0,:,:]==0).sum().item()
    correct_ones = ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==1)).sum().item()
    correct_zeros = ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==0)).sum().item()
    print(correct_ones, total_ones, correct_zeros, total_zeros)
    print('accuracy:', correct/total*100, '%')
    print('semi_correct:', semi_correct/total*100, '%')
    print('correct ones:', correct_ones/total_ones*100, '%')
    print('correct zeros:', correct_zeros/total_zeros*100, '%')
    
## test accuracy
    print('#########')
    print('test results')
    test_pred = model.predict(np.expand_dims(test_loader,4))
    predicted = test_pred.round()
    labels = test_labels
    total = labels.shape[0]*labels.shape[2]*labels.shape[3]
    print('label shape', labels.shape)
    print('total', total)
    m = (labels[:,0,:,:]>0)*(predicted[:,0,:,:] == labels[:,0,:,:])
    correct = (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()-(m*(predicted[:,1,:,:] != labels[:,1,:,:])).sum().item()
    semi_correct = (predicted[:,0,:,:] == labels[:,0,:,:]).sum().item()
    print('correct', correct)
    print('semi correct', semi_correct)
    total_ones =  (labels[:,0,:,:]==1).sum().item()
    total_zeros = (labels[:,0,:,:]==0).sum().item()
    correct_ones = ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==1)).sum().item()
    correct_zeros = ((predicted[:,0,:,:] == labels[:,0,:,:])*(labels[:,0,:,:]==0)).sum().item()
    print(correct_ones, total_ones, correct_zeros, total_zeros)
    print('accuracy:', correct/total*100, '%')
    print('semi_correct:', semi_correct/total*100, '%')
    print('correct ones:', correct_ones/total_ones*100, '%')
    print('correct zeros:', correct_zeros/total_zeros*100, '%')


# In[ ]:



