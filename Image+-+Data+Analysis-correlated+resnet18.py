
# coding: utf-8

# ## Import packages

# In[1]:

from __future__ import print_function

import sys
sys.path.append('/home/eym16/anaconda3/lib/python3.6/site-packages')

import numpy as np
import bisect

import keras.backend as K
import keras.layers
from keras.layers import LSTM
import keras.models
import keras.regularizers

import keras_resnet.blocks
import keras_resnet.layers
import keras_resnet.models

import numpy as np
import bisect

keras.backend.set_image_data_format('channels_last')


# ## Data loading and preparation

# In[2]:

import pandas as pd
import bisect
import numpy as np
import os


dim = 10
Path = '/home/eym16/Simulated_Data/Data/'
filelist = os.listdir(Path)
print(filelist)
filelist.remove('.ipynb_checkpoints')
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
#     print(names)
    labels_file[names[0]] = labels_file[names[0]]/38000
    labels_file[names[1]] = labels_file[names[1]]/38000
#     print(labels_file.iat[5,1])
    intervals = np.arange(0,1,1/dim)
    loc1 = []
    loc2 = []
#     print(labels_file)
    for n in range(labels_file.shape[0]):
        loc1.append(bisect.bisect_left(intervals, labels_file.iat[n,0])-1)
        loc2.append(bisect.bisect_left(intervals, labels_file.iat[n,1])-1)
    for h,j in zip(loc1,loc2):
        labels[i][1][h][j] += 1
    m = (labels[i][1][:][:]!=0)
    labels[i][0][:][:]=1*m
#     print(labels)
    
    #read frames data
    data_file = pd.read_csv(Path + filelist[i+int(len(filelist)/2)], delimiter=',')
    #normalise columns
    data_file['xnano'] = data_file['xnano']/38000
    data_file['ynano'] = data_file['ynano']/38000
    data_file['intensity'] = data_file['intensity']/38000
#     print(data_file)
    if i==0:
        max_frame = data_file['frame'].max()
#         print(max_frame)
    frames_data = np.zeros((max_frame,dim,dim))
#     print(frames_data.shape)
    for frame in range(max_frame):
        df_tmp = data_file[data_file.frame==frame]
        loc1 = []
        loc2 = []
        intervals = np.arange(0,1,1/dim)
#     print(labels_file)
        for n in range(df_tmp.shape[0]):
            loc1.append(bisect.bisect_left(intervals, df_tmp.iat[n,1])-1)
            loc2.append(bisect.bisect_left(intervals, df_tmp.iat[n,2])-1)
        for h,j,k in zip(loc1,loc2,df_tmp['intensity']):
            frames[i][frame-1][h][j] += k
            
#         print(frames_data)
#     for l in range(max_frame):   
#         print((frames_data[l,:,:]>0).sum())
#     print(frames_data[0,:,:])
#     print((frames_data[:,:,:]==1).sum())
# print(labels)
print(frames.shape)
print(labels.shape)
print((frames[0,:,:,:]>1).sum())
print((labels[1,:,:,:]>0).sum())

data = frames


# ## Hyper parameters

# In[3]:

# Hyper-parameters
num_epochs = 100
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

# datasets shapes
#print(train_loader.shape)
#print(test_loader.shape)


# ## Define and run the model

# In[ ]:

from keras.models import Model
from keras.optimizers import Nadam
from keras.layers import BatchNormalization, Convolution2D, Input, merge
from keras.layers.core import Activation, Layer
from keras.utils.vis_utils import plot_model
import pydot
import tensorflow as tf

'''
Keras Customizable Residual Unit
This is a simplified implementation of the basic (no bottlenecks) full pre-activation residual unit from He, K., Zhang, X., Ren, S., Sun, J., "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027v2).
'''
#hyper parameters
epochs = 55
batch_size = 64


def conv_block(feat_maps_out, prev):
    prev = keras.layers.TimeDistributed(BatchNormalization(axis=1))(prev) # Specifying the axis and mode allows for later merging
    print('bn1', prev.shape)
    prev = keras.layers.TimeDistributed(Activation('relu'))(prev)
    print('ac1', prev.shape)
    prev = keras.layers.TimeDistributed(Convolution2D(feat_maps_out, kernel_size=3, strides=1, padding='same'))(prev)
    print('conv1', prev.shape)
    prev = keras.layers.TimeDistributed(BatchNormalization(axis=1))(prev) # Specifying the axis and mode allows for later merging
    print('bn2', prev.shape)
    prev = keras.layers.TimeDistributed(Activation('relu'))(prev)
    print('ac2', prev.shape)
    prev = keras.layers.TimeDistributed(Convolution2D(feat_maps_out, 3, strides=1, padding='same'))(prev)
    print('conv2', prev.shape)
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = keras.layers.TimeDistributed(Convolution2D(feat_maps_out, 1, strides=1, padding='same'))(prev)
        print(prev.shape)
    return prev 


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)
    
    print('skip shape', skip.shape)
    print('conv shape', conv.shape)
    print('prev shape', prev_layer.shape)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([skip, conv], mode='sum') # the residual connection


if __name__ == "__main__":
    # NOTE: Toy example shows structure
    img_rows = dim  
    img_cols = dim 

    inp = Input((n_frames, img_rows, img_cols, 1))
    print('inp shape', inp.shape)
    cnv1 = keras.layers.TimeDistributed(Convolution2D(16, kernel_size=3, strides=1, activation='relu', input_shape=(n_frames, img_rows, img_cols, 1), padding='same'))(inp)
    r1 = Residual(16, 32, cnv1)
    # An example residual unit coming after a convolutional layer. NOTE: the above residual takes the 64 output channels
    # from the Convolutional2D layer as the first argument to the Residual function
    r2 = Residual(32, 64, r1)
    r3 = Residual(64, 64, r2)
    print('r3 shape', r3.shape)
    r4 = keras.layers.Reshape((dim,dim,int(r3.shape[1])*int(r3.shape[4])))(r3)
    print('r4 shape', r3.shape)
    r5 = keras.layers.Reshape((n_frames,dim*dim*64))(r4)
    r6 = keras.layers.LSTM(2*dim*dim, return_sequences=False, input_shape=(n_frames, dim*dim))(r5)
    r7 = Activation('sigmoid')(r6)
    r8 = keras.layers.Reshape((2,dim,dim))(r7)
    print('r8 shape', r8.shape)

    model = Model(input=inp, output=r8)
#     model.compile(optimizer=Nadam(lr=1e-5), loss='mean_squared_error')

#     plot_model(model, to_file='./toy_model.png', show_shapes=True)
# print(model.summary())


####define a loss function

def customloss(y_true, y_pred):
    print('y_pred', y_pred[:,:,:,0])
    loss_layer0 = -(y_true[:,0,:,:]*K.log(keras.backend.clip(y_pred[:,0,:,:], 0.001,0.999))*10+(1-y_true[:,0,:,:])*K.log(1-keras.backend.clip(y_pred[:,0,:,:], 0.001, 0.999)))
    print(loss_layer0)
    loss_layer0 = K.sum(loss_layer0)
    loss_layer1 = 0
    print('y_pred shape', y_pred.shape)
    loss_layer1 = K.abs(y_true[:,1,:,:]-y_pred[:,1,:,:])/K.cast(K.pow(y_pred.shape[2],2), 'float32')*(y_true[:,0,:,:]==1)
    totloss = loss_layer0+loss_layer1
    return totloss
    
####

####compile the model

model.compile(loss=customloss,
              optimizer='adam'
             )

####
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Train...')
    
    print('train shape:', (np.expand_dims(train_loader,4)).shape)
    print('lables shape', train_labels.shape)
    
    model.fit(np.expand_dims(train_loader,4), train_labels,
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
    print('predicted', predicted[:,:,:,:].shape)
    print('labels', labels[:,:,:,:].shape)
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

tf.__version__


# In[ ]:

a = np.array([1, 2])
b = np.array([1, 3])
print((a==b).sum().item())


# In[ ]:

keras.backend.set_image_data_format('channels_first')


# In[ ]:



