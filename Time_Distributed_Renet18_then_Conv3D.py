
# coding: utf-8

# ## Import packages

# In[15]:

import sys
sys.path.append('/home/eym16/anaconda3/lib/python3.6/site-packages')

import numpy as np
import bisect
from __future__ import print_function

import keras.backend as K
import keras.layers
import keras.models
import keras.regularizers

import keras_resnet.blocks
import keras_resnet.layers
import keras_resnet.models

import numpy as np
import bisect
from __future__ import print_function

keras.backend.set_image_data_format('channels_last')


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
n_images = 500
n_frames = 199
dim = 10
res = frames_data_fun(n_images=n_images, n_frames=n_frames, dim=dim)
labels = res[0]
data = res[1]

print(data.shape)
# print('data frame:', data[2,3,:,:])
# print('lable:', labels[2,:,:])


# ## Train and test datasets

# In[ ]:

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

input_dim = train_loader.shape[2]
n_frames = train_loader.shape[1]


# ## Define the model

# ## Run the model

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
epochs = 100
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
    r4 = Residual(64, 1, r3)
    print('r4 shape', r4.shape)
    r5 = keras.layers.Reshape((1,10,10,n_frames))(r4)
    print('r5 shape', r5.shape)
    #conv3D
    r6 = keras.layers.Convolution3D(16, kernel_size=(1,1,3), strides=1, activation='relu', padding='same')(r5)
    print('r6 shape', r6.shape)
    r7 = keras.layers.Convolution3D(32, kernel_size=(1,1,3), strides=1, activation='relu', padding='same')(r6)
    print('r7 shape', r7.shape)
    r8 = keras.layers.Convolution3D(64, kernel_size=(1,1,3), strides=1, activation='relu', padding='same')(r7)
    print('r8 shape', r8.shape)
    out0 = keras.layers.Convolution3D(1, kernel_size=(1,1,3), strides=1, activation='sigmoid', padding='same')(r8)
    print('out0.a', out0.shape)
    out0 = keras.layers.Reshape((10,10,1))(out0)
    print('out0.b', out0.shape)
    out1 = keras.layers.Convolution3D(1, kernel_size=(1,1,3), strides=1, activation='relu', padding='same')(r8)
    print('out1', out1.shape)
    out1 = keras.layers.Reshape((10,10,1))(out1)
    print('out1.b', out1.shape)
    out = keras.layers.concatenate([out0, out1], axis=3)
    print('out shape', out.shape)

    model = Model(input=inp, output=out)
#     model.compile(optimizer=Nadam(lr=1e-5), loss='mean_squared_error')

#     plot_model(model, to_file='./toy_model.png', show_shapes=True)
# print(model.summary())


####define a loss function
#due to cnn output shape, loss and metrics are modified compared to TimeDistributed_Resnet+LSTM
def customloss(y_true, y_pred):
    print('y_pred', y_pred[:,:,:,0])
    loss_layer0 = -(y_true[:,:,:,0]*K.log(keras.backend.clip(y_pred[:,:,:,0], 0.001,0.999))*10+(1-y_true[:,:,:,0])*K.log(1-keras.backend.clip(y_pred[:,:,:,0], 0.001, 0.999)))
    print(loss_layer0)
    loss_layer0 = K.sum(loss_layer0)
    loss_layer1 = 0
    print('y_pred shape', y_pred.shape)
    loss_layer1 = K.abs(y_true[:,:,:,1]-y_pred[:,:,:,1])/K.cast(K.pow(y_pred.shape[2],2), 'float32')*(y_true[:,:,:,0]==1)
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
    
    model.fit(np.expand_dims(train_loader,4), train_labels.reshape((-1,dim,dim,2)),
              batch_size=batch_size,
              epochs=epochs)

    ## train accuracy
    print('#########')
    print('train results')
    train_pred = model.predict(np.expand_dims(train_loader,4))
    predicted = train_pred.round()
    labels = train_labels.reshape((-1,dim,dim,2))
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
    labels = test_labels.reshape((-1,dim,dim,2))
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



