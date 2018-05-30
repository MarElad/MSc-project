
# coding: utf-8

# # Attempt Jan

# In[2]:

import numpy as nm


# In[3]:

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
    
# images generator function
def frames_data_fun(n_images, n_frames, dim):
    labels = nm.empty((n_images, dim, dim))
    data = nm.empty((n_images, n_frames, dim, dim))
    for n in range(n_images):
        x = nm.random.poisson(lam=1, size=(dim,dim))
        x = nm.minimum(x,1)
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

nm.save('labels.npy', labels)
nm.save('data.npy', data)

