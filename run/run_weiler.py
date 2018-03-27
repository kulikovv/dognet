from __future__ import division

import sys
sys.path.insert(0, '..')

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.measure import label,regionprops
from skimage.transform import rescale

import dognet
import torch
from torch.autograd import Variable
from skimage.draw import circle
import pandas as pd
from skimage.io import imsave


collman = np.load('../datasets/collman15/collman_small.npy')

def inference(net,image):
    x = np.expand_dims(image,0)
    
    vx = Variable(torch.from_numpy(x).float())
    if torch.cuda.is_available():
        vx = vx.cuda()
    res,_ = net(vx)
    return res.data.cpu().numpy()

def make_labels(img,xs,ys,radius=2):
    labels = np.zeros(img.shape[1:])
    for xv,yv in zip(xs,ys):
        rr,cc = circle(yv,xv,radius)
        labels[rr,cc]=1
    return labels

def load_annotation(path,scale):
    anno = np.load(path)['collman15v2_annotation']
    
    layer = []
    for i in range(anno.shape[0]):
        props = regionprops(anno[i])
        yy = [p.centroid[0] * scale for p in props]
        xx = [p.centroid[1] * scale for p in props]
        layer.append([xx, yy])
    return layer

print("training on collman15 dataset")

layer = load_annotation('../datasets/collman15/collman15v2_annotation.npz',2.23e-9/1e-7)

#Train set
train_images = []
train_labels = []
for i in range(9,14):
    d = make_labels(collman[:,0],layer[i][0],layer[i][1])
    train_images.append(collman[:,i-2:i+2].mean(axis=1))
    train_labels.append(d)
    
n = dognet.SimpleAnisotropic(3,9,6,return_intermediate=True,learn_amplitude=False)
n.weights_init()
if torch.cuda.is_available():
    print('with cuda')
    n = n.cuda()
else:
    print('CUDA is not detected, running on CPU')
net,errors =dognet.train_routine(n,dognet.create_generator(train_images,train_labels),n_iter=3000,margin=3)

weiler = np.load('../datasets/weiler14/weiler.npz')['weiler']


new_weiler = []
for i in range(3):
    frag = collman[i,0,20:80,20:100]
    intfrag = weiler[i,0,300:900,350:900]
    k = (np.std(frag)/np.std(intfrag))
    shift = (np.mean(frag)-np.mean(intfrag))
    wfrag = intfrag*k+shift
    
    domain_shift = []
    for j in range(weiler.shape[1]):
        domain_shift.append(weiler[i,j]*k+shift)
        
    new_weiler.append(np.stack(domain_shift))
new_weiler = np.stack(new_weiler)

print("processing weiler14 dataset")

name = 'weiler'
req_channels=['Ex3R43C2_Synapsin1_3', 'Ex3R43C2_vGluT1_2', 'Ex3R43C2_PSD95_2']
dm=[]
for silce in range(1,2):
    x = new_weiler[:,silce-2:silce+2].mean(axis=1).astype(np.float32)
    y  = inference(net,x)
    xx,yy,_ = dognet.find_peaks(y[0,0],3)
    
    imsave("../results/weiler_prob_"+str(silce)+".png",y[0,0])
    pic = x[:3].transpose(1,2,0)
    
    pic = np.copy(pic)
    pic[pic>0.5]=0.5
    pic = (pic-np.min(pic,(0,1)))/(np.max(pic,(0,1))-np.min(pic,(0,1))).astype(np.float)
    for x,y in zip(xx,yy):
        x = int(x)
        y = int(y)
        pic[x-1:x+2,y,0]=1
        pic[x-1:x+2,y,1]=1
        pic[x-1:x+2,y,2]=0
        pic[x,y-1:y+2,0]=1
        pic[x,y-1:y+2,1]=1
        pic[x,y-1:y+2,2]=0
    imsave("../results/weiler_loc_"+str(silce)+".png",pic)
    for c in range(len(req_channels)):    
            desc = dognet.extract_descriptor(weiler[c,silce-2:silce+2].mean(axis=0),xx,yy,3)   
            dm+=[[silce,req_channels[c]]+d for d in desc]
            
dm = np.array(dm)            
d = {'fov': dm[:,0] , 'marker': dm[:,1],'x': dm[:,2] ,'y': dm[:,3] ,'A': dm[:,4] ,'L1': dm[:,5]
     ,'L2': dm[:,6] ,'sigmax2': dm[:,7],'sigmay2': dm[:,8],'sigmaxy': dm[:,9],'angle': dm[:,10],
     'x_dog': dm[:,11],'y_dog': dm[:,12]}
df = pd.DataFrame(data=d)
df.to_csv(name+".csv")
