from __future__ import division
import sys
sys.path.insert(0, '..')

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.measure import label,regionprops
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.transform import rescale
import dognet
import pandas as pd

import torch
from torch.autograd import Variable

net =dognet.DeepIsotropic(3,11,4,3,learn_amplitude=False)

def load_annotation(path,scale,synapses_indexes=None):
    anno = np.load(path)['collman15v2_annotation']
    layer = []
    for i in range(anno.shape[0]):
        pt = []
        props = regionprops(anno[i])
        for p in props:
            if (synapses_indexes is not None):
                if (p.label in synapses_indexes):
                    pt.append(np.array(p.centroid)*scale)
            else:
                pt.append(np.array(p.centroid)*scale)
        layer.append(pt)
    return layer
collman = np.load('../datasets/collman15/collman_large.npy')
b = load_annotation("../datasets/collman15/collman15v2_annotation.npz",0.05)

from skimage.draw import circle
def make_labels(img,xs,ys,radius=5):
    labels = np.zeros(img.shape[1:])
    for xv,yv in zip(xs,ys):
        rr,cc = circle(xv,yv,radius,labels.shape)
        
        labels[rr,cc]=1
    return labels

#Train set
def make_training_set(labels,indexes):
    train_images = []
    train_labels = []
    for i in indexes:
        d = make_labels(collman[:,0],np.array(labels[i])[:,0],np.array(labels[i])[:,1])
        train_images.append(collman[:,i-2:i+2].mean(axis=1))
        train_labels.append(d)
    return train_images,train_labels


def inference(net,image,get_inter = False):
    x = np.expand_dims(image,0)
    vx = Variable(torch.from_numpy(x).float())
    if torch.cuda.is_available():
        vx = vx.cuda()
        
    res,inter = net(vx)
    if get_inter:
        return res.data.cpu().numpy(),inter.data.cpu().numpy()
    return res.data.cpu().numpy()

def estimate_quality(collman,net,layer,slices=[2,3,4,5,6],th=0.5):
    mprecision=[]
    mrecall=[]
    mf1_score=[]
    for s in slices:
        y  = inference(net,collman[:,s-2:s+2].mean(axis=1))
        gt_pts = np.array([np.array(layer[s])[:,1],np.array(layer[s])[:,0]]).transpose(1,0)
        #print(gt_pts)
        coords = np.array([ list(p.centroid) for p in regionprops(label(y[0,0]>th)) if p.area>5])
        dog_pts = np.array([coords[:,1],coords[:,0]]).transpose(1,0)
        
        precision,recall,f1_score,_ = dognet.get_metric(gt_pts,dog_pts,s=10.)
        
        mprecision.append(precision)
        mrecall.append(recall)
        mf1_score.append(f1_score)
    return np.mean(mf1_score),np.mean(mprecision),np.mean(mrecall)


#net = dognet.SimpleAnisotropic(3,11,3,return_intermediate=True)
#net.weights_init()
if torch.cuda.is_available():
    print('with CUDA')
    net = net.cuda()
else:
    print('CUDA is not detected, running on CPU. Take a cup of tea or coffe.')
    
train_images,train_labels = make_training_set(b,range(2,5))
net,errors =dognet.train_routine(net,dognet.create_generator(train_images,train_labels),n_iter=3000,margin=5)
print(estimate_quality(collman,net,b,slices=range(8,25)))


name = 'collman'
req_channels=[b'collman15v2_Synapsin647', b'collman15v2_VGluT1_647', b'collman15v2_PSD95_488']
dm=[]
print("Proccessing Collman15 dataset")
for silce in range(2,25):
    x = collman[:,silce-2:silce+2].mean(axis=1)
    y  = inference(net,x)
    coords = np.array([ list(p.centroid) for p in regionprops(label(y[0,0]>0.5)) if p.area>1])
    
    for c in range(len(req_channels)):    
            desc = dognet.extract_descriptor(x[c],coords[:,0],coords[:,1],10,get_gaussian=dognet.get_gaussian)   
            dm+=[[silce,req_channels[c]]+d for d in desc]
            
dm = np.array(dm)            
d = {'fov': dm[:,0] , 'marker': dm[:,1],'x': dm[:,2] ,'y': dm[:,3] ,'A': dm[:,4] ,'L1': dm[:,5]
     ,'L2': dm[:,6] ,'sigmax2': dm[:,7],'sigmay2': dm[:,8],'sigmaxy': dm[:,9],'angle': dm[:,10],
     'x_dog': dm[:,11],'y_dog': dm[:,12]}
df = pd.DataFrame(data=d)
df.to_csv("../results/collman.csv")


