from __future__ import division
import sys
sys.path.insert(0, '..')

import numpy as np

import pandas as pd
import dognet
import torch
from torch.autograd import Variable
import pandas as pd
from skimage.io import imsave


req_channels=[b'synapsin-1', b'PSD-95', b'VGLUT1',b'bassoon']
print(req_channels)
rep21 = dognet.data.Prism("../datasets/prism17/rep2-1/Rep2-1.npz",req_channels=req_channels)
rep31 = dognet.data.Prism("../datasets/prism17/rep3-1/Rep3-1.npz",req_channels=req_channels)
rep31_map = dognet.data.Prism("../datasets/prism17/rep3-1/Rep3-1.npz",req_channels=[b'MAP2'])

#train
rep21_anno = rep21.get_annotations("../datasets/prism17/rep2-1/annotation_data_mbs_final.r2-1.mat")\
+ rep21.get_annotations("../datasets/prism17/rep2-1/annotation_data_smg.r2-1.mat")

#test
rep31_anno = rep31.get_annotations("../datasets/prism17/rep3-1/annotation_data_mbs.r3-1.mat")\
+ rep31.get_annotations("../datasets/prism17/rep3-1/annotation_data_smg.r3-1.mat")


meanx = rep21.data.mean(axis=(0,2,3))
minx = rep21.data.min(axis=(0,2,3))
maxx = rep21.data.max(axis=(0,2,3))

def get_normparams(data):
    return data.mean(axis=(0,2,3)),data.min(axis=(0,2,3)),data.max(axis=(0,2,3))

def normalize(im,norm_data):
    meanx,minx,maxx = norm_data
    x = np.copy(im.astype(np.float32))
    x = x.transpose(1,2,0)
    x = (x - meanx - minx)/(maxx - minx)
    return x.transpose(2,0,1)

def inference(net,image,get_intermediate=False):
    x = np.expand_dims(image,0)
    vx = Variable(torch.from_numpy(x).float())
    if torch.cuda.is_available():
        vx = vx.cuda()
    
    res,inter = net(vx)
    if get_intermediate:
        return res.data.cpu().numpy(),inter.data.cpu().numpy()
    return res.data.cpu().numpy()

def estimate(net,test_set):
    mf1=[]
    mprec =[]
    mrec = []
    for test in test_set:
        y = inference(net,normalize(test.image,get_normparams(rep21.data)))
        xx,yy,_ = dognet.find_peaks(y[0,0],3)
        dog_pts = np.array([yy,xx]).transpose(1,0)
        xs,ys = test.x,test.y
        gt_pts = np.array([xs,ys]).transpose(1,0)
        prec = 0
        rec =0
        f1 = 0 
        if len(xx)>0:
            prec,rec,f1,_ = dognet.get_metric(gt_pts,dog_pts,s=4.)
        mf1.append(f1)
        mprec.append(prec)
        mrec.append(rec)
    return np.mean(mf1),np.mean(mprec),np.mean(mrec)

gen = dognet.create_generator([normalize(r.image,get_normparams(rep21.data)) for r in rep21_anno[:2]],
                                                 [l.labels for l in rep21_anno[:2]])
net = dognet.SimpleIsotropic(len(req_channels),11,4,learn_amplitude=True,return_intermediate=True)
net.weights_init()
if torch.cuda.is_available():
    print('with CUDA')
    net = net.cuda()
else:
    print('CUDA is not detected, running on CPU.')
net,errors = dognet.train_routine(net,gen,n_iter=3000,margin=5)
print(estimate(net,rep31_anno))

print("processing PRISM dataset")
for rep,name in zip([rep21,rep31],['rep21','rep31']):
    dm = []
    for fov in range(0,6):
        x = rep.data[fov]
        y = inference(net,normalize(x,get_normparams(rep21.data))) 
        xx,yy,_ = dognet.find_peaks(y[0,0],3)
        imsave("../results/prism_prob_"+name+"_"+str(fov)+".png",y[0,0])
        pic = x[:3].transpose(1,2,0)
        pic = np.copy(pic)
        for x,y in zip(xx,yy):
            x = int(x)
            y = int(y)
            pic[x-1:x+2,y,0]=32000
            pic[x-1:x+2,y,1]=32000
            pic[x-1:x+2,y,2]=0
            pic[x,y-1:y+2,0]=32000
            pic[x,y-1:y+2,1]=32000
            pic[x,y-1:y+2,2]=0
        imsave("../results/prism_loc_"+name+"_"+str(fov)+".png",pic)
        
        for c in range(len(req_channels)):          
            desc = dognet.extract_descriptor(x[c],xx,yy,5)
            dm+=[[fov,req_channels[c]]+d for d in desc]
    dm = np.array(dm)            
    d = {'fov': dm[:,0] , 'marker': dm[:,1],'x': dm[:,2] ,'y': dm[:,3] ,'A': dm[:,4] ,'L1': dm[:,5]
         ,'L2': dm[:,6] ,'sigmax2': dm[:,7],'sigmay2': dm[:,8],'sigmaxy': dm[:,9],'angle': dm[:,10],
         'x_dog': dm[:,11],'y_dog': dm[:,12]}
    df = pd.DataFrame(data=d)
    df.to_csv("../results/prism17.csv")