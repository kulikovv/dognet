from skimage.filters import median, threshold_otsu, gaussian
from skimage.morphology import white_tophat,disk,black_tophat
import numpy as np

def cellprofiler_pipeline_single(img,sz=8,th=None):
    #1. estimate bg
    img = (img-img.min())/(img.max()-img.min())*255.
    bg = median(img.astype(np.uint8),selem=np.ones((100,100))).astype(np.float32)
    #2. find the difference we are intrested in only higher values
    diff = img-bg
    diff[diff<0]=0.
    
    if th is None:
        th = threshold_otsu(diff,255)
    #3. apply tophat    
    tophat = white_tophat(diff,selem=disk(sz))
    return tophat

def cellprofiler_pipeline(pre,post,sz=8,th=None):
    pre_process = np.zeros_like(pre[0])
    post_process = np.zeros_like(pre[0])
    for p in pre:
        pre_process+=cellprofiler_pipeline_single(p,sz,th)
    for p in post:
        post_process+=cellprofiler_pipeline_single(p,sz,th)
        
    pre_process/=len(pre)
    post_process/=len(post)
    return pre_process+post_process