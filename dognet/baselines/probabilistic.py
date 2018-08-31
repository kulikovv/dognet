"""
Synapse probabilistic m
"""
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from scipy.stats import norm
from skimage.morphology import remove_small_objects


def fg_prob(im):
    im = im.astype(np.float64)
    probs = np.zeros_like(im)

    for i in range(im.shape[0]):
        mean = np.mean(im[i])
        sigma = np.std(im[i])
        probs[i] = norm.cdf(im[i], loc=mean, scale=sigma)
    return probs


def convolve(im, size):
    prim = np.ones((size, size))
    im[im==0]=0.01
    log_image = np.log(im)

    for i in range(im.shape[0]):
        log_image[i] = ndimage.convolve(log_image[i], prim, mode='constant')

    return np.exp(log_image / (size ** 2))


def factor(vol):
    factors = []
    for n in range(len(vol)):
        if n == 0:
            diff = np.exp(-(vol[n] - vol[n + 1]) ** 2)
        elif n == len(vol) - 1:
            diff = np.exp(-(vol[n] - vol[n - 1]) ** 2)
        else:
            diff = np.exp(-(vol[n] - vol[n - 1]) ** 2 - (vol[n] - vol[n + 1]) ** 2)
        factors.append(diff)
    return factors


def factor_2(vol):
    factors = np.zeros_like(vol)
    for n in range(len(vol)):

        if n == len(vol) - 1:
            factors[n] = np.exp(-(vol[n] - vol[n - 1]) ** 2)
        else:
            factors[n] = np.exp(-(vol[n] - vol[n + 1]) ** 2)

    return factors


def remove_blobs(im, maxSize, th):
    probs = im.copy()
    for i in range(im.shape[0]):
        ter = im[i] > th
        ter = (remove_small_objects(ter, min_size=maxSize, connectivity=8))
        probs[i, ter] = 0
    return probs


def max_pooling(im, size):
    prim = np.ones((size, size))
    pos = np.multiply([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)], size)
    temp_maxes = np.zeros_like(im)
    maxes = np.zeros_like(im)
    for i in range(im.shape[0]):
        # convolution with boxfilter
        res = ndimage.convolve(im[i], prim, mode='constant') / (size ** 2)
        # max for 9 loc
        temp_maxes[i] = np.stack([shift(res, p) for p in pos]).max(0)

    for i in range(im.shape[0]):
        maxes[i] = temp_maxes[max(i - 1, 0):min(i + 1, im.shape[0])].max(0)
    return maxes


def combine_volumes(post, pre, base_threshold, size):
    maxes = max_pooling(pre, size)
    finalvolume = post * maxes
    finalvolume[post < base_threshold] = 0
    return finalvolume


def pipeline(raw, max_size=500, conf_theshold=0.7, window_size=2):
    prob_my = fg_prob(raw)
    prob_my = remove_blobs(prob_my, max_size, conf_theshold)
    prob_my = convolve(prob_my, window_size)
    factor_my = factor_2(prob_my)
    prob_my = prob_my * factor_my
    return prob_my


def probabilistic_synapse_segmentation(synapsin, psd95, max_size=500, conf_threshold=0.7, window_size=2, base_threshold=0.01):
    synapsin = pipeline(synapsin, max_size, conf_threshold, window_size)
    psd95 = pipeline(psd95, max_size, conf_threshold, window_size)
    return combine_volumes(psd95, synapsin, base_threshold, window_size)
