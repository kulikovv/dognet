import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def create_generator(data, labels, s=0 , size=(64, 64), n=10):
    """
    Generates training patches
    :param data: list of images
    :param labels: list of labels
    :return: function generator
    """

    def f(n=n, size=size, s=s):

        d = np.zeros((n, data[0].shape[0], size[0], size[1]))
        l = np.zeros((n, 1, size[0] - 2 * s, size[1] - 2 * s))
        rot = np.random.randint(0, 1, n)
        flip = np.random.randint(0, 1, n)
        for i in range(n):
            index = np.random.randint(0, len(data))
            x = np.random.randint(0, data[index].shape[1] - size[0])
            y = np.random.randint(0, data[index].shape[2] - size[1])
            dd = data[index][:, x:x + size[0], y:y + size[1]].copy()
            ll = labels[index][x + s:x + size[0] - s, y + s:y + size[1] - s].copy()

            if rot[i] > 0:
                dd = np.rot90(dd)
                ll = np.rot90(ll)
            if flip[i] > 0:
                dd = np.flipud(dd)
                ll = np.flipud(ll)
            d[i] = dd
            l[i, 0] = ll
        return d, l

    return f

def create_generator_3d(data, labels, s=0 , size=(64, 64), n=10, depth=1):
    """
    Generates training patches
    :param data: list of images
    :param labels: list of labels
    :return: function generator
    """

    def f(n=n, size=size, s=s):

        d = np.zeros((n, data[0].shape[0],1+2*depth, size[0], size[1]))
        l = np.zeros((n, 1, size[0] - 2 * s, size[1] - 2 * s))
        rot = np.random.randint(0, 1, n)
        flip = np.random.randint(0, 1, n)
        for i in range(n):
            index = np.random.randint(depth, len(data)-depth)
            x = np.random.randint(0, data[index].shape[1] - size[0])
            y = np.random.randint(0, data[index].shape[2] - size[1])
            dd = np.stack([data[a][:, x:x + size[0], y:y + size[1]].copy() for a in range(index-depth,index+depth+1) ],1) 
            ll = labels[index][x + s:x + size[0] - s, y + s:y + size[1] - s].copy()

            if rot[i] > 0:
                dd = np.rot90(dd)
                ll = np.rot90(ll)
            if flip[i] > 0:
                dd = np.flipud(dd)
                ll = np.flipud(ll)
            d[i] = dd
            l[i, 0] = ll
        return d, l

    return f

def update_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_percent(percent):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('=' * percent, 5 * percent))
    sys.stdout.flush()

    
def soft_dice_loss(y_pred, y_true, epsilon=1e-6):
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_pred: b x N x X x Y Network output, must sum to 1 over c channel (such as after softmax) 
        y_true: b x N x X x Y  One hot encoding of ground truth       
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score

    numerator = 2. * torch.sum(y_pred * y_true)
    denominator = torch.sum(y_pred.pow(2) + y_true)
    
    return 1 - torch.mean(numerator / (denominator + epsilon))  # average over classes and batch

def create_weight(pos_weight):
    def weighted_binary_cross_entropy(sigmoid_x, targets, size_average=True, reduce=True):
        """
        Args:
            sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
            targets: true value, one-hot-like vector of size [N,C]
            pos_weight: Weight for postive sample
        """
        if not (targets.size() == sigmoid_x.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

        loss = -  pos_weight*targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()



        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()
    return weighted_binary_cross_entropy

def train_routine(detector,
                  generator,
                  n_iter=5000,
                  loss="bce",
                  lr=0.01,
                  margin=10,
                  decay_schedule=(3000, 0.1),
                  optimizer=None,
                  regk=0.,
                  verbose=True):
    """
    Train a detector with respect to the data from generator
    :param detector: A detector network
    :param generator: A generator
    :param n_iter: number of iterations
    :param loss: loss function
    :param lr: starting learning rate
    :param margin: margin for ignore
    :param decay_schedule: pair of a which iteration and how should we decay
    :param use_gpu: if system has cuda support, the training can be run on GPU
    :return: trained network, errors
    """
    
    detector.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, detector.parameters()), lr=lr)
     
    device = detector.parameters().next().device
    
    _, y = generator()
    sumy = y[0,0].sum()
    if 0==sumy:
        sumy=1
    ky = torch.FloatTensor([(y.shape[-1]*y.shape[-2]-y.sum()/y.shape[0])/sumy*y.shape[0]]).to(device)
    
    floss = soft_dice_loss
    if loss=="bce":
        floss = nn.BCELoss()
    elif loss=="softdice":
        floss = soft_dice_loss
    elif loss=="weightbce":
        floss = create_weight(ky)
    if verbose:
        print(ky,y.shape,y.shape[-1]*y.shape[-2],y[0,0].sum(),y[0,0].max())
        print(floss) 
    
        
    
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, detector.parameters()), lr=lr)
    errors = []
    if verbose:
        print("Training started!")
    percent_old = 0
    for i in range(n_iter):
        x, y = generator()
        vx, vy = torch.from_numpy(x).float().to(device), \
                 torch.from_numpy(y).float().to(device)


        p, _ = detector(vx)
        optimizer.zero_grad()
        l = floss(p[:, :, margin:-margin, margin:-margin], vy[:, :, margin:-margin, margin:-margin])+regk*torch.sum(torch.pow(detector.get_reg_params().next(),2))

        errors.append(l.item())
        l.backward()
        optimizer.step()

        if 0 == (i + 1) % decay_schedule[0]:
            lr = lr * decay_schedule[1]
            update_rate(optimizer, lr)

        percent = int(float(i) / float(n_iter) * 20.)
        if percent_old != percent:
            percent_old = percent
            if verbose:
                print_percent(percent)

    
    detector.eval()
    if verbose:
        print_percent(20)
        print("\nTraining finished!")
    return detector, errors
