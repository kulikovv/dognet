import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def create_generator(data, labels, s=0):
    """
    Generates training patches
    :param data: list of images
    :param labels: list of labels
    :return: function generator
    """

    def f(n=10, size=(64, 64), s=s):

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


def update_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_percent(percent):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('=' * percent, 5 * percent))
    sys.stdout.flush()


def train_routine(detector,
                  generator,
                  n_iter=5000,
                  loss=nn.BCELoss(),
                  lr=0.01,
                  margin=10,
                  decay_schedule=(3000, 0.1),
                  use_gpu=True):
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
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, detector.parameters()), lr=lr)
    detector.train()
    errors = []
    print("Training started!")
    percent_old = 0
    for i in range(n_iter):
        x, y = generator()
        vx, vy = Variable(torch.from_numpy(x).float()), \
                 Variable(torch.from_numpy(y).float(), requires_grad=False)

        if use_gpu:
            vx = vx.cuda()
            vy = vy.cuda()

        p, _ = detector(vx)
        optimizer.zero_grad()
        l = loss(p[:, :, margin:-margin, margin:-margin], vy[:, :, margin:-margin, margin:-margin])

        errors.append(l.data.select(0, 0))
        l.backward()
        optimizer.step()

        if 0 == (i + 1) % decay_schedule[0]:
            lr = lr * decay_schedule[1]
            update_rate(optimizer, lr)

        percent = int(float(i) / float(n_iter) * 20.)
        if percent_old != percent:
            percent_old = percent
            print_percent(percent)

    print_percent(20)
    detector.eval()
    print("\nTraining finished!")
    return detector, errors
