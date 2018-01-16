import numpy as np
from skimage.measure import regionprops
from skimage.transform import rescale


class Weiler:
    def __init__(self, path,
                 req_channels=['Ex3R43C2_Synapsin1_3', 'Ex3R43C2_vGluT1_2', 'Ex3R43C2_PSD95_2'],
                 scale=1.):
        channels = []
        self.path = path
        self.scale = scale
        for channel in req_channels:
            print('Loading channel:', channel)
            data = np.transpose(np.load(self.path + channel + ".npy"), (1, 2, 0))
            channels.append(rescale(data, scale, mode='reflect'))

        self.data = np.stack(channels)