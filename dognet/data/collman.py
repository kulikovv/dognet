import numpy as np
from skimage.measure import regionprops
from skimage.transform import rescale


class Collman:
    def __init__(self, path,
                 req_channels=['collman15v2_Synapsin647', 'collman15v2_VGluT1_647', 'collman15v2_PSD95_488'],
                 scale=0.1):
        channels = []
        self.path = path
        self.scale = scale
        for channel in req_channels:
            print('Loading channel:', channel)
            data = np.transpose(np.load(self.path + channel + ".npy"), (1, 2, 0))
            channels.append(rescale(data, scale, mode='reflect'))

        self.data = np.stack(channels)
        self.anno = None

    def get_annotation(self, name="collman15v2_annotation", scale=None):
        if not self.anno:
            self.anno = np.load(self.path + name + ".npy")

        # Annotation = namedtuple('Annotation', ['image', 'labels', 'r', 'x', 'y'])
        layer = []
        if not scale:
            scale = self.scale

        for i in range(self.anno.shape[0]):
            props = regionprops(self.anno[i])
            yy = [p.centroid[0] * scale for p in props]
            xx = [p.centroid[1] * scale for p in props]
            layer.append(xx, yy)
        return layer
