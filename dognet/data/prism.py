from collections import namedtuple

import numpy as np
import scipy.io as sio
from skimage.draw import circle


class Prism:
    def __init__(self, path,
                 req_channels=['synapsin-1', 'PSD-95', 'bassoon', 'VGLUT1', 'SHANK3', 'Homer-1b/c']):
        full = np.load(path)
        self.channel_names = full['channel_names'].tolist()
        print(self.channel_names)
        req_indexes = [self.channel_names.index(c) for c in req_channels]
        self.data = full['data'][:, req_indexes, :, :]

    def get_annotations(self, annotation, margin=10, radius=2):
        annotations = []
        if annotation:
            for fov in range(self.data.shape[0]):
                try:
                    r, x, y = self.read_annotations(annotation, fov)
                    image = self.data[fov, :, int(r[1] - margin):int(r[1] + r[3] + margin),
                            int(r[0] - margin):int(r[0] + r[2] + margin)]
                    x = (x - r[0] + margin)
                    y = (y - r[1] + margin)
                    Annotation = namedtuple('Annotation', ['image', 'labels', 'r', 'x', 'y'])
                    annotations.append(Annotation(image, self.make_labels(image, x, y, radius), r, x, y))
                except:
                    pass
        return annotations

    def get_fov(self, fov):
        return self.data[fov]

    @staticmethod
    def make_labels(img, xs, ys, size=2):
        labels = np.zeros(img.shape[1:])
        for xv, yv in zip(xs, ys):
            rr, cc = circle(yv, xv, size)
            labels[rr, cc] = 1
        return labels

    @staticmethod
    def read_annotations(path, fov=0):
        annotation = sio.loadmat(path)
        rect = annotation['field'][0][fov][1][0][0][0][0]
        x, y = annotation['field'][0][fov][0]

        # Because of Matlab
        rect[0] = rect[0] - 1
        rect[1] = rect[1] - 1
        x = x - 1
        y = y - 1
        return rect, x, y


"""
p = Prism('/media/hpc-4_Raid/vkulikov/PRISM_annotation/Rep3-1/Rep3-1.npz')
smg_31 = p.get_annotations("/media/hpc-4_Raid/vkulikov/PRISM_annotation/Rep3-1/annotation_data_smg.r3-1.mat")
mbs_31 = p.get_annotations("/media/hpc-4_Raid/vkulikov/PRISM_annotation/Rep3-1/annotation_data_mbs.r3-1.mat")
print(len(smg_31), len(mbs_31))
plt.figure()
plt.imshow(smg_31[0].image[0])
plt.scatter(smg_31[0].x, smg_31[0].y)
plt.scatter(mbs_31[0].x, mbs_31[0].y)
plt.waitforbuttonpress()
"""
