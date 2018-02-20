import numpy as np
from matplotlib.patches import Ellipse
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from sklearn import mixture
import sys

def get_n_params(model):
    """
    Get number of trainable parameters in the model
    :param model: Pytorch model
    :return: number of trainable parameters in the model
    """
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])

def get_gaussian(data):
    patch = data
    A = np.mean(patch)
    patch = patch - np.min(patch)

    xx = np.repeat(np.expand_dims(range(patch.shape[1]), 0), patch.shape[0], 0) + 1
    yy = np.transpose(np.repeat(np.expand_dims(range(patch.shape[0]), 0), patch.shape[1], 0) + 1)

    x_est = np.sum(xx * patch) / np.sum(patch) - 1.
    y_est = np.sum(yy * patch) / np.sum(patch) - 1.

    x_std = np.sum((xx - x_est) ** 2 * patch) / np.sum(patch) - 1.
    y_std = np.sum((yy - y_est) ** 2 * patch) / np.sum(patch) - 1.
    xy_std = np.sum((xx - x_est) * (yy - y_est) * patch) / np.sum(patch) - 1.

    return x_est, y_est, x_std, y_std, xy_std, A

def get_gmm(data):
    def im2samples(im,max_samples=30):
        imx = ((im-np.min(im))/(np.max(im)-np.min(im))*max_samples).astype(np.int32)
        x = np.repeat(np.expand_dims(np.array(range(0,im.shape[0])),0),im.shape[1],0)
        y = np.transpose(np.repeat(np.expand_dims(np.array(range(0,im.shape[1])),0),im.shape[0],0),(1,0))
        coords = np.squeeze(np.dstack([x.flatten(),y.flatten()])).tolist()
        samples = [ [coord]*ndouble for coord,ndouble in zip(coords,imx.flatten()) if  ndouble>0]
        return np.array(np.sum(samples[::2])),np.array(np.sum(samples[1::2]))
    
    samples,valid = im2samples(data)
    n_components =  range(1,4)
    models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(samples)
          for n in n_components]
    
    bics = [m.bic(valid) for m in models]
    gmm = models[np.argmin(bics)]
    bigest = np.argmax(gmm.weights_)
    covariances = gmm.covariances_[bigest][:2, :2]
    indexes =gmm.predict(samples)
    amp = np.mean(data[samples[bigest==indexes]])
        
    return gmm.means_[bigest, 0], gmm.means_[bigest, 1],covariances[0,0],covariances[1,1],covariances[0,1],amp
        
def extract_descriptor(data_image, x_list, y_list, window_size=5,get_gaussian=get_gaussian):
    """
    Extract descriptors from an image using a giving point set
    :param data_image: the image from which the descriptors will be extracted
    :param x_list: list of synapse x coordinates  
    :param y_list: list of synapse y coordinates
    :param window_size: the size of window in which the descriptor is estimated
    :return: array of descripors [x,y,amp,ax1_width,ax2_width,stdxy, angle_rad, y, x]
    """
    result = []
    for x, y in zip(x_list, y_list):
        x = int(round(x))
        y = int(round(y))
        patch = data_image[max(x - window_size, 0):x + window_size, max(y - window_size, 0):y + window_size]
        x_est, y_est, stdx, stdy, stdxy, A = get_gaussian(patch)
        cov = np.array([[stdx, stdxy], [stdxy, stdy]])
        U, E, V = np.linalg.svd(cov)
        angle_rad = 0
        if U[0, 0] < U[1, 1]:
            angle_rad = np.arccos((U[0, 1])) - np.pi / 2.
        else:
            angle_rad = np.arccos((-U[0, 1])) - np.pi / 2.
        result.append([max(y - window_size, 0) + x_est, max(x - window_size, 0) + y_est, A, E[0], E[1], stdx, stdy, stdxy, angle_rad, y, x])
    return result


def find_peaks(probimage, distance=4, minval=0.5):
    """
    Detects peaks with a confidence threshold
    :param probimage: probabilities produced by dognet
    :param distance: minimal distance between peaks
    :param minval: minimal confidence from 0 to 1
    :return x,y,binary
    """
    norm = (probimage - probimage.min()) / (probimage.max() - probimage.min()) * 255.
    norm = norm.astype(np.uint8)

    binary = peak_local_max(norm, threshold_abs=int(255. * minval), min_distance=distance, indices=False)
    objects = label(binary)
    props = regionprops(objects)

    cc = np.array([p.centroid for p in props])
    
    if 0==cc.size:
        return [],[],None
        
    cc = cc[cc[:, 0] > 10]
    cc = cc[cc[:, 0] < probimage.shape[0] - 10]
    cc = cc[cc[:, 1] > 10]
    cc = cc[cc[:, 1] < probimage.shape[1] - 10]

    return cc[:, 0].tolist(), cc[:, 1].tolist(), binary


def draw_descriptors(ax, desciptor, color='m'):
    """
    Draw descriptor as ellipsoid
    :param ax: axis to draw
    :param descriptor: descriptor form utils.extract_descriptor function
    :param color: color used to draw ellipse
    """
    for x, y, A, sx, sy, tdx, stdy, stdxy, angle_rad, y_orig, x_orig in desciptor:
        ax.plot(x, y, color + 'o')
        el = Ellipse((x, y), sx, sy, color=color, fill=False, angle=180. * angle_rad / np.pi)
        ax.add_artist(el)


def calc_fitting(pts1, pts2, tau):
    """
    Calc how well the pts2 are related to pts1
    :param pts1: point ground truth
    :param pts2: point detected
    :param tau:  maximal distance between points
    :return: matched pairs
    """
    cost = np.zeros((pts1.shape[0], pts2.shape[0]))
    good1 = []
    good2 = []
    gt_res = []
    for index1, p1 in enumerate(pts1.astype(np.float32)):
        for index2, p2 in enumerate(pts2.astype(np.float32)):
            cost[index1, index2] = np.linalg.norm(p1 - p2)
            if cost[index1, index2] < tau and index1 not in good1 and index2 not in good2:
                good1.append(index1)
                good2.append(index2)
                gt_res.append((index1, index2))

    return gt_res


def get_metric(pts1, pts2, s=5.):
    """
    Calc how well the pts2 are related to pts1
    :param pts1: point ground truth
    :param pts2: point detected
    :param s: maximal distance between pair of points
    :return precision,recall,f1_score and point pairs
    """
    gt_res = calc_fitting(pts1, pts2, s)
    precision = float(len(gt_res)) / float(pts1.shape[0])
    recall = float(len(gt_res)) / float(pts2.shape[0])
    if  0==(precision + recall):
        return 0,0,0,[]
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1, gt_res

