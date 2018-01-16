import numpy as np
from skimage.draw import circle, ellipse
from skimage.filters import gaussian

w = 6
h = 1.5
noise = 0.5


def generate(n_obj=10, n_dist=20, w=6., h=1.5, noise=0.5, glob_noise=0.1):
    image = np.zeros((100, 100))
    gt = np.zeros_like(image)
    # traget objects
    for i in range(n_obj):
        x = np.random.randint(0, image.shape[0])
        y = np.random.randint(0, image.shape[1])
        a = np.random.randn() * 3.14
        nx = np.random.randn() * noise
        ny = np.random.randn() * noise
        rr, cc = ellipse(x, y, w + nx, h + ny, shape=image.shape, rotation=a)
        image[rr, cc] = 1
        rr, cc = ellipse(x, y, w, h, shape=image.shape, rotation=a)
        gt[rr, cc] = 1
    image = gaussian(image, 1)
    # distractors
    distract = np.zeros_like(image)
    for i in range(n_dist):
        x = np.random.randint(0, distract.shape[0])
        y = np.random.randint(0, distract.shape[1])
        rr, cc = circle(x, y, 3, shape=distract.shape)
        distract[rr, cc] = 1.

    distract = gaussian(distract, 1) * 1.
    return image + distract + np.random.randn(image.shape[0], image.shape[1]) * glob_noise, gt


def create_toy_generator(n_images=10,generate_func=generate):
    def f(n_img=n_images, gen=generate_func):
        l = []
        for i in range(n_img):
            l.append(gen())
        return np.stack(l)[:, 0:1, :, :], np.stack(l)[:, 1:2, :, :]
    return f


