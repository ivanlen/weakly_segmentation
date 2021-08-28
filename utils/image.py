import numpy as np


def torch_to_numpy(torch_image):
    npy_im = torch_image.numpy()
    return np.rollaxis(npy_im, 0, 3)
