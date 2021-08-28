import numpy as np


def torch_to_numpy(torch_image):
    npy_im = torch_image.numpy()
    return np.rollaxis(npy_im, 0, 3)


class UnNormalize(object):
    """
    https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: RGB image.
        """
        _tensor = tensor.detach().clone()
        for t, m, s in zip(_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return _tensor
