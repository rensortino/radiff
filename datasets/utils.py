import torch
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval
import numpy as np
from torch.utils.data import DataLoader


CLASSES = ['background', 'spurious', 'compact', 'extended']
COLORS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]


class RemoveNaNs(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img[np.isnan(img)] = 0
        return img


class ZScale(object):
    def __init__(self, contrast=0.15):
        self.contrast = contrast

    def __call__(self, img):
        interval = ZScaleInterval(contrast=self.contrast)
        min, max = interval.get_limits(img)

        img = (img - min) / (max - min)
        return img


class SigmaClip(object):
    def __init__(self, sigma=3, masked=True):
        self.sigma = sigma
        self.masked = masked

    def __call__(self, img):
        img = sigma_clip(img, sigma=self.sigma, masked=self.masked)
        return img


class MinMaxNormalize(object):
    def __init__(self):
        pass

    def __call__(self, img):
        img = (img - img.min()) / (img.max() - img.min())
        return img


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32)


class Unsqueeze(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.unsqueeze(0)


class FromNumpy(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.from_numpy(img).type(torch.float32)


def get_data_loader(dataset, batch_size, split="train", workers=8):
    batch_size = batch_size
    workers = workers
    is_train = split == "train"
    return DataLoader(dataset, shuffle=is_train, batch_size=batch_size,
                      num_workers=workers, persistent_workers=workers > 0,
                      drop_last=is_train
                      )
