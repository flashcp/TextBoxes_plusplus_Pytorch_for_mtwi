from .mtwi2018 import MTWIDetection, MTWIAnnotationTransform, MTWI_CLASSES
from .mtwi2018_test import MTWIDetectionTest

from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size), boxes, labels
