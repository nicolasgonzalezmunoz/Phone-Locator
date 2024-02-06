"""Implement torch transformations that are position-aware"""
from typing import TypeAlias
import numpy as np
import torchvision.transforms.functional as tf


LabelTuple: TypeAlias = tuple[float, float]


class RandomHorizontalFlip:
    '''Horizontal flip the image and labels with probability p.'''
    def __init__(self, p: float = 0.5) -> None:
        if not 0 <= p <= 1:
            msg = 'Variable p is a probability, should be float between 0 to 1'
            raise ValueError(msg)
        self.p = p

    def __call__(
        self,
        image_label_sample: tuple[np.ndarray, LabelTuple]
    ) -> tuple[np.ndarray, LabelTuple]:
        image = image_label_sample[0]
        label = image_label_sample[1]
        c_x, c_y = label
        if np.random.random() < self.p:
            image = tf.hflip(image)
            label = 1- c_x, c_y
        return image, label


class RandomVerticalFlip:
    '''Vertically flip the image and labels with probability p'''
    def __init__(self, p: float = 0.5) -> None:
        if not 0 <= p <= 1:
            msg = 'Variable p is a probability, should be float between 0 to 1'
            raise ValueError(msg)
        self.p = p

    def __call__(
        self,
        image_label_sample: tuple[np.ndarray, LabelTuple]
    ) -> tuple[np.ndarray, LabelTuple]:
        image = image_label_sample[0]
        label = image_label_sample[1]
        c_x, c_y = label
        if np.random.random() < self.p:
            image = tf.vflip(image)
            label = c_x, 1 - c_y
        return image, label


class RandomTranslation:
    '''Translate the image and labels by a random amount.'''
    def __init__(
        self,
        max_translation: tuple[float, float] = (0.2, 0.2),
        eps: float = 0.1
    ) -> None:
        is_valid = (0 <= max_translation[0] <= 1)
        is_valid = is_valid and (0 <= max_translation[1] <= 1)
        if not is_valid:
            msg = 'Variable max_translation should be float between 0 to 1'
            raise ValueError(msg)
        self.max_translation_x = max_translation[0]
        self.max_translation_y = max_translation[1]
        self.eps = eps

    def __call__(
        self,
        image_label_sample
    ) -> tuple[np.ndarray, LabelTuple]:
        image = image_label_sample[0]
        label = image_label_sample[1]
        w, h = image.size(dim=1), image.size(dim=2)
        c_x, c_y = label
        max_translation_x = min(
            self.max_translation_x,
            c_x - self.eps,
            1 - c_x - self.eps
        )
        max_translation_y = min(
            self.max_translation_y,
            c_y - self.eps,
            1 - c_y - self.eps
        )

        max_translation_x = max(0, max_translation_x)
        max_translation_y = max(0, max_translation_y)

        if max_translation_x <= 0:
            x_translate = 0
        else:
            x_translate = int(
                np.random.uniform(-max_translation_x, max_translation_x) * w
            )
    
        if max_translation_y <= 0:
            y_translate = 0
        else:
            y_translate = int(
                np.random.uniform(-max_translation_y, max_translation_y) * h
            )
        image = tf.affine(
            image,
            translate=(x_translate, y_translate),
            angle=0,
            scale=1,
            shear=0
        )
        label = c_x + (x_translate / w), c_y + (y_translate / h)
        return image, label
