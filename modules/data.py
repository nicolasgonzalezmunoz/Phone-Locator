"""Implement a class for building pytorch Datasets for phone location."""
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
from typing import Optional, TypeAlias, Any


Location: TypeAlias = tuple[float, float]
LocationsDict: TypeAlias = dict[os.PathLike, Location]
ImagePathsList: TypeAlias = list[os.PathLike]


class PhoneDataset(Dataset):
    """
    Implement a Dataset to get images from a file path with its location.

    Parameters
    ----------
    
    """
    def __init__(
        self,
        image_paths: ImagePathsList,
        locations: LocationsDict,
        position_aware_transform: Optional[Any] = None,
        position_unaware_transform: Optional[Any] = None
    ):
        self.image_paths = image_paths
        self.locations = locations
        self.position_aware_transform = position_aware_transform
        self.position_unaware_transform = position_unaware_transform


    def __len__(self) -> int:
        return len(self.image_paths)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        filename = os.path.basename(image_path)
        location = self.locations[filename]
        image = cv2.imread(image_path) / 255.0
        image = tf.to_tensor(image.astype('float32'))
        location = [location[1], location[0]]
        if self.position_aware_transform is not None:
            image, location = self.position_aware_transform((image, location))
        if self.position_unaware_transform is not None:
            image = self.position_unaware_transform(image)
        return image, torch.Tensor(location)
