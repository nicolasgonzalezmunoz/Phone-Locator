"""Implement utility functions for other modules."""
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import TypeAlias, Optional
from modules.data import PhoneDataset

Location: TypeAlias = tuple[float, float]
LocationsDict: TypeAlias = dict[os.PathLike, Location]
ImagePathsList: TypeAlias = list[os.PathLike]


def get_data(
    folder: os.PathLike
) -> tuple[ImagePathsList, LocationsDict]:
    """
    Get a list with paths to images and a dict with their labels.

    Parameters
    ----------
    folder: os.PathLike
        A path to the folder where the images and the labels.txt file are.

    Return
    ------
    image_paths: list[os.PathLike]
        A list with the paths to the image files.
    locations: dict[float, float]
        A dict where the keys are paths to an image and its value is a tuple
        with its label.
    """
    labels = pd.read_csv(
        os.path.join(folder, 'labels.txt'),
        header=None,
        delimiter=' ',
        names=['filename', 'x', 'y']
    )
    image_paths = []
    locations = dict()
    for row in labels.itertuples():
        image_paths.append(os.path.join(folder, row.filename))
        locations[row.filename] = (row.x, row.y)

    return image_paths, locations


def visualize_augmentations(
    dataset: PhoneDataset,
    locations: LocationsDict,
    idx: int = 0,
    samples: int = 10,
    cols: int = 5,
    random_img: bool = False
) -> None:
    """
    Visualize dataset images after image augmentation.

    Parameters
    ----------
    idx: int, default=0
        idx of the image to plot. Ignored if random_img is True.
    samples: int, default=10
        Number of samples to plot.
    cols: int, default=5
        Number of columns of the subplot instance.
    random_img: bool, default=False
        Whether to take random samples of images or not. If True, then
        'idx' is ignored.

    Return
    ------
    None
    """
    
    dataset = copy.deepcopy(dataset)
    rows = samples // cols
    size = len(dataset)
    
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1, size)
        image, _ = dataset[idx]
        image = np.swapaxes(image, 2, 1)
        image = np.swapaxes(image, 2, 0)
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(f'file={list(locations.keys())[idx]}')
    plt.tight_layout(pad=1)
    plt.show()


def get_layer_output_size(
    h_in: int,
    w_in: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> tuple[int, int]:
    """
    Compute the dimensions of a Conv2d or MaxPool2d output.

    Parameters
    ----------
    h_in, w_in: int
        The height and width of the image.
    kernel_size, stride, padding, dilation: int
        Parameters to be passed to the layer.

    Return
    ------
    h_out, w_out: int
        Dimensions of the layer's output.
    """
    h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
    h_out = int(h_out)
    w_out = (w_in + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
    w_out = int(w_out)
    return h_out, w_out


def get_max_conv_depth(
    h_in: int,
    w_in: int,
    conv_init_kernel_size: int = 5,
    conv_kernel_size: int = 3,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    pool_kernel_size: int = 3,
    pool_stride: int = 1,
    pool_padding: int = 0,
    pool_dilation: int = 1
) -> int:
    """
    Get the maximum possible amount of convolutional layers with the given parameters.

    Parameters
    ----------
    h_in, w_in: int
        The height and width of the images.
    conv_init_kernel_size: int
        Kernel size of the first Conv2d layer.
    conv_kernel_size, conv_stride, conv_padding, conv_dilation: int
        Parameters passed to the Conv2d layers.
    pool_kernel_size, pool_stride, pool_padding, pool_dilation: int
        Parameters passed to the MaxPool2d layers.

    Return
    ------
    max_n_layers: int
        The maximum number of convolutional layers with the current configuration.
    """
    max_n_layers = 0
    h_out, w_out = get_layer_output_size(
        h_in,
        w_in,
        conv_init_kernel_size,
        conv_stride,
        conv_padding,
        conv_dilation
    )
    if h_out > 0 and w_out > 0:
        max_n_layers += 1
    while h_out > 0 and w_out > 0:
        h_out, w_out = get_layer_output_size(
            h_out,
            w_out,
            pool_kernel_size,
            pool_stride,
            pool_padding,
            pool_dilation
        )
        h_out, w_out = get_layer_output_size(
            h_out,
            w_out,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            conv_dilation
        )
        if h_out > 0 and w_out > 0:
            max_n_layers += 1
    return max_n_layers


def get_block_output_size(h_in: int, w_in: int) -> tuple[int, int]:
    """
    Compute the dimensions of a Conv2d or MaxPool2d output.

    Parameters
    ----------
    h_in, w_in: int
        The height and width of the image.

    Return
    ------
    h_out, w_out: int
        Dimensions of the layer's output.
    """
    #kernel_size = 3
    pool_kernel_size = 2
    pool_stride = pool_kernel_size

    #MaxPool2d output
    h_out = (h_in - (pool_kernel_size - 1) - 1)/pool_stride + 1
    h_out = int(h_out)

    w_out = (w_in - (pool_kernel_size - 1) - 1)/pool_stride + 1
    w_out = int(w_out)
    return h_out, w_out


def train_test_split(
    image_paths: ImagePathsList,
    locations: LocationsDict,
    test_size: float = 0.1,
    random_state: Optional[int] = None
) -> tuple[ImagePathsList, ImagePathsList, LocationsDict, LocationsDict]:
    """
    Split data into train and test set.

    Parameters
    ----------
    image_paths: list[os.PathLike]
        A list with paths to images.
    locations: dict[os.PathLike, tuple[float, float]]
        A dict with image paths as keys and labels as values.
    test_size: float, default=0.1
        Size of the test set, relative to the total size of the dataset.
    random_state: int
        Seed of the random generator.

    Return
    ------
    train_image_paths, test_image_paths: list[os.PathLike]
        Lists of paths to images from the train and test set, respectively.
    train_locations, test_locations: dict[os.PathLike, tuple[float, float]]
        Dictionary with image paths as keys and labels as values.
    """
    size = len(image_paths)
    train_size = int((1 - test_size) * size)
    idxs = np.random.default_rng(random_state).permutation(size)
    train_idxs = idxs[:train_size]
    test_idxs = idxs[train_size:]
    train_image_paths = list(np.array(image_paths)[train_idxs])
    test_image_paths = list(np.array(image_paths)[test_idxs])
    train_locations = {
        os.path.basename(image_path): locations[os.path.basename(image_path)] for image_path in train_image_paths
    }
    test_locations = {
        os.path.basename(image_path): locations[os.path.basename(image_path)] for image_path in test_image_paths
    }
    return train_image_paths, test_image_paths, train_locations, test_locations
