import os
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure


def make_dirs(paths):
    """Make directories if they do not exist

    Args:
        paths (list): List of paths to create
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def make_gridlines(image_shape, nrtiles):
    """Make gridlines for an image

    Args:
        image_shape (tuple): Shape of the image
        nrtiles (int): Number of tiles along each dimension

    Returns:
        list: List of gridlines, for use in napari
        int: Grid spacing
    """
    grid_spacing = image_shape[0] // nrtiles

    horizontal_lines = [
        [[y, 0], [y, image_shape[1]]] for y in range(0, image_shape[0], grid_spacing)
    ]

    vertical_lines = [
        [[0, x], [image_shape[1], x]] for x in range(0, image_shape[1], grid_spacing)
    ]

    grid_lines = horizontal_lines + vertical_lines

    return grid_lines, grid_spacing


def polygon_to_mask(polygon, image_shape):
    """Convert a napari polygon to a mask

    Args:
        polygon (list): List of points in the polygon, from napari
        image_shape (tuple): Shape of the image

    Returns:
        np.array: Mask of the polygon
    """
    polygon = [(e[0], e[1]) for e in polygon]
    img = Image.new("L", image_shape[:2], 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img).T
    return mask


def discretize_mask(mask, grid_spacing):
    """Discritize a mask using a max pooling operation

    Args:
        mask (np.array): Binary mask
        grid_spacing (int): Number of pixels to pool over

    Returns:
        np.array: Mask after discretization with same shape as input
        np.array: Mask after discretization with shape reduced by grid_spacing
    """
    mask_ds = measure.block_reduce(mask, block_size=grid_spacing, func=np.max)
    mask = np.repeat(mask_ds, grid_spacing, axis=0)
    mask = np.repeat(mask, grid_spacing, axis=1)

    return mask, mask_ds
