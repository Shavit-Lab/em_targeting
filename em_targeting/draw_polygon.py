import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import napari
import numpy as np
import argparse
from pathlib import Path
from skimage import measure
import os
import h5py


def polygons_to_mask(layer_data, image_shape):
    """Convert polygons to a binary mask

    Args:
        layer_data (list): Napari shape layer data that contains polygons.
        image (tuple): Base image shape

    Returns:
        np.array: Binary mask
    """
    mask = np.zeros(image_shape[:2], dtype=int)
    for polygon in layer_data:
        polygon = [(e[0], e[1]) for e in polygon]
        img = Image.new("L", image_shape[:2], 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask += np.array(img).T
    mask = mask > 0
    return mask


def napari_seg(image, nrtiles=None):
    viewer = napari.Viewer()
    viewer.add_image(
        [image, image[::2, ::2], image[::4, ::4]], interpolation2d="linear"
    )

    if nrtiles:
        viewer = display_grid(viewer, image, nrtiles)

    # add shapes layer for selection
    layer_draw = viewer.add_shapes(name="tissue")
    napari.run()

    return layer_draw.data


def display_grid(viewer, image, nrtiles):
    # add grid lines
    image_shape = image.shape
    grid_lines = make_gridlines(image_shape, nrtiles)

    viewer.add_shapes(
        grid_lines,
        shape_type="line",
        edge_color="red",
        edge_width=image_shape[0] // 500,
        name="Grid Lines",
    )

    return viewer


def make_gridlines(image_shape, nrtilesh, nrtilesv):
    """Make gridlines for an image. Assumes overview image is the smallest square that contains the mosaic of square tiles.

    Args:
        image_shape (tuple): Shape of the image
        nrtilesh (int): Number of tiles along horizontal dimension
        nrtilesv (int): Number of tiles along vertical dimension

    Returns:
        list: List of gridlines, for use in napari
        int: Grid spacing
    """
    assert image_shape[0] == image_shape[1], "Overview image must be square"

    if nrtilesh >= nrtilesv:
        grid_spacing = int(np.ceil(image_shape[1] / nrtilesh))
        hmin = 0
        hmax = image_shape[1]

        height = grid_spacing * nrtilesv
        vmin = (image_shape[0] - height) // 2
        vmax = vmin + height
    else:
        grid_spacing = int(np.ceil(image_shape[0] / nrtilesv))
        vmin = 0
        vmax = image_shape[0]

        width = grid_spacing * nrtilesh
        hmin = (image_shape[1] - width) // 2
        hmax = hmin + width


    horizontal_lines = [
        [[v, hmin], [v, hmax]] for v in range(vmin, vmax, grid_spacing)
    ]

    vertical_lines = [
        [[vmin, h], [vmax, h]] for h in range(hmin, hmax, grid_spacing)
    ]

    grid_lines = horizontal_lines + vertical_lines

    return grid_lines, grid_spacing


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


def display_drawing(image, nrtiles, layer_data):
    image_shape = image.shape
    grid_spacing = image_shape[0] // nrtiles

    mask = polygons_to_mask(image_shape, layer_data)

    mask_ds = mask_to_tiles(mask, grid_spacing)

    viewer_show = napari.Viewer()
    viewer_show.add_image(image, interpolation2d="linear")
    viewer_show.add_labels(mask)
    napari.run()

    return mask_ds
