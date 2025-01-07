import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import napari
import numpy as np
import argparse
from pathlib import Path
from skimage import measure
import os
import h5py


def read_image(path_im):
    path_im = Path(path_im)
    if path_im.suffix == ".h5":
        with h5py.File(path_im) as f:
            image = f["exported_data"][:]
            image = np.squeeze(image)
            image = image == 2
    else:
        image = Image.open(path_im)
        image = np.array(image)
    return image


def polygons_to_mask(image, layer_data):
    image_shape = image.shape
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
    grid_spacing = image_shape[0] // nrtiles

    horizontal_lines = [
        [[y, 0], [y, image_shape[1]]] for y in range(0, image_shape[0], grid_spacing)
    ]

    vertical_lines = [
        [[0, x], [image_shape[1], x]] for x in range(0, image_shape[1], grid_spacing)
    ]

    grid_lines = horizontal_lines + vertical_lines

    viewer.add_shapes(
        grid_lines,
        shape_type="line",
        edge_color="red",
        edge_width=image_shape[0] // 500,
        name="Grid Lines",
    )

    return viewer


def mask_to_tiles(mask, ds_factor):
    mask_ds = measure.block_reduce(mask, block_size=ds_factor, func=np.max)
    mask = np.repeat(mask_ds, ds_factor, axis=0)
    mask = np.repeat(mask, ds_factor, axis=1)

    return mask


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
