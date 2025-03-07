import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import napari
import numpy as np
import argparse
from pathlib import Path
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
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
    """Make gridlines for an image. Assumptions:
    - Image is square
    - Gridlines are equally spaced
    - Image is smallest bounding square for the gridlines

    Args:
        image_shape (tuple): Shape of the image. Must be square/
        nrtilesh (int): Number of tiles along horizontal dimension
        nrtilesv (int): Number of tiles along vertical dimension

    Returns:
        list: List of gridlines, for use in napari
        int: Grid spacing
    """
    assert image_shape[0] == image_shape[1], "Overview image must be square"

    grid_spacing, bbox = _ntiles_to_spacing(nrtilesh, nrtilesv, image_shape)
    hmin, hmax, vmin, vmax = bbox

    horizontal_lines = [
        [[v, hmin], [v, hmax]] for v in np.arange(vmin, vmax, grid_spacing)
    ]
    if len(horizontal_lines) < nrtilesv + 1:
        horizontal_lines += [[[vmax, hmin], [vmax, hmax]]]

    vertical_lines = [
        [[vmin, h], [vmax, h]] for h in np.arange(hmin, hmax, grid_spacing)
    ]
    if len(vertical_lines) < nrtilesh + 1:
        vertical_lines += [[[vmin, hmax], [vmax, hmax]]]

    grid_lines = horizontal_lines + vertical_lines

    return grid_lines


def _ntiles_to_spacing(nrtilesh, nrtilesv, image_shape):
    """Given an image shape and number of tiles, calculate the grid spacing and bounding box.
        Note that returned value may not be a whole number. Assumptions:
        - Image is square
        - Gridlines are equally spaced
        - Image is smallest bounding square for the gridlines

    Args:
        nrtilesh (int): number of tiles along horizontal dimension
        nrtilesv (int): number of tiles along vertical dimension
        image_shape (tuple): shape of the image

    Returns:
        float: width of a tile in pixels
        tuple: bounding box of the gridlines
    """
    assert image_shape[0] == image_shape[1], "Overview image must be square"

    if nrtilesh >= nrtilesv:
        grid_spacing = image_shape[1] / nrtilesh
        hmin = 0
        hmax = image_shape[1]

        height = grid_spacing * nrtilesv
        vmin = (image_shape[0] - height) / 2
        vmax = vmin + height
    else:
        grid_spacing = image_shape[0] / nrtilesv
        vmin = 0
        vmax = image_shape[0]

        width = grid_spacing * nrtilesh
        hmin = (image_shape[1] - width) / 2
        hmax = hmin + width

    bbox = (hmin, hmax, vmin, vmax)
    return grid_spacing, bbox


def discretize_mask(mask, nrtilesh, nrtilesv):
    """Discritize a mask using a max pooling operation

    Args:
        mask (np.array): Binary mask
        nrtilesh (int): Number of tiles along horizontal dimension
        nrtilesv (int): Number of tiles along vertical dimension

    Returns:
        np.array: Mask after discretization with same shape as input
        np.array: Mask after discretization with shape reduced by grid_spacing
    """
    image_shape = mask.shape
    grid_spacing, bbox = _ntiles_to_spacing(nrtilesh, nrtilesv, image_shape)
    grid_spacing_ceil = int(np.ceil(grid_spacing))
    hmin, hmax, vmin, vmax = bbox

    # interp so there are grid_spacing_ceil points per tile
    interp = RegularGridInterpolator(
        (np.arange(image_shape[0]), np.arange(image_shape[1])),
        mask,
        method="nearest",
        bounds_error=False,
        fill_value=None,
    )

    spacing = grid_spacing / grid_spacing_ceil

    v_upsample = np.arange(vmin, vmax, spacing)
    if len(v_upsample) % grid_spacing_ceil == 1:
        v_upsample = v_upsample[:-1]

    h_upsample = np.arange(hmin, hmax, spacing)
    if len(h_upsample) % grid_spacing_ceil == 1:
        h_upsample = h_upsample[:-1]

    v, h = np.meshgrid(v_upsample, h_upsample, indexing="ij")
    mask_interp = interp((v, h))

    assert mask_interp.shape == (
        grid_spacing_ceil * nrtilesv,
        grid_spacing_ceil * nrtilesh,
    )

    # downsample
    mask_ds = measure.block_reduce(
        mask_interp, block_size=grid_spacing_ceil, func=np.max
    )

    assert mask_ds.shape == (nrtilesv, nrtilesh)

    # # resample to original size
    v_downsample = np.arange(vmin, vmax, grid_spacing) + grid_spacing / 2
    if len(v_downsample) % nrtilesv == 1:
        v_downsample = v_downsample[:-1]
    h_downsample = np.arange(hmin, hmax, grid_spacing) + grid_spacing / 2
    if len(h_downsample) % nrtilesh == 1:
        h_downsample = h_downsample[:-1]

    interp_inv = RegularGridInterpolator(
        (v_downsample, h_downsample),
        mask_ds,
        method="nearest",
        fill_value=None,
        bounds_error=False,
    )

    v_sample_og = np.arange(image_shape[0])
    h_sample_og = np.arange(image_shape[1])
    v, h = np.meshgrid(v_sample_og, h_sample_og, indexing="ij")
    mask = interp_inv((v, h))

    # mask the edges
    mask[:, : int(np.round(hmin))] = 0
    mask[:, int(np.round(hmax)) :] = 0
    mask[: int(np.round(vmin)), :] = 0
    mask[int(np.round(vmax)) :, :] = 0

    mask = mask > 0

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
