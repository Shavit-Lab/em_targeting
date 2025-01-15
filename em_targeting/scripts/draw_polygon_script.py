import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import napari
import numpy as np
import argparse
from pathlib import Path
from skimage import measure
import os


def main():
    parser = argparse.ArgumentParser(description="Perform spline selection of ROI")
    parser.add_argument("--path_im", type=str, help="Image Path")
    parser.add_argument("--nrtiles", type=int, help="Number of tiles")
    args = parser.parse_args()
    path_im = args.path_im
    nrtiles = args.nrtiles

    path_im = Path(path_im)

    # Make various paths
    mask_dir = path_im.parent / "masks"
    overview_dir = path_im.parent / "overviews"
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(overview_dir):
        os.makedirs(overview_dir)
    path_mask = mask_dir / f"{path_im.stem}_mask.tif"
    path_overview = overview_dir / f"{path_im.stem}_overview.tif"

    # Get the grid
    image = read_image(path_im)
    mask = get_mask(image, nrtiles)

    # Save the mask and overview
    Image.fromarray(mask).save(path_mask)
    path_im.rename(path_overview)

    print(f"Saving {path_mask}, {path_overview}")


def read_image(path_im):
    image = Image.open(path_im)
    image = np.array(image)
    return image


def get_mask(image, nrtiles):
    viewer = napari.Viewer()
    viewer.add_image(
        [image, image[::2, ::2], image[::4, ::4]], interpolation2d="linear"
    )
    # viewer.add_image(image, interpolation2d='linear')

    # add grid lines
    image_shape = image.shape
    grid_spacing = int(np.ceil(image_shape[0] / nrtiles))

    horizontal_lines = [
        [[y, 0], [y, image_shape[1]]] for y in range(0, image_shape[0], grid_spacing)
    ]

    vertical_lines = [
        [[0, x], [image_shape[1], x]] for x in range(0, image_shape[1], grid_spacing)
    ]

    grid_lines = horizontal_lines + vertical_lines

    edge_width = np.amax([image_shape[0] // 500, 1])

    viewer.add_shapes(
        grid_lines,
        shape_type="line",
        edge_color="red",
        edge_width=edge_width,
        name="Grid Lines",
    )

    # add shapes layer for selection
    viewer.add_shapes(name="tissue")
    napari.run()

    mask = np.zeros(image.shape[:2], dtype=int)
    for polygon in viewer.layers["tissue"].data:
        polygon = [(e[0], e[1]) for e in polygon]
        img = Image.new("L", image.shape[:2], 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask += np.array(img).T
    mask = mask > 0

    print(mask.shape)
    print(grid_spacing)
    mask_ds = measure.block_reduce(mask, block_size=grid_spacing, func=np.max)
    mask = np.repeat(mask_ds, grid_spacing, axis=0)
    mask = np.repeat(mask, grid_spacing, axis=1)

    viewer = napari.Viewer()
    viewer.add_image(image, interpolation2d="linear")
    viewer.add_labels(mask)
    napari.run()

    print(mask_ds.shape)

    return mask_ds


if __name__ == "__main__":
    main()
