from PIL import Image
import napari
import numpy as np
import argparse
from pathlib import Path
from em_targeting.image_io import read_image
from em_targeting.utils import (
    make_dirs,
    make_gridlines,
    polygon_to_mask,
    discretize_mask,
)


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
    make_dirs([mask_dir, overview_dir])
    path_mask = mask_dir / f"{path_im.stem}_mask.tif"
    path_overview = overview_dir / f"{path_im.stem}_overview.tif"

    # Get the grid
    image = read_image(path_im)
    mask = get_mask(image, nrtiles)

    # Save the mask and overview
    Image.fromarray(mask).save(path_mask)
    path_im.rename(path_overview)

    print(f"Saving {path_mask}, {path_overview}")


def get_mask(image, nrtiles):
    viewer = napari.Viewer()

    # Add image
    viewer.add_image(
        [image, image[::2, ::2], image[::4, ::4]], interpolation2d="linear"
    )
    # viewer.add_image(image, interpolation2d='linear')

    # Add grid lines
    image_shape = image.shape
    grid_lines, grid_spacing = make_gridlines(image_shape, nrtiles)
    viewer.add_shapes(
        grid_lines,
        shape_type="line",
        edge_color="red",
        edge_width=image_shape[0] // 500,
        name="Grid Lines",
    )

    # Add shapes layer for selection
    viewer.add_shapes(name="tissue")
    napari.run()

    # Aggregate the polygons into a mask
    mask = np.zeros(image_shape[:2], dtype=int)
    for polygon in viewer.layers["tissue"].data:
        mask += polygon_to_mask(polygon, image_shape)
    mask = mask > 0

    # Discretize the mask
    mask, mask_ds = discretize_mask(mask, grid_spacing)

    # Show the mask
    viewer = napari.Viewer()
    viewer.add_image(image, interpolation2d="linear")
    viewer.add_labels(mask)
    napari.run()

    return mask_ds


if __name__ == "__main__":
    main()