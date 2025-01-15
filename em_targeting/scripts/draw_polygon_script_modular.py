from PIL import Image
import napari
import numpy as np
import argparse
from pathlib import Path
from em_targeting.image_io import read_image
from em_targeting.utils import (
    make_dirs,
)
from em_targeting.draw_polygon import (
    make_gridlines,
    polygons_to_mask,
    discretize_mask,
)


def main():
    parser = argparse.ArgumentParser(description="Perform spline selection of ROI. Assumes the overview image is the minimum size square that contains the mosaic of square tiles.")
    parser.add_argument("--path_im", type=str, help="Image Path")
    parser.add_argument("--nrtilesx", type=int, help="Number of tiles in x")
    parser.add_argument("--nrtilesy", type=int, help="Number of tiles in y")
    args = parser.parse_args()
    path_im = args.path_im
    nrtilesx = args.nrtilesx
    nrtilesy = args.nrtilesy

    path_im = Path(path_im)

    # Make various paths
    mask_dir = path_im.parent / "masks"
    overview_dir = path_im.parent / "overviews"
    make_dirs([mask_dir, overview_dir])
    path_mask = mask_dir / f"{path_im.stem}_mask.tif"
    path_overview = overview_dir / f"{path_im.stem}_overview.tif"

    # Get the grid
    image = read_image(path_im)
    mask = get_mask(image, nrtilesx, nrtilesy)

    # Save the mask and overview
    Image.fromarray(mask).save(path_mask)
    path_im.rename(path_overview)

    print(f"Saving {path_mask}, {path_overview}")


def get_mask(image, nrtilesx, nrtilesy):
    viewer = napari.Viewer()

    # Add image
    viewer.add_image(
        [image, image[::2, ::2], image[::4, ::4]], interpolation2d="linear"
    )
    # viewer.add_image(image, interpolation2d='linear')

    # Make grid lines
    image_shape = image.shape
    grid_lines, grid_spacing = make_gridlines(image_shape, nrtilesx, nrtilesy)
    edge_width = np.amax([image_shape[0] // 500, 1])

    # Add grid lines and shapes layer
    viewer.add_shapes(
        grid_lines,
        shape_type="line",
        edge_color="red",
        edge_width=edge_width,
        name="Grid Lines",
    )
    viewer.add_shapes(name="tissue")
    napari.run()

    # Aggregate the polygons into a mask
    mask = polygons_to_mask(viewer.layers["tissue"].data, image_shape)

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
