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
from em_targeting.mask_processing import (
    postprocess_mask_bell,
    postprocess_mask_tentacles,
)
from em_targeting.apply_ilastik import apply_ilastik

# ilastik_models = {
#     "bell": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/24_12_04_multi_res/images/overview_bell_123.ilp",
#     "tentacle": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/24_12_04_multi_res/images/overview_tentacle_123.ilp",
# }
ilastik_models = {
    "bell": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/25_01_22_ilastik_section/section_from_wafer.ilp",
    "tentacle": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/25_01_22_ilastik_section/section_from_wafer.ilp",
}


def main():
    parser = argparse.ArgumentParser(
        description="Perform spline selection of ROI. Assumes the overview image is the minimum size square that contains the mosaic of square tiles."
    )
    parser.add_argument("--path_im", type=str, help="Image Path")
    parser.add_argument("--nrtilesh", type=int, help="Number of tiles in x")
    parser.add_argument("--nrtilesv", type=int, help="Number of tiles in y")
    args = parser.parse_args()
    path_im = args.path_im
    nrtilesh = args.nrtilesh
    nrtilesv = args.nrtilesv

    path_im = Path(path_im)

    # Make various paths
    mask_dir = path_im.parent / "masks"
    overview_dir = path_im.parent / "overviews"
    make_dirs([mask_dir, overview_dir])
    path_mask = mask_dir / f"{path_im.stem}_mask.tif"
    path_overview = overview_dir / f"{path_im.stem}_overview.tif"

    # Get the grid
    image = read_image(path_im)

    init_mask = get_ilastik_mask(path_im)
    mask = get_mask(image, nrtilesh, nrtilesv, init_mask=init_mask)

    # Save the mask and overview
    Image.fromarray(mask).save(path_mask)
    path_im.rename(path_overview)

    print(f"Saving {path_mask}, {path_overview}")


def get_ilastik_mask(path_im):
    ilastik_path = apply_ilastik(
        ilastik_models["bell"],
        path_im,
        export_source="Simple Segmentation",
        suffix="bell",
    )
    seg_pred_bell = read_image(ilastik_path)

    ilastik_path = apply_ilastik(
        ilastik_models["tentacle"],
        path_im,
        export_source="Simple Segmentation",
        suffix="tentacle",
    )
    seg_pred_tent = read_image(ilastik_path)

    seg_pred_postprocess_bell, _ = postprocess_mask_bell(seg_pred_bell)
    seg_pred_postprocess_tent, _ = postprocess_mask_tentacles(seg_pred_tent)
    seg_pred_postprocess = np.logical_or(
        seg_pred_postprocess_bell, seg_pred_postprocess_tent
    )

    return seg_pred_postprocess


def get_mask(image, nrtilesh, nrtilesv, init_mask=None):
    viewer = napari.Viewer()

    # Add image
    viewer.add_image(
        [image, image[::2, ::2], image[::4, ::4]], interpolation2d="linear"
    )
    if init_mask is not None:
        viewer.add_labels(init_mask, name="ilastik prediction")
    # viewer.add_image(image, interpolation2d='linear')

    # Make grid lines
    image_shape = image.shape
    grid_lines = make_gridlines(image_shape, nrtilesh, nrtilesv)
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
    if init_mask is not None:
        mask = np.logical_or(mask, init_mask)

    # Discretize the mask
    mask, mask_ds = discretize_mask(mask, nrtilesh, nrtilesv)

    assert mask_ds.shape == (nrtilesv, nrtilesh)

    # Show the mask
    viewer = napari.Viewer()
    viewer.add_image(image, interpolation2d="linear")
    viewer.add_labels(mask)
    napari.run()

    return mask_ds


if __name__ == "__main__":
    main()
