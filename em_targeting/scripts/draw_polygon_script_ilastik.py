from PIL import Image
import napari
import numpy as np
import argparse
from pathlib import Path
from em_targeting.image_io import read_image
from em_targeting.utils import (
    make_dirs,
    convert_to_hdf5
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
import time
from scipy.ndimage import zoom
from skimage.transform import resize



ilastik_models = {
    "bell": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/24_12_04_multi_res/images/overview_bell_123.ilp",
    "tentacle": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/24_12_04_multi_res/images/overview_tentacle_123.ilp",
}
ilastik_models = {
    "bell": "/Users/thomasathey/Documents/shavit-lab/jellyfish/images/p123_ds_targeting_experiment/p123_ds2_bell.ilp",
    "tentacle": "/Users/thomasathey/Documents/shavit-lab/jellyfish/images/p123_ds_targeting_experiment/p123_ds2_tentacle.ilp",
}
ilastik_models = {
    "bell": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/25_01_22_ilastik_section/section_from_wafer_gray.ilp",
    "tentacle": "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/25_01_22_ilastik_section/section_from_wafer_gray.ilp",
}
zoom_factor = 1


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timer
        result = func(*args, **kwargs)  # Call the actual function
        end_time = time.time()  # End timer
        print(f"Function '{func.__name__}' took {end_time - start_time:.6f} seconds to run.")
        return result  # Return the original function's result
    return wrapper



def main():
    parser = argparse.ArgumentParser(
        description="Perform spline selection of ROI. Assumes the overview image is the minimum size square that contains the mosaic of square tiles."
    )
    parser.add_argument("--path_im", type=str, help="Image Path")
    parser.add_argument("--nrtilesh", type=int, help="Number of tiles in x")
    parser.add_argument("--nrtilesv", type=int, help="Number of tiles in y")
    parser.add_argument('--dontmove', action="store_true", help="don't move the overview image")
    parser.add_argument('--dontpredict', action="store_true", help="don't run ilastik")
    
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

    path_im_hdf5 = path_im.with_suffix(".hdf5")
    convert_to_hdf5(path_im, path_im_hdf5, zoom_factor = zoom_factor)

    if not args.dontpredict:
        init_mask = get_ilastik_mask(path_im_hdf5, image.shape)
    else:
        init_mask = None
        
    mask = get_mask(image, nrtilesh, nrtilesv, init_mask=init_mask)

    # Save the mask and overview
    Image.fromarray(mask).save(path_mask)
    if not args.dontmove:
        path_im.rename(path_overview)

    print(f"Saving {path_mask}, {path_overview}")

@timing_decorator
def get_ilastik_mask(path_im, final_shape):
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

    seg_pred_postprocess_bell, _ = postprocess_mask_bell(seg_pred_bell, major_axis_length=100, min_size=1250)
    seg_pred_postprocess_tent, _ = postprocess_mask_tentacles(seg_pred_tent, min_size=1250)
    seg_pred_postprocess = np.logical_or(
        seg_pred_postprocess_bell, seg_pred_postprocess_tent
    )

    seg_pred_postprocess = resize(seg_pred_postprocess, final_shape, order=0)

    return seg_pred_postprocess

@timing_decorator
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
    viewer.add_shapes(name="ADD")
    viewer.add_shapes(name="REMOVE")
    napari.run()

    # Aggregate the polygons into a mask
    if init_mask is not None:
        mask = polygons_to_mask(viewer.layers["REMOVE"].data, image_shape)
        init_mask = np.logical_and(np.logical_not(mask), init_mask)

    mask = polygons_to_mask(viewer.layers["ADD"].data, image_shape)
    if init_mask is not None:
        mask = np.logical_or(mask, init_mask)

    

    # Discretize the mask
    mask, mask_ds = discretize_mask(mask, nrtilesh, nrtilesv)

    assert mask_ds.shape == (nrtilesv, nrtilesh)

    # Show the mask
    viewer = napari.Viewer()
    viewer.add_image(image, interpolation2d="linear")
    viewer.add_labels(mask)
    #viewer.add_image(mask_ds, scale=(image.shape[0] / mask_ds.shape[0], image.shape[1] / mask_ds.shape[1]))
    napari.run()

    return mask_ds


if __name__ == "__main__":
    main()
