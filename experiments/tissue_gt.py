# list all files in image_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from em_targeting.draw_polygon import napari_seg, polygons_to_mask, read_image
import time
import pandas as pd
import napari
from skimage import io
from tqdm import tqdm

image_dir = Path(
    "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/24_12_04_multi_res/images"
)

# list name of all folders in image_dir
folders = [
    f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))
]

for res in tqdm(folders, desc="Resolution"):
    if res == "4":
        continue

    path_dir = image_dir / res
    path_ims = [
        f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))
    ]
    path_ims = [path_dir / f for f in path_ims if f.endswith(".tif") and "_gt" not in f]

    nrtiles = int(res) // 4

    for path_im in tqdm(path_ims, desc="Image", leave=False):

        path_out = path_im.parent / (path_im.stem + "_tissue_gt.tif")

        if not path_out.exists():
            image = read_image(path_im)

            data = napari_seg(image)
            mask = polygons_to_mask(image, data)

            print(f"{mask.dtype} {mask.shape}")

            io.imsave(path_out, mask.astype(np.uint8))
