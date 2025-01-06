# list all files in image_dir
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from em_targeting.draw_polygon import display_overview, display_drawing, read_image
import time
import pandas as pd

image_dir = Path(
    "/Users/thomasathey/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/jellyfish-imaging/24_12_04_multi_res/images"
)

# list name of all folders in image_dir
folders = [
    f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))
]

data_res = []
data_time_select = []
data_area_px_select = []
data_image_path = []

for res in folders:
    if res == "4":
        continue

    path_dir = image_dir / res
    path_ims = [
        f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))
    ]
    path_ims = [path_dir / f for f in path_ims if f.endswith(".tif")]

    nrtiles = int(res) // 4

    for path_im in path_ims:
        image = read_image(path_im)

        assert image.shape[0] == image.shape[1], "Image is not square"

        tic = time.time()
        data = display_overview(image, nrtiles)
        time_select = time.time() - tic

        mask = display_drawing(image, nrtiles=nrtiles, layer_data=data)

        data_res.append(int(res))
        data_time_select.append(time_select)
        data_area_px_select.append(np.sum(mask) * (image.shape[0] / nrtiles) ** 2)
        data_image_path.append(path_im)


data = {
    "Resolution (nm)": data_res,
    "Time to Select (s)": data_time_select,
    "Area Selected (px)": data_area_px_select,
    "Image Path": data_image_path,
}
df = pd.DataFrame(data)
df.to_csv("manual_selection.csv", index=False)
