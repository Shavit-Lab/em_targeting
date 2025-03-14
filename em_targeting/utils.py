import os
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure, io
import h5py
from scipy.ndimage import zoom


def make_dirs(paths):
    """Make directories if they do not exist

    Args:
        paths (list): List of paths to create
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def convert_to_hdf5(path_im, path_hdf5, zoom_factor = 1):
    image = io.imread(path_im)

    if zoom_factor != 1:
        image = zoom(image, zoom_factor, order=1)


    with h5py.File(path_hdf5, "w") as f:
        chunks = tuple([int(np.amin([256, s])) for s in image.shape])
        f.create_dataset("image", data=image, chunks=chunks)

    