import os
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure, io
import h5py


def make_dirs(paths):
    """Make directories if they do not exist

    Args:
        paths (list): List of paths to create
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def convert_to_hdf5(path_im, path_hdf5):
    image = io.imread(path_im)

    with h5py.File(path_hdf5, "w") as f:
        f.create_dataset("image", data=image, chunks=(256,256))

    