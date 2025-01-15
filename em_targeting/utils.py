import os
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure


def make_dirs(paths):
    """Make directories if they do not exist

    Args:
        paths (list): List of paths to create
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
