import numpy as np
from PIL import Image
from pathlib import Path


def read_image(path_im):
    """Read an image from a path

    Args:
        path_im (str): Path to the image

    Returns:
        np.array: Image as a numpy array
    """
    path_im = Path(path_im)

    image = Image.open(path_im)
    image = np.array(image)
    return image
