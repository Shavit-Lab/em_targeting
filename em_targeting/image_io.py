import numpy as np
from PIL import Image
from pathlib import Path
import h5py


def read_image(path_im):
    """Read an image from a path

    Args:
        path_im (str): Path to the image

    Returns:
        np.array: Image as a numpy array
    """
    path_im = Path(path_im)
    if path_im.suffix == ".h5":
        with h5py.File(path_im) as f:
            image = f["exported_data"][:]
            image = np.squeeze(image)
            unq = np.unique(image)
            if 2 in unq:  #
                image = image == 2
            else:
                print(f"Warning, label 2 not found in {path_im}, using second value")
                image = image == unq[1]
    else:
        image = Image.open(path_im)
        image = np.array(image)

    return image
