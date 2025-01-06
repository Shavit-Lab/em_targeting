import numpy as np
import napari
from skimage import data  # For a sample image


def test():
    # Load a sample image (use your own image if needed)
    image = data.camera()

    # Create a napari viewer to display the original image
    viewer = napari.Viewer()
    viewer.add_image(image, name="Original Image")

    # Block execution until the user closes the window
    napari.run()

    # Flip the image upside down
    flipped_image = np.flipud(image)

    # Create another napari viewer to display the flipped image
    viewer = napari.Viewer()
    viewer.add_image(flipped_image, name="Upside Down Image")

    # Block execution again until the user closes the second window
    napari.run()
