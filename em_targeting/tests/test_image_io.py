from em_targeting import image_io
from PIL import Image
import numpy as np
import pytest


# write a pytest fixture that creates a temporary directory and writes an image
# to it
@pytest.fixture
def image(tmpdir):
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    path = tmpdir / "image.tif"
    Image.fromarray(image).save(path)
    return path


# write a test that reads the image and checks that the shape is correct
def test_read_image(image):
    image = image_io.read_image(image)
    assert image.shape == (100, 100)
