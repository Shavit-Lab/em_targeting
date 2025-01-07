from em_targeting import utils
import pytest
import numpy as np


# write a pytest fixture that creates a temporary directory and makes two paths of subfolders
# to test the make_dirs function
@pytest.fixture
def paths(tmpdir):
    paths = [tmpdir / "dir1", tmpdir / "dir2"]
    return paths


# write a test that checks that the make_dirs function creates the directories
def test_make_dirs(paths):
    utils.make_dirs(paths)
    for path in paths:
        assert path.exists()


# write a test that checks that the make_gridlines function returns the correct gridlines
def test_make_gridlines():
    image_shape = (100, 100)
    nrtiles = 10
    grid_lines, grid_spacing = utils.make_gridlines(image_shape, nrtiles)
    assert len(grid_lines) == 20
    assert grid_spacing == 10


# write a test that checks that the polygon_to_mask function returns the correct mask
def test_polygon_to_mask():
    polygon = [[0, 0], [0, 9], [9, 9], [9, 0]]
    image_shape = (100, 100)
    mask = utils.polygon_to_mask(polygon, image_shape)
    assert mask.shape == image_shape
    assert mask.sum() == 100


# write a test that checks that the discretize_mask function returns the correct mask
def test_discretize_mask():
    mask = np.zeros((100, 100))
    mask[-1, -1] = 1
    grid_spacing = 10

    mask, mask_ds = utils.discretize_mask(mask, grid_spacing)

    assert mask_ds.shape == (10, 10)
    assert mask_ds[-1, -1] == 1
    assert np.sum(mask_ds) == 1

    assert mask.shape == (100, 100)
    assert mask.sum() == 100
