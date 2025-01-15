from em_targeting import draw_polygon
import numpy as np


# write a test that checks that the polygon_to_mask function returns the correct mask
def test_polygons_to_mask():
    polygons = [[[0, 0], [0, 9], [9, 9], [9, 0]], [[5, 5], [5, 10], [10, 10], [10, 5]]]
    image_shape = (100, 100)
    mask = draw_polygon.polygons_to_mask(polygons, image_shape)
    assert mask.shape == image_shape
    assert mask.sum() == 121 - 10


# write a test that checks that the make_gridlines function returns the correct gridlines
def test_make_gridlines():
    image_shape = (100, 100)
    grid_lines, grid_spacing = draw_polygon.make_gridlines(image_shape, 10, 10)
    assert len(grid_lines) == 20
    assert grid_spacing == 10

    grid_lines, grid_spacing = draw_polygon.make_gridlines(image_shape, 10, 5)
    assert len(grid_lines) == 20
    assert grid_spacing == 10

    grid_lines, grid_spacing = draw_polygon.make_gridlines(image_shape, 5, 10)
    assert len(grid_lines) == 20
    assert grid_spacing == 10

# write a test that checks that the discretize_mask function returns the correct mask
def test_discretize_mask():
    mask = np.zeros((100, 100))
    mask[-1, -1] = 1
    grid_spacing = 10

    mask, mask_ds = draw_polygon.discretize_mask(mask, grid_spacing)

    assert mask_ds.shape == (10, 10)
    assert mask_ds[-1, -1] == 1
    assert np.sum(mask_ds) == 1

    assert mask.shape == (100, 100)
    assert mask.sum() == 100
