from em_targeting import draw_polygon
import numpy as np


# write a test that checks that the polygon_to_mask function returns the correct mask
def test_polygons_to_mask():

    image_shape = (10, 10)
    polygons = [
        [[0, 0], [0, 1], [1, 1], [1, 0]],
        [[5, 5], [5, 6], [6, 6], [6, 5]],
        [[11, 11], [11, 12], [12, 12]],
    ]
    mask = draw_polygon.polygons_to_mask(polygons, image_shape)
    assert mask.shape == image_shape
    assert mask.sum() == 8


# write a test that checks that the make_gridlines function returns the correct gridlines
def test_make_gridlines():
    # Divisible
    # Square
    image_shape = (10, 10)

    nrtilesh = 2
    nrtilesv = 2
    grid_lines = draw_polygon.make_gridlines(image_shape, nrtilesh, nrtilesv)
    assert len(grid_lines) == (nrtilesh + 1) + (nrtilesv + 1)
    grid_lines_true = np.array(
        [[[0, 0], [0, 10]], [[5, 0], [5, 10]], [[10, 0], [10, 10]]]
    )
    np.testing.assert_allclose(grid_lines[:3], grid_lines_true)
    grid_lines_true = np.array(
        [[[0, 0], [10, 0]], [[0, 5], [10, 5]], [[0, 10], [10, 10]]]
    )
    np.testing.assert_allclose(grid_lines[3:], grid_lines_true)

    # Non-square
    nrtilesh = 1
    nrtilesv = 2
    grid_lines = draw_polygon.make_gridlines(image_shape, nrtilesh, nrtilesv)
    assert len(grid_lines) == (nrtilesh + 1) + (nrtilesv + 1)
    grid_lines_true = np.array(
        [[[0, 2.5], [0, 7.5]], [[5, 2.5], [5, 7.5]], [[10, 2.5], [10, 7.5]]]
    )
    np.testing.assert_allclose(grid_lines[:3], grid_lines_true)
    grid_lines_true = np.array([[[0, 2.5], [10, 2.5]], [[0, 7.5], [10, 7.5]]])
    np.testing.assert_allclose(grid_lines[3:], grid_lines_true)

    nrtilesh = 2
    nrtilesv = 1
    grid_lines = draw_polygon.make_gridlines(image_shape, nrtilesh, nrtilesv)
    assert len(grid_lines) == (nrtilesh + 1) + (nrtilesv + 1)
    grid_lines_true = np.array([[[2.5, 0], [2.5, 10]], [[7.5, 0], [7.5, 10]]])
    np.testing.assert_allclose(grid_lines[:2], grid_lines_true)
    grid_lines_true = np.array(
        [[[2.5, 0], [7.5, 0]], [[2.5, 5], [7.5, 5]], [[2.5, 10], [7.5, 10]]]
    )
    np.testing.assert_allclose(grid_lines[2:], grid_lines_true)

    # Non-divisible
    # Square
    nrtilesh = 3
    nrtilesv = 3
    grid_lines = draw_polygon.make_gridlines(image_shape, nrtilesh, nrtilesv)
    assert len(grid_lines) == (nrtilesh + 1) + (nrtilesv + 1)
    grid_lines_true = np.array(
        [
            [[0, 0], [0, 10]],
            [[10 / 3, 0], [10 / 3, 10]],
            [[20 / 3, 0], [20 / 3, 10]],
            [[10, 0], [10, 10]],
        ]
    )
    np.testing.assert_allclose(grid_lines[:4], grid_lines_true)
    grid_lines_true = np.array(
        [
            [[0, 0], [10, 0]],
            [[0, 10 / 3], [10, 10 / 3]],
            [[0, 20 / 3], [10, 20 / 3]],
            [[0, 10], [10, 10]],
        ]
    )
    np.testing.assert_allclose(grid_lines[4:], grid_lines_true)

    # Non-square
    nrtilesh = 2
    nrtilesv = 3
    grid_lines = draw_polygon.make_gridlines(image_shape, nrtilesh, nrtilesv)
    assert len(grid_lines) == (nrtilesh + 1) + (nrtilesv + 1)
    grid_lines_true = np.array(
        [
            [[0, 10 / 6], [0, 50 / 6]],
            [[10 / 3, 10 / 6], [10 / 3, 50 / 6]],
            [[20 / 3, 10 / 6], [20 / 3, 50 / 6]],
            [[10, 10 / 6], [10, 50 / 6]],
        ]
    )
    np.testing.assert_allclose(grid_lines[:4], grid_lines_true)
    grid_lines_true = np.array(
        [[[0, 10 / 6], [10, 10 / 6]], [[0, 5], [10, 5]], [[0, 50 / 6], [10, 50 / 6]]]
    )
    np.testing.assert_allclose(grid_lines[4:], grid_lines_true)


def test_discretize_mask():
    # Divisible
    # Square
    mask = np.zeros((10, 10))
    nrtilesh, nrtilesv = 2, 2
    mask[-1, -1] = 1
    mask, mask_ds = draw_polygon.discretize_mask(mask, nrtilesh, nrtilesv)

    assert mask_ds.shape == (nrtilesv, nrtilesh)
    assert mask_ds[-1, -1] == 1
    assert np.sum(mask_ds) == 1
    assert mask.shape == (10, 10)
    assert mask.sum() == 25

    # Non-square
    mask = np.zeros((10, 10))
    nrtilesh, nrtilesv = 2, 1
    mask[5, -1] = 1
    mask[-1, 0] = 1
    mask, mask_ds = draw_polygon.discretize_mask(mask, nrtilesh, nrtilesv)

    assert mask_ds.shape == (nrtilesv, nrtilesh)
    assert mask_ds[-1, -1] == 1
    assert np.sum(mask_ds) == 1
    assert mask.shape == (10, 10)
    assert mask.sum() == 25

    # Non-divisible
    # Square
    mask = np.zeros((10, 10))
    nrtilesh, nrtilesv = 3, 3
    mask[-1, -1] = 1
    mask[5, 5] = 1
    mask, mask_ds = draw_polygon.discretize_mask(mask, nrtilesh, nrtilesv)

    assert mask_ds.shape == (nrtilesv, nrtilesh)
    assert mask_ds[1, 1] == 1
    assert np.sum(mask_ds) == 2
    assert mask.shape == (10, 10)
    assert mask.sum() >= 18
    assert mask.sum() <= 32

    # Non-square
    mask = np.zeros((10, 10))
    nrtilesh, nrtilesv = 2, 3
    mask[-1, -1] = 1
    mask[5, 5] = 1
    mask, mask_ds = draw_polygon.discretize_mask(mask, nrtilesh, nrtilesv)

    assert mask_ds.shape == (nrtilesv, nrtilesh)
    assert mask_ds[1, 1] == 1
    assert np.sum(mask_ds) == 1
    assert mask.shape == (10, 10)
    assert mask.sum() >= 9
    assert mask.sum() <= 16
