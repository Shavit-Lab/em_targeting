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
