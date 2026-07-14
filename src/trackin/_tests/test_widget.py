import numpy as np
import pandas as pd

from trackin._widget import add_white_border, check_and_add_displ_cols


def test_check_and_add_displ_cols_adds_missing_columns():
    df = pd.DataFrame({"tframe": [0, 1], "y": [1, 2], "x": [3, 4]})
    result = check_and_add_displ_cols(df)
    assert (result["displ_x"] == 0).all()
    assert (result["displ_y"] == 0).all()


def test_check_and_add_displ_cols_preserves_existing_columns():
    df = pd.DataFrame(
        {"tframe": [0], "y": [1], "x": [3], "displ_x": [5], "displ_y": [7]}
    )
    result = check_and_add_displ_cols(df)
    assert result["displ_x"].tolist() == [5]
    assert result["displ_y"].tolist() == [7]


def test_add_white_border_grayscale():
    image = np.zeros((10, 10), dtype=np.uint8)
    bordered = add_white_border(image, border_size=2)
    assert bordered.shape == (14, 14)
    assert (bordered[0, :] == 255).all()  # border row is white
    assert (bordered[2:-2, 2:-2] == 0).all()  # original data preserved


def test_add_white_border_rgb():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bordered = add_white_border(image, border_size=2)
    assert bordered.shape == (14, 14, 3)
    assert (bordered[0, :, :] == 255).all()
    assert (bordered[2:-2, 2:-2, :] == 0).all()
