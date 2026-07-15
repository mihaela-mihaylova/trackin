import numpy as np
import pandas as pd

from trackin._widget import (
    add_white_border,
    check_and_add_displ_cols,
    write_updated_detections_to_file,
)


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


def test_write_updated_detections_to_file(tmp_path):
    data = [
        [(10, 20, 1, -1), (-1000000, -1000000, 0, 0)],  # frame 0: one real, one deleted
        [(30, 40, 0, 2)],  # frame 1: one real
    ]
    write_updated_detections_to_file(data, "out.csv", str(tmp_path))

    lines = (tmp_path / "out.csv").read_text().splitlines()

    # Header row
    assert lines[0] == "tframe,y,x,displ_y,displ_x"
    # Deleted detection (-1000000 sentinel) excluded, only the two real ones written
    assert len(lines) == 3
    # Real detections written with correct tframe,y,x,displ_y,displ_x values
    assert lines[1] == "0,10,20,1,-1"
    assert lines[2] == "1,30,40,0,2"


def test_write_updated_detections_to_file_empty_data_writes_header_only(tmp_path):
    write_updated_detections_to_file([], "out.csv", str(tmp_path))

    lines = (tmp_path / "out.csv").read_text().splitlines()

    assert lines == ["tframe,y,x,displ_y,displ_x"]
