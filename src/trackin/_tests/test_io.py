import pandas as pd
import pytest

from trackin.core.detection import Detection
from trackin.core.io import TrackFileError, combine_tracks, read_track_file, tracks_table


def test_read_track_file_refuses_file_not_starting_with_standard_columns(tmp_path):
    """A track file must start with tframe, y, x, displ_y, displ_x,
    track_no, in that order. Otherwise reading it raises TrackFileError,
    whose message says what the file starts with."""
    path = tmp_path / "tracks.csv"
    path.write_text("label,tframe,y,x\n7,0,5,5\n")

    with pytest.raises(TrackFileError) as error:
        read_track_file(path)

    assert "It starts with: label, tframe, y, x." in str(error.value)


def test_read_track_file_refuses_empty_file(tmp_path):
    """A track file with only a header has no tracks to add."""
    path = tmp_path / "tracks.csv"
    path.write_text("tframe,y,x,displ_y,displ_x,track_no\n")

    with pytest.raises(TrackFileError) as error:
        read_track_file(path)

    assert str(error.value) == "The loaded file is empty or only contains a header."


def test_read_track_file_keeps_extra_values_as_written(tmp_path):
    """The tracking columns are read as whole numbers; extra columns are
    read as text exactly as written (007 stays 007), so they are copied
    unchanged into the combined file."""
    path = tmp_path / "tracks.csv"
    path.write_text("tframe,y,x,displ_y,displ_x,track_no,label\n0,5,5,0,0,3,007\n")

    df = read_track_file(path)

    assert df["track_no"].tolist() == [3]
    assert df["label"].tolist() == ["007"]


def test_tracks_table_numbers_tracks_from_first_track_no():
    """Accepted tracks become rows as in a session file, numbered from
    first_track_no (e.g. after a loaded track file's highest number). Extra
    values are given as the text written in a file: a missing value is
    empty."""
    detections = [
        [Detection(10, 10, extra=(7, "cell_A")), Detection(50, 50, extra=(8, float("nan")))],
        [Detection(12, 12, extra=(7, "cell_A"))],
    ]
    accepted = [[0, 0], [1, -1]]  # track 1: frames 0 and 1; track 2: frame 0 only

    df = tracks_table(accepted, detections, ["label", "note"], first_track_no=4)

    assert df.values.tolist() == [
        [0, 10, 10, 0, 0, 4, "7", "cell_A"],
        [1, 12, 12, 0, 0, 4, "7", "cell_A"],
        [0, 50, 50, 0, 0, 5, "8", ""],
    ]


def test_combine_tracks_puts_the_track_file_first():
    """The combined file has the track file's tracks first, then this
    session's. Its extra columns are both files' (each once, the track
    file's first), with -1 where a file doesn't have a column."""
    track_df = pd.DataFrame([[0, 5, 5, 0, 0, 3, "good"]],
                            columns=["tframe", "y", "x", "displ_y", "displ_x", "track_no", "quality"])
    session_df = pd.DataFrame([[0, 10, 10, 0, 0, 4, "cell_A"]],
                              columns=["tframe", "y", "x", "displ_y", "displ_x", "track_no", "note"])

    combined, extra_columns = combine_tracks(track_df, session_df)

    assert extra_columns == ["quality", "note"]
    assert combined.values.tolist() == [
        [0, 5, 5, 0, 0, 3, "good", -1],
        [0, 10, 10, 0, 0, 4, -1, "cell_A"],
    ]


def test_read_track_file_refuses_file_that_is_not_a_valid_csv(tmp_path):
    """A track file that pandas can't read (here a line with more values than
    the header has columns, or a completely empty file) raises
    TrackFileError too, so the plugin shows it like the other problems."""
    broken = tmp_path / "broken.csv"
    broken.write_text("tframe,y,x,displ_y,displ_x,track_no\n0,5,5,0,0,3\n1,6,6,0,0,3,9\n")
    empty = tmp_path / "empty.csv"
    empty.write_text("")

    for path in (broken, empty):
        with pytest.raises(TrackFileError) as error:
            read_track_file(path)
        assert str(error.value).startswith("The selected file is not a valid CSV or is malformed:")
