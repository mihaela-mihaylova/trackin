from types import SimpleNamespace

import numpy as np
import pandas as pd

import trackin._widget as widget
from trackin._widget import (
    add_white_border,
    check_and_add_displ_cols,
    generate_upd_track_filename,
    write_updated_detections_to_file,
)
from trackin.shared_state import shared_state
from trackin.tracking import generate_graph

SHARED_STATE_FIELDS = [
    "DATA",
    "track",
    "track_lines",
    "NUM_DET_PER_FRAME",
    "TRACKED",
    "MAX_TRACK_ID",
    "G",
    "N_TRACKS",
    "SESSION_FILE",
    "UPDATED_DATA_FILE",
    "UPD_TRACK_FILE",
    "NUM_DETS",
    "NUM_CONN",
]


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


def test_generate_upd_track_filename_normal_path():
    result = generate_upd_track_filename(
        "/home/user/data/tracks.csv", "20250101_120000"
    )
    assert result == "with_new_tracks_added_tracks_20250101_120000.csv"


def test_generate_upd_track_filename_windows_style_path():
    result = generate_upd_track_filename(
        "C:\\Users\\test\\my_tracks.csv", "20250101_120000"
    )
    assert result == "with_new_tracks_added_my_tracks_20250101_120000.csv"


def test_generate_upd_track_filename_multiple_dots_in_filename():
    result = generate_upd_track_filename(
        "/home/user/data/my.tracks.v2.csv", "20250101_120000"
    )
    assert result == "with_new_tracks_added_my.tracks.v2_20250101_120000.csv"


class _FakeViewer:
    """Minimal stand-in for a napari viewer -- clear_detections_and_tracks()
    only needs `layers` to support `in`, it never touches anything else."""

    def __init__(self):
        self.layers = {}


def test_clear_detections_and_tracks_resets_shared_state_and_ui():
    original_viewer = widget.viewer
    original_widgets = {
        name: getattr(widget, name)
        for name in (
            "track_file_label",
            "leftover_row",
            "leftover_path_field",
            "session_row",
            "session_path_field",
            "upd_track_row",
            "upd_track_path_field",
            "dets_label",
            "conns_label",
            "session_files_header",
            "session_files_content",
        )
    }
    original_shared_state = {field: getattr(shared_state, field) for field in SHARED_STATE_FIELDS}

    try:
        widget.viewer = _FakeViewer()

        widget.track_file_label = widget.ElidedPathLabel()
        widget.track_file_label.set_path("Track file loaded: old_tracks.csv")

        widget.leftover_row, widget.leftover_path_field = widget.create_path_row(
            "Leftover detections", "..."
        )
        widget.session_row, widget.session_path_field = widget.create_path_row(
            "Tracks accepted this session", "..."
        )
        widget.upd_track_row, widget.upd_track_path_field = widget.create_path_row(
            "Combined tracks", "..."
        )
        for row in (widget.leftover_row, widget.session_row, widget.upd_track_row):
            row.setVisible(True)  # simulate them being shown from a prior load

        widget.dets_label = widget.QLabel("Detections: 42")
        widget.conns_label = widget.QLabel("Connections: 99")

        widget.session_files_header = widget.QPushButton("")
        widget.session_files_header.setVisible(True)  # simulate it showing from a prior load
        widget.session_files_content = widget.QWidget()
        widget.set_session_files_expanded(True)  # simulate it being expanded from a prior load

        # Seed state as if a full session (images, detections, and a loaded
        # track file) had already happened for the *previous* dataset
        shared_state.DATA = [[(1, 2, 0, 0)]]
        shared_state.track = [0]
        shared_state.track_lines = np.array([[[1, 2], [3, 4]]])
        shared_state.NUM_DET_PER_FRAME = [1]
        shared_state.TRACKED = True
        shared_state.MAX_TRACK_ID = 7
        shared_state.G = object()  # value doesn't matter, only that it's reset to None
        shared_state.N_TRACKS = 3
        shared_state.SESSION_FILE = "track_session_old_20250101_000000.csv"
        shared_state.UPDATED_DATA_FILE = "upd_old_20250101_000000.csv"
        shared_state.UPD_TRACK_FILE = "with_new_tracks_added_old_20250101_000000.csv"
        shared_state.NUM_DETS = 42
        shared_state.NUM_CONN = 99

        widget.clear_detections_and_tracks()

        assert shared_state.DATA == []
        assert shared_state.track == []
        assert shared_state.track_lines is None
        assert shared_state.NUM_DET_PER_FRAME == []
        assert shared_state.TRACKED is False
        assert shared_state.MAX_TRACK_ID is None
        assert shared_state.G is None
        assert shared_state.N_TRACKS == 0
        assert shared_state.SESSION_FILE == ''
        assert shared_state.UPDATED_DATA_FILE == ''
        assert shared_state.UPD_TRACK_FILE == ''
        assert shared_state.NUM_DETS is None
        assert shared_state.NUM_CONN is None

        assert widget.track_file_label.full_path == ""
        assert widget.leftover_row.isVisible() is False
        assert widget.session_row.isVisible() is False
        assert widget.upd_track_row.isVisible() is False

        # The Detections/Connections labels only update in response to
        # graph_updated_event -- clear_detections_and_tracks() must fire it
        # itself, or these stay showing the *previous* dataset's stale
        # counts until some unrelated keypress happens to fire it later.
        assert widget.dets_label is None
        assert widget.conns_label is None

        # The Session Files card should disappear entirely (not just
        # collapse) -- there's nothing left in it to show until a new CSV
        # is loaded.
        assert widget.session_files_header.isVisible() is False
        assert widget.session_files_content.isVisible() is False
        assert widget.session_files_header.text() == "▸ Session Files"
    finally:
        widget.viewer = original_viewer
        for name, value in original_widgets.items():
            setattr(widget, name, value)
        for field, value in original_shared_state.items():
            setattr(shared_state, field, value)


def test_delete_detection_d_key_no_op_when_track_is_empty_list():
    """Right after images are reloaded (before a new CSV is loaded),
    shared_state.track is [] -- the 'D' key path indexed into it with
    curr_track[shared_state.current_index] unconditionally, an IndexError
    crash on an empty list."""
    original_track = shared_state.track
    try:
        shared_state.track = []
        widget.delete_detection(None, use_key=True)  # must not raise
    finally:
        shared_state.track = original_track


class _FakeLayer:
    """Minimal stand-in for a napari Points layer -- delete_det_by_key/
    delete_det_by_mouse only need .data (mutable, indexable) and .refresh()."""

    def __init__(self, data):
        self.data = data

    def refresh(self):
        pass


def test_delete_det_by_key_writes_updated_detections_file(tmp_path):
    """Deleting a detection via the 'D' key marked it as removed in
    shared_state.DATA but never persisted that to UPDATED_DATA_FILE --
    unlike adding a detection, which already did via the same function."""
    original = {
        field: getattr(shared_state, field)
        for field in (
            "current_index", "DATA", "TRACKED", "G", "track",
            "csv_folder_to_save", "UPDATED_DATA_FILE",
        )
    }
    original_clicked_index = getattr(widget, "clicked_index", None)

    try:
        shared_state.current_index = 0
        shared_state.DATA = [[(10, 20, 0, 0), (30, 40, 0, 0)]]
        shared_state.TRACKED = False
        shared_state.track = [-1]
        shared_state.csv_folder_to_save = str(tmp_path)
        shared_state.UPDATED_DATA_FILE = "updated.csv"
        shared_state.G = generate_graph(
            shared_state.DATA, max_score=1600, score_func="squared", tracked=False
        )
        widget.clicked_index = 0

        layer = _FakeLayer(np.array([[10.0, 20.0], [30.0, 40.0]]))
        widget.delete_det_by_key(layer)

        lines = (tmp_path / "updated.csv").read_text().splitlines()
        assert lines[0] == "tframe,y,x,displ_y,displ_x"
        assert lines[1] == "0,30,40,0,0"  # the deleted detection is excluded
    finally:
        for field, value in original.items():
            setattr(shared_state, field, value)
        widget.clicked_index = original_clicked_index


def test_delete_det_by_mouse_writes_updated_detections_file(tmp_path):
    """Same bug as the 'D' key path, via right-click deletion instead."""
    original = {
        field: getattr(shared_state, field)
        for field in (
            "current_index", "DATA", "TRACKED", "G",
            "csv_folder_to_save", "UPDATED_DATA_FILE",
        )
    }
    original_clicked_index = getattr(widget, "clicked_index", None)

    try:
        shared_state.current_index = 0
        shared_state.DATA = [[(10, 20, 0, 0), (30, 40, 0, 0)]]
        shared_state.TRACKED = False
        shared_state.csv_folder_to_save = str(tmp_path)
        shared_state.UPDATED_DATA_FILE = "updated.csv"
        shared_state.G = generate_graph(
            shared_state.DATA, max_score=1600, score_func="squared", tracked=False
        )
        widget.clicked_index = 1

        layer = _FakeLayer(np.array([[10.0, 20.0], [30.0, 40.0]]))
        event = SimpleNamespace(handled=False)
        widget.delete_det_by_mouse(layer, event)

        lines = (tmp_path / "updated.csv").read_text().splitlines()
        assert lines[0] == "tframe,y,x,displ_y,displ_x"
        assert lines[1] == "0,10,20,0,0"  # the deleted detection (index 1) is excluded
        assert event.handled is True
    finally:
        for field, value in original.items():
            setattr(shared_state, field, value)
        widget.clicked_index = original_clicked_index


def test_update_file_paths_display_auto_expands_session_files_card(tmp_path):
    """A newly added file (e.g. from "Add Track File") should be immediately
    visible, not hidden behind a collapsed header the user has to think to
    open -- update_file_paths_display() must expand the card whenever it
    actually updates a row's content."""
    original_csv_loaded = widget.csv_loaded
    original_widgets = {
        name: getattr(widget, name)
        for name in (
            "leftover_row",
            "leftover_path_field",
            "session_row",
            "session_path_field",
            "upd_track_row",
            "upd_track_path_field",
            "session_files_header",
            "session_files_content",
        )
    }
    original_shared_state = {
        field: getattr(shared_state, field)
        for field in ("csv_folder_to_save", "UPDATED_DATA_FILE", "SESSION_FILE", "MAX_TRACK_ID", "UPD_TRACK_FILE")
    }

    try:
        widget.csv_loaded = True
        shared_state.csv_folder_to_save = str(tmp_path)
        shared_state.UPDATED_DATA_FILE = "updated.csv"
        shared_state.SESSION_FILE = "session.csv"
        shared_state.MAX_TRACK_ID = None

        widget.leftover_row, widget.leftover_path_field = widget.create_path_row(
            "Leftover detections", "..."
        )
        widget.session_row, widget.session_path_field = widget.create_path_row(
            "Tracks accepted this session", "..."
        )
        widget.upd_track_row, widget.upd_track_path_field = widget.create_path_row(
            "Combined tracks", "..."
        )

        widget.session_files_header = widget.QPushButton("")
        widget.session_files_header.setVisible(False)  # hidden until a CSV is loaded, as it would be by default
        widget.session_files_content = widget.QWidget()
        widget.set_session_files_expanded(False)  # start collapsed, as it would be by default

        widget.update_file_paths_display()

        assert widget.session_files_header.isVisible() is True
        assert widget.session_files_content.isVisible() is True
        assert widget.session_files_header.text() == "▾ Session Files"
    finally:
        widget.csv_loaded = original_csv_loaded
        for name, value in original_widgets.items():
            setattr(widget, name, value)
        for field, value in original_shared_state.items():
            setattr(shared_state, field, value)
