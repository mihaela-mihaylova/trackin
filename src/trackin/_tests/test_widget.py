from types import SimpleNamespace
import pytest

import numpy as np
import pandas as pd
from skimage.io import imsave

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


def test_generate_positions_list_keeps_empty_frames(tmp_path, monkeypatch):
    # 5 images; the CSV has no detections in frame 2 and none after frame 3
    monkeypatch.setattr(shared_state, "DATA", [])
    monkeypatch.setattr(shared_state, "TRACKED", False)
    monkeypatch.setattr(shared_state, "SESSION_FILE", "session.csv")
    df = pd.DataFrame(
        {
            "tframe": [0, 1, 1, 3],
            "y": [10, 20, 21, 30],
            "x": [11, 22, 23, 33],
            "displ_y": [0, 0, 0, 0],
            "displ_x": [0, 0, 0, 0],
        }
    )
    data = widget.generate_positions_list(df, str(tmp_path), n_frames=5)

    assert len(data) == 5  # one entry per image
    assert data[0] == [(10, 11, 0, 0)]
    assert data[1] == [(20, 22, 0, 0), (21, 23, 0, 0)]
    assert data[2] == []  # empty frame stays empty instead of shifting
    assert data[3] == [(30, 33, 0, 0)]
    assert data[4] == []


def test_check_image_numbering_accepts_consecutive_names():
    assert widget.check_image_numbering(["2.png", "0.png", "1.png"]) is None


def test_check_image_numbering_rejects_missing_frame():
    """One missing image: name the frame and say how to fix it."""
    message = widget.check_image_numbering(["0.png", "1.png", "3.png"])
    assert "There is no image for frame 2." in message
    assert "Add the missing images, or renumber the images" in message
    assert widget.IMAGE_NAMING_RULE in message


def test_check_image_numbering_rejects_several_missing_frames():
    """Many missing images: give the total and list only the first 10."""
    message = widget.check_image_numbering(["0.png", "100.png"])
    assert "There is no image for 99 frames: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and 89 more." in message


def test_check_image_numbering_rejects_not_starting_at_zero():
    """A folder numbered 1, 2, 3, ... is told it must start at 0, rather
    than that frame 0 is missing."""
    message = widget.check_image_numbering(["1.png", "2.png", "3.png"])
    assert "The images are numbered from 1, but must start at 0." in message
    assert "no image for frame" not in message


def test_check_image_numbering_rejects_duplicates():
    """Name the files that share a frame number."""
    message = widget.check_image_numbering(["0.png", "1.png", "01.png"])
    assert "More than one image has frame number 1: 01.png, 1.png." in message
    assert "Keep one image per frame number." in message


def test_check_image_numbering_rejects_several_duplicates():
    message = widget.check_image_numbering(["0.png", "1.png", "01.png", "2.png", "02.png"])
    assert "same frame number: 1 (01.png, 1.png), 2 (02.png, 2.png)." in message


def test_check_image_numbering_rejects_names_without_number():
    """Name the files and suggest the new name for the first one."""
    message = widget.check_image_numbering(["0.png", "frame_1.png"])
    assert "not named with only a frame number: frame_1.png." in message
    assert "e.g. frame_1.png → 1.png." in message


def test_check_image_numbering_rejects_empty_folder():
    """An empty folder (or one with only other files) lists the supported
    formats instead of the naming rule."""
    message = widget.check_image_numbering([])
    assert "No images found in this folder." in message
    assert "png" in message
    assert widget.IMAGE_NAMING_RULE not in message


def test_format_list_shortens_long_lists():
    assert widget.format_list([1, 2, 3]) == "1, 2, 3"
    assert widget.format_list(range(12)) == "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, and 2 more"


def test_check_detection_frames_rejects_frames_after_last_image():
    df = pd.DataFrame({"tframe": [0, 1, 3, 3, 4], "y": [1] * 5, "x": [1] * 5})
    message = widget.check_detection_frames(df, n_images=3)
    assert "3 detection(s) are in frame(s) 3, 4" in message
    assert "only images for frames 0 to 2" in message
    assert "every tframe must be between 0 and 2" in message


def test_check_detection_frames_rejects_negative_frames():
    df = pd.DataFrame({"tframe": [-1, 0], "y": [1, 1], "x": [1, 1]})
    message = widget.check_detection_frames(df, n_images=3)
    assert "1 detection(s) are in frame(s) -1" in message


def test_check_detection_frames_accepts_when_all_frames_have_images():
    df = pd.DataFrame({"tframe": [0, 2], "y": [1, 1], "x": [1, 1]})
    assert widget.check_detection_frames(df, n_images=3) is None


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


def test_update_image_clears_points_on_frame_without_detections(make_napari_viewer, monkeypatch):
    """Moving from a frame with detections to a frame without any must
    clear the points from the screen. Before, nothing was redrawn for an
    empty frame, so the previous frame's points (and the yellow track
    marker) stayed visible on it."""
    viewer = make_napari_viewer()
    monkeypatch.setattr(widget, "viewer", viewer)
    monkeypatch.setattr(widget, "images", [np.zeros((20, 20)), np.zeros((20, 20))])
    monkeypatch.setattr(widget, "image_files", ["0.png", "1.png"])
    monkeypatch.setattr(widget, "csv_data", pd.DataFrame())  # only needs to be "loaded"
    monkeypatch.setattr(shared_state, "TRACKED", False)
    # frame 0 has one detection, which is on the current track; frame 1 has none
    monkeypatch.setattr(shared_state, "DATA", [[(10, 10, 0, 0)], []])
    monkeypatch.setattr(shared_state, "track", [0, -1])

    monkeypatch.setattr(shared_state, "current_index", 0)
    widget.update_image()
    assert len(viewer.layers["detections"].data) == 1
    assert len(viewer.layers["track_point"].data) == 1

    monkeypatch.setattr(shared_state, "current_index", 1)
    widget.update_image()
    assert len(viewer.layers["detections"].data) == 0
    assert len(viewer.layers["track_point"].data) == 0


def test_list_image_files_ignores_other_files(tmp_path):
    """Only image files count as frames. Other files in the folder, such as
    macOS's hidden .DS_Store or a notes.txt, must be ignored -- otherwise
    they would be reported as images not named with a frame number."""
    for name in ["0.png", "1.TIF", ".DS_Store", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    assert sorted(widget.list_image_files(str(tmp_path))) == ["0.png", "1.TIF"]


# Session file name of the dataset that is open before a test tries to load
# another folder or CSV. A stand-in only: no file with this name is created.
OPEN_SESSION_FILE = "track_session_open_dataset_20250101_000000.csv"


def _seed_open_dataset(monkeypatch):
    """State as if a dataset (images and detections) were already open, to
    check that a refused folder or file leaves it untouched."""
    monkeypatch.setattr(widget, "images", ["open image"])
    monkeypatch.setattr(widget, "image_files", ["0.png"])
    monkeypatch.setattr(widget, "images_loaded", True)
    monkeypatch.setattr(widget, "csv_loaded", True)
    monkeypatch.setattr(shared_state, "DATA", [[(1, 2, 0, 0)]])
    monkeypatch.setattr(shared_state, "track", [0])
    monkeypatch.setattr(shared_state, "SESSION_FILE", OPEN_SESSION_FILE)


def _record_critical_messages(monkeypatch):
    """Replace the error dialog with one that records its title and text,
    so a test can check the message without anyone clicking OK."""
    messages = []
    monkeypatch.setattr(
        widget.QMessageBox, "critical",
        lambda parent, title, text: messages.append((title, text)),
    )
    return messages


def _assert_open_dataset_untouched():
    """The dataset set up by _seed_open_dataset is still loaded as it was.
    An unchanged SESSION_FILE means no new session was started, so accepted
    tracks would still be saved to the open dataset's session file."""
    assert widget.images == ["open image"]
    assert widget.image_files == ["0.png"]
    assert shared_state.DATA == [[(1, 2, 0, 0)]]
    assert shared_state.track == [0]
    assert shared_state.SESSION_FILE == OPEN_SESSION_FILE


def test_choose_folder_refuses_missing_image_and_keeps_open_dataset(tmp_path, monkeypatch):
    """Load Images on a folder with 0.png, 1.png and 3.png (no 2.png) must
    show the "Images Not Loaded" error naming frame 2, and leave the dataset
    that was already open exactly as it was. The image names are checked
    before anything is read, so the (empty) files never need to be opened."""
    for name in ["0.png", "1.png", "3.png"]:
        (tmp_path / name).write_bytes(b"")
    _seed_open_dataset(monkeypatch)
    messages = _record_critical_messages(monkeypatch)
    # the folder dialog picks the test folder instead of asking the user
    monkeypatch.setattr(
        widget.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path)
    )

    widget.choose_folder()

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Images Not Loaded"
    assert "There is no image for frame 2." in text
    _assert_open_dataset_untouched()


def test_choose_folder_refuses_unreadable_image_and_keeps_open_dataset(tmp_path, monkeypatch):
    """Load Images on a correctly numbered folder where one file is not a
    valid image (1.png) must refuse the folder -- every later image would
    otherwise shift one frame -- and leave the dataset that was already
    open exactly as it was."""
    imsave(str(tmp_path / "0.png"), np.zeros((5, 5), dtype=np.uint8))
    (tmp_path / "1.png").write_bytes(b"not an image")
    _seed_open_dataset(monkeypatch)
    messages = _record_critical_messages(monkeypatch)
    # the folder dialog picks the test folder instead of asking the user
    monkeypatch.setattr(
        widget.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path)
    )

    widget.choose_folder()

    # one message for all unreadable images, naming them
    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Images Not Loaded"
    assert "These images could not be opened: 1.png." in text
    _assert_open_dataset_untouched()


def test_load_csv_refuses_detections_without_image_and_keeps_open_dataset(tmp_path, monkeypatch):
    """With 2 images loaded (frames 0 and 1), Load Detections on a CSV with
    a detection in frame 5 must show the "Detections Not Loaded" error with
    that frame and the allowed range, and leave the detections and track
    that were already loaded exactly as they were."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("tframe,y,x\n0,10,10\n1,20,20\n5,30,30\n")
    _seed_open_dataset(monkeypatch)
    monkeypatch.setattr(widget, "images", ["image 0", "image 1"])  # frames 0 and 1
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test CSV instead of asking the user
    monkeypatch.setattr(
        widget.QFileDialog, "getOpenFileName", lambda *args: (str(csv_path), "")
    )

    widget.load_csv()

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Detections Not Loaded"
    assert "1 detection(s) are in frame(s) 5" in text
    assert "only images for frames 0 to 1" in text
    assert widget.images == ["image 0", "image 1"]
    assert shared_state.DATA == [[(1, 2, 0, 0)]]
    assert shared_state.track == [0]
    assert shared_state.SESSION_FILE == OPEN_SESSION_FILE


# --- Extra columns of the detections file (e.g. label, area) ---------------

def test_read_detections_csv_keeps_extra_columns_as_they_are(tmp_path):
    """Only the tracking columns are converted to whole numbers. Extra
    columns keep their values: before, every column was converted, so a
    text column (e.g. a label name) made loading crash."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("tframe,y,x,label,area\n0,10.7,20,cell_A,350.5\n")

    df = widget.read_detections_csv(str(csv_path))

    assert df["y"].tolist() == [10]  # tracking column: whole number, as before
    assert df["label"].tolist() == ["cell_A"]  # text kept
    assert df["area"].tolist() == [350.5]  # decimals kept


def test_generate_positions_list_stores_extra_values(tmp_path, monkeypatch):
    """Each detection's extra values are stored in shared_state.EXTRA, in the
    same place as its position in DATA. The positions stay whole numbers even
    though the extra columns hold text and decimals."""
    monkeypatch.setattr(shared_state, "DATA", [])
    monkeypatch.setattr(shared_state, "TRACKED", False)
    monkeypatch.setattr(shared_state, "SESSION_FILE", "session.csv")
    monkeypatch.setattr(shared_state, "EXTRA_COLUMNS", ["label", "area"])
    df = pd.DataFrame(
        {
            "tframe": [0, 2],
            "y": [10, 30],
            "x": [20, 40],
            "displ_y": [0, 0],
            "displ_x": [0, 0],
            "label": ["cell_A", "cell_B"],
            "area": [350.5, 410.0],
        }
    )

    data = widget.generate_positions_list(df, str(tmp_path), n_frames=3)

    assert data[0] == [(10, 20, 0, 0)]
    assert type(data[0][0][0]) is int  # not turned into a decimal or text
    assert shared_state.EXTRA[0] == [("cell_A", 350.5)]
    assert shared_state.EXTRA[1] == []  # empty frame
    assert shared_state.EXTRA[2] == [("cell_B", 410.0)]


def test_write_updated_detections_to_file_writes_extra_columns(tmp_path, monkeypatch):
    """The leftover file has the extra columns after the standard ones. A
    value containing a comma is put in quotes so it stays one cell, a missing
    value is left blank, and a deleted detection is still left out."""
    monkeypatch.setattr(shared_state, "EXTRA_COLUMNS", ["label", "note"])
    monkeypatch.setattr(shared_state, "EXTRA", [[(7, "cell_A"), (8, "x, y")], [(9, float("nan"))]])
    data = [
        [(10, 20, 0, 0), (-1000000, -1000000, 0, 0)],  # frame 0: one real, one deleted
        [(30, 40, 0, 0)],
    ]

    write_updated_detections_to_file(data, "out.csv", str(tmp_path))

    lines = (tmp_path / "out.csv").read_text().splitlines()
    assert lines == [
        "tframe,y,x,displ_y,displ_x,label,note",
        "0,10,20,0,0,7,cell_A",
        "1,30,40,0,0,9,",
    ]


def test_accept_track_writes_extra_columns(tmp_path, monkeypatch):
    """Accepting a track writes its detections' extra values into the
    session file (after track_no), and the remaining detections' extra values
    into the leftover file. A value containing a comma is put in quotes."""
    from trackin import utils

    monkeypatch.setattr(shared_state, "TRACKED", False)
    monkeypatch.setattr(shared_state, "DATA", [[(10, 10, 0, 0), (50, 50, 0, 0)], [(12, 12, 0, 0)]])
    monkeypatch.setattr(shared_state, "EXTRA_COLUMNS", ["label", "note"])
    monkeypatch.setattr(shared_state, "EXTRA", [[(7, "cell_A"), (8, "x, y")], [(9, float("nan"))]])
    monkeypatch.setattr(shared_state, "track", [0, 0])  # first detection of frame 0, then of frame 1
    monkeypatch.setattr(shared_state, "N_TRACKS", 0)
    monkeypatch.setattr(shared_state, "MAX_TRACK_ID", None)
    monkeypatch.setattr(shared_state, "csv_folder_to_save", str(tmp_path))
    monkeypatch.setattr(shared_state, "SESSION_FILE", "session.csv")
    monkeypatch.setattr(shared_state, "UPDATED_DATA_FILE", "leftover.csv")
    monkeypatch.setattr(
        shared_state, "G",
        generate_graph(shared_state.DATA, max_score=1600, score_func="squared", tracked=False),
    )

    utils.accept_track()

    assert (tmp_path / "session.csv").read_text().splitlines() == [
        "tframe,y,x,displ_y,displ_x,track_no,label,note",
        "0,10,10,0,0,1,7,cell_A",
        "1,12,12,0,0,1,9,",
    ]
    assert (tmp_path / "leftover.csv").read_text().splitlines() == [
        "tframe,y,x,displ_y,displ_x,label,note",
        '0,50,50,0,0,8,"x, y"',
    ]


def test_add_detection_gets_no_value_in_extra_columns(tmp_path, monkeypatch):
    """A detection added by shift-clicking didn't come from the detections
    file, so it gets -1 (NO_VALUE) in every extra column, in memory and in
    the leftover file."""
    monkeypatch.setattr(shared_state, "TRACKED", False)
    monkeypatch.setattr(shared_state, "DATA", [[(10, 10, 0, 0)], [(20, 20, 0, 0)]])
    monkeypatch.setattr(shared_state, "EXTRA_COLUMNS", ["label"])
    monkeypatch.setattr(shared_state, "EXTRA", [[(7,)], [(8,)]])
    monkeypatch.setattr(shared_state, "NUM_DET_PER_FRAME", [1, 1])
    monkeypatch.setattr(shared_state, "track", [0, 0])
    monkeypatch.setattr(shared_state, "current_index", 0)
    monkeypatch.setattr(shared_state, "csv_folder_to_save", str(tmp_path))
    monkeypatch.setattr(shared_state, "UPDATED_DATA_FILE", "leftover.csv")
    monkeypatch.setattr(
        shared_state, "G",
        generate_graph(shared_state.DATA, max_score=1600, score_func="squared", tracked=False),
    )
    monkeypatch.setattr(widget, "images", [np.zeros((50, 50)), np.zeros((50, 50))])
    monkeypatch.setattr(widget, "new_point_added", False)
    # existing labels, so the detection counter just updates them
    monkeypatch.setattr(widget, "dets_label", widget.QLabel(""))
    monkeypatch.setattr(widget, "conns_label", widget.QLabel(""))

    # stand-ins for the napari points layer and the shift-click
    layer = SimpleNamespace(
        data=np.array([[10.0, 10.0]]),
        size=np.array([20]),
        border_width_is_relative=True,
        world_to_data=lambda position: position,
        refresh=lambda: None,
    )
    event = SimpleNamespace(button=1, modifiers=["Shift"], position=(30, 40))

    widget.add_detection(layer, event)

    assert shared_state.EXTRA[0] == [(7,), (-1,)]
    assert (tmp_path / "leftover.csv").read_text().splitlines() == [
        "tframe,y,x,displ_y,displ_x,label",
        "0,10,10,0,0,7",
        "0,30,40,0,0,-1",
        "1,20,20,0,0,8",
    ]


def test_load_track_file_combined_file_has_extra_columns(tmp_path, monkeypatch):
    """Add Track File writes a combined file whose header includes the
    detections file's extra columns, so tracks accepted later line up with
    it. Rows from the loaded track file (which has no extra columns) get -1
    (NO_VALUE) there, and rows from this session keep their values exactly
    (007 stays 007, it is not turned into the number 7)."""
    (tmp_path / "session.csv").write_text(
        "tframe,y,x,displ_y,displ_x,track_no,label,note\n"
        "0,10,10,0,0,1,007,cell_A\n"
    )
    track_file = tmp_path / "previous_tracks.csv"
    track_file.write_text("tframe,y,x,displ_y,displ_x,track_no\n0,5,5,0,0,3\n")

    monkeypatch.setattr(widget, "images_loaded", True)
    monkeypatch.setattr(widget, "csv_loaded", True)
    monkeypatch.setattr(widget, "track_file_label", None)
    monkeypatch.setattr(widget, "update_file_paths_display", lambda: None)
    monkeypatch.setattr(shared_state, "EXTRA_COLUMNS", ["label", "note"])
    monkeypatch.setattr(shared_state, "csv_folder_to_save", str(tmp_path))
    monkeypatch.setattr(shared_state, "SESSION_FILE", "session.csv")
    monkeypatch.setattr(shared_state, "UPD_TRACK_FILE", "")
    monkeypatch.setattr(shared_state, "MAX_TRACK_ID", None)
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test track file instead of asking the user
    monkeypatch.setattr(
        widget.QFileDialog, "getOpenFileName", lambda *args: (str(track_file), "")
    )

    widget.load_track_file()

    assert messages == []
    combined = (tmp_path / shared_state.UPD_TRACK_FILE).read_text().splitlines()
    assert combined == [
        "tframe,y,x,displ_y,displ_x,track_no,label,note",
        "0,5,5,0,0,3,-1,-1",  # from the loaded track file: no extra values
        "0,10,10,0,0,4,007,cell_A",  # this session's track, renumbered after 3
    ]


def test_load_track_file_accepts_track_file_with_its_own_extra_columns(tmp_path, monkeypatch):
    """A track file may have extra columns after the standard ones, e.g. a
    session file from an earlier sitting. The combined file gets the extra
    columns of both files, each once and the track file's first. Where a
    file doesn't have a column, its rows get -1 there: the track file has no
    "note", this session has no "quality"."""
    (tmp_path / "session.csv").write_text(
        "tframe,y,x,displ_y,displ_x,track_no,label,note\n"
        "0,10,10,0,0,1,7,cell_A\n"
    )
    track_file = tmp_path / "previous_tracks.csv"
    track_file.write_text(
        "tframe,y,x,displ_y,displ_x,track_no,label,quality\n"
        "0,5,5,0,0,3,4,good\n"
    )

    monkeypatch.setattr(widget, "images_loaded", True)
    monkeypatch.setattr(widget, "csv_loaded", True)
    monkeypatch.setattr(widget, "track_file_label", None)
    monkeypatch.setattr(widget, "update_file_paths_display", lambda: None)
    monkeypatch.setattr(shared_state, "EXTRA_COLUMNS", ["label", "note"])
    monkeypatch.setattr(shared_state, "COMBINED_EXTRA_COLUMNS", [])
    monkeypatch.setattr(shared_state, "csv_folder_to_save", str(tmp_path))
    monkeypatch.setattr(shared_state, "SESSION_FILE", "session.csv")
    monkeypatch.setattr(shared_state, "UPD_TRACK_FILE", "")
    monkeypatch.setattr(shared_state, "MAX_TRACK_ID", None)
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test track file instead of asking the user
    monkeypatch.setattr(
        widget.QFileDialog, "getOpenFileName", lambda *args: (str(track_file), "")
    )

    widget.load_track_file()

    assert messages == []
    assert shared_state.COMBINED_EXTRA_COLUMNS == ["label", "quality", "note"]
    combined = (tmp_path / shared_state.UPD_TRACK_FILE).read_text().splitlines()
    assert combined == [
        "tframe,y,x,displ_y,displ_x,track_no,label,quality,note",
        "0,5,5,0,0,3,4,good,-1",  # from the track file: no "note"
        "0,10,10,0,0,4,7,-1,cell_A",  # from this session: no "quality"
    ]


def test_load_track_file_refuses_file_not_starting_with_standard_columns(tmp_path, monkeypatch):
    """A track file that doesn't start with the standard columns, in order,
    is refused with a message saying what it should start with. The combined
    file that was in use (name and track numbering) stays as it was, so
    tracks accepted later still go to the right file."""
    track_file = tmp_path / "not_a_track_file.csv"
    track_file.write_text("label,tframe,y,x\n7,0,5,5\n")

    monkeypatch.setattr(widget, "images_loaded", True)
    monkeypatch.setattr(widget, "csv_loaded", True)
    monkeypatch.setattr(shared_state, "csv_folder_to_save", str(tmp_path))
    monkeypatch.setattr(shared_state, "UPD_TRACK_FILE", "with_new_tracks_added_in_use.csv")
    monkeypatch.setattr(shared_state, "MAX_TRACK_ID", 3)
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test file instead of asking the user
    monkeypatch.setattr(
        widget.QFileDialog, "getOpenFileName", lambda *args: (str(track_file), "")
    )

    widget.load_track_file()

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Track File Not Loaded"
    assert "must start with these columns, in this order: tframe, y, x, displ_y, displ_x, track_no" in text
    assert "It starts with: label, tframe, y, x." in text
    assert shared_state.UPD_TRACK_FILE == "with_new_tracks_added_in_use.csv"
    assert shared_state.MAX_TRACK_ID == 3


def test_accept_track_writes_combined_file_in_its_column_order(tmp_path, monkeypatch):
    """After Add Track File, an accepted track is appended to the combined
    file in that file's column order. Here the combined file has a column
    ("quality") that only the loaded track file had, so the new rows get -1
    there; the session file keeps only the detections file's columns."""
    from trackin import utils

    (tmp_path / "combined.csv").write_text(
        "tframe,y,x,displ_y,displ_x,track_no,label,quality,note\n"
        "0,5,5,0,0,3,4,good,-1\n"
    )
    monkeypatch.setattr(shared_state, "TRACKED", False)
    monkeypatch.setattr(shared_state, "DATA", [[(10, 10, 0, 0)], [(12, 12, 0, 0)]])
    monkeypatch.setattr(shared_state, "EXTRA_COLUMNS", ["label", "note"])
    monkeypatch.setattr(shared_state, "EXTRA", [[(7, "cell_A")], [(8, "cell_A")]])
    monkeypatch.setattr(shared_state, "COMBINED_EXTRA_COLUMNS", ["label", "quality", "note"])
    monkeypatch.setattr(shared_state, "track", [0, 0])
    monkeypatch.setattr(shared_state, "N_TRACKS", 0)
    monkeypatch.setattr(shared_state, "MAX_TRACK_ID", 3)
    monkeypatch.setattr(shared_state, "csv_folder_to_save", str(tmp_path))
    monkeypatch.setattr(shared_state, "SESSION_FILE", "session.csv")
    monkeypatch.setattr(shared_state, "UPDATED_DATA_FILE", "leftover.csv")
    monkeypatch.setattr(shared_state, "UPD_TRACK_FILE", "combined.csv")
    monkeypatch.setattr(
        shared_state, "G",
        generate_graph(shared_state.DATA, max_score=1600, score_func="squared", tracked=False),
    )

    utils.accept_track()

    assert (tmp_path / "combined.csv").read_text().splitlines() == [
        "tframe,y,x,displ_y,displ_x,track_no,label,quality,note",
        "0,5,5,0,0,3,4,good,-1",
        "0,10,10,0,0,4,7,-1,cell_A",  # track 4 = 3 + this session's track 1
        "1,12,12,0,0,4,8,-1,cell_A",
    ]
    assert (tmp_path / "session.csv").read_text().splitlines()[0] == (
        "tframe,y,x,displ_y,displ_x,track_no,label,note"
    )


def test_load_csv_shows_error_for_unreadable_file_and_keeps_open_dataset(tmp_path, monkeypatch):
    """Load Detections on a CSV that can't be read -- here line 3 has 4
    values but the header only 3 columns, as happens with a decimal comma
    (2,5 instead of 2.5) -- must show the "Detections Not Loaded" error with
    the reason, and leave the dataset that was already open as it was.
    Before, the error only appeared in the terminal and nothing in the app."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("tframe,y,x\n0,10,10\n0,2,5,20\n")
    _seed_open_dataset(monkeypatch)
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test CSV instead of asking the user
    monkeypatch.setattr(
        widget.QFileDialog, "getOpenFileName", lambda *args: (str(csv_path), "")
    )

    widget.load_csv()

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Detections Not Loaded"
    assert text.startswith("The detections file could not be read:")
    assert "Expected 3 fields in line 3, saw 4" in text
    _assert_open_dataset_untouched()


def test_read_detections_csv_refuses_extra_value_on_first_line(tmp_path):
    """When only the first line after the header has one value too many,
    pandas gives no error but shifts all columns one place to the left (the
    y values end up in tframe), so the detections would be read wrongly.
    This must raise an error instead, which Load Detections shows."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("tframe,y,x\n0,10,2,5\n1,20,20\n")

    with pytest.raises(ValueError, match="Line 2 has more values than the header has columns"):
        widget.read_detections_csv(str(csv_path))
