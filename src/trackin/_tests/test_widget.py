from types import SimpleNamespace
import pytest

import numpy as np
import pandas as pd
from napari.components import ViewerModel
from skimage.io import imsave

import trackin._interaction as _interaction
import trackin._layers as _layers
import trackin._loading as _loading
import trackin._navigation as _navigation
import trackin.core.io as core_io
import trackin._widget as widget
from trackin.core.detection import Detection
from trackin.core.session import Session
from trackin._loading import add_white_border
from trackin.core.io import check_and_add_displ_cols, generate_upd_track_filename, write_leftover_file
from trackin._tests.stand_in_widget import plugin_state
from trackin.plugin_state import PluginState
from trackin._viewer_setup import initialize_viewer
from trackin.core.tracking import generate_graph
from trackin._tests.stand_in_widget import stand_in_widget


# two cells over three frames
DETECTIONS_CSV = "tframe,y,x\n0,10,10\n0,30,10\n1,10,12\n1,30,12\n2,10,14\n2,30,14\n"


def write_dataset(folder, detections_csv, n_frames):
    """Write n_frames blank images (0.png, 1.png, ...) into folder/images and
    the detections file (CSV text) into folder; return the images folder and
    the detections file."""
    images_folder = folder / "images"
    images_folder.mkdir()
    for frame in range(n_frames):
        imsave(images_folder / f"{frame}.png", np.zeros((40, 40), dtype=np.uint8), check_contrast=False)
    detections_file = folder / "detections.csv"
    detections_file.write_text(detections_csv)
    return images_folder, detections_file


def load_dataset(folder, detections_csv, n_frames):
    """A stand-in widget, with napari's viewer without a window, that has
    loaded the dataset written into folder through the real
    open_images_folder and open_detections, as the app does."""
    images_folder, detections_file = write_dataset(folder, detections_csv, n_frames)
    w = stand_in_widget(ViewerModel())
    _loading.open_images_folder(w, str(images_folder))
    _loading.open_detections(w, str(detections_file))
    return w


@pytest.fixture
def dataset(tmp_path):
    """The standard small dataset written into the test's folder."""
    return write_dataset(tmp_path, DETECTIONS_CSV, n_frames=3)


@pytest.fixture
def loaded_widget(tmp_path):
    """A stand-in widget with the standard small dataset loaded."""
    return load_dataset(tmp_path, DETECTIONS_CSV, n_frames=3)


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


def test_session_from_dataframe_keeps_empty_frames():
    # 5 images; the CSV has no detections in frame 2 and none after frame 3
    df = pd.DataFrame(
        {
            "tframe": [0, 1, 1, 3],
            "y": [10, 20, 21, 30],
            "x": [11, 22, 23, 33],
            "displ_y": [0, 0, 0, 0],
            "displ_x": [0, 0, 0, 0],
        }
    )
    data = Session.from_dataframe(df, n_frames=5).detections

    assert len(data) == 5  # one entry per image
    assert data[0] == [Detection(10, 11)]
    assert data[1] == [Detection(20, 22), Detection(21, 23)]
    assert data[2] == []  # empty frame stays empty instead of shifting
    assert data[3] == [Detection(30, 33)]
    assert data[4] == []


def test_check_image_numbering_accepts_consecutive_names():
    assert core_io.check_image_numbering(["2.png", "0.png", "1.png"]) is None


def test_check_image_numbering_rejects_missing_frame():
    """One missing image: name the frame and say how to fix it."""
    message = core_io.check_image_numbering(["0.png", "1.png", "3.png"])
    assert "There is no image for frame 2." in message
    assert "Add the missing images, or renumber the images" in message
    assert core_io.IMAGE_NAMING_RULE in message


def test_check_image_numbering_rejects_several_missing_frames():
    """Many missing images: give the total and list only the first 10."""
    message = core_io.check_image_numbering(["0.png", "100.png"])
    assert "There is no image for 99 frames: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, and 89 more." in message


def test_check_image_numbering_rejects_not_starting_at_zero():
    """A folder numbered 1, 2, 3, ... is told it must start at 0, rather
    than that frame 0 is missing."""
    message = core_io.check_image_numbering(["1.png", "2.png", "3.png"])
    assert "The images are numbered from 1, but must start at 0." in message
    assert "no image for frame" not in message


def test_check_image_numbering_rejects_duplicates():
    """Name the files that share a frame number."""
    message = core_io.check_image_numbering(["0.png", "1.png", "01.png"])
    assert "More than one image has frame number 1: 01.png, 1.png." in message
    assert "Keep one image per frame number." in message


def test_check_image_numbering_rejects_several_duplicates():
    message = core_io.check_image_numbering(["0.png", "1.png", "01.png", "2.png", "02.png"])
    assert "same frame number: 1 (01.png, 1.png), 2 (02.png, 2.png)." in message


def test_check_image_numbering_rejects_names_without_number():
    """Name the files and suggest the new name for the first one."""
    message = core_io.check_image_numbering(["0.png", "frame_1.png"])
    assert "not named with only a frame number: frame_1.png." in message
    assert "e.g. frame_1.png → 1.png." in message


def test_check_image_numbering_rejects_empty_folder():
    """An empty folder (or one with only other files) lists the supported
    formats instead of the naming rule."""
    message = core_io.check_image_numbering([])
    assert "No images found in this folder." in message
    assert "png" in message
    assert core_io.IMAGE_NAMING_RULE not in message


def test_format_list_shortens_long_lists():
    assert core_io.format_list([1, 2, 3]) == "1, 2, 3"
    assert core_io.format_list(range(12)) == "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, and 2 more"


def test_check_detection_frames_rejects_frames_after_last_image():
    df = pd.DataFrame({"tframe": [0, 1, 3, 3, 4], "y": [1] * 5, "x": [1] * 5})
    message = core_io.check_detection_frames(df, n_images=3)
    assert "3 detection(s) are in frame(s) 3, 4" in message
    assert "only images for frames 0 to 2" in message
    assert "every tframe must be between 0 and 2" in message


def test_check_detection_frames_rejects_negative_frames():
    df = pd.DataFrame({"tframe": [-1, 0], "y": [1, 1], "x": [1, 1]})
    message = core_io.check_detection_frames(df, n_images=3)
    assert "1 detection(s) are in frame(s) -1" in message


def test_check_detection_frames_accepts_when_all_frames_have_images():
    df = pd.DataFrame({"tframe": [0, 2], "y": [1, 1], "x": [1, 1]})
    assert core_io.check_detection_frames(df, n_images=3) is None


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


def test_write_leftover_file(tmp_path):
    data = [
        [Detection(10, 20, 1, -1), Detection(50, 60, active=False)],  # frame 0: one real, one deleted
        [Detection(30, 40, 0, 2)],  # frame 1: one real
    ]
    write_leftover_file(tmp_path / "out.csv", data, [])

    lines = (tmp_path / "out.csv").read_text().splitlines()

    # Header row
    assert lines[0] == "tframe,y,x,displ_y,displ_x"
    # Deleted (inactive) detection excluded, only the two real ones written
    assert len(lines) == 3
    # Real detections written with correct tframe,y,x,displ_y,displ_x values
    assert lines[1] == "0,10,20,1,-1"
    assert lines[2] == "1,30,40,0,2"


def test_write_leftover_file_empty_data_writes_header_only(tmp_path):
    write_leftover_file(tmp_path / "out.csv", [], [])

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


def test_refresh_stats_shows_the_counts_only_when_there_is_a_graph():
    """The widget's Detections and Connections counts are hidden while
    there is no graph, and show the session's numbers once there is one."""
    parent = widget.QWidget()  # never shown, so the labels open no window
    stand_in = SimpleNamespace(state=plugin_state,
                               dets_label=widget.QLabel(parent), conns_label=widget.QLabel(parent))

    widget.TrackinWidget.refresh_stats(stand_in)
    assert stand_in.dets_label.isHidden() and stand_in.conns_label.isHidden()

    plugin_state.session.detections = [[Detection(10, 10)], [Detection(12, 12)]]
    plugin_state.session.graph = generate_graph(
        plugin_state.session.detections, max_score=1600, score_func="squared", tracked=False)
    counts = plugin_state.session.stats()

    widget.TrackinWidget.refresh_stats(stand_in)
    assert not stand_in.dets_label.isHidden() and not stand_in.conns_label.isHidden()
    assert stand_in.dets_label.text() == f"Detections: {counts['detections']}"
    assert stand_in.conns_label.text() == f"Connections: {counts['connections']}"


def test_clear_detections_and_tracks_resets_plugin_state_and_ui():
    w = stand_in_widget(_FakeViewer())
    # the widget's counts are refreshed by its refresh_stats
    refreshed = []
    w.refresh_stats = lambda: refreshed.append(True)

    # the track file label and the Session Files panel as after a prior load
    w.track_file_label.set_path("Track file loaded: old_tracks.csv")
    w.session_files.show_paths("/old/upd.csv", "/old/session.csv", "/old/combined.csv")

    # Seed state as if a full session (images, detections, and a loaded
    # track file) had already happened for the *previous* dataset
    plugin_state.session.detections = [[(1, 2, 0, 0)]]
    plugin_state.track = [0]
    plugin_state.session.has_prior_tracks = True
    plugin_state.max_prior_track_id = 7
    plugin_state.session.graph = object()  # value doesn't matter, only that it's reset to None
    plugin_state.session.accepted = [[0], [0], [0]]  # three tracks accepted before
    plugin_state.session_file = "track_session_old_20250101_000000.csv"
    plugin_state.leftover_file = "upd_old_20250101_000000.csv"
    plugin_state.combined_file = "with_new_tracks_added_old_20250101_000000.csv"
    plugin_state.combined_extra_columns = ["label"]
    plugin_state.output_folder = "/old/dataset"
    plugin_state.n_detections_shown = 42
    plugin_state.n_connections_shown = 99

    w.clear_detections_and_tracks()

    assert plugin_state.session.detections == []
    assert plugin_state.track == []
    assert plugin_state.session.has_prior_tracks is False
    assert plugin_state.max_prior_track_id is None
    assert plugin_state.session.graph is None
    assert plugin_state.session.accepted == []
    assert plugin_state.session_file == ''
    assert plugin_state.leftover_file == ''
    assert plugin_state.combined_file == ''
    assert plugin_state.combined_extra_columns == []
    assert plugin_state.output_folder == ''
    assert plugin_state.n_detections_shown is None
    assert plugin_state.n_connections_shown is None

    assert w.track_file_label.full_path == ""

    # There is no graph left, so the widget's counts are refreshed to hide them
    assert len(refreshed) == 1

    # The Session Files panel should disappear entirely (not just
    # collapse) -- there's nothing left in it to show until a new CSV
    # is loaded.
    panel = w.session_files
    assert panel.isHidden()
    assert panel.leftover_row.isHidden() and panel.session_row.isHidden() and panel.combined_row.isHidden()
    assert panel.content.isHidden()
    assert panel.header.text() == "▸ Session Files"


def test_delete_detection_d_key_no_op_when_track_is_empty_list():
    """Right after images are reloaded (before a new CSV is loaded),
    plugin_state.track is [] -- the 'D' key path indexed into it with
    curr_track[plugin_state.current_frame] unconditionally, an IndexError
    crash on an empty list."""
    original_track = plugin_state.track
    try:
        plugin_state.track = []
        _interaction.delete_detection(stand_in_widget(), None, use_key=True)  # must not raise
    finally:
        plugin_state.track = original_track


class _FakeLayer:
    """Minimal stand-in for a napari Points layer -- delete_det_by_key/
    delete_det_by_mouse only need .data (mutable, indexable) and .refresh()."""

    def __init__(self, data):
        self.data = data

    def refresh(self):
        pass


def test_delete_det_by_key_writes_updated_detections_file(tmp_path):
    """Deleting a detection via the 'D' key marked it as removed in
    plugin_state.session.detections but never persisted that to the leftover file --
    unlike adding a detection, which already did via the same function."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,20\n0,30,40\n", n_frames=1)
    w.state.track = [-1]
    layer = _FakeLayer(np.array([[10.0, 20.0], [30.0, 40.0]]))
    _interaction.delete_det_by_key(w, layer, 0)

    lines = (tmp_path / w.state.leftover_file).read_text().splitlines()
    assert lines[0] == "tframe,y,x,displ_y,displ_x"
    assert lines[1] == "0,30,40,0,0"  # the deleted detection is excluded


def test_delete_det_by_key_shows_new_track_from_frame_0(tmp_path):
    """Deleting a detection of the current track (D, or right-click on it)
    proposes a new track, which can differ in earlier frames too, so the
    view goes back to frame 0, as after accepting or saving a segment."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,20\n1,12,22\n1,30,40\n", n_frames=2)
    w.state.track = [0, 0]
    # the view and the slider are on frame 1
    w.navigator.image_slider.image_index.value = 1
    assert w.state.current_frame == 1

    _interaction.delete_det_by_key(w, _FakeLayer(np.array([[12.0, 22.0], [30.0, 40.0]])), 0)

    assert w.state.current_frame == 0
    assert w.navigator.image_slider.image_index.value == 0


def test_each_widget_has_its_own_navigator():
    """Two widgets each get their own Frame slider, arrow-key timers and
    held-key flags, so moving or holding a key in one leaves the other alone."""
    w1, w2 = stand_in_widget(), stand_in_widget()
    assert w1.navigator.image_slider is not w2.navigator.image_slider
    assert w1.navigator.right_timer is not w2.navigator.right_timer

    w1.navigator.image_slider.image_index.max = 5
    assert w2.navigator.image_slider.image_index.max == 0

    w1.navigator.right_key_held = True
    _navigation.start_timer(w1.navigator.right_timer)
    try:
        assert w2.navigator.right_key_held is False
        assert not w2.navigator.right_timer.isActive()
    finally:
        _navigation.stop_timer(w1.navigator.right_timer)


def _two_widgets_on_one_viewer():
    """Two stand-in widgets, each with its own state, set up one after the
    other on the same napari viewer (without a window), as when Trackin is
    opened a second time."""
    viewer = ViewerModel()
    w1, w2 = stand_in_widget(viewer), stand_in_widget(viewer)
    w1.state, w2.state = PluginState(), PluginState()
    initialize_viewer(w1)
    initialize_viewer(w2)  # used to fail: keybinding Right already used
    return viewer, w1, w2


def test_opening_a_second_widget_on_the_same_viewer_works():
    """Setting up a second widget on the same viewer gives it the keys and
    replaces the first widget's zoom handler instead of adding another."""
    viewer, w1, w2 = _two_widgets_on_one_viewer()
    right = next(f for k, f in viewer.keymap.items() if str(k) == "Right")
    assert right == w2.navigator.handle_right
    wheel_handlers = [f for f in viewer.mouse_wheel_callbacks if f.__name__ == "on_mouse_wheel_zoom"]
    assert len(wheel_handlers) == 1


def test_zoom_is_kept_within_the_newest_widgets_limits():
    """The zoom limits come from the state of the widget set up on the
    viewer, not from another widget's state."""
    viewer, w1, w2 = _two_widgets_on_one_viewer()
    w1.state.min_zoom, w1.state.max_zoom = 0.01, 100
    w2.state.min_zoom, w2.state.max_zoom = 1, 2
    viewer.camera.zoom = 10
    assert viewer.camera.zoom == 2


def test_delete_det_by_mouse_writes_updated_detections_file(tmp_path):
    """Same bug as the 'D' key path, via right-click deletion instead."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,20\n0,30,40\n", n_frames=1)
    layer = _FakeLayer(np.array([[10.0, 20.0], [30.0, 40.0]]))
    event = SimpleNamespace(handled=False)
    _interaction.delete_det_by_mouse(w, layer, 1, event)

    lines = (tmp_path / w.state.leftover_file).read_text().splitlines()
    assert lines[0] == "tframe,y,x,displ_y,displ_x"
    assert lines[1] == "0,10,20,0,0"  # the deleted detection (index 1) is excluded
    assert event.handled is True


def test_update_file_paths_display_auto_expands_session_files_panel(tmp_path, loaded_widget):
    """A newly added file (e.g. from "Add Track File") should be immediately
    visible, not hidden behind a collapsed header the user has to think to
    open -- update_file_paths_display() must expand the panel whenever it
    actually updates a row's content."""
    w = loaded_widget
    w.session_files.hide_all()  # the panel hidden and closed, as before a detections file is loaded

    w.update_file_paths_display()

    panel = w.session_files
    assert not panel.isHidden()
    assert not panel.content.isHidden()
    assert panel.header.text() == "▾ Session Files"
    assert panel.leftover_path_field.full_path == str(tmp_path / w.state.leftover_file)
    assert panel.combined_row.isHidden()  # no track file added


def test_session_files_panel_shows_and_hides_its_rows():
    """The Session Files panel shows the leftover and session files' rows,
    the combined file's row only when there is one, and hides everything
    again when there are no output files."""
    panel = widget.SessionFilesPanel()
    assert panel.isHidden()

    panel.show_paths("/data/upd.csv", "/data/session.csv")
    assert not panel.isHidden() and not panel.content.isHidden()
    assert not panel.leftover_row.isHidden() and not panel.session_row.isHidden()
    assert panel.combined_row.isHidden()

    panel.show_paths("/data/upd.csv", "/data/session.csv", "/data/combined.csv")
    assert not panel.combined_row.isHidden()
    assert panel.combined_path_field.full_path == "/data/combined.csv"

    panel.header.click()  # closes the panel's rows
    assert panel.content.isHidden() and panel.header.text() == "▸ Session Files"
    panel.header.click()  # and opens them again
    assert not panel.content.isHidden() and panel.header.text() == "▾ Session Files"

    panel.hide_all()
    assert panel.isHidden()
    assert panel.leftover_row.isHidden() and panel.session_row.isHidden() and panel.combined_row.isHidden()


def test_update_image_draws_detections_of_file_without_track_no(tmp_path):
    """Detections from a file without a track_no column must be drawn.
    A Detection always has prior_track as its 5th value, which made
    drawing fail for such files ("4 columns passed, passed data had 5")."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,10\n0,15,5\n", n_frames=1)
    w.state.track = [0]

    _layers.update_image(w)

    assert len(w.viewer.layers["detections"].data) == 2


def test_update_image_hides_deleted_detections(tmp_path):
    """A deleted detection keeps its position in the session's detections (active=False), but
    must not be seen: it is drawn far off-screen, at the same place in the
    layer, so the index of a clicked point still matches its detection."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,10\n0,15,5\n", n_frames=1)
    w.state.session.delete_detection(0, 1)  # the second detection, as D or right-click do
    w.state.track = [0]

    _layers.update_image(w)

    points = w.viewer.layers["detections"].data.tolist()
    assert points == [[10, 10], [-1000000, -1000000]]


def test_update_image_clears_points_on_frame_without_detections(tmp_path):
    """Moving from a frame with detections to a frame without any must
    clear the points from the screen. Before, nothing was redrawn for an
    empty frame, so the previous frame's points (and the yellow track
    marker) stayed visible on it.

    Uses napari's viewer without a window (ViewerModel), which has the same
    layers but draws nothing, so the test also runs on machines without
    graphics support."""
    # frame 0 has one detection, which is on the current track; frame 1 has none
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,10\n", n_frames=2)
    w.state.track = [0, -1]

    w.state.current_frame = 0
    _layers.update_image(w)
    assert len(w.viewer.layers["detections"].data) == 1
    assert len(w.viewer.layers["track_point"].data) == 1

    w.state.current_frame = 1
    _layers.update_image(w)
    assert len(w.viewer.layers["detections"].data) == 0
    assert len(w.viewer.layers["track_point"].data) == 0


def test_track_lines_are_drawn_again_after_being_emptied(tmp_path):
    """The track lines layer is emptied when a track has fewer than two
    points (and when new detections are loaded). napari then takes shapes
    put into it for polygons, which two points don't draw, so the next
    track's lines were invisible. They must be lines again."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,10\n1,20,20\n2,30,25\n", n_frames=3)

    w.state.track = [0, 0, 0]   # a track over 3 frames: 2 lines
    _layers.update_image(w)
    w.state.track = [0, -1, -1]  # a single point: no lines, layer emptied
    _layers.update_image(w)
    assert len(w.viewer.layers["track_lines"].data) == 0
    w.state.track = [0, 0, 0]   # a track over 3 frames again
    _layers.update_image(w)

    assert w.viewer.layers["track_lines"].shape_type == ["line", "line"]


def test_list_image_files_ignores_other_files(tmp_path):
    """Only image files count as frames. Other files in the folder, such as
    macOS's hidden .DS_Store or a notes.txt, must be ignored -- otherwise
    they would be reported as images not named with a frame number."""
    for name in ["0.png", "1.TIF", ".DS_Store", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    assert sorted(core_io.list_image_files(str(tmp_path))) == ["0.png", "1.TIF"]


def _open_dataset(w):
    """What is open in the widget w: its images, image file names, session
    (with its detections), track and session file name."""
    return (w.state.images, list(w.state.image_files), w.state.session,
            w.state.session.detections, list(w.state.track), w.state.session_file)


def _record_critical_messages(monkeypatch):
    """Replace the error dialog with one that records its title and text,
    so a test can check the message without anyone clicking OK."""
    messages = []
    monkeypatch.setattr(
        _loading.QMessageBox, "critical",
        lambda parent, title, text: messages.append((title, text)),
    )
    return messages


def _assert_open_dataset_untouched(w, before):
    """The dataset noted down by _open_dataset is still loaded as it was.
    An unchanged session file name means no new session was started, so accepted
    tracks would still be saved to the open dataset's session file."""
    images, image_files, session, detections, track, session_file = before
    assert w.state.images is images
    assert w.state.image_files == image_files
    assert w.state.session is session
    assert w.state.session.detections is detections
    assert w.state.track == track
    assert w.state.session_file == session_file


def test_choose_folder_refuses_missing_image_and_keeps_open_dataset(tmp_path, monkeypatch, loaded_widget):
    """Load Images on a folder with 0.png, 1.png and 3.png (no 2.png) must
    show the "Images Not Loaded" error naming frame 2, and leave the dataset
    that was already open exactly as it was. The image names are checked
    before anything is read, so the (empty) files never need to be opened."""
    w = loaded_widget
    before = _open_dataset(w)
    folder = tmp_path / "new_images"
    folder.mkdir()
    for name in ["0.png", "1.png", "3.png"]:
        (folder / name).write_bytes(b"")
    messages = _record_critical_messages(monkeypatch)
    # the folder dialog picks the test folder instead of asking the user
    monkeypatch.setattr(
        _loading.QFileDialog, "getExistingDirectory", lambda *args: str(folder)
    )

    _loading.choose_folder(w)

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Images Not Loaded"
    assert "There is no image for frame 2." in text
    _assert_open_dataset_untouched(w, before)


def test_choose_folder_refuses_unreadable_image_and_keeps_open_dataset(tmp_path, monkeypatch, loaded_widget):
    """Load Images on a correctly numbered folder where one file is not a
    valid image (1.png) must refuse the folder -- every later image would
    otherwise shift one frame -- and leave the dataset that was already
    open exactly as it was."""
    w = loaded_widget
    before = _open_dataset(w)
    folder = tmp_path / "new_images"
    folder.mkdir()
    imsave(str(folder / "0.png"), np.zeros((5, 5), dtype=np.uint8))
    (folder / "1.png").write_bytes(b"not an image")
    messages = _record_critical_messages(monkeypatch)
    # the folder dialog picks the test folder instead of asking the user
    monkeypatch.setattr(
        _loading.QFileDialog, "getExistingDirectory", lambda *args: str(folder)
    )

    _loading.choose_folder(w)

    # one message for all unreadable images, naming them
    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Images Not Loaded"
    assert "These images could not be opened: 1.png." in text
    _assert_open_dataset_untouched(w, before)


def test_load_detections_refuses_detections_without_image_and_keeps_open_dataset(tmp_path, monkeypatch):
    """With 2 images loaded (frames 0 and 1), Load Detections on a CSV with
    a detection in frame 5 must show the "Detections Not Loaded" error with
    that frame and the allowed range, and leave the detections and track
    that were already loaded exactly as they were."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,10\n1,10,12\n", n_frames=2)  # frames 0 and 1
    before = _open_dataset(w)
    csv_path = tmp_path / "new_detections.csv"
    csv_path.write_text("tframe,y,x\n0,10,10\n1,20,20\n5,30,30\n")
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test CSV instead of asking the user
    monkeypatch.setattr(
        _loading.QFileDialog, "getOpenFileName", lambda *args: (str(csv_path), "")
    )

    _loading.load_detections(w)

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Detections Not Loaded"
    assert "1 detection(s) are in frame(s) 5" in text
    assert "only images for frames 0 to 1" in text
    _assert_open_dataset_untouched(w, before)


# --- Extra columns of the detections file (e.g. label, area) ---------------

def test_read_detections_csv_keeps_extra_columns_as_they_are(tmp_path):
    """Only the tracking columns are converted to whole numbers. Extra
    columns keep their values: before, every column was converted, so a
    text column (e.g. a label name) made loading crash."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("tframe,y,x,label,area\n0,10.7,20,cell_A,350.5\n")

    df = core_io.read_detections_csv(str(csv_path))

    assert df["y"].tolist() == [10]  # tracking column: whole number, as before
    assert df["label"].tolist() == ["cell_A"]  # text kept
    assert df["area"].tolist() == [350.5]  # decimals kept


def test_session_from_dataframe_stores_extra_values():
    """Each detection's extra values are stored in the detection itself
    (its `extra`), and the session knows the extra columns' names. The
    positions stay whole numbers even though the extra columns hold text and
    decimals. Without a track_no column, there are no prior tracks."""
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

    session = Session.from_dataframe(df, n_frames=3)
    data = session.detections

    assert session.extra_columns == ["label", "area"]
    assert session.has_prior_tracks is False
    assert data[0] == [Detection(10, 20, extra=("cell_A", 350.5))]
    assert type(data[0][0].y) is int  # not turned into a decimal or text
    assert data[1] == []  # empty frame
    assert data[2] == [Detection(30, 40, extra=("cell_B", 410.0))]


def test_write_leftover_file_writes_extra_columns(tmp_path):
    """The leftover file has the extra columns after the standard ones. A
    value containing a comma is put in quotes so it stays one cell, a missing
    value is left blank, and a deleted detection is still left out."""
    data = [
        # frame 0: one real, one deleted
        [Detection(10, 20, extra=(7, "cell_A")), Detection(50, 60, active=False, extra=(8, "x, y"))],
        [Detection(30, 40, extra=(9, float("nan")))],
    ]

    write_leftover_file(tmp_path / "out.csv", data, ["label", "note"])

    lines = (tmp_path / "out.csv").read_text().splitlines()
    assert lines == [
        "tframe,y,x,displ_y,displ_x,label,note",
        "0,10,20,0,0,7,cell_A",
        "1,30,40,0,0,9,",
    ]


def test_accept_track_writes_extra_columns(tmp_path, monkeypatch):
    """Accepting a track writes its detections' extra values into the
    session file (after track_no), and the plugin then writes the remaining
    detections' extra values into the leftover file. A value containing a
    comma is put in quotes."""
    monkeypatch.setattr(plugin_state.session, "has_prior_tracks", False)
    monkeypatch.setattr(plugin_state.session, "detections", [
        [Detection(10, 10, extra=(7, "cell_A")), Detection(50, 50, extra=(8, "x, y"))],
        [Detection(12, 12, extra=(9, float("nan")))],
    ])
    monkeypatch.setattr(plugin_state.session, "extra_columns", ["label", "note"])
    monkeypatch.setattr(plugin_state, "track", [0, 0])  # first detection of frame 0, then of frame 1
    monkeypatch.setattr(plugin_state, "max_prior_track_id", None)
    monkeypatch.setattr(plugin_state, "output_folder", str(tmp_path))
    monkeypatch.setattr(plugin_state, "session_file", "session.csv")
    monkeypatch.setattr(plugin_state, "leftover_file", "leftover.csv")
    monkeypatch.setattr(
        plugin_state.session, "graph",
        generate_graph(plugin_state.session.detections, max_score=1600, score_func="squared", tracked=False),
    )

    plugin_state.session.accept_track(plugin_state.track)
    _interaction.save_accepted_track(stand_in_widget())
    _interaction.save_leftover_file(stand_in_widget())

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
    monkeypatch.setattr(plugin_state.session, "has_prior_tracks", False)
    monkeypatch.setattr(plugin_state.session, "detections",
                        [[Detection(10, 10, extra=(7,))], [Detection(20, 20, extra=(8,))]])
    monkeypatch.setattr(plugin_state.session, "extra_columns", ["label"])
    monkeypatch.setattr(plugin_state, "track", [0, 0])
    monkeypatch.setattr(plugin_state, "current_frame", 0)
    monkeypatch.setattr(plugin_state, "output_folder", str(tmp_path))
    monkeypatch.setattr(plugin_state, "leftover_file", "leftover.csv")
    monkeypatch.setattr(
        plugin_state.session, "graph",
        generate_graph(plugin_state.session.detections, max_score=1600, score_func="squared", tracked=False),
    )
    monkeypatch.setattr(plugin_state, "images", [np.zeros((50, 50)), np.zeros((50, 50))])
    monkeypatch.setattr(plugin_state, "new_point_added", False)

    # stand-ins for the napari points layer and the shift-click
    layer = SimpleNamespace(
        data=np.array([[10.0, 10.0]]),
        size=np.array([20]),
        border_width_is_relative=True,
        world_to_data=lambda position: position,
        refresh=lambda: None,
    )
    event = SimpleNamespace(button=1, modifiers=["Shift"], position=(30, 40))

    _interaction.add_detection(stand_in_widget(), layer, event)

    added = plugin_state.session.detections[0][1]
    assert added.extra == (-1,)
    assert added.added is True
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
    (NO_VALUE) there, and this session's accepted tracks keep their extra
    values exactly as the session has them."""
    track_file = tmp_path / "previous_tracks.csv"
    track_file.write_text("tframe,y,x,displ_y,displ_x,track_no\n0,5,5,0,0,3\n")

    w = load_dataset(tmp_path, "tframe,y,x,label,note\n0,10,10,7,cell_A\n", n_frames=1)
    # one track accepted in this session (detection 0 of frame 0)
    w.state.session.accept_track([0])
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test track file instead of asking the user
    monkeypatch.setattr(
        _loading.QFileDialog, "getOpenFileName", lambda *args: (str(track_file), "")
    )

    _loading.load_track_file(w)

    assert messages == []
    combined = (tmp_path / w.state.combined_file).read_text().splitlines()
    assert combined == [
        "tframe,y,x,displ_y,displ_x,track_no,label,note",
        "0,5,5,0,0,3,-1,-1",  # from the loaded track file: no extra values
        "0,10,10,0,0,4,7,cell_A",  # this session's track, renumbered after 3
    ]


def test_load_track_file_accepts_track_file_with_its_own_extra_columns(tmp_path, monkeypatch):
    """A track file may have extra columns after the standard ones, e.g. a
    session file from an earlier sitting. The combined file gets the extra
    columns of both files, each once and the track file's first. Where a
    file doesn't have a column, its rows get -1 there: the track file has no
    "note", this session has no "quality"."""
    track_file = tmp_path / "previous_tracks.csv"
    track_file.write_text(
        "tframe,y,x,displ_y,displ_x,track_no,label,quality\n"
        "0,5,5,0,0,3,4,good\n"
    )

    w = load_dataset(tmp_path, "tframe,y,x,label,note\n0,10,10,7,cell_A\n", n_frames=1)
    # one track accepted in this session (detection 0 of frame 0)
    w.state.session.accept_track([0])
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test track file instead of asking the user
    monkeypatch.setattr(
        _loading.QFileDialog, "getOpenFileName", lambda *args: (str(track_file), "")
    )

    _loading.load_track_file(w)

    assert messages == []
    assert w.state.combined_extra_columns == ["label", "quality", "note"]
    combined = (tmp_path / w.state.combined_file).read_text().splitlines()
    assert combined == [
        "tframe,y,x,displ_y,displ_x,track_no,label,quality,note",
        "0,5,5,0,0,3,4,good,-1",  # from the track file: no "note"
        "0,10,10,0,0,4,7,-1,cell_A",  # from this session: no "quality"
    ]


def test_load_track_file_refuses_file_not_starting_with_standard_columns(tmp_path, monkeypatch, loaded_widget):
    """A track file that doesn't start with the standard columns, in order,
    is refused with a message saying what it should start with. The combined
    file that was in use (name and track numbering) stays as it was, so
    tracks accepted later still go to the right file."""
    track_file = tmp_path / "not_a_track_file.csv"
    track_file.write_text("label,tframe,y,x\n7,0,5,5\n")

    w = loaded_widget
    # a track file was added earlier: its combined file is in use
    w.state.combined_file = "with_new_tracks_added_in_use.csv"
    w.state.max_prior_track_id = 3
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test file instead of asking the user
    monkeypatch.setattr(
        _loading.QFileDialog, "getOpenFileName", lambda *args: (str(track_file), "")
    )

    _loading.load_track_file(w)

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Track File Not Loaded"
    assert "must start with these columns, in this order: tframe, y, x, displ_y, displ_x, track_no" in text
    assert "It starts with: label, tframe, y, x." in text
    assert w.state.combined_file == "with_new_tracks_added_in_use.csv"
    assert w.state.max_prior_track_id == 3


def test_accept_track_writes_combined_file_in_its_column_order(tmp_path, monkeypatch):
    """After Add Track File, an accepted track is appended to the combined
    file in that file's column order. Here the combined file has a column
    ("quality") that only the loaded track file had, so the new rows get -1
    there; the session file keeps only the detections file's columns."""
    (tmp_path / "combined.csv").write_text(
        "tframe,y,x,displ_y,displ_x,track_no,label,quality,note\n"
        "0,5,5,0,0,3,4,good,-1\n"
    )
    monkeypatch.setattr(plugin_state.session, "has_prior_tracks", False)
    monkeypatch.setattr(plugin_state.session, "detections",
                        [[Detection(10, 10, extra=(7, "cell_A"))], [Detection(12, 12, extra=(8, "cell_A"))]])
    monkeypatch.setattr(plugin_state.session, "extra_columns", ["label", "note"])
    monkeypatch.setattr(plugin_state, "combined_extra_columns", ["label", "quality", "note"])
    monkeypatch.setattr(plugin_state, "track", [0, 0])
    monkeypatch.setattr(plugin_state, "max_prior_track_id", 3)
    monkeypatch.setattr(plugin_state, "output_folder", str(tmp_path))
    monkeypatch.setattr(plugin_state, "session_file", "session.csv")
    monkeypatch.setattr(plugin_state, "leftover_file", "leftover.csv")
    monkeypatch.setattr(plugin_state, "combined_file", "combined.csv")
    monkeypatch.setattr(
        plugin_state.session, "graph",
        generate_graph(plugin_state.session.detections, max_score=1600, score_func="squared", tracked=False),
    )

    plugin_state.session.accept_track(plugin_state.track)
    _interaction.save_accepted_track(stand_in_widget())

    assert (tmp_path / "combined.csv").read_text().splitlines() == [
        "tframe,y,x,displ_y,displ_x,track_no,label,quality,note",
        "0,5,5,0,0,3,4,good,-1",
        "0,10,10,0,0,4,7,-1,cell_A",  # track 4 = 3 + this session's track 1
        "1,12,12,0,0,4,8,-1,cell_A",
    ]
    assert (tmp_path / "session.csv").read_text().splitlines()[0] == (
        "tframe,y,x,displ_y,displ_x,track_no,label,note"
    )


def test_load_detections_shows_error_for_unreadable_file_and_keeps_open_dataset(tmp_path, monkeypatch, loaded_widget):
    """Load Detections on a CSV that can't be read -- here line 3 has 4
    values but the header only 3 columns, as happens with a decimal comma
    (2,5 instead of 2.5) -- must show the "Detections Not Loaded" error with
    the reason, and leave the dataset that was already open as it was.
    Before, the error only appeared in the terminal and nothing in the app."""
    w = loaded_widget
    before = _open_dataset(w)
    csv_path = tmp_path / "new_detections.csv"
    csv_path.write_text("tframe,y,x\n0,10,10\n0,2,5,20\n")
    messages = _record_critical_messages(monkeypatch)
    # the file dialog picks the test CSV instead of asking the user
    monkeypatch.setattr(
        _loading.QFileDialog, "getOpenFileName", lambda *args: (str(csv_path), "")
    )

    _loading.load_detections(w)

    assert len(messages) == 1
    title, text = messages[0]
    assert title == "Detections Not Loaded"
    assert text.startswith("The detections file could not be read:")
    assert "Expected 3 fields in line 3, saw 4" in text
    _assert_open_dataset_untouched(w, before)


def test_read_detections_csv_refuses_extra_value_on_first_line(tmp_path):
    """When only the first line after the header has one value too many,
    pandas gives no error but shifts all columns one place to the left (the
    y values end up in tframe), so the detections would be read wrongly.
    This must raise an error instead, which Load Detections shows."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("tframe,y,x\n0,10,2,5\n1,20,20\n")

    with pytest.raises(ValueError, match="Line 2 has more values than the header has columns"):
        core_io.read_detections_csv(str(csv_path))


def test_read_detections_csv_refuses_file_without_required_columns(tmp_path):
    """A detections file must have the columns tframe, y and x. Without one
    of them, the error names the missing column and the columns the file
    has. Before, loading failed later with an error only in the terminal."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("frame,y,x\n0,10,10\n")

    with pytest.raises(ValueError) as error:
        core_io.read_detections_csv(str(csv_path))

    assert "must have the columns tframe, y and x, but it has no tframe" in str(error.value)
    assert "Its columns are: frame, y, x." in str(error.value)


def test_read_detections_csv_refuses_file_separated_by_semicolons(tmp_path):
    """A file separated by semicolons is read as one single column, so it
    is refused by the same check. The column name in the message shows the
    semicolons."""
    csv_path = tmp_path / "detections.csv"
    csv_path.write_text("tframe;y;x\n0;10;10\n")

    with pytest.raises(ValueError) as error:
        core_io.read_detections_csv(str(csv_path))

    assert "but it has no tframe, y, x" in str(error.value)
    assert "Its columns are: tframe;y;x." in str(error.value)


def test_click_changes_the_edited_track_not_the_proposal(monkeypatch):
    """The track being edited is a copy of the session's proposal: left-
    clicking a detection changes that copy (which W, Shift-Z, X and accept
    are given), while the session's proposal stays as the session made it."""
    monkeypatch.setattr(plugin_state.session, "proposal", [0, 0])  # the proposal
    _interaction.use_proposed_track(stand_in_widget())
    monkeypatch.setattr(plugin_state, "current_frame", 1)
    monkeypatch.setattr(plugin_state, "new_point_added", False)
    layer = SimpleNamespace(get_value=lambda position, world: 1)  # the click hits detection 1

    _interaction.on_click(stand_in_widget(), layer, SimpleNamespace(button=1, position=(5, 5)))

    assert plugin_state.track == [0, 1]
    assert plugin_state.session.proposal == [0, 0]


def test_session_from_dataframe_with_track_no_has_prior_tracks():
    """A table with a track_no column comes from an earlier tracking run:
    the session notes that it has prior tracks, keeps each detection's
    earlier track number (prior_track), and track_no is not an extra column."""
    df = pd.DataFrame({"tframe": [0, 1], "y": [10, 12], "x": [20, 22],
                       "displ_y": [0, 0], "displ_x": [0, 0], "track_no": [7, 7]})

    session = Session.from_dataframe(df, n_frames=2)

    assert session.has_prior_tracks is True
    assert session.detections[0][0].prior_track == 7
    assert session.extra_columns == []


def test_accept_and_delete_segment_write_the_leftover_file(tmp_path):
    """The session no longer writes files itself: after Shift-Q and Shift-Z,
    the plugin writes the leftover file, which then holds only the
    detections that are still available."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,10\n0,50,50\n1,12,12\n1,52,52\n", n_frames=2)
    leftover = tmp_path / w.state.leftover_file

    w.state.track = [0, 0]
    _interaction.acceptTrack(w)  # Shift-Q
    assert leftover.read_text().splitlines() == ["tframe,y,x,displ_y,displ_x", "0,50,50,0,0", "1,52,52,0,0"]

    w.state.track = [1, 1]
    w.state.current_frame = 1
    _interaction.deleteSegment(w)  # Shift-Z
    assert leftover.read_text().splitlines() == ["tframe,y,x,displ_y,displ_x"]


def test_accept_writes_the_session_and_combined_files(tmp_path):
    """The session no longer writes files itself: after Shift-Q, the plugin
    appends the accepted track to the session file (numbered 1, 2, ...) and,
    once a track file has been added, to the combined file (numbered after
    that file's highest track number)."""
    w = load_dataset(tmp_path, "tframe,y,x\n0,10,10\n0,50,50\n1,12,12\n1,52,52\n", n_frames=2)
    w.state.combined_file = "combined.csv"
    w.state.max_prior_track_id = 3  # a track file with tracks 1-3 was added

    w.state.track = [0, 0]
    _interaction.acceptTrack(w)  # Shift-Q

    assert (tmp_path / w.state.session_file).read_text().splitlines() == [
        "tframe,y,x,displ_y,displ_x,track_no", "0,10,10,0,0,1", "1,12,12,0,0,1"]
    assert (tmp_path / "combined.csv").read_text().splitlines() == ["0,10,10,0,0,4", "1,12,12,0,0,4"]



def test_open_images_folder_loads_the_images(tmp_path):
    """open_images_folder loads a folder of images without a dialog: the
    images in frame order, shown from frame 0, with the slider fitted."""
    images_folder, _ = write_dataset(tmp_path, "tframe,y,x\n0,10,10\n1,10,11\n2,10,12\n", n_frames=3)
    w = stand_in_widget(ViewerModel())

    _loading.open_images_folder(w, str(images_folder))

    assert w.state.images_loaded
    assert w.state.image_files == ["0.png", "1.png", "2.png"]
    assert len(w.state.images) == 3
    assert w.state.current_frame == 0
    assert w.navigator.image_slider.image_index.max == 2


def test_open_detections_starts_a_tracking_session(tmp_path):
    """open_detections loads a detections file without a dialog: a new
    session with the file's detections, a proposed track, and the output
    files named after the file in its folder (the session file created
    empty)."""
    images_folder, detections_file = write_dataset(tmp_path, "tframe,y,x\n0,10,10\n1,10,11\n", n_frames=2)
    w = stand_in_widget(ViewerModel())
    _loading.open_images_folder(w, str(images_folder))

    _loading.open_detections(w, str(detections_file))

    assert w.state.csv_loaded
    assert [len(frame) for frame in w.state.session.detections] == [1, 1]
    assert w.state.track == [0, 0]
    assert w.state.output_folder == str(tmp_path)
    assert w.state.session_file.startswith("track_session_detections_")
    assert w.state.leftover_file.startswith("upd_detections_")
    assert (tmp_path / w.state.session_file).read_text() == ""


def test_open_track_file_writes_the_combined_file(tmp_path):
    """open_track_file adds a track file without a dialog: it writes the
    combined file next to the detections file, numbers new tracks after
    the file's highest track number and shows the file's name."""
    images_folder, detections_file = write_dataset(tmp_path, "tframe,y,x\n0,10,10\n1,10,11\n", n_frames=2)
    w = stand_in_widget(ViewerModel())
    _loading.open_images_folder(w, str(images_folder))
    _loading.open_detections(w, str(detections_file))
    track_file = tmp_path / "earlier_tracks.csv"
    track_file.write_text("tframe,y,x,displ_y,displ_x,track_no\n0,5,5,0,0,4\n1,6,6,0,0,4\n")

    _loading.open_track_file(w, str(track_file))

    assert w.state.max_prior_track_id == 4
    assert (tmp_path / w.state.combined_file).exists()
    assert w.track_file_label.full_path == "Track file loaded: earlier_tracks.csv"


def test_cancelled_dialogs_load_nothing(tmp_path, monkeypatch):
    """Closing a load button's dialog without choosing anything leaves the
    open dataset as it was."""
    images_folder, detections_file = write_dataset(tmp_path, "tframe,y,x\n0,10,10\n1,10,11\n", n_frames=2)
    w = stand_in_widget(ViewerModel())
    _loading.open_images_folder(w, str(images_folder))
    _loading.open_detections(w, str(detections_file))
    session, image_files = w.state.session, w.state.image_files
    monkeypatch.setattr(_loading.QFileDialog, "getExistingDirectory", lambda *args: "")
    monkeypatch.setattr(_loading.QFileDialog, "getOpenFileName", lambda *args: ("", ""))

    _loading.choose_folder(w)
    _loading.load_detections(w)
    _loading.load_track_file(w)

    assert w.state.session is session
    assert w.state.image_files == image_files
    assert w.state.combined_file == ""
