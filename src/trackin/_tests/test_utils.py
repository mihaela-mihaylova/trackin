import pytest

from trackin.core.detection import Detection
from trackin.shared_state import shared_state
from trackin.core.tracking import generate_graph

SHARED_STATE_FIELDS = [
    "DATA",
    "track",
    "csv_folder_to_save",
    "SESSION_FILE",
    "UPDATED_DATA_FILE",
    "N_TRACKS",
    "TRACKED",
    "MAX_TRACK_ID",
    "UPD_TRACK_FILE",
    "NUM_DETS",
    "G",
]


@pytest.fixture
def staged_shared_state(tmp_path):
    """Seed shared_state with a minimal two-frame, two-detection scenario
    for accept_track(), and restore the original values afterward -- it's
    a module-level singleton shared across tests."""
    original = {field: getattr(shared_state, field) for field in SHARED_STATE_FIELDS}

    shared_state.DATA = [[Detection(10, 20)], [Detection(30, 40)]]
    shared_state.track = [0, 0]  # accept the single detection in each frame
    shared_state.csv_folder_to_save = str(tmp_path)
    shared_state.SESSION_FILE = "session.csv"
    shared_state.UPDATED_DATA_FILE = "updated.csv"
    shared_state.N_TRACKS = 0
    shared_state.TRACKED = False
    shared_state.MAX_TRACK_ID = None  # no prior track file loaded -> skip UPD_TRACK_FILE
    shared_state.G = generate_graph(
        shared_state.DATA, max_score=1600, score_func="squared", tracked=False
    )

    yield tmp_path

    for field, value in original.items():
        setattr(shared_state, field, value)


def test_accept_track_writes_session_file_and_marks_detections_consumed(
    staged_shared_state,
):
    tmp_path = staged_shared_state

    shared_state.accept_track(shared_state.track)

    session_lines = (tmp_path / "session.csv").read_text().splitlines()
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,10,20,0,0,1"
    assert session_lines[2] == "1,30,40,0,0,1"

    updated_lines = (tmp_path / "updated.csv").read_text().splitlines()
    assert updated_lines == ["tframe,y,x,displ_y,displ_x"]  # both detections consumed

    # kept with their position, but no longer available
    assert shared_state.DATA[0][0] == Detection(10, 20, active=False)
    assert shared_state.DATA[1][0] == Detection(30, 40, active=False)


def test_accept_track_does_not_duplicate_header_when_session_file_has_content(
    staged_shared_state,
):
    tmp_path = staged_shared_state
    session_path = tmp_path / "session.csv"
    session_path.write_text("tframe,y,x,displ_y,displ_x,track_no\n0,1,2,0,0,1\n")

    shared_state.accept_track(shared_state.track)

    session_lines = session_path.read_text().splitlines()
    assert session_lines.count("tframe,y,x,displ_y,displ_x,track_no") == 1
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,1,2,0,0,1"  # pre-existing row untouched
    assert session_lines[2] == "0,10,20,0,0,1"  # newly accepted row appended
    assert session_lines[3] == "1,30,40,0,0,1"


def test_accept_track_adds_header_when_session_file_exists_but_empty(
    staged_shared_state,
):
    tmp_path = staged_shared_state
    session_path = tmp_path / "session.csv"
    session_path.touch()  # exists on disk, but 0 bytes

    shared_state.accept_track(shared_state.track)

    session_lines = session_path.read_text().splitlines()
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,10,20,0,0,1"
    assert session_lines[2] == "1,30,40,0,0,1"


def test_accept_track_writes_upd_track_file_with_offset_track_id(staged_shared_state):
    tmp_path = staged_shared_state
    shared_state.MAX_TRACK_ID = 5
    shared_state.UPD_TRACK_FILE = "upd_tracks.csv"

    shared_state.accept_track(shared_state.track)

    # updated_track_id = MAX_TRACK_ID + N_TRACKS = 5 + 1 (accept_track increments
    # N_TRACKS from 0 to 1 before this file is written). Unlike SESSION_FILE, this
    # file has no header -- it's opened in append mode and rows are written as-is.
    upd_track_lines = (tmp_path / "upd_tracks.csv").read_text().splitlines()
    assert upd_track_lines == [
        "0,10,20,0,0,6",
        "1,30,40,0,0,6",
    ]


def test_accept_track_no_op_when_track_is_empty_list(staged_shared_state):
    """Right after images are reloaded (before a new CSV is loaded),
    shared_state.track is [] rather than None -- accept_track() must treat
    that the same way, not silently create/write into the *previous*
    dataset's SESSION_FILE."""
    tmp_path = staged_shared_state
    shared_state.track = []

    shared_state.accept_track(shared_state.track)

    assert not (tmp_path / "session.csv").exists()
    assert shared_state.N_TRACKS == 0  # must not increment on a no-op


def test_delete_segment_no_op_when_track_is_empty_list_even_with_stale_num_dets(
    staged_shared_state,
):
    """Right after images are reloaded there is no track, while the panel's
    counter (NUM_DETS) can still hold a value from the *previous* dataset.
    shared_state.delete_segment() must do nothing then, and not truncate-and-overwrite
    that dataset's UPDATED_DATA_FILE."""
    tmp_path = staged_shared_state
    shared_state.track = []
    shared_state.NUM_DETS = 5  # stale nonzero value

    updated_path = tmp_path / "updated.csv"
    updated_path.write_text("tframe,y,x,displ_y,displ_x\n0,99,99,0,0\n")

    shared_state.delete_segment(shared_state.track, 0)

    assert updated_path.read_text() == "tframe,y,x,displ_y,displ_x\n0,99,99,0,0\n"


def test_save_segment_no_op_when_track_is_empty_list(staged_shared_state):
    """Right after images are reloaded (before a new CSV is loaded),
    shared_state.track is [] and shared_state.G is None. The segment-
    building part is a harmless no-op on an empty track either way, but
    shared_state.save_segment() unconditionally ended with send_track(), which calls
    .copy() on shared_state.G -- an AttributeError crash when G is None."""
    shared_state.track = []
    shared_state.G = None

    shared_state.save_segment(shared_state.track, 0)  # must not raise

    assert shared_state.track == []  # send_track() never ran to reassign it


def test_delete_all_connections_no_op_when_track_is_empty_list(staged_shared_state):
    """Right after images are reloaded (before a new CSV is loaded),
    shared_state.track is [] -- delete_all_connections() indexed into it
    with shared_state.track[shared_state.current_index] unconditionally,
    an IndexError crash on an empty list."""
    shared_state.track = []

    shared_state.delete_all_connections(shared_state.track, 0)  # must not raise


def test_accept_track_returns_messages_for_the_user(staged_shared_state):
    """accept_track() returns the messages to show instead of showing them
    itself: "Accepted track.", and here also "No detections left." because
    the accepted track used every detection, so no new track is left."""
    assert shared_state.accept_track(shared_state.track) == ["Accepted track.", "No detections left."]


def test_actions_without_a_track_return_no_detections_left(staged_shared_state):
    """Without a proposed track (e.g. right after loading new images), each
    action does nothing and returns the message "No detections left." for
    the plugin to show."""
    shared_state.track = []

    assert shared_state.accept_track(shared_state.track) == ["No detections left."]
    for action in (shared_state.save_segment, shared_state.delete_segment, shared_state.delete_all_connections):
        assert action(shared_state.track, 0) == ["No detections left."]


def test_session_does_not_need_napari():
    """The tracking session must load without napari, so it can be used
    outside the plugin. Checked in a separate Python process, since napari
    is already loaded in this one by the other tests."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, trackin.core.session; "
         "print(any(m.startswith('napari') for m in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False"


def test_delete_segment_works_without_the_panel_counting(staged_shared_state):
    """Shift-Z must not depend on the panel's "Detections" counter
    (NUM_DETS), which is only set when the panel is shown. Before, it did
    nothing while NUM_DETS was unset, e.g. when the tracking is used without
    the plugin's panel."""
    tmp_path = staged_shared_state
    shared_state.NUM_DETS = None  # the panel hasn't counted anything

    assert shared_state.delete_segment(shared_state.track, 1) == ["No detections left."]  # nothing left to propose
    assert shared_state.DATA[0][0].active is False  # both detections deleted
    assert shared_state.DATA[1][0].active is False
    assert (tmp_path / "updated.csv").read_text().splitlines() == ["tframe,y,x,displ_y,displ_x"]
