import pytest

from trackin.shared_state import shared_state
from trackin.tracking import generate_graph
from trackin.utils import accept_track, delete_all_connections, delete_segment, save_segment

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

    shared_state.DATA = [[(10, 20, 0, 0)], [(30, 40, 0, 0)]]
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

    accept_track()

    session_lines = (tmp_path / "session.csv").read_text().splitlines()
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,10,20,0,0,1"
    assert session_lines[2] == "1,30,40,0,0,1"

    updated_lines = (tmp_path / "updated.csv").read_text().splitlines()
    assert updated_lines == ["tframe,y,x,displ_y,displ_x"]  # both detections consumed

    assert shared_state.DATA[0][0] == (-1000000, -1000000, 0, 0)
    assert shared_state.DATA[1][0] == (-1000000, -1000000, 0, 0)


def test_accept_track_does_not_duplicate_header_when_session_file_has_content(
    staged_shared_state,
):
    tmp_path = staged_shared_state
    session_path = tmp_path / "session.csv"
    session_path.write_text("tframe,y,x,displ_y,displ_x,track_no\n0,1,2,0,0,1\n")

    accept_track()

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

    accept_track()

    session_lines = session_path.read_text().splitlines()
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,10,20,0,0,1"
    assert session_lines[2] == "1,30,40,0,0,1"


def test_accept_track_writes_upd_track_file_with_offset_track_id(staged_shared_state):
    tmp_path = staged_shared_state
    shared_state.MAX_TRACK_ID = 5
    shared_state.UPD_TRACK_FILE = "upd_tracks.csv"

    accept_track()

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

    accept_track()

    assert not (tmp_path / "session.csv").exists()
    assert shared_state.N_TRACKS == 0  # must not increment on a no-op


def test_delete_segment_no_op_when_track_is_empty_list_even_with_stale_num_dets(
    staged_shared_state,
):
    """NUM_DETS can still hold a stale nonzero value from the *previous*
    dataset right after images are reloaded -- delete_segment() must not
    rely on NUM_DETS alone, since that would let it through to
    truncate-and-overwrite that dataset's UPDATED_DATA_FILE."""
    tmp_path = staged_shared_state
    shared_state.track = []
    shared_state.NUM_DETS = 5  # stale nonzero value

    updated_path = tmp_path / "updated.csv"
    updated_path.write_text("tframe,y,x,displ_y,displ_x\n0,99,99,0,0\n")

    delete_segment()

    assert updated_path.read_text() == "tframe,y,x,displ_y,displ_x\n0,99,99,0,0\n"


def test_save_segment_no_op_when_track_is_empty_list(staged_shared_state):
    """Right after images are reloaded (before a new CSV is loaded),
    shared_state.track is [] and shared_state.G is None. The segment-
    building part is a harmless no-op on an empty track either way, but
    save_segment() unconditionally ended with send_track(), which calls
    .copy() on shared_state.G -- an AttributeError crash when G is None."""
    shared_state.track = []
    shared_state.G = None

    save_segment()  # must not raise

    assert shared_state.track == []  # send_track() never ran to reassign it


def test_delete_all_connections_no_op_when_track_is_empty_list(staged_shared_state):
    """Right after images are reloaded (before a new CSV is loaded),
    shared_state.track is [] -- delete_all_connections() indexed into it
    with shared_state.track[shared_state.current_index] unconditionally,
    an IndexError crash on an empty list."""
    shared_state.track = []

    delete_all_connections()  # must not raise
