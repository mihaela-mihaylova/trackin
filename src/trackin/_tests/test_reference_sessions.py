"""Reference tests: scripted tracking sessions on a small made-up movie,
checked against what trackin produced when they were recorded (the files
in data/reference/).

They protect the restructuring of the code: if a change alters which
tracks are proposed, or what is written to the session and leftover files,
these tests fail. Each session replays what the plugin does for a key
press or click, without napari's window.

To record them again, on purpose, after an intended change in behaviour:

    python -m trackin._tests.test_reference_sessions
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from trackin import utils
from trackin._widget import generate_positions_list, write_updated_detections_to_file
from trackin.core.io import TRACKING_COLUMNS, check_and_add_displ_cols, read_detections_csv
from trackin.shared_state import SharedState, shared_state

REFERENCE_DIR = Path(__file__).parent / "data" / "reference"
N_FRAMES = 6

# A small made-up movie: three cells moving through 6 frames, with the
# predicted movement to the next frame (displ_y, displ_x). Cell B is not
# detected in frame 3, cell C appears in frame 2, there is one noise
# detection in frame 3, and a "label" column that the tracking doesn't use.
DETECTIONS = pd.DataFrame(
    [
        # tframe, y, x, displ_y, displ_x, label
        (0, 10, 10, 3, 2, "A"), (1, 13, 12, 3, 3, "A"), (2, 16, 15, 3, 2, "A"),
        (3, 19, 17, 3, 3, "A"), (4, 22, 20, 3, 2, "A"), (5, 25, 22, 0, 0, "A"),
        (0, 50, 50, 2, 4, "B"), (1, 52, 54, 3, 3, "B"), (2, 55, 57, 0, 0, "B"),
        (4, 60, 64, 2, 4, "B"), (5, 62, 68, 0, 0, "B"),
        (2, 80, 30, -2, 4, "C"), (3, 78, 34, -3, 3, "C"), (4, 75, 37, -2, 4, "C"),
        (5, 73, 41, 0, 0, "C"),
        (3, 35, 85, 0, 0, "noise"),
    ],
    columns=["tframe", "y", "x", "displ_y", "displ_x", "label"],
)

# The same movie as an already-tracked file (with track_no)
TRACKED_DETECTIONS = DETECTIONS.assign(
    track_no=DETECTIONS["label"].map({"A": 1, "B": 2, "C": 3, "noise": 4})
)


# --- What the plugin does for each action (the tracking part only) --------

def load(detections, folder, max_score=1600, score_func="squared"):
    """What Load Detections does: read the file, build the graph and
    propose the first track."""
    csv_path = folder / "detections.csv"
    detections.to_csv(csv_path, index=False)
    df = check_and_add_displ_cols(read_detections_csv(csv_path))
    shared_state.MAX_SCORE = max_score
    shared_state.SCORE_FUNC = score_func
    shared_state.TRACKED = "track_no" in df.columns
    shared_state.EXTRA_COLUMNS = [c for c in df.columns if c not in TRACKING_COLUMNS]
    shared_state.csv_folder_to_save = str(folder)
    shared_state.SESSION_FILE = "session.csv"
    shared_state.UPDATED_DATA_FILE = "leftover.csv"
    shared_state.DATA = generate_positions_list(df, str(folder), n_frames=N_FRAMES)
    utils.track()


def update_counters():
    """What the panel does after every action: count the detections and
    connections in the graph (the "Detections" and "Connections" shown).
    Shift-Z only works once these are counted."""
    shared_state.NUM_DETS = shared_state.G.number_of_nodes() - 2 - 2 * (len(shared_state.DATA) - 1)
    shared_state.NUM_CONN = shared_state.G.number_of_edges() - 2 * (len(shared_state.DATA) - 1)


def index_at(frame, y, x):
    """Which detection in `frame` is at (y, x)."""
    return next(i for i, d in enumerate(shared_state.DATA[frame]) if (d.y, d.x) == (y, x))


def save_leftover_file():
    """What the plugin does after deleting or adding a detection."""
    write_updated_detections_to_file(
        shared_state.DATA, shared_state.UPDATED_DATA_FILE, shared_state.csv_folder_to_save
    )


def delete_with_d(frame):
    """D: delete the track's detection in `frame`, and propose again."""
    utils.delete_detection(frame, shared_state.track[frame])
    shared_state.track[frame] = -1
    utils.track()
    save_leftover_file()


def delete_with_right_click(frame, index):
    """Right-click on a detection that is not on the track: delete it,
    without proposing again."""
    utils.delete_detection(frame, index)
    save_leftover_file()


# --- The sessions ----------------------------------------------------------

def replay_edits(folder):
    """Every kind of edit on the made-up movie, noting the proposed track
    after each action."""
    steps = []

    def note(action):
        update_counters()
        steps.append({
            "action": action,
            "track": list(shared_state.track or []),
            "detections": shared_state.NUM_DETS,
            "connections": shared_state.NUM_CONN,
        })

    # Load Detections
    load(DETECTIONS, folder)
    note("load")

    # W at frame 2: save the track up to frame 2
    shared_state.current_index = 2
    utils.save_segment()
    note("W at frame 2")

    # Left-click at frame 3: pick the noise detection instead of the proposed one
    shared_state.track[3] = index_at(3, 35, 85)
    note("click noise at frame 3")

    # D at frame 3: delete it again, and get a new proposal
    delete_with_d(3)
    note("D at frame 3")

    # Right-click at frame 2: delete cell C's detection there
    delete_with_right_click(2, index_at(2, 80, 30))
    note("right-click C at frame 2")

    # Shift-click at frame 4: add a detection by hand at (40, 40)
    utils.add_detection(4, 40, 40)
    save_leftover_file()
    note("add at frame 4")

    # X at frame 2: delete the connections after frame 2
    shared_state.current_index = 2
    utils.delete_all_connections()
    note("X at frame 2")

    # Shift-Q: accept the track
    utils.accept_track()
    note("accept")

    # Shift-Z at frame 3: delete the segment up to frame 3
    shared_state.current_index = 3
    utils.delete_segment()
    note("Shift-Z at frame 3")

    # Shift-Q until nothing is left
    while shared_state.track:
        utils.accept_track()
        note("accept")

    return steps


def replay_accept_all(folder, detections, **settings):
    """Load, then accept until nothing is left: the tracking on its own."""
    load(detections, folder, **settings)
    tracks = []
    while shared_state.track:
        tracks.append(list(shared_state.track))
        utils.accept_track()
    return tracks


SESSIONS = {
    "edits": replay_edits,
    "accept_all": lambda folder: replay_accept_all(folder, DETECTIONS),
    "accept_all_linear": lambda folder: replay_accept_all(
        folder, DETECTIONS, max_score=40, score_func="linear"),
    "tracked_file": lambda folder: replay_accept_all(folder, TRACKED_DETECTIONS),
}


# --- Recording and checking ------------------------------------------------

def read_lines(path):
    return path.read_text().splitlines() if path.is_file() else None


def replay(name, folder):
    """Play a session, starting from an empty state, and return everything
    it produced."""
    shared_state.__dict__.update(SharedState().__dict__)
    return {
        "steps": SESSIONS[name](folder),
        "session_file": read_lines(folder / "session.csv"),
        "leftover_file": read_lines(folder / "leftover.csv"),
    }


@pytest.fixture
def fresh_shared_state():
    """Puts shared_state back as it was after the test, as other tests
    share it."""
    original = dict(shared_state.__dict__)
    yield
    shared_state.__dict__.clear()
    shared_state.__dict__.update(original)


@pytest.mark.parametrize("name", SESSIONS)
def test_session_gives_the_recorded_results(name, tmp_path, fresh_shared_state):
    """Replays the session and checks that the proposed tracks and the
    session and leftover files are exactly as recorded. If this fails after
    a change, that change altered what trackin does."""
    recorded = json.loads((REFERENCE_DIR / f"{name}.json").read_text())
    assert replay(name, tmp_path) == recorded


def record():
    """Save what the current code produces as the recording."""
    import tempfile

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name in SESSIONS:
        with tempfile.TemporaryDirectory() as folder:
            result = replay(name, Path(folder))
        (REFERENCE_DIR / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(f"recorded {name}")


if __name__ == "__main__":
    record()
