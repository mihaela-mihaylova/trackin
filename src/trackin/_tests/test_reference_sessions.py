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

import pytest

from trackin._widget import save_accepted_track, save_leftover_file, use_proposed_track
from trackin.core.io import check_and_add_displ_cols, read_detections_csv
from trackin.core.session import Session
from trackin.shared_state import SharedState, shared_state
from trackin._tests.reference_data import DETECTIONS, N_FRAMES, REFERENCE_DIR, TRACKED_DETECTIONS


# --- What the plugin does for each action (the tracking part only) --------

def load(detections, folder, max_score=1600, score_func="squared"):
    """What Load Detections does: read the file, build the graph and
    propose the first track."""
    csv_path = folder / "detections.csv"
    detections.to_csv(csv_path, index=False)
    df = check_and_add_displ_cols(read_detections_csv(csv_path))
    shared_state.session = Session.from_dataframe(df, n_frames=N_FRAMES)
    shared_state.session.max_score = max_score
    shared_state.session.score_func = score_func
    shared_state.output_folder = str(folder)
    shared_state.session_file = "session.csv"
    shared_state.leftover_file = "leftover.csv"
    (folder / "session.csv").write_text("")  # the empty session file, as the plugin creates it
    shared_state.session.propose()
    use_proposed_track()


def index_at(frame, y, x):
    """Which detection in `frame` is at (y, x)."""
    return next(i for i, d in enumerate(shared_state.session.detections[frame]) if (d.y, d.x) == (y, x))


def delete_with_d(frame):
    """D: delete the track's detection in `frame`, and propose again."""
    shared_state.session.delete_detection(frame, shared_state.track[frame])
    shared_state.track[frame] = -1
    shared_state.session.propose()
    use_proposed_track()
    save_leftover_file()


def delete_with_right_click(frame, index):
    """Right-click on a detection that is not on the track: delete it,
    without proposing again."""
    shared_state.session.delete_detection(frame, index)
    save_leftover_file()


# --- The sessions ----------------------------------------------------------

def replay_edits(folder):
    """Every kind of edit on the made-up movie, noting the proposed track
    after each action."""
    steps = []

    def note(action):
        counts = shared_state.session.stats()  # the "Detections" and "Connections" shown in the panel
        steps.append({
            "action": action,
            "track": list(shared_state.track or []),
            "detections": counts["detections"],
            "connections": counts["connections"],
        })

    # Load Detections
    load(DETECTIONS, folder)
    note("load")

    # W at frame 2: save the track up to frame 2
    shared_state.session.save_segment(shared_state.track, 2)
    use_proposed_track()
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
    shared_state.session.add_detection(4, 40, 40)
    save_leftover_file()
    note("add at frame 4")

    # X at frame 2: delete the connections after frame 2
    shared_state.session.delete_all_connections(shared_state.track, 2)
    note("X at frame 2")

    # Shift-Q: accept the track
    shared_state.session.accept_track(shared_state.track)
    save_accepted_track()
    save_leftover_file()
    use_proposed_track()
    note("accept")

    # Shift-Z at frame 3: delete the segment up to frame 3
    shared_state.session.delete_segment(shared_state.track, 3)
    save_leftover_file()
    use_proposed_track()
    note("Shift-Z at frame 3")

    # Shift-Q until nothing is left
    while shared_state.track:
        shared_state.session.accept_track(shared_state.track)
        save_accepted_track()
        save_leftover_file()
        use_proposed_track()
        note("accept")

    return steps


def replay_accept_all(folder, detections, **settings):
    """Load, then accept until nothing is left: the tracking on its own."""
    load(detections, folder, **settings)
    tracks = []
    while shared_state.track:
        tracks.append(list(shared_state.track))
        shared_state.session.accept_track(shared_state.track)
        save_accepted_track()
        save_leftover_file()
        use_proposed_track()
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


@pytest.mark.parametrize("name", SESSIONS)
def test_session_gives_the_recorded_results(name, tmp_path):
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
