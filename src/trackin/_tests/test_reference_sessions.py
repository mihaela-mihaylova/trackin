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

from trackin._interaction import save_accepted_track, save_leftover_file, use_proposed_track
from trackin._tests.stand_in_widget import stand_in_widget
from trackin.core.io import check_and_add_displ_cols, read_detections_csv
from trackin.core.session import Session
from trackin._tests.reference_data import DETECTIONS, N_FRAMES, REFERENCE_DIR, TRACKED_DETECTIONS


# --- What the plugin does for each action (the tracking part only) --------

def load(w, detections, folder, max_score=1600, score_func="squared"):
    """What Load Detections does: read the file, build the graph and
    propose the first track."""
    csv_path = folder / "detections.csv"
    detections.to_csv(csv_path, index=False)
    df = check_and_add_displ_cols(read_detections_csv(csv_path))
    w.state.session = Session.from_dataframe(df, n_frames=N_FRAMES)
    w.state.session.max_score = max_score
    w.state.session.score_func = score_func
    w.state.output_folder = str(folder)
    w.state.session_file = "session.csv"
    w.state.leftover_file = "leftover.csv"
    (folder / "session.csv").write_text("")  # the empty session file, as the plugin creates it
    w.state.session.propose()
    use_proposed_track(w)


def index_at(w, frame, y, x):
    """Which detection in `frame` is at (y, x)."""
    return next(i for i, d in enumerate(w.state.session.detections[frame]) if (d.y, d.x) == (y, x))


def delete_with_d(w, frame):
    """D: delete the track's detection in `frame`, and propose again."""
    w.state.session.delete_detection(frame, w.state.track[frame])
    w.state.track[frame] = -1
    w.state.session.propose()
    use_proposed_track(w)
    save_leftover_file(w)


def delete_with_right_click(w, frame, index):
    """Right-click on a detection that is not on the track: delete it,
    without proposing again."""
    w.state.session.delete_detection(frame, index)
    save_leftover_file(w)


# --- The sessions ----------------------------------------------------------

def replay_edits(w, folder):
    """Every kind of edit on the made-up movie, noting the proposed track
    after each action."""
    steps = []

    def note(action):
        counts = w.state.session.stats()  # the "Detections" and "Connections" shown in the panel
        steps.append({
            "action": action,
            "track": list(w.state.track or []),
            "detections": counts["detections"],
            "connections": counts["connections"],
        })

    # Load Detections
    load(w, DETECTIONS, folder)
    note("load")

    # W at frame 2: save the track up to frame 2
    w.state.session.save_segment(w.state.track, 2)
    use_proposed_track(w)
    note("W at frame 2")

    # Left-click at frame 3: pick the noise detection instead of the proposed one
    w.state.track[3] = index_at(w, 3, 35, 85)
    note("click noise at frame 3")

    # D at frame 3: delete it again, and get a new proposal
    delete_with_d(w, 3)
    note("D at frame 3")

    # Right-click at frame 2: delete cell C's detection there
    delete_with_right_click(w, 2, index_at(w, 2, 80, 30))
    note("right-click C at frame 2")

    # Shift-click at frame 4: add a detection by hand at (40, 40)
    w.state.session.add_detection(4, 40, 40)
    save_leftover_file(w)
    note("add at frame 4")

    # X at frame 2: delete the connections after frame 2
    w.state.session.delete_all_connections(w.state.track, 2)
    note("X at frame 2")

    # Shift-Q: accept the track
    w.state.session.accept_track(w.state.track)
    save_accepted_track(w)
    save_leftover_file(w)
    use_proposed_track(w)
    note("accept")

    # Shift-Z at frame 3: delete the segment up to frame 3
    w.state.session.delete_segment(w.state.track, 3)
    save_leftover_file(w)
    use_proposed_track(w)
    note("Shift-Z at frame 3")

    # Shift-Q until nothing is left
    while w.state.track:
        w.state.session.accept_track(w.state.track)
        save_accepted_track(w)
        save_leftover_file(w)
        use_proposed_track(w)
        note("accept")

    return steps


def replay_accept_all(w, folder, detections, **settings):
    """Load, then accept until nothing is left: the tracking on its own."""
    load(w, detections, folder, **settings)
    tracks = []
    while w.state.track:
        tracks.append(list(w.state.track))
        w.state.session.accept_track(w.state.track)
        save_accepted_track(w)
        save_leftover_file(w)
        use_proposed_track(w)
    return tracks


SESSIONS = {
    "edits": replay_edits,
    "accept_all": lambda w, folder: replay_accept_all(w, folder, DETECTIONS),
    "accept_all_linear": lambda w, folder: replay_accept_all(
        w, folder, DETECTIONS, max_score=40, score_func="linear"),
    "tracked_file": lambda w, folder: replay_accept_all(w, folder, TRACKED_DETECTIONS),
}


# --- Recording and checking ------------------------------------------------

def read_lines(path):
    return path.read_text().splitlines() if path.is_file() else None


def replay(name, folder):
    """Play a session, starting from an empty state, and return everything
    it produced."""
    w = stand_in_widget()
    return {
        "steps": SESSIONS[name](w, folder),
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
