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
from trackin.plugin_state import PluginState
from trackin._tests.stand_in_widget import plugin_state
from trackin._tests.reference_data import DETECTIONS, N_FRAMES, REFERENCE_DIR, TRACKED_DETECTIONS


# --- What the plugin does for each action (the tracking part only) --------

def load(detections, folder, max_score=1600, score_func="squared"):
    """What Load Detections does: read the file, build the graph and
    propose the first track."""
    csv_path = folder / "detections.csv"
    detections.to_csv(csv_path, index=False)
    df = check_and_add_displ_cols(read_detections_csv(csv_path))
    plugin_state.session = Session.from_dataframe(df, n_frames=N_FRAMES)
    plugin_state.session.max_score = max_score
    plugin_state.session.score_func = score_func
    plugin_state.output_folder = str(folder)
    plugin_state.session_file = "session.csv"
    plugin_state.leftover_file = "leftover.csv"
    (folder / "session.csv").write_text("")  # the empty session file, as the plugin creates it
    plugin_state.session.propose()
    use_proposed_track(stand_in_widget())


def index_at(frame, y, x):
    """Which detection in `frame` is at (y, x)."""
    return next(i for i, d in enumerate(plugin_state.session.detections[frame]) if (d.y, d.x) == (y, x))


def delete_with_d(frame):
    """D: delete the track's detection in `frame`, and propose again."""
    plugin_state.session.delete_detection(frame, plugin_state.track[frame])
    plugin_state.track[frame] = -1
    plugin_state.session.propose()
    use_proposed_track(stand_in_widget())
    save_leftover_file(stand_in_widget())


def delete_with_right_click(frame, index):
    """Right-click on a detection that is not on the track: delete it,
    without proposing again."""
    plugin_state.session.delete_detection(frame, index)
    save_leftover_file(stand_in_widget())


# --- The sessions ----------------------------------------------------------

def replay_edits(folder):
    """Every kind of edit on the made-up movie, noting the proposed track
    after each action."""
    steps = []

    def note(action):
        counts = plugin_state.session.stats()  # the "Detections" and "Connections" shown in the panel
        steps.append({
            "action": action,
            "track": list(plugin_state.track or []),
            "detections": counts["detections"],
            "connections": counts["connections"],
        })

    # Load Detections
    load(DETECTIONS, folder)
    note("load")

    # W at frame 2: save the track up to frame 2
    plugin_state.session.save_segment(plugin_state.track, 2)
    use_proposed_track(stand_in_widget())
    note("W at frame 2")

    # Left-click at frame 3: pick the noise detection instead of the proposed one
    plugin_state.track[3] = index_at(3, 35, 85)
    note("click noise at frame 3")

    # D at frame 3: delete it again, and get a new proposal
    delete_with_d(3)
    note("D at frame 3")

    # Right-click at frame 2: delete cell C's detection there
    delete_with_right_click(2, index_at(2, 80, 30))
    note("right-click C at frame 2")

    # Shift-click at frame 4: add a detection by hand at (40, 40)
    plugin_state.session.add_detection(4, 40, 40)
    save_leftover_file(stand_in_widget())
    note("add at frame 4")

    # X at frame 2: delete the connections after frame 2
    plugin_state.session.delete_all_connections(plugin_state.track, 2)
    note("X at frame 2")

    # Shift-Q: accept the track
    plugin_state.session.accept_track(plugin_state.track)
    save_accepted_track(stand_in_widget())
    save_leftover_file(stand_in_widget())
    use_proposed_track(stand_in_widget())
    note("accept")

    # Shift-Z at frame 3: delete the segment up to frame 3
    plugin_state.session.delete_segment(plugin_state.track, 3)
    save_leftover_file(stand_in_widget())
    use_proposed_track(stand_in_widget())
    note("Shift-Z at frame 3")

    # Shift-Q until nothing is left
    while plugin_state.track:
        plugin_state.session.accept_track(plugin_state.track)
        save_accepted_track(stand_in_widget())
        save_leftover_file(stand_in_widget())
        use_proposed_track(stand_in_widget())
        note("accept")

    return steps


def replay_accept_all(folder, detections, **settings):
    """Load, then accept until nothing is left: the tracking on its own."""
    load(detections, folder, **settings)
    tracks = []
    while plugin_state.track:
        tracks.append(list(plugin_state.track))
        plugin_state.session.accept_track(plugin_state.track)
        save_accepted_track(stand_in_widget())
        save_leftover_file(stand_in_widget())
        use_proposed_track(stand_in_widget())
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
    plugin_state.__dict__.update(PluginState().__dict__)
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
