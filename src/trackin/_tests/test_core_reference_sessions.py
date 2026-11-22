"""The reference sessions again, played on the tracking core alone: each
replay creates its own Session, as a program using trackin's core would,
without napari or the plugin's shared state. The results must be exactly
the recorded ones (data/reference/), like the plugin's reference tests."""

import json

import pytest

from trackin.core.io import append_to_session_file, check_and_add_displ_cols, read_detections_csv, write_leftover_file
from trackin.core.session import Session
from trackin._tests.reference_data import DETECTIONS, N_FRAMES, REFERENCE_DIR, TRACKED_DETECTIONS


def load(detections, folder, max_score=1600, score_func="squared"):
    """A session for the detections file, with its first proposal."""
    csv_path = folder / "detections.csv"
    detections.to_csv(csv_path, index=False)
    session = Session.from_dataframe(
        check_and_add_displ_cols(read_detections_csv(csv_path)), n_frames=N_FRAMES)
    session.max_score = max_score
    session.score_func = score_func
    session.propose()
    return session


def save_accepted_track(session, folder):
    """Append the track just accepted to the session file, as a program
    using the core does after accepting."""
    append_to_session_file(folder / "session.csv", len(session.accepted), session.accepted[-1],
                           session.detections, session.extra_columns)


def save_leftover_file(session, folder):
    """Write the session's available detections to the leftover file, as a
    program using the core does after accepting or deleting a segment."""
    write_leftover_file(folder / "leftover.csv", session.detections, session.extra_columns)


def copy_of_proposal(session):
    """The track a user edits: a copy of the session's proposal."""
    return None if session.proposal is None else list(session.proposal)


def index_at(session, frame, y, x):
    """Which detection in `frame` is at (y, x)."""
    return next(i for i, d in enumerate(session.detections[frame]) if (d.y, d.x) == (y, x))


def replay_edits(folder):
    """Every kind of edit on the made-up movie, noting the edited track
    after each action."""
    session = load(DETECTIONS, folder)
    track = copy_of_proposal(session)
    steps = []

    def note(action):
        counts = session.stats()
        steps.append({"action": action, "track": list(track or []),
                      "detections": counts["detections"], "connections": counts["connections"]})

    note("load")

    # W at frame 2
    session.save_segment(track, 2)
    track = copy_of_proposal(session)
    note("W at frame 2")

    # Left-click at frame 3: the noise detection instead of the proposed one
    track[3] = index_at(session, 3, 35, 85)
    note("click noise at frame 3")

    # D at frame 3: delete it, and get a new proposal
    session.delete_detection(3, track[3])
    session.propose()
    track = copy_of_proposal(session)
    note("D at frame 3")

    # Right-click at frame 2: delete cell C's detection there (no new proposal)
    session.delete_detection(2, index_at(session, 2, 80, 30))
    note("right-click C at frame 2")

    # Shift-click at frame 4: add a detection by hand
    session.add_detection(4, 40, 40)
    note("add at frame 4")

    # X at frame 2
    session.delete_all_connections(track, 2)
    note("X at frame 2")

    # Shift-Q
    session.accept_track(track)
    save_accepted_track(session, folder)
    save_leftover_file(session, folder)
    track = copy_of_proposal(session)
    note("accept")

    # Shift-Z at frame 3
    session.delete_segment(track, 3)
    save_leftover_file(session, folder)
    track = copy_of_proposal(session)
    note("Shift-Z at frame 3")

    # Shift-Q until nothing is left
    while track:
        session.accept_track(track)
        save_accepted_track(session, folder)
        save_leftover_file(session, folder)
        track = copy_of_proposal(session)
        note("accept")

    return steps


def replay_accept_all(folder, detections, **settings):
    """Load, then accept until nothing is left."""
    session = load(detections, folder, **settings)
    tracks = []
    while session.proposal:
        tracks.append(list(session.proposal))
        session.accept_track(session.proposal)
        save_accepted_track(session, folder)
        save_leftover_file(session, folder)
    return tracks


SESSIONS = {
    "edits": replay_edits,
    "accept_all": lambda folder: replay_accept_all(folder, DETECTIONS),
    "accept_all_linear": lambda folder: replay_accept_all(
        folder, DETECTIONS, max_score=40, score_func="linear"),
    "tracked_file": lambda folder: replay_accept_all(folder, TRACKED_DETECTIONS),
}


def read_lines(path):
    return path.read_text().splitlines() if path.is_file() else None


@pytest.mark.parametrize("name", SESSIONS)
def test_core_session_gives_the_recorded_results(name, tmp_path):
    """Replays the session on the core alone and checks that the tracks,
    the counts and the session and leftover files are exactly as recorded."""
    result = {
        "steps": SESSIONS[name](tmp_path),
        "session_file": read_lines(tmp_path / "session.csv"),
        "leftover_file": read_lines(tmp_path / "leftover.csv"),
    }
    recorded = json.loads((REFERENCE_DIR / f"{name}.json").read_text())
    assert result == recorded
