import pytest

from trackin.core.detection import Detection
from trackin._interaction import save_accepted_track
from trackin._tests.stand_in_widget import stand_in_widget
from trackin.plugin_state import plugin_state
from trackin.core.tracking import generate_graph

def accept_and_save(track):
    """Accept `track`, then write it to the session file (and the combined
    file), as the plugin does after Shift-Q."""
    n_accepted = len(plugin_state.session.accepted)
    messages = plugin_state.session.accept_track(track)
    if len(plugin_state.session.accepted) > n_accepted:
        save_accepted_track(stand_in_widget())
    return messages


@pytest.fixture
def staged_plugin_state(tmp_path):
    """Seed the session with a minimal two-frame, two-detection scenario
    for accept_track()."""

    plugin_state.session.detections = [[Detection(10, 20)], [Detection(30, 40)]]
    plugin_state.session.proposal = [0, 0]  # accept the single detection in each frame
    plugin_state.output_folder = str(tmp_path)
    plugin_state.session_file = "session.csv"
    plugin_state.leftover_file = "updated.csv"
    plugin_state.session.has_prior_tracks = False
    plugin_state.max_prior_track_id = None  # no prior track file loaded -> skip the combined file
    plugin_state.session.graph = generate_graph(
        plugin_state.session.detections, max_score=1600, score_func="squared", tracked=False
    )

    yield tmp_path


def test_accept_track_writes_session_file_and_marks_detections_consumed(
    staged_plugin_state,
):
    tmp_path = staged_plugin_state

    accept_and_save(plugin_state.session.proposal)

    session_lines = (tmp_path / "session.csv").read_text().splitlines()
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,10,20,0,0,1"
    assert session_lines[2] == "1,30,40,0,0,1"

    # both detections used: kept with their position, but no longer available
    assert plugin_state.session.detections[0][0] == Detection(10, 20, active=False)
    assert plugin_state.session.detections[1][0] == Detection(30, 40, active=False)


def test_accept_track_does_not_duplicate_header_when_session_file_has_content(
    staged_plugin_state,
):
    tmp_path = staged_plugin_state
    session_path = tmp_path / "session.csv"
    session_path.write_text("tframe,y,x,displ_y,displ_x,track_no\n0,1,2,0,0,1\n")

    accept_and_save(plugin_state.session.proposal)

    session_lines = session_path.read_text().splitlines()
    assert session_lines.count("tframe,y,x,displ_y,displ_x,track_no") == 1
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,1,2,0,0,1"  # pre-existing row untouched
    assert session_lines[2] == "0,10,20,0,0,1"  # newly accepted row appended
    assert session_lines[3] == "1,30,40,0,0,1"


def test_accept_track_adds_header_when_session_file_exists_but_empty(
    staged_plugin_state,
):
    tmp_path = staged_plugin_state
    session_path = tmp_path / "session.csv"
    session_path.touch()  # exists on disk, but 0 bytes

    accept_and_save(plugin_state.session.proposal)

    session_lines = session_path.read_text().splitlines()
    assert session_lines[0] == "tframe,y,x,displ_y,displ_x,track_no"
    assert session_lines[1] == "0,10,20,0,0,1"
    assert session_lines[2] == "1,30,40,0,0,1"


def test_accept_track_writes_upd_track_file_with_offset_track_id(staged_plugin_state):
    tmp_path = staged_plugin_state
    plugin_state.max_prior_track_id = 5
    plugin_state.combined_file = "upd_tracks.csv"

    accept_and_save(plugin_state.session.proposal)

    # updated_track_id = max_prior_track_id + this track's number = 5 + 1
    # (it's the first track accepted in this session). Unlike the session file, this
    # file has no header -- it's opened in append mode and rows are written as-is.
    upd_track_lines = (tmp_path / "upd_tracks.csv").read_text().splitlines()
    assert upd_track_lines == [
        "0,10,20,0,0,6",
        "1,30,40,0,0,6",
    ]


def test_accept_track_no_op_when_track_is_empty_list(staged_plugin_state):
    """Right after images are reloaded (before a new CSV is loaded),
    plugin_state.session.proposal is [] rather than None -- accept_track() must treat
    that the same way, not silently create/write into the *previous*
    dataset's session file."""
    tmp_path = staged_plugin_state
    plugin_state.session.proposal = []

    accept_and_save(plugin_state.session.proposal)

    assert not (tmp_path / "session.csv").exists()
    assert plugin_state.session.accepted == []  # nothing accepted on a no-op


def test_delete_segment_no_op_when_track_is_empty_list_even_with_stale_num_dets(
    staged_plugin_state,
):
    """Right after images are reloaded there is no track, while the panel's
    counter (n_detections_shown) can still hold a value from the *previous* dataset.
    plugin_state.session.delete_segment() must do nothing then, and not truncate-and-overwrite
    that dataset's leftover file."""
    tmp_path = staged_plugin_state
    plugin_state.session.proposal = []
    plugin_state.n_detections_shown = 5  # stale nonzero value

    updated_path = tmp_path / "updated.csv"
    updated_path.write_text("tframe,y,x,displ_y,displ_x\n0,99,99,0,0\n")

    plugin_state.session.delete_segment(plugin_state.session.proposal, 0)

    assert updated_path.read_text() == "tframe,y,x,displ_y,displ_x\n0,99,99,0,0\n"


def test_save_segment_no_op_when_track_is_empty_list(staged_plugin_state):
    """Right after images are reloaded (before a new CSV is loaded),
    plugin_state.session.proposal is [] and plugin_state.session.graph is None. The segment-
    building part is a harmless no-op on an empty track either way, but
    plugin_state.session.save_segment() unconditionally ended with propose_next(), which calls
    .copy() on plugin_state.session.graph -- an AttributeError crash when there is no graph."""
    plugin_state.session.proposal = []
    plugin_state.session.graph = None

    plugin_state.session.save_segment(plugin_state.session.proposal, 0)  # must not raise

    assert plugin_state.session.proposal == []  # propose_next() never ran to reassign it


def test_delete_all_connections_no_op_when_track_is_empty_list(staged_plugin_state):
    """Right after images are reloaded (before a new CSV is loaded),
    plugin_state.session.proposal is [] -- delete_all_connections() indexed into it
    with plugin_state.session.proposal[plugin_state.current_frame] unconditionally,
    an IndexError crash on an empty list."""
    plugin_state.session.proposal = []

    plugin_state.session.delete_all_connections(plugin_state.session.proposal, 0)  # must not raise


def test_accept_track_returns_messages_for_the_user(staged_plugin_state):
    """accept_track() returns the messages to show instead of showing them
    itself: "Accepted track.", and here also "No detections left." because
    the accepted track used every detection, so no new track is left."""
    assert plugin_state.session.accept_track(plugin_state.session.proposal) == ["Accepted track.", "No detections left."]


def test_actions_without_a_track_return_no_detections_left(staged_plugin_state):
    """Without a proposed track (e.g. right after loading new images), each
    action does nothing and returns the message "No detections left." for
    the plugin to show."""
    plugin_state.session.proposal = []

    assert plugin_state.session.accept_track(plugin_state.session.proposal) == ["No detections left."]
    for action in (plugin_state.session.save_segment, plugin_state.session.delete_segment, plugin_state.session.delete_all_connections):
        assert action(plugin_state.session.proposal, 0) == ["No detections left."]


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


def test_delete_segment_works_without_the_panel_counting(staged_plugin_state):
    """Shift-Z must not depend on the panel's "Detections" counter
    (n_detections_shown), which is only set when the panel is shown. Before,
    it did nothing while that counter was unset, e.g. when the tracking is used without
    the plugin's panel."""
    tmp_path = staged_plugin_state
    plugin_state.n_detections_shown = None  # the panel hasn't counted anything

    assert plugin_state.session.delete_segment(plugin_state.session.proposal, 1) == ["No detections left."]  # nothing left to propose
    assert plugin_state.session.detections[0][0].active is False  # both detections deleted
    assert plugin_state.session.detections[1][0].active is False


def test_accept_track_keeps_the_accepted_tracks(staged_plugin_state):
    """The session keeps each accepted track, in order; a track's number in
    the session file is its place in that list (1 for the first)."""
    tmp_path = staged_plugin_state
    plugin_state.session.detections = [[Detection(10, 20), Detection(50, 50)], [Detection(30, 40), Detection(60, 60)]]
    plugin_state.session.graph = generate_graph(
        plugin_state.session.detections, max_score=1600, score_func="squared", tracked=False
    )

    accept_and_save([0, 0])
    accept_and_save([1, 1])

    assert plugin_state.session.accepted == [[0, 0], [1, 1]]
    lines = (tmp_path / "session.csv").read_text().splitlines()
    assert [line.split(",")[5] for line in lines[1:]] == ["1", "1", "2", "2"]  # track_no column
