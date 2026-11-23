"""A stand-in for the Trackin widget, for tests of the plugin's functions."""

from types import MethodType, SimpleNamespace

from trackin._navigation import Navigator
from trackin._session_files import ElidedPathLabel, SessionFilesPanel
from trackin._widget import TrackinWidget
from trackin.plugin_state import PluginState

# The state the stand-in widgets use; conftest.py resets it before every test
plugin_state = PluginState()


def stand_in_widget(viewer=None):
    """A stand-in for the Trackin widget, with the parts the plugin's
    functions use: the tests' state, the given viewer, its own Navigator,
    track file label and Session Files panel, the widget's own
    update_file_paths_display and clear_detections_and_tracks, and
    focus/refresh_stats doing nothing. A real widget would need a napari
    window, which the test machines don't all have."""
    w = SimpleNamespace(state=plugin_state, viewer=viewer,
                        focus=lambda: None, refresh_stats=lambda: None)
    w.navigator = Navigator(w)
    w.track_file_label = ElidedPathLabel()
    w.session_files = SessionFilesPanel()
    w.update_file_paths_display = MethodType(TrackinWidget.update_file_paths_display, w)
    w.clear_detections_and_tracks = MethodType(TrackinWidget.clear_detections_and_tracks, w)
    return w
