"""A stand-in for the Trackin widget, for tests of the plugin's functions."""

from types import SimpleNamespace

from trackin._navigation import Navigator
from trackin.plugin_state import plugin_state


def stand_in_widget():
    """A stand-in for the Trackin widget, with the parts the plugin's
    functions use: the state, the viewer, its own Navigator, and
    focus/refresh_stats doing nothing. A real widget would need a napari
    window, which the test machines don't all have."""
    w = SimpleNamespace(state=plugin_state, viewer=plugin_state.viewer,
                        focus=lambda: None, refresh_stats=lambda: None)
    w.navigator = Navigator(w)
    return w
