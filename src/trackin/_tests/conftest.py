import pytest

from trackin.plugin_state import PluginState, plugin_state


@pytest.fixture(autouse=True)
def fresh_plugin_state():
    """Every test starts with an empty tracking session and the screen state
    as at startup, and plugin_state is put back as it was afterwards, as it
    is shared by all tests."""
    original = dict(plugin_state.__dict__)
    plugin_state.__dict__.update(PluginState().__dict__)
    yield
    plugin_state.__dict__.clear()
    plugin_state.__dict__.update(original)
