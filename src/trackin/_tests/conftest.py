import pytest

from trackin.shared_state import SharedState, shared_state


@pytest.fixture(autouse=True)
def fresh_shared_state():
    """Every test starts with an empty tracking session and the screen state
    as at startup, and shared_state is put back as it was afterwards, as it
    is shared by all tests."""
    original = dict(shared_state.__dict__)
    shared_state.__dict__.update(SharedState().__dict__)
    yield
    shared_state.__dict__.clear()
    shared_state.__dict__.update(original)
