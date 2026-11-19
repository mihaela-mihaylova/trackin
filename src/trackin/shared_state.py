"""The state the plugin shares between its parts: the tracking session,
plus what the screen shows."""

from .core.session import Session, NO_VALUE, csv_value


class SharedState(Session):
    def __init__(self):
        super().__init__()
        self.current_index = 0      # the frame shown
        self.track_lines = None     # the lines drawn for the current track
        self.NUM_DETS = None        # the "Detections" number shown
        self.NUM_CONN = None        # the "Connections" number shown


shared_state = SharedState()
