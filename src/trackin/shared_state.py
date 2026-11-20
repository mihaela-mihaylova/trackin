"""The state the plugin shares between its parts: what the screen shows,
and the tracking session it shows."""

from .core.io import NO_VALUE, csv_value
from .core.session import Session


class SharedState:
    def __init__(self):
        self.session = Session()    # the tracking: data and actions
        self.current_frame = 0      # the frame shown
        self.track = []             # the track being edited: a copy of the session's proposal, with the user's clicks
        self.track_lines = None     # the lines drawn for the current track
        self.n_detections_shown = None        # the "Detections" number shown
        self.n_connections_shown = None        # the "Connections" number shown


shared_state = SharedState()
