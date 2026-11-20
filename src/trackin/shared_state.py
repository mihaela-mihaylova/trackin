"""The state the plugin shares between its parts: what the screen shows,
and the tracking session it shows."""

from .core.session import Session, NO_VALUE, csv_value


class SharedState:
    def __init__(self):
        self.session = Session()    # the tracking: data and actions
        self.current_index = 0      # the frame shown
        self.track = []             # the track being edited: a copy of the session's proposal, with the user's clicks
        self.track_lines = None     # the lines drawn for the current track
        self.NUM_DETS = None        # the "Detections" number shown
        self.NUM_CONN = None        # the "Connections" number shown


shared_state = SharedState()
