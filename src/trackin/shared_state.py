# this files declares all the variables that are shared between widgets and utils
# this allows for sharing this variables in both files and avoding circular imports

import math

# value written in the extra columns for a detection added by hand, which
# didn't come from the detections file and so has no values of its own there
NO_VALUE = -1

# shared_state.py
class SharedState:
    def __init__(self):
        self.track = []
        self.track_lines = None
        self.G = None
        self.DATA = []
        # number of detections that have been in frame (orig dets num + any added ones, used for D-nodes generation
        self.NUM_DET_PER_FRAME = []
        self.TRACKED = False
        self.MAX_SCORE = 1600
        self.SCORE_FUNC = 'squared'
        self.N_TRACKS = 0
        self.SESSION_FILE = ''
        self.UPDATED_DATA_FILE = ''
        self.csv_folder_to_save = ''
        self.current_index = 0
        self.UPD_TRACK_FILE = ''
        self.MAX_TRACK_ID = None
        self.NUM_DETS = None
        self.NUM_CONN = None
        # columns of the detections file that the tracking doesn't use (e.g.
        # frame, label, area), in their original order; they are written back
        # out after the standard columns in every output file
        self.EXTRA_COLUMNS = []
        # values of those columns for each detection, indexed like DATA:
        # EXTRA[t][i] belongs to DATA[t][i]
        self.EXTRA = []

    def extra_header(self):
        """The extra columns' names for a CSV header row, each preceded by a
        comma, or "" if there are none."""
        return "".join("," + csv_value(c) for c in self.EXTRA_COLUMNS)

    def extra_values(self, t, i):
        """The extra columns' values of detection i in frame t for a CSV row,
        each preceded by a comma, or "" if there are none."""
        if not self.EXTRA_COLUMNS:
            return ""
        return "".join("," + csv_value(v) for v in self.EXTRA[t][i])

def csv_value(value):
    """A value as written in a CSV cell: empty for a missing value, and in
    quotes if it contains a comma, quote or line break, so it stays one cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if any(char in text for char in ',"\n'):
        text = '"' + text.replace('"', '""') + '"'
    return text

shared_state = SharedState()
