# this files declares all the variables that are shared between widgets and utils
# this allows for sharing this variables in both files and avoding circular imports

# shared_state.py
class SharedState:
    def __init__(self):
        self.track = []
        self.track_dict = {}
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

shared_state = SharedState()