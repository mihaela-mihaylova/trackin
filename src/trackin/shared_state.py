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
        self.TRACKED = False
        self.MAX_SCORE = 1600
        self.SCORE_FUNC = 'squared'
        self.N_TRACKS = 0

shared_state = SharedState()