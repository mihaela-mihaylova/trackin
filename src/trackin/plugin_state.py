"""Everything the plugin keeps between clicks: the tracking session it
shows, what the screen shows, and the files the plugin writes for it."""

from .core.session import Session


class PluginState:
    def __init__(self):
        self.viewer = None          # the napari viewer the panel is in
        self.images = []            # the loaded images (with a white border), in frame order
        self.image_files = []       # their file names, in the same order
        self.images_loaded = False  # a folder of images has been loaded
        self.csv_loaded = False     # a detections file has been loaded
        # napari's camera has no zoom limits, so scrolling could zoom out until
        # the view is a blank speck. These are set from each image folder's
        # fit-to-view zoom when it is loaded.
        self.min_zoom = 0.05
        self.max_zoom = 100.0
        self.session = Session()    # the tracking: data and actions
        self.current_frame = 0      # the frame shown
        self.track = []             # the track being edited: a copy of the session's proposal, with the user's clicks
        self.n_detections_shown = None        # the "Detections" number shown
        self.n_connections_shown = None        # the "Connections" number shown
        self.clear_files()

    def clear_files(self):
        """No output files: the state before a detections file is loaded.
        The blank names make any stray write fail loudly (e.g.
        IsADirectoryError) rather than write into an earlier dataset's files."""
        self.output_folder = ''     # the detections file's folder, where all output files go
        self.session_file = ''      # the tracks accepted in this session
        self.leftover_file = ''     # the detections not yet used
        self.combined_file = ''     # a loaded track file's tracks plus this session's ('' until one is added)
        # the combined file's extra columns: the track file's, then the detections file's
        self.combined_extra_columns = []
        # the highest track number in the loaded track file; None until one is added
        self.max_prior_track_id = None


plugin_state = PluginState()
