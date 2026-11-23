import os
from functools import partial
from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from qtpy.QtCore import Qt
import napari
import numpy as np
from .plugin_state import plugin_state
from . import _loading
from ._help_dialog import show_help
from ._session_files import ElidedPathLabel, SessionFilesPanel
from ._navigation import Navigator
from ._viewer_setup import initialize_viewer
from .core.session import Session

class TrackinWidget(QWidget):
    """Dock widget for the plugin interface. napari's dock-widget injection
    only introspects a widget *class's* __init__ signature for a
    napari_viewer parameter (a plain function's own .__init__ is just the
    generic one every function object inherits, so this used to be a bare
    function and never actually received the real viewer -- see
    _instantiate_dock_widget in napari's qt_main_window.py). napari
    automatically passes in the currently active viewer when this is
    instantiated as a dock widget via the Plugins menu."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.state = plugin_state
        self.navigator = Navigator(self)
        initialize_viewer(self)

        layout = QVBoxLayout(self)  # Create a vertical layout

        # Small round help button, opens a dialog listing keyboard/mouse shortcuts
        help_button = QPushButton("?")
        help_button.setFixedSize(24, 24)
        help_button.setToolTip("Help")
        help_button.setCursor(Qt.PointingHandCursor)
        help_button.setStyleSheet(
            "QPushButton {"
            "  background-color: #2b7de9;"
            "  color: white;"
            "  border: none;"
            "  border-radius: 12px;"
            "  font-weight: bold;"
            "}"
            "QPushButton:hover { background-color: #4a90f0; }"
            "QPushButton:pressed { background-color: #1c5fc2; }"
        )
        help_button.clicked.connect(show_help)
        layout.addWidget(help_button, alignment=Qt.AlignRight)

        # The three load buttons, each calling its function with this widget
        self.load_images_button = magicgui(
            partial(_loading.choose_folder, self), call_button="Load Images", auto_call=True)
        self.load_images_button.call_button.native.setToolTip(
            "Select a folder of image frames.\n"
            "Files must be named with numeric filenames (e.g. 0.png, 1.png, ...),\n"
            "one per frame, using a supported format: jpg, jpeg, png, tif, tiff, bmp, gif."
        )
        self.load_detections_button = magicgui(
            partial(_loading.load_csv, self), call_button="Load Detections", auto_call=True)
        self.load_detections_button.call_button.native.setToolTip(
            "Select a CSV of detections for the loaded image frames.\n"
            "Required columns: tframe, y, x.\n"
            "Optional columns: displ_y, displ_x (default to 0 if omitted),\n"
            "and track_no (include it if these detections are already tracked).\n"
            "Any other columns (e.g. label, area) are kept and written to the\n"
            "output files; detections added by hand get -1 in them."
        )
        self.add_track_file_button = magicgui(
            partial(_loading.load_track_file, self), call_button="Add Track File", auto_call=True)
        self.add_track_file_button.call_button.native.setToolTip(
            "Select a CSV of tracks accepted in a previous session, so new tracks\n"
            "continue numbering from where that session left off.\n"
            "Must start with these columns, in this order:\n"
            "tframe, y, x, displ_y, displ_x, track_no.\n"
            "Any further columns (e.g. label, area) are kept."
        )
        for button in (self.load_images_button, self.load_detections_button, self.add_track_file_button):
            layout.addWidget(button.native)

        # Shows the loaded track file's name once "Add Track File" succeeds,
        # since that success is otherwise only reported via a transient toast
        self.track_file_label = ElidedPathLabel()
        self.track_file_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.track_file_label)

        layout.addWidget(self.navigator.image_slider.native)  # Add the slider widget

        # The Session Files panel with the output files' paths, hidden until a
        # detections file is loaded; update_file_paths_display() fills it in
        self.session_files = SessionFilesPanel()
        layout.addWidget(self.session_files)

        # The Detections and Connections counts, shown once there is a graph
        self.dets_label = QLabel()
        self.conns_label = QLabel()
        layout.addWidget(self.dets_label)
        layout.addWidget(self.conns_label)
        self.refresh_stats()

        # Set layout to the widget
        self.setLayout(layout)

        # Set focus policy and initially focus the widget
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)

    def focus(self):
        """Move keyboard focus back to the widget so shortcuts keep working."""
        self.setFocus()

    def refresh_stats(self):
        """Show the current number of detections and connections, or hide
        the counts when there is no graph."""
        state = self.state
        has_graph = state.session.graph is not None
        if has_graph:
            counts = state.session.stats()
            state.n_detections_shown = counts["detections"]
            state.n_connections_shown = counts["connections"]
            self.dets_label.setText(f"Detections: {state.n_detections_shown}")
            self.conns_label.setText(f"Connections: {state.n_connections_shown}")
        self.dets_label.setVisible(has_graph)
        self.conns_label.setVisible(has_graph)

    def update_file_paths_display(self):
        """Show the output files' paths once a detections file is loaded,
        or hide the Session Files panel when there is none."""
        state = self.state
        if not state.csv_loaded:
            self.session_files.hide_all()
            return
        combined = (os.path.join(state.output_folder, state.combined_file)
                    if state.max_prior_track_id is not None else None)
        self.session_files.show_paths(
            os.path.join(state.output_folder, state.leftover_file),
            os.path.join(state.output_folder, state.session_file),
            combined,
        )

    def clear_detections_and_tracks(self):
        """Explicitly clear detection points and track lines from the Napari viewer."""

        # Check and clear the 'detections' layer if it exists
        if 'detections' in self.viewer.layers:
            self.viewer.layers['detections'].data = np.empty((0, 2))
            self.viewer.layers['detections'].refresh()  # Ensure the layer refreshes

        # Check and clear the 'track_lines' layer if it exists
        if 'track_lines' in self.viewer.layers:
            self.viewer.layers['track_lines'].data = np.empty((0, 2, 2))
            self.viewer.layers['track_lines'].refresh()  # Ensure the layer refreshes

        # A fresh session: no detections, graph or tracks, and no output files.
        self.state.session = Session()
        self.state.clear_files()
        self.state.track = []
        self.state.n_detections_shown = None
        self.state.n_connections_shown = None

        # Hide the Session Files panel directly rather than through
        # update_file_paths_display(): that decides on state.csv_loaded, which
        # is not reset here or in choose_folder(), so after images are
        # reloaded it would show the previous dataset's file paths again. A
        # caller that goes on to load a new detections file shows the panel
        # again through its own update_file_paths_display() call.
        self.session_files.hide_all()

        # The "Track file loaded: ..." label belongs to the previous dataset too
        self.track_file_label.set_path("")

        # There is no graph now, so this hides the widget's counts
        self.refresh_stats()

        print("Cleared all detections and tracks.")
