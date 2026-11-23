import os
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from qtpy.QtCore import Qt
import napari
import numpy as np
from napari.utils.events import EventEmitter
from .plugin_state import plugin_state
from ._help_dialog import show_help
from ._session_files import ElidedPathLabel, create_path_row
from ._loading import choose_folder, load_csv, load_track_file
from ._interaction import acceptTrack, deleteAllConnections, deleteSegment, delete_detection, saveSegment
from ._navigation import (
    handle_left, handle_right, image_slider, left_timer, next_image, previous_image, right_timer,
)
from .core.session import Session

container = None  # Global reference to the container

ZOOM_STEP = 1.1  # multiplicative zoom change per wheel notch


# EVENT EMITTERS
data_updated_event = EventEmitter(source=None, type_name='data_updated')
graph_updated_event = EventEmitter(source=None, type_name='graph_updated')

def clamp_camera_zoom(event=None):
    """Keep the camera zoom within the zoom limits (plugin_state.min_zoom, max_zoom)."""
    zoom = plugin_state.viewer.camera.zoom
    clamped = min(max(zoom, plugin_state.min_zoom), plugin_state.max_zoom)
    if clamped != zoom:
        plugin_state.viewer.camera.zoom = clamped

def on_mouse_wheel_zoom(viewer, event):
    """Zoom the camera on plain mouse scroll, replacing napari's default
    wheel-zoom so we control the direction and clamp range ourselves,
    regardless of the OS/trackpad scroll convention."""
    if event.modifiers:
        return  # leave modified scroll (e.g. Control = change frame) alone

    delta = event.delta[1]
    if event.native is not None and event.native.inverted():
        delta = -delta
    if delta == 0:
        return

    factor = ZOOM_STEP if delta > 0 else 1 / ZOOM_STEP
    new_zoom = viewer.camera.zoom * factor
    viewer.camera.zoom = min(max(new_zoom, plugin_state.min_zoom), plugin_state.max_zoom)
    event.handled = True

def initialize_viewer(napari_viewer):
    """Initialize the Napari viewer object."""
    plugin_state.viewer = napari_viewer

    # Set up key bindings for the viewer
    setup_keybindings()

    # Take over scroll-to-zoom so direction and range are under our control
    plugin_state.viewer.mouse_wheel_callbacks.append(on_mouse_wheel_zoom)

    # Backstop in case zoom changes through some other path (e.g. reset_view)
    plugin_state.viewer.camera.events.zoom.connect(clamp_camera_zoom)

    # Set up the timers for handling repeated key presses
    right_timer.timeout.connect(next_image)
    left_timer.timeout.connect(previous_image)


def setup_keybindings():
    """Set up key bindings for the viewer."""
    plugin_state.viewer.bind_key('Right', handle_right)  # Right arrow key to move to the next image
    plugin_state.viewer.bind_key('Left', handle_left)  # Left arrow key to move to the previous image
    plugin_state.viewer.bind_key('D', lambda event: delete_detection(plugin_state.viewer.layers.selection.active, use_key=True))  # Trigger only for D key
    plugin_state.viewer.bind_key('Shift-Q', acceptTrack)  
    plugin_state.viewer.bind_key('W', saveSegment) # save correct segment
    plugin_state.viewer.bind_key('Shift-Z', deleteSegment) # delete a whole segment
    plugin_state.viewer.bind_key('X', deleteAllConnections) # delete all connections from a node, apart from the D-X edge

track_file_label = None  # Shows which track file was last loaded, if any

# Persistent, copyable output-file-path rows (see create_path_row / update_file_paths_display)
leftover_row = None
leftover_path_field = None
session_row = None
session_path_field = None
upd_track_row = None
upd_track_path_field = None

# Collapsible "Session Files" card wrapping the three rows above
session_files_header = None
session_files_content = None

def set_session_files_expanded(expanded):
    """Show/hide the Session Files card's content and update its header arrow."""
    session_files_content.setVisible(expanded)
    arrow = "▾" if expanded else "▸"  # ▾ expanded, ▸ collapsed
    session_files_header.setText(f"{arrow} Session Files")

def toggle_session_files():
    set_session_files_expanded(not session_files_content.isVisible())

def update_file_paths_display():
    """Show/hide and refresh the output-file-path rows based on current state."""
    if plugin_state.csv_loaded:
        leftover_path_field.set_path(os.path.join(plugin_state.output_folder, plugin_state.leftover_file))
        leftover_row.setVisible(True)
        session_path_field.set_path(os.path.join(plugin_state.output_folder, plugin_state.session_file))
        session_row.setVisible(True)
        # The whole Session Files card (header included) only makes sense
        # once there's a CSV loaded -- with just images loaded, there's
        # nothing to show yet.
        if session_files_header is not None:
            session_files_header.setVisible(True)
    else:
        leftover_row.setVisible(False)
        session_row.setVisible(False)
        if session_files_header is not None:
            session_files_header.setVisible(False)

    if plugin_state.max_prior_track_id is not None:
        upd_track_path_field.set_path(os.path.join(plugin_state.output_folder, plugin_state.combined_file))
        upd_track_row.setVisible(True)
    else:
        upd_track_row.setVisible(False)

    # Auto-expand the card whenever a row's content actually changed, so a
    # newly added file (e.g. from "Add Track File") is immediately visible
    # rather than hidden behind a collapsed header the user has to think to
    # open. Only relevant once the header itself is showing (csv_loaded).
    if plugin_state.csv_loaded and session_files_content is not None:
        set_session_files_expanded(True)

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
        initialize_viewer(napari_viewer)

        global container, dets_label, conns_label, track_file_label
        global leftover_row, leftover_path_field, session_row, session_path_field
        global upd_track_row, upd_track_path_field
        global session_files_header, session_files_content
        container = self
        layout = QVBoxLayout(container)  # Create a vertical layout

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

        # Use the magicgui widgets and add them directly to the layout
        layout.addWidget(choose_folder.native)  # Add the magicgui widget's native Qt widget
        layout.addWidget(load_csv.native)
        layout.addWidget(load_track_file.native)

        # Shows the loaded track file's name once "Add Track File" succeeds,
        # since that success is otherwise only reported via a transient toast
        track_file_label = ElidedPathLabel()
        track_file_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(track_file_label)

        layout.addWidget(image_slider.native)  # Add the slider widget

        # Collapsible "Session Files" card wrapping the copyable output-file-path
        # rows, shown/updated as state changes. Starts collapsed since there's
        # nothing to show until a CSV is loaded; update_file_paths_display()
        # auto-expands it whenever a row's content actually changes.
        session_files_header = QPushButton("▸ Session Files")
        session_files_header.setFlat(True)
        session_files_header.setStyleSheet(
            "QPushButton { text-align: left; font-weight: 600; border: none; padding: 2px 0; }"
        )
        session_files_header.setCursor(Qt.PointingHandCursor)
        session_files_header.clicked.connect(toggle_session_files)
        session_files_header.setVisible(False)  # nothing to show until a CSV is loaded
        layout.addWidget(session_files_header)

        session_files_content = QWidget()
        file_paths_layout = QVBoxLayout(session_files_content)
        file_paths_layout.setContentsMargins(0, 4, 0, 14)
        file_paths_layout.setSpacing(8)

        leftover_row, leftover_path_field = create_path_row(
            "Leftover detections",
            "Contains the detections still available for tracking: detections in accepted tracks "
            "and detections you deleted (D, right-click or Shift-Z) are left out, and detections "
            "you added (Shift-click) are included.",
        )
        session_row, session_path_field = create_path_row(
            "Tracks accepted this session",
            "Contains the detections included in the accepted tracks, together with the respective generated track ids.",
        )
        upd_track_row, upd_track_path_field = create_path_row(
            "Combined tracks",
            "Contains the tracks from the added track file, together with the accepted tracks in this session.",
        )

        for row in (leftover_row, session_row, upd_track_row):
            row.setVisible(False)
            file_paths_layout.addWidget(row)

        session_files_content.setVisible(False)
        layout.addWidget(session_files_content)

        # Set layout to the container
        container.setLayout(layout)

        # Set focus policy and initially focus the container
        container.setFocusPolicy(Qt.StrongFocus)
        container.setFocus()
        plugin_state.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        plugin_state.viewer.window.qt_viewer.dockLayerControls.setVisible(False)


def clear_detections_and_tracks():
    """Explicitly clear detection points and track lines from the Napari viewer."""
    
    # Check and clear the 'detections' layer if it exists
    if 'detections' in plugin_state.viewer.layers:
        plugin_state.viewer.layers['detections'].data = np.empty((0, 2))
        plugin_state.viewer.layers['detections'].refresh()  # Ensure the layer refreshes
    
    # Check and clear the 'track_lines' layer if it exists
    if 'track_lines' in plugin_state.viewer.layers:
        plugin_state.viewer.layers['track_lines'].data = np.empty((0, 2, 2))
        plugin_state.viewer.layers['track_lines'].refresh()  # Ensure the layer refreshes
    
    # A fresh session: no detections, graph or tracks, and no output files.
    plugin_state.session = Session()
    plugin_state.clear_files()
    plugin_state.track = []
    plugin_state.n_detections_shown = None
    plugin_state.n_connections_shown = None

    # Hide all three rows unconditionally rather than routing through
    # update_file_paths_display(): that function's leftover/session
    # visibility is gated on plugin_state.csv_loaded, which is never
    # reset to False here or in choose_folder() -- so if a CSV had
    # already been loaded once this session and images are reloaded
    # without a new CSV yet, csv_loaded is still stale True and would
    # re-show the *previous* dataset's file paths. Any caller that goes on
    # to load a fresh CSV (e.g. load_csv() reloading) will re-show the
    # rows correctly via its own update_file_paths_display() call right after.
    if leftover_row is not None:
        leftover_row.setVisible(False)
        session_row.setVisible(False)
        upd_track_row.setVisible(False)

    # Hide the whole Session Files card (not just collapse it) too, since
    # there's nothing left in it to show until a new CSV is loaded.
    if session_files_content is not None:
        set_session_files_expanded(False)
        session_files_header.setVisible(False)

    # Clear the "Track file loaded: ..." label too -- it previously wasn't
    # touched here at all, so it kept showing the *previous* dataset's
    # loaded track file even after images were reloaded.
    if track_file_label is not None:
        track_file_label.set_path("")

    # update_labels() (the Detections/Connections counts) only runs in
    # response to this event -- without firing it here, those labels kept
    # showing the *previous* dataset's stale counts until some unrelated
    # keypress happened to fire the event afterward.
    graph_updated_event()

    print("Cleared all detections and tracks.")

# Global references for the QLabel widgets
dets_label = None
conns_label = None

def update_labels():
    """Update the QLabel widgets with the current count of detections and connections."""
    global dets_label, conns_label  # Declare the variables as global to avoid NameError

    if plugin_state.session.graph is not None:
        # Compute the number of detections and connections
        counts = plugin_state.session.stats()
        plugin_state.n_detections_shown = counts["detections"]
        plugin_state.n_connections_shown = counts["connections"]

        # Check if labels exist, if not create and add them to the layout
        if dets_label is None:
            dets_label = QLabel(f"Detections: {plugin_state.n_detections_shown}")
            conns_label = QLabel(f"Connections: {plugin_state.n_connections_shown}")
            container.layout().addWidget(dets_label)  # Add the labels to the layout dynamically
            container.layout().addWidget(conns_label)
        else:
            # If the labels already exist, just update the text
            dets_label.setText(f"Detections: {plugin_state.n_detections_shown}")
            conns_label.setText(f"Connections: {plugin_state.n_connections_shown}")
    else:
        # If there is no graph, remove or hide the labels (if necessary)
        if dets_label is not None and conns_label is not None:
            dets_label.setParent(None)  # Remove the label from the layout
            conns_label.setParent(None)
            dets_label = None  # Reset the label references
            conns_label = None

graph_updated_event.connect(update_labels)