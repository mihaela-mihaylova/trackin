"""Viewer-wide behaviour: scroll-to-zoom within the zoom limits, and the
keyboard shortcuts."""

from ._interaction import acceptTrack, deleteAllConnections, deleteSegment, delete_detection, saveSegment
from ._navigation import handle_left, handle_right, left_timer, next_image, previous_image, right_timer
from .plugin_state import plugin_state

ZOOM_STEP = 1.1  # multiplicative zoom change per wheel notch


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
