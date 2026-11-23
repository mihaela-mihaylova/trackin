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

def initialize_viewer(w):
    """Set up the viewer of the widget w: its key bindings, scroll-to-zoom
    and the arrow-key timers."""
    plugin_state.viewer = w.viewer

    # Set up key bindings for the viewer
    setup_keybindings(w)

    # Take over scroll-to-zoom so direction and range are under our control
    w.viewer.mouse_wheel_callbacks.append(on_mouse_wheel_zoom)

    # Backstop in case zoom changes through some other path (e.g. reset_view)
    w.viewer.camera.events.zoom.connect(clamp_camera_zoom)

    # Set up the timers for handling repeated key presses
    right_timer.timeout.connect(next_image)
    left_timer.timeout.connect(previous_image)


def setup_keybindings(w):
    """Set up key bindings for the viewer, each action called with the widget w."""
    w.viewer.bind_key('Right', handle_right)  # Right arrow key to move to the next image
    w.viewer.bind_key('Left', handle_left)  # Left arrow key to move to the previous image
    w.viewer.bind_key('D', lambda viewer: delete_detection(w, viewer.layers.selection.active, use_key=True))  # Trigger only for D key
    w.viewer.bind_key('Shift-Q', lambda viewer: acceptTrack(w))
    w.viewer.bind_key('W', lambda viewer: saveSegment(w)) # save correct segment
    w.viewer.bind_key('Shift-Z', lambda viewer: deleteSegment(w)) # delete a whole segment
    w.viewer.bind_key('X', lambda viewer: deleteAllConnections(w)) # delete all connections from a node, apart from the D-X edge
