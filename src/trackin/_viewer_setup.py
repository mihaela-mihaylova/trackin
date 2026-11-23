"""Viewer-wide behaviour: scroll-to-zoom within the zoom limits, and the
keyboard shortcuts."""

import weakref

from ._interaction import acceptTrack, deleteAllConnections, deleteSegment, delete_detection, saveSegment

ZOOM_STEP = 1.1  # multiplicative zoom change per wheel notch

# the zoom handlers put on each viewer, so opening the widget again replaces them
_installed_zoom_handlers = weakref.WeakKeyDictionary()


def initialize_viewer(w):
    """Set up the viewer of the widget w: its key bindings and scroll-to-zoom."""
    state = w.state

    def clamp_camera_zoom(event=None):
        """Keep the camera zoom within the zoom limits (state.min_zoom, max_zoom)."""
        zoom = w.viewer.camera.zoom
        clamped = min(max(zoom, state.min_zoom), state.max_zoom)
        if clamped != zoom:
            w.viewer.camera.zoom = clamped

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
        viewer.camera.zoom = min(max(new_zoom, state.min_zoom), state.max_zoom)
        event.handled = True

    # Set up key bindings for the viewer
    setup_keybindings(w)

    # Remove the zoom handlers an earlier widget put on this viewer
    previous = _installed_zoom_handlers.get(w.viewer)
    if previous is not None:
        old_wheel, old_clamp = previous
        if old_wheel in w.viewer.mouse_wheel_callbacks:
            w.viewer.mouse_wheel_callbacks.remove(old_wheel)
        w.viewer.camera.events.zoom.disconnect(old_clamp)

    # Take over scroll-to-zoom so direction and range are under our control
    w.viewer.mouse_wheel_callbacks.append(on_mouse_wheel_zoom)

    # Backstop in case zoom changes through some other path (e.g. reset_view)
    w.viewer.camera.events.zoom.connect(clamp_camera_zoom)

    _installed_zoom_handlers[w.viewer] = (on_mouse_wheel_zoom, clamp_camera_zoom)


def setup_keybindings(w):
    """Set up key bindings for the viewer, each action called with the widget
    w. A widget opened later on the same viewer takes the keys over."""
    w.viewer.bind_key('Right', w.navigator.handle_right, overwrite=True)  # Right arrow key to move to the next image
    w.viewer.bind_key('Left', w.navigator.handle_left, overwrite=True)  # Left arrow key to move to the previous image
    w.viewer.bind_key('D', lambda viewer: delete_detection(w, viewer.layers.selection.active, use_key=True), overwrite=True)  # Trigger only for D key
    w.viewer.bind_key('Shift-Q', lambda viewer: acceptTrack(w), overwrite=True)
    w.viewer.bind_key('W', lambda viewer: saveSegment(w), overwrite=True) # save correct segment
    w.viewer.bind_key('Shift-Z', lambda viewer: deleteSegment(w), overwrite=True) # delete a whole segment
    w.viewer.bind_key('X', lambda viewer: deleteAllConnections(w), overwrite=True) # delete all connections from a node, apart from the D-X edge
