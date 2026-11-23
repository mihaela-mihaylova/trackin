"""Moving between frames: the Frame slider, and the Right and Left arrow
keys (tap for one frame, hold to keep going)."""

from magicgui import magicgui
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QSlider

from ._layers import update_image
from .plugin_state import plugin_state


# Create timers for handling continuous key press events
right_timer = QTimer()
left_timer = QTimer()
REPEAT_INITIAL_DELAY = 350  # ms to hold a key before continuous scrolling kicks in
right_key_held = False
left_key_held = False


# go to following image
def next_image(event=None):
    """Display the next image in the sequence and overlay CSV data."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    if plugin_state.images:
        plugin_state.current_frame = (plugin_state.current_frame + 1) % len(plugin_state.images)
        update_image()
        image_slider.image_index.value = plugin_state.current_frame  # Sync slider value
    
    # Refocus the container after key press
    if _widget.container:
        _widget.container.setFocus()

# go to previous image
def previous_image(event=None):
    """Display the previous image in the sequence and overlay CSV data."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    if plugin_state.images:
        plugin_state.current_frame = (plugin_state.current_frame - 1) % len(plugin_state.images)
        update_image()
        image_slider.image_index.value = plugin_state.current_frame  # Sync slider value
    
    # Refocus the container after key press
    if _widget.container:
        _widget.container.setFocus()

# responsible for the image slider which moves when images are changed
@magicgui(image_index={"widget_type": "Slider", "min": 0, "max": 0, "step": 1, "label": "Frame"}, auto_call=True)
def image_slider(image_index: int = 0):
    """Update the displayed image based on the slider value."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    if image_index != plugin_state.current_frame:
        plugin_state.current_frame = image_index
        update_image()
        # Refocus the container after slider change
        if _widget.container:
            _widget.container.setFocus()

def _frame_slider_wheel_event(qevent):
    """Step exactly one frame per wheel event, ignoring the reported scroll
    magnitude -- trackpads can report a delta worth more than one notch for
    a single gesture, which otherwise skips frames."""
    delta = qevent.angleDelta().y()
    if delta > 0:
        previous_image()
    elif delta < 0:
        next_image()
    qevent.accept()

for _qslider in image_slider.image_index.native.findChildren(QSlider):
    _qslider.wheelEvent = _frame_slider_wheel_event

def update_slider_max():
    """Update the maximum value of the slider based on the number of images."""
    if plugin_state.images:
        image_slider.image_index.max = len(plugin_state.images) - 1
        image_slider.image_index.value = plugin_state.current_frame
    else:
        image_slider.image_index.max = 0
        image_slider.image_index.value = 0


def _maybe_start_repeat_right():
    """Only start auto-repeat if the key is still held after the initial delay."""
    if right_key_held:
        start_timer(right_timer)

def _maybe_start_repeat_left():
    """Only start auto-repeat if the key is still held after the initial delay."""
    if left_key_held:
        start_timer(left_timer)

def handle_right(viewer):
    """Handle the Right arrow key press and release."""
    global right_key_held
    right_key_held = True
    next_image()  # A single tap always advances exactly one frame
    QTimer.singleShot(REPEAT_INITIAL_DELAY, _maybe_start_repeat_right)  # Auto-repeat only if held
    yield  # Wait for key release
    right_key_held = False
    stop_timer(right_timer)  # Stop on key release

def handle_left(viewer):
    """Handle the Left arrow key press and release."""
    global left_key_held
    left_key_held = True
    previous_image()  # A single tap always advances exactly one frame
    QTimer.singleShot(REPEAT_INITIAL_DELAY, _maybe_start_repeat_left)  # Auto-repeat only if held
    yield  # Wait for key release
    left_key_held = False
    stop_timer(left_timer)  # Stop on key release

def start_timer(timer):
    """Start the timer to continuously update images."""
    if not timer.isActive():
        timer.start(50)  

def stop_timer(timer):
    """Stop the timer when the key is released."""
    if timer.isActive():
        timer.stop()
