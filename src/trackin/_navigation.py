"""Moving between frames: the Frame slider, and the Right and Left arrow
keys (tap for one frame, hold to keep going)."""

from magicgui import magicgui
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QSlider

from ._layers import update_image


REPEAT_INITIAL_DELAY = 350  # ms to hold a key before continuous scrolling kicks in


class Navigator:
    """Moving between frames for one widget: its Frame slider, and the
    Right and Left arrow keys (tap for one frame, hold to keep going)."""

    def __init__(self, w):
        self.w = w
        # Create timers for handling continuous key press events
        self.right_timer = QTimer()
        self.left_timer = QTimer()
        self.right_timer.timeout.connect(self.next_image)
        self.left_timer.timeout.connect(self.previous_image)
        self.right_key_held = False
        self.left_key_held = False
        # responsible for the image slider which moves when images are changed
        self.image_slider = magicgui(
            self.go_to_frame,
            image_index={"widget_type": "Slider", "min": 0, "max": 0, "step": 1, "label": "Frame"},
            auto_call=True,
        )
        for qslider in self.image_slider.image_index.native.findChildren(QSlider):
            qslider.wheelEvent = self._frame_slider_wheel_event

    # go to following image
    def next_image(self, event=None):
        """Display the next image in the sequence and overlay CSV data."""
        state = self.w.state
        if state.images:
            state.current_frame = (state.current_frame + 1) % len(state.images)
            update_image(self.w)
            self.image_slider.image_index.value = state.current_frame  # Sync slider value

        # Refocus the widget after key press
        self.w.focus()

    # go to previous image
    def previous_image(self, event=None):
        """Display the previous image in the sequence and overlay CSV data."""
        state = self.w.state
        if state.images:
            state.current_frame = (state.current_frame - 1) % len(state.images)
            update_image(self.w)
            self.image_slider.image_index.value = state.current_frame  # Sync slider value

        # Refocus the widget after key press
        self.w.focus()

    def go_to_frame(self, image_index: int = 0):
        """Update the displayed image based on the slider value."""
        state = self.w.state
        if image_index != state.current_frame:
            state.current_frame = image_index
            update_image(self.w)
            # Refocus the widget after slider change
            self.w.focus()

    def _frame_slider_wheel_event(self, qevent):
        """Step exactly one frame per wheel event, ignoring the reported scroll
        magnitude -- trackpads can report a delta worth more than one notch for
        a single gesture, which otherwise skips frames."""
        delta = qevent.angleDelta().y()
        if delta > 0:
            self.previous_image()
        elif delta < 0:
            self.next_image()
        qevent.accept()

    def update_slider_max(self):
        """Update the maximum value of the slider based on the number of images."""
        state = self.w.state
        if state.images:
            self.image_slider.image_index.max = len(state.images) - 1
            self.image_slider.image_index.value = state.current_frame
        else:
            self.image_slider.image_index.max = 0
            self.image_slider.image_index.value = 0

    def _maybe_start_repeat_right(self):
        """Only start auto-repeat if the key is still held after the initial delay."""
        if self.right_key_held:
            start_timer(self.right_timer)

    def _maybe_start_repeat_left(self):
        """Only start auto-repeat if the key is still held after the initial delay."""
        if self.left_key_held:
            start_timer(self.left_timer)

    def handle_right(self, viewer):
        """Handle the Right arrow key press and release."""
        self.right_key_held = True
        self.next_image()  # A single tap always advances exactly one frame
        QTimer.singleShot(REPEAT_INITIAL_DELAY, self._maybe_start_repeat_right)  # Auto-repeat only if held
        yield  # Wait for key release
        self.right_key_held = False
        stop_timer(self.right_timer)  # Stop on key release

    def handle_left(self, viewer):
        """Handle the Left arrow key press and release."""
        self.left_key_held = True
        self.previous_image()  # A single tap always advances exactly one frame
        QTimer.singleShot(REPEAT_INITIAL_DELAY, self._maybe_start_repeat_left)  # Auto-repeat only if held
        yield  # Wait for key release
        self.left_key_held = False
        stop_timer(self.left_timer)  # Stop on key release


def start_timer(timer):
    """Start the timer to continuously update images."""
    if not timer.isActive():
        timer.start(50)

def stop_timer(timer):
    """Stop the timer when the key is released."""
    if timer.isActive():
        timer.stop()
