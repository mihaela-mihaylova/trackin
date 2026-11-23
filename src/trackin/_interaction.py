"""The user's actions on the image and their shortcuts, turned into calls
on the tracking session: right-click or D deletes a detection, Shift-click
adds one, left-click puts a detection on the current track, Shift-Q accepts
the track, W fixes the track up to the current frame in its current form,
Shift-Z deletes it, and X deletes the current detection's connections.
After a change it hands the leftover detections and any accepted track to
core/io.py, which writes them to the output files."""

import os

import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer

from ._layers import update_image
from ._navigation import image_slider, next_image
from .core.io import append_to_combined_file, append_to_session_file, write_leftover_file
from .plugin_state import plugin_state


def save_accepted_track():
    """Write the track just accepted to the session file, and to the
    combined file once a track file has been added."""
    session = plugin_state.session
    track_no = len(session.accepted)
    track = session.accepted[-1]
    append_to_session_file(os.path.join(plugin_state.output_folder, plugin_state.session_file),
                           track_no, track, session.detections, session.extra_columns)
    if plugin_state.max_prior_track_id is not None:
        append_to_combined_file(os.path.join(plugin_state.output_folder, plugin_state.combined_file),
                                plugin_state.max_prior_track_id + track_no, track,
                                session.detections, session.extra_columns, plugin_state.combined_extra_columns)

def save_leftover_file():
    """Write the session's available detections to the leftover file
    (nothing before a detections file is loaded, as there is no file name)."""
    session = plugin_state.session
    if plugin_state.leftover_file:
        write_leftover_file(os.path.join(plugin_state.output_folder, plugin_state.leftover_file),
                            session.detections, session.extra_columns)

def use_proposed_track():
    """The track being edited becomes a copy of the session's newly
    proposed track (None when no track is left). Clicks then change this
    copy, not the session's proposal."""
    proposal = plugin_state.session.proposal
    plugin_state.track = None if proposal is None else list(proposal)

def show_messages(messages):
    """Show the messages returned by the tracking session's actions
    (e.g. 'Accepted track.') as napari notifications."""
    for message in messages:
        show_info(message)


def delete_detection(layer, event=None, use_key=False):
    """Delete detection or track detection based on right-click or 'D' key."""
    global clicked_index
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    clicked_index = None
    # Determine the clicked index based on input type
    if not use_key:  # Called from mouse event
        if event.button != 2:  # Not a right-click: nothing to do here, let the
            return              # event fall through to napari's default pan/zoom.
        click_position = event.position
        print(f'clicked position:{click_position}')
        clicked_index = layer.get_value(click_position, world=True)
        if clicked_index is None:
            return  # No valid point was clicked, exit
    else:  # Called using the 'D' key
        curr_track = plugin_state.track
        # e.g. right after images are reloaded, before a new CSV is loaded
        if not curr_track:
            print("No detections loaded to delete with 'D' key.")
            return
        if curr_track[plugin_state.current_frame] != -1:
            clicked_index = curr_track[plugin_state.current_frame]  # Use the tracked index if available
        else:
            print("No detection found to delete with 'D' key.")
            return

    # If a valid point is found, determine if it's part of a track or a standard detection
    if clicked_index is not None:
        curr_track = plugin_state.track
        if curr_track[plugin_state.current_frame] == clicked_index:
            # Call the function for deleting track detection
            delete_det_by_key(layer)
        else:
            delete_det_by_mouse(layer, event)

        _widget.graph_updated_event()

    # Mark the event as handled if called from a mouse event, otherwise mouse gets locked
    if event is not None:
        event.handled = True

def delete_det_by_key(layer):
    """Delete track detection using the 'D' key (or right-click on it), and
    show the new proposed track from frame 0."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    point_coords = layer.data[clicked_index].copy()
    layer.data[clicked_index] = [-1000000, -1000000]

    node_name = f'D_{plugin_state.current_frame}_{clicked_index}'
    plugin_state.session.delete_detection(plugin_state.current_frame, clicked_index)

    # Update the track state
    plugin_state.track[plugin_state.current_frame] = -1
    show_messages(plugin_state.session.propose())  # propose a new track without the deleted detection
    use_proposed_track()

    # Refresh the layer to update the visual display
    layer.refresh()
    # The new track can differ in earlier frames too, so show it from
    # frame 0, as after accepting a track or saving a segment
    plugin_state.current_frame = 0
    update_image()
    image_slider.image_index.value = plugin_state.current_frame  # Sync the slider value
    # Move keyboard focus off the slider's number box, back to the panel,
    # so keys like D and the arrows keep working
    if _widget.container:
        _widget.container.setFocus()

    # save up-to-date version of the detections
    save_leftover_file()

    print(f"Removed node {node_name} with coordinates {point_coords}.")


def delete_det_by_mouse(layer, event):
    """Delete track detection using right-click."""
    point_coords = layer.data[clicked_index].copy()
    layer.data[clicked_index] = [-1000000, -1000000]

    node_name = f'D_{plugin_state.current_frame}_{clicked_index}'
    plugin_state.session.delete_detection(plugin_state.current_frame, clicked_index)

    layer.refresh()
    update_image()

    # save up-to-date version of the detections
    save_leftover_file()

    print(f"Removed node {node_name} with coordinates {point_coords}.")

    # Mark the event as handled to stop further processing
    event.handled = True
    return  # Exit here to stop any further mouse events

def on_click(layer, event):
    """Handle left mouse click and make clicked-on detection part of current track only if clicked directly on a point."""
    global new_point_added

    if new_point_added:
        # Ignore the click if a new point was just added
        return

    if event.button == 1:  # Check for left mouse button click
        click_position = event.position

        # Check if the click is on a valid point using get_value() method
        clicked_index = layer.get_value(click_position, world=True)

        if clicked_index is not None and clicked_index >= 0:
            # If a valid point was clicked, update the track and go to the next image
            plugin_state.track[plugin_state.current_frame] = clicked_index
            next_image()
        else:
            print("Clicked outside of any detection point.")


new_point_added = False

def add_detection(layer, event):
    """Handle shift + left mouse click events to add a new point within image bounds."""
    global new_point_added
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget

    if event.button == 1 and 'Shift' in event.modifiers:
        # Convert world coordinates to data coordinates (pixel coordinates)
        data_coords = layer.world_to_data(event.position)

        # Get the dimensions of the currently displayed image
        image_shape = plugin_state.images[plugin_state.current_frame].shape  # This gets (height, width) for the image

        # Ensure the click is within the image bounds
        if (0 <= data_coords[0] < image_shape[0]) and (0 <= data_coords[1] < image_shape[1]):
            # Append the new point to the layer's data in data coordinates (pixels)
            new_point = [int(data_coords[0]), int(data_coords[1])]  # [y, x] format

            # Add the new point to the existing points layer data
            layer.data = np.vstack([layer.data, new_point])

            # Determine the default point size (20) or get the minimum size of existing points
            if len(layer.size) > 0:
                point_size = 20

            # Create a new size array matching the updated data length
            new_size_array = np.full((len(layer.data),), point_size)

            # Ensure the size of the track detection in the current frame is changed to 30
            curr_track = plugin_state.track
            if curr_track[plugin_state.current_frame] != -1:
                track_index = curr_track[plugin_state.current_frame]  # Get the correct track index
                new_size_array[track_index] = 30  # Set track point size to 30

            # Update the size array with the correct shape
            layer.size = new_size_array
            layer.border_width_is_relative = False  # Ensure the border width is absolute

            # Add the new detection to the tracking data and the graph
            plugin_state.session.add_detection(plugin_state.current_frame, int(data_coords[0]), int(data_coords[1]))

            # Refresh the layer to display the new point
            layer.refresh()

            print(f"Added new point at data coordinates: {new_point}, with size: {point_size}")

            # save up-to-date version of the detections
            save_leftover_file()

            # Set the flag to indicate a new point was added
            new_point_added = True

            # Use a QTimer to reset the flag after 200 milliseconds
            QTimer.singleShot(100, reset_new_point_flag)
            _widget.graph_updated_event()
        else:
            print("Clicked outside the image bounds. Point not added.")

def acceptTrack(event=None):
    """Accept the current track and move to frame 0."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    n_accepted = len(plugin_state.session.accepted)
    show_messages(plugin_state.session.accept_track(plugin_state.track))
    if len(plugin_state.session.accepted) > n_accepted:
        save_accepted_track()
    save_leftover_file()
    use_proposed_track()
    plugin_state.current_frame = 0  # Reset to the first frame (frame 0)
    update_image()  # Update the displayed image
    image_slider.image_index.value = plugin_state.current_frame  # Sync the slider value
    _widget.graph_updated_event()
    # Refocus the main container after handling the event
    if _widget.container:
        _widget.container.setFocus()

     
def saveSegment(event=None):
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    print("saveSegment triggered!")  # Add this line to check if the function is called
    show_messages(plugin_state.session.save_segment(plugin_state.track, plugin_state.current_frame))
    use_proposed_track()
    plugin_state.current_frame = 0  # Reset to the first frame (frame 0)
    update_image()
    image_slider.image_index.value = plugin_state.current_frame  # Sync the slider value
    # Refocus the main container after handling the event
    if _widget.container:
        _widget.container.setFocus()

def deleteSegment(event=None):
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    show_messages(plugin_state.session.delete_segment(plugin_state.track, plugin_state.current_frame))
    save_leftover_file()
    use_proposed_track()
    plugin_state.current_frame = 0  # Reset to the first frame (frame 0)
    update_image()  # Update the displayed image
    image_slider.image_index.value = plugin_state.current_frame  # Sync the slider value
    _widget.graph_updated_event()
    # Refocus the main container after handling the event
    if _widget.container:
        _widget.container.setFocus()

def deleteAllConnections(event=None):
    """Triggers the function which would delete all outgoing edges (apart from an X- or T-edge)"""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    show_messages(plugin_state.session.delete_all_connections(plugin_state.track, plugin_state.current_frame))
    update_image()
    _widget.graph_updated_event()
    # Refocus the main container after handling the event
    if _widget.container:
        _widget.container.setFocus()

def reset_new_point_flag():
    """Reset the flag indicating a new point was added."""
    global new_point_added
    new_point_added = False
