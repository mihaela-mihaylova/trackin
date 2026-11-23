"""Drawing the current frame: the image, the detections, the current
track's point in this frame and the lines of the current track."""

import os

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info

from .plugin_state import plugin_state


def update_image():
    """Update the displayed image and overlay CSV data without changing zoom level."""
    if plugin_state.images:
        # Store the current camera settings (zoom and center)
        current_zoom = plugin_state.viewer.camera.zoom
        current_center = plugin_state.viewer.camera.center

        if 'Image' in plugin_state.viewer.layers:
            # Just update the data of the existing image layer
            plugin_state.viewer.layers['Image'].data = plugin_state.images[plugin_state.current_frame]
        else:
            # Create a new image layer if it doesn't exist
            plugin_state.viewer.add_image(plugin_state.images[plugin_state.current_frame], name='Image')

        # Restore the camera zoom and center position
        plugin_state.viewer.camera.zoom = current_zoom
        plugin_state.viewer.camera.center = current_center

    # Ensure the detections are loaded, then overlay points and lines
    if plugin_state.csv_loaded and plugin_state.session.detections is not None:
        # Get the frame number from the current image file name
        frame_number = int(os.path.splitext(os.path.basename(plugin_state.image_files[plugin_state.current_frame]))[0])
        # Ensure the frame number is within the range of the detections
        if 0 <= frame_number < len(plugin_state.session.detections):
            frame_data = plugin_state.session.detections[frame_number]
            # Also called for a frame without detections, so the previous
            # frame's points are cleared instead of staying on screen
            # the positions of the frame's detections, which are drawn; a
            # deleted detection is put far off-screen, so it isn't seen but
            # keeps its place (a clicked point's place is its index)
            overlay_points(pd.DataFrame({
                'y': [d.y if d.active else -1000000 for d in frame_data],
                'x': [d.x if d.active else -1000000 for d in frame_data],
            }))


def overlay_points(frame_data):
    """Overlay circles on the image for each (x, y) in the frame data, with a separate layer for the track point.

    For a frame without detections the layers are emptied, so no points of
    another frame stay on screen."""
    # The click handlers are imported here rather than at the top, since
    # _interaction.py imports this file: imported at the top, each would wait
    # for the other to load first.
    from ._interaction import add_detection, delete_detection, on_click
    # _widget.py imports this file too, so it is imported here as well
    from . import _widget

    # Extract (y, x) positions from the dataframe: one row [y, x] per detection.
    # For a frame without detections this is an empty array with no rows,
    # which empties the layers below.
    points = np.array([frame_data['y'], frame_data['x']]).T
    curr_track = plugin_state.track
    if curr_track is None:
        show_info('No detections left.')
        return
    
    # --- Main Points Layer ---
    if 'detections' in plugin_state.viewer.layers:
        points_layer = plugin_state.viewer.layers['detections']
        # Update only the data if it has changed
        if not np.array_equal(points_layer.data, points):
            points_layer.data = points
    else:
        points_layer = plugin_state.viewer.add_points(
            points,
            size=20,  # Default size for all detections
            face_color='transparent',
            border_color='white',
            border_width=1,
            border_width_is_relative=False,
            name='detections'
        )
        # Attach the click handlers, each called with the widget
        w = _widget.container
        points_layer.mouse_drag_callbacks.append(lambda layer, event: delete_detection(w, layer, event))
        points_layer.mouse_drag_callbacks.append(lambda layer, event: add_detection(w, layer, event))
        points_layer.mouse_drag_callbacks.append(lambda layer, event: on_click(w, layer, event))

    # Prepare attributes for the main points layer
    border_colors = ['white'] * len(points)
    sizes = [20] * len(points)
    border_widths = [1] * len(points)

    # Check if track contains detection in this frame and apply custom styles for the track point
    track_index = curr_track[plugin_state.current_frame]
    if track_index != -1:
        border_colors[track_index] = 'yellow'
        sizes[track_index] = 35  # Increase the size of the track point
        border_widths[track_index] = 3  # Thicker border for the track point

    # --- Apply updates only if attributes have changed ---
    if not np.array_equal(points_layer.size, sizes):
        points_layer.size = sizes
    if not np.array_equal(points_layer.border_color, border_colors):
        points_layer.border_color = border_colors
    if not np.array_equal(points_layer.border_width, border_widths):
        points_layer.border_width = border_widths

    # --- Separate Track Point Layer ---
    # Check if there is a valid track point in this frame
    if track_index != -1:
        track_point = np.array([points[track_index]])  # Select only the current track point
    else:
        track_point = np.empty((0, 2))  # Empty array if no track point is present

    # Create or update the separate track point layer
    if 'track_point' in plugin_state.viewer.layers:
        track_layer = plugin_state.viewer.layers['track_point']
        if not np.array_equal(track_layer.data, track_point):
            track_layer.data = track_point
    else:
        track_layer = plugin_state.viewer.add_points(
            track_point,
            size=35,  # Larger size for the track point
            face_color='transparent',
            border_color='yellow',
            border_width=3,
            border_width_is_relative=False,
            name='track_point'
        )

    # --- Construct and update track lines ---
    track_points = []
    for t in range(len(plugin_state.track)):
        track_idx = plugin_state.track[t]
        if track_idx != -1:
            d = plugin_state.session.detections[t][track_idx]
            y, x = d.y, d.x
            track_points.append([y, x])

    if len(track_points) > 1:
        lines = np.array([[track_points[i], track_points[i + 1]] for i in range(len(track_points) - 1)])
        if 'track_lines' in plugin_state.viewer.layers:
            track_lines_layer = plugin_state.viewer.layers['track_lines']
            if not np.array_equal(track_lines_layer.data, lines):
                track_lines_layer.data = lines
                # Once the layer has been emptied, napari takes new shapes
                # for polygons, which two points don't draw
                track_lines_layer.shape_type = 'line'
        else:
            # Create the shapes layer for the track lines if it doesn't exist
            plugin_state.viewer.add_shapes(
                lines,
                shape_type='line',
                edge_width=1,
                edge_color='white',
                name='track_lines',
                face_color='transparent'
            )
    else:
        if 'track_lines' in plugin_state.viewer.layers:
            plugin_state.viewer.layers['track_lines'].data = np.empty((0, 2, 2))

    # Set the points layer as active to enable further interactions
    plugin_state.viewer.layers.selection.active = points_layer
