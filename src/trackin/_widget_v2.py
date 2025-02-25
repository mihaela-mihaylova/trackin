import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from qtpy.QtCore import Qt, QTimer
import napari
import numpy as np
from napari.utils.events import EventEmitter
from .shared_state import shared_state
from .utils import build_graph
from datetime import datetime
from .tracking import find_node_by_attributes, add_node_with_dummy_edges
from napari.utils.notifications import show_info
from napari_plugin_engine import (
    napari_hook_implementation,
)  # Make sure this is imported


# Add buttons to the viewer
class TrackingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()

        # Set up a vertical layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create buttons
        self.button_add_detect = QPushButton("Add Detections")
        self.button_add_track = QPushButton("Add Track File")

        # Add buttons to the layout
        layout.addWidget(self.button_add_detect)
        layout.addWidget(self.button_add_track)

        # Connect button signals to callback methods
        self.button_add_detect.clicked.connect(self.add_detections)
        self.button_add_track.clicked.connect(self.add_tracks)

        # Attributes
        self.detections = None

    @staticmethod
    def _check_and_add_displ_cols(df):
        """Add displ_x and add displ_y if these are not present in loaded df"""
        if 'displ_x' not in df.columns:
            df['displ_x'] = 0
        if 'displ_y' not in df.columns:
            df['displ_y'] = 0
        return df

    def _compute_track_lines(self):
        """Compute line segments connecting consecutive points in the track."""
        track_points = []
        for t in range(len(shared_state.track)):
            track_idx = shared_state.track[t]
            if track_idx is not None and track_idx != -1:
                # Ensure track index is within bounds of shared_state.DATA[t]
                if 0 <= track_idx < len(shared_state.DATA[t]):
                    y, x = shared_state.DATA[t][track_idx][:2]
                    track_points.append([y, x])

        # Create line segments only if there are at least two points in the track
        if len(track_points) > 1:
            shared_state.track_lines = np.array(
                [[track_points[i], track_points[i + 1]] for i in range(len(track_points) - 1)]
            )
        else:
            shared_state.track_lines = None  # No valid lines to draw

    def update_image(self):
        """Update the displayed image and overlay CSV data without changing zoom level."""
        global viewer
        if images:
            # Get the current image and add a white border to it
            bordered_image = add_white_border(images[shared_state.current_index], border_size=2)
    
            # Store the current camera settings (zoom and center)
            current_zoom = viewer.camera.zoom
            current_center = viewer.camera.center
    
            if 'Image' in viewer.layers:
                # Just update the data of the existing image layer
                viewer.layers['Image'].data = images[shared_state.current_index]
            else:
                # Create a new image layer if it doesn't exist
                viewer.add_image(images[shared_state.current_index], name='Image')
    
            # Restore the camera zoom and center position
            viewer.camera.zoom = current_zoom
            viewer.camera.center = current_center
    
        # Ensure DATA and csv_data are loaded, then overlay points and lines
        if csv_data is not None and shared_state.DATA is not None:
            # Get the frame number from the current image file name
            frame_number = int(os.path.splitext(os.path.basename(image_files[shared_state.current_index]))[0])
            # Ensure the frame number is within the range of DATA
            if 0 <= frame_number < len(shared_state.DATA):
                frame_data = shared_state.DATA[frame_number]
                if frame_data:  # Only overlay points if data is present
                    columns = ['y', 'x', 'displ_y', 'displ_x'] if not shared_state.TRACKED else ['y', 'x', 'displ_y', 'displ_x', 'track_no']
                    overlay_points(pd.DataFrame(frame_data, columns=columns))

    def overlay_points(self, frame_data):
        """Overlay circles on the image for each (x, y) in the frame data, with a separate layer for the track point."""
        global viewer, points_layer
    
        if frame_data.empty:
            print("No frame data provided")
            return
    
        # Extract (y, x) positions from the dataframe
        points = np.array([frame_data['y'], frame_data['x']]).T
        curr_track = shared_state.track
        if curr_track is None:
            show_info('No detections left.')
            return
        
        # --- Main Points Layer ---
        if 'detections' in viewer.layers:
            points_layer = viewer.layers['detections']
            # Update only the data if it has changed
            if not np.array_equal(points_layer.data, points):
                points_layer.data = points
        else:
            points_layer = viewer.add_points(
                points,
                size=20,  # Default size for all detections
                face_color='transparent',
                border_color='white',
                border_width=1,
                border_width_is_relative=False,
                name='detections'
            )
            # Attach event handlers if necessary
            points_layer.mouse_drag_callbacks.append(delete_detection)
            points_layer.mouse_drag_callbacks.append(add_detection)
            points_layer.mouse_drag_callbacks.append(on_click)
    
        # Prepare attributes for the main points layer
        border_colors = ['white'] * len(points)
        sizes = [20] * len(points)
        border_widths = [1] * len(points)
    
        # Check if track contains detection in this frame and apply custom styles for the track point
        track_index = curr_track[shared_state.current_index]
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
        if 'track_point' in viewer.layers:
            track_layer = viewer.layers['track_point']
            if not np.array_equal(track_layer.data, track_point):
                track_layer.data = track_point
        else:
            track_layer = viewer.add_points(
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
        for t in range(len(shared_state.track)):
            track_idx = shared_state.track[t]
            if track_idx != -1:
                y, x = shared_state.DATA[t][track_idx][:2]  # Get (y, x) coordinates
                track_points.append([y, x])
    
        if len(track_points) > 1:
            lines = np.array([[track_points[i], track_points[i + 1]] for i in range(len(track_points) - 1)])
            if 'track_lines' in viewer.layers:
                if not np.array_equal(viewer.layers['track_lines'].data, lines):
                    viewer.layers['track_lines'].data = lines
            else:
                # Create the shapes layer for the track lines if it doesn't exist
                viewer.add_shapes(
                    lines,
                    shape_type='line',
                    edge_width=1,
                    edge_color='white',
                    name='track_lines',
                    face_color='transparent'
                )
        else:
            if 'track_lines' in viewer.layers:
                viewer.layers['track_lines'].data = np.empty((0, 2, 2))
    
        # Set the points layer as active to enable further interactions
        viewer.layers.selection.active = points_layer

    def add_detections(self):
        # Check if images are already loaded.
        if len(self.viewer.layers) == 0:
            QMessageBox.information(
                None,
                "Please load the images",
                "Before adding detections, images need to be loaded. File > Open Files as Stack ",
            )
        else:
            csv_path, _ = QFileDialog.getOpenFileName(
                None, "Select CSV File", "", "CSV Files (*.csv)"
            )
            if csv_path:
                show_info("Detections are being loaded...")

            # TODO: If there are detections already plotted remove them.

            path, filename = os.path.split(csv_path)
            csv_data = pd.read_csv(csv_path).astype(int)
            csv_data = self._check_and_add_displ_cols(csv_data)

            G = build_graph(data)

            self._compute_track_lines()

            check_and_update_image()

    def add_tracks(self):
        print("Button 2 clicked!")


def trackin_main():
    widget = TrackingWidget()
    widget.setWindowTitle("Tracking")
    return widget
