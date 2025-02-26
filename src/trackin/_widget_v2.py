import os

import napari
import pandas as pd
import numpy as np
from qtpy.QtCore import Qt, QTimer
from napari.utils.events import EventEmitter
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QPushButton,
)

from .shared_state import shared_state
from .tracking import find_node_by_attributes, add_node_with_dummy_edges, generate_track
from .utils_v2 import build_graph_v2, generate_positions_list, track_to_posarray, track_to_lines
from .config import MAX_SCORE, SCORE_FUNC

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
        self.button_add_detect.clicked.connect(self.add_detections_file)
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

    def add_detections_file(self):
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

            show_info("Detections are being loaded...")

            # TODO: If there are detections already plotted remove them.

            csv_data = pd.read_csv(csv_path).astype(int)
            csv_data = self._check_and_add_displ_cols(csv_data)

            if "track_no" in csv_data.columns:
                tracked = True
            else:
                tracked = False

            data = generate_positions_list(df=csv_data, tracked=tracked)
            G = build_graph_v2(data, MAX_SCORE, SCORE_FUNC, tracked)

            track = generate_track(G)
            track_pos = track_to_posarray(G, track, data)

            self.viewer.add_points(csv_data.loc[:, ['tframe', 'y', 'x']].values, size=20, face_color='transparent',
                                   border_color='white', border_width=1,
                                   border_width_is_relative=False,
                                   name='Detections')

            lines = track_to_lines(G, track)
            self.viewer.add_shapes(lines, shape_type='line', edge_width=1, edge_color='white', name='Tracks', face_color='transparent')

    def add_tracks(self):
        print("Button 2 clicked!")


def trackin_main():
    widget = TrackingWidget()
    widget.setWindowTitle("Tracking")
    return widget
