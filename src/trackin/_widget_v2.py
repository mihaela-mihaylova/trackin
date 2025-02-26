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
from .tracking import (
    find_node_by_attributes,
    add_node_with_dummy_edges,
    generate_track,
)
from .utils_v2 import (
    build_graph_v2,
    generate_positions_list,
    track_to_posarray,
    track_to_lines,
)
from .config import MAX_SCORE, SCORE_FUNC


class TrackingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()

        # Show the first frame.
        self.viewer.dims.set_current_step(axis=0, value=0)

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

    def on_points_data_change(self, event):
        """Event handler for adding and removing detections"""

        if event.action == "adding":
            pass

        elif event.action == "added":
            tframe, y, x = event.value[-1]

            tframe = int(tframe)
            y = int(y)
            x = int(x)

            self.data[tframe].append([y, x, 0, 0])

            self.G = add_node_with_dummy_edges(
                node=f"D_{tframe}_{self.n_detect_per_frame[tframe]}",
                time_point=tframe,
                idx=self.n_detect_per_frame[tframe],
                y=int(y),
                x=int(x),
                displ_y=0,
                displ_x=0,
                G=self.G,
                highest_frame_id=len(self.data) - 1,
                max_score=MAX_SCORE,
            )
            self.n_detect_per_frame[tframe] += 1

            # TODO: How to update the graph and the tracks?

        elif event.action == "removing":
            # TODO: Can use remove multiple detections at once?
            tframe, y, x = event.source._data[event.data_indices[0]]

            # TODO: How to update the graph and the tracks?

    def on_frame_change(self, event):
        print("Changed frame")

    @staticmethod
    def _check_and_add_displ_cols(df):
        """Add displ_x and add displ_y if these are not present in loaded df"""
        if "displ_x" not in df.columns:
            df["displ_x"] = 0
        if "displ_y" not in df.columns:
            df["displ_y"] = 0
        return df

    def add_detections_file(self):
        """
        Functionality for adding detections from a CSV file. Used by the "Add Detections" button.
        """
        # Step 0: Check if images are already loaded. If not prompt to load images.
        if len(self.viewer.layers) == 0:
            QMessageBox.information(
                None,
                "Please load the images",
                "Before adding detections, images need to be loaded. File > Open Files as Stack ",
            )
        else:
            # Step 1: Load the detections file
            csv_path, _ = QFileDialog.getOpenFileName(
                None, "Select CSV File", "", "CSV Files (*.csv)"
            )

            show_info("Loading detections and generating tracks. Please wait...")

            # TODO: If there are detections already plotted remove them.

            csv_data = pd.read_csv(csv_path).astype(int)
            csv_data = self._check_and_add_displ_cols(csv_data)

            if "track_no" in csv_data.columns:
                tracked = True
            else:
                tracked = False

            # Step 2: Generate the graph and one track
            self.data = generate_positions_list(df=csv_data, tracked=tracked)
            self.n_detect_per_frame = [len(frame) for frame in self.data]
            self.G = build_graph_v2(self.data, MAX_SCORE, SCORE_FUNC, tracked)

            track = generate_track(self.G)
            track_pos = track_to_posarray(self.G, track, self.data)

            # Step 3: Add all the detections.
            points_layer = self.viewer.add_points(
                csv_data.loc[:, ["tframe", "y", "x"]].values,
                size=20,
                face_color="transparent",
                border_color="white",
                border_width=1,
                border_width_is_relative=False,
                name="Detections",
            )

            # Step 4: Add event hander for changes to point layer
            points_layer.events.data.connect(self.on_points_data_change)

            # Step 5: Add the first track
            lines = track_to_lines(self.G, track)
            self.viewer.add_shapes(
                lines,
                shape_type="line",
                edge_width=1,
                edge_color="white",
                name="Tracks",
                face_color="transparent",
            )

            # Step 6: Add event handler for moving between frames.
            self.viewer.dims.events.current_step.connect(self.on_frame_change)

            show_info("Done!")

    def add_tracks(self):
        """
        Functionality for adding tracks to the images. Used by the "Add Track File" button.
        """
        print("Button 2 clicked!")


def trackin_main():
    widget = TrackingWidget()
    widget.setWindowTitle("Tracking")
    return widget
