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

    def add_detections(self):
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

    def add_tracks(self):
        print("Button 2 clicked!")


def trackin_main():
    widget = TrackingWidget()
    widget.setWindowTitle("Tracking")
    return widget
