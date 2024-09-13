# _widget.py

import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import QFileDialog, QMessageBox

# Global variables to hold the state
viewer = None  # This will be set when initializing the Napari viewer
current_index = 0
images = []
image_files = []
csv_data = None

def initialize_viewer(napari_viewer):
    """Initialize the Napari viewer object."""
    global viewer
    viewer = napari_viewer

def load_images_from_folder(folder_path):
    """Load images in numerical order from a given folder."""
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif', 'png'))],
        key=lambda x: int(os.path.splitext(x)[0])  # Sort by the numeric value of the filename
    )
    images = []
    for img in image_files:
        try:
            image = imread(os.path.join(folder_path, img))
            images.append(image)
        except Exception as e:
            print(f"Could not open image {img}: {e}")
            QMessageBox.critical(None, "Image Load Error", f"Could not open image {img}: {e}")
    return images, image_files

@magicgui(call_button="Choose Folder")
def choose_folder():
    """Open a dialog to select a folder and load images from it."""
    global images, image_files, current_index, viewer
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder with Images")
    if folder_path:
        images, image_files = load_images_from_folder(folder_path)
        current_index = 0
        if images:
            if len(viewer.layers) > 0:
                viewer.layers.clear()  # Clear any existing layers
            viewer.add_image(images[current_index], name=os.path.basename(image_files[current_index]))

@magicgui(call_button="Next Image")
def next_image():
    """Display the next image in the sequence."""
    global current_index, viewer
    if images:
        current_index = (current_index + 1) % len(images)  # Wrap around to the first image
        viewer.layers[0].data = images[current_index]
        viewer.layers[0].name = os.path.basename(image_files[current_index])

@magicgui(call_button="Previous Image")
def previous_image():
    """Display the previous image in the sequence."""
    global current_index, viewer
    if images:
        current_index = (current_index - 1) % len(images)  # Wrap around to the last image
        viewer.layers[0].data = images[current_index]
        viewer.layers[0].name = os.path.basename(image_files[current_index])

@magicgui(call_button="Load CSV")
def load_csv():
    """Open a dialog to select a CSV file and load its data."""
    global csv_data
    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        try:
            csv_data = pd.read_csv(csv_path)
            print(f"CSV Data Loaded: {csv_path}")
            #QMessageBox.information(None, "CSV Load Success", f"CSV file loaded successfully: {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"Could not load CSV file: {e}")
            QMessageBox.critical(None, "CSV Load Error", f"Could not load CSV file: {e}")
