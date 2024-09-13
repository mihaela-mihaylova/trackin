import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import QFileDialog, QMessageBox
import napari
import numpy as np

# Global variables to hold the state
viewer = None  # Will be set when initializing the Napari viewer
current_index = 0
images = []
image_files = []
csv_data = None
images_loaded = False
csv_loaded = False

def initialize_viewer(napari_viewer):
    """Initialize the Napari viewer object."""
    global viewer
    viewer = napari_viewer

def load_images_from_folder(folder_path):
    """Load images in numerical order from a given folder."""
    global images, image_files
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif'))],
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

def check_and_update_image():
    """Check if both images and CSV data are loaded, then update the image."""
    global images_loaded, csv_loaded
    if images_loaded and csv_loaded:
        update_image()

@magicgui(call_button="Choose Folder")
def choose_folder():
    """Open a dialog to select a folder and load images from it."""
    global images, image_files, current_index, viewer, images_loaded
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder with Images")
    if folder_path:
        images, image_files = load_images_from_folder(folder_path)
        current_index = 0
        images_loaded = True
        if images:
            # Clear existing layers and add the new image layer
            viewer.layers.clear()
            viewer.add_image(images[current_index], name=os.path.basename(image_files[current_index]))
            check_and_update_image()

@magicgui(call_button="Load CSV")
def load_csv():
    """Open a dialog to select a CSV file and load its data."""
    global csv_data, csv_loaded
    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        try:
            csv_data = pd.read_csv(csv_path)
            print(f"CSV Data Loaded: {csv_path}")
            QMessageBox.information(None, "CSV Load Success", f"CSV file loaded successfully: {os.path.basename(csv_path)}")
            csv_loaded = True
            check_and_update_image()
        except Exception as e:
            print(f"Could not load CSV file: {e}")
            QMessageBox.critical(None, "CSV Load Error", f"Could not load CSV file: {e}")

@magicgui(call_button="Next Image")
def next_image():
    """Display the next image in the sequence and overlay CSV data."""
    global current_index, viewer
    if images:
        current_index = (current_index + 1) % len(images)  # Wrap around to the first image
        update_image()

@magicgui(call_button="Previous Image")
def previous_image():
    """Display the previous image in the sequence and overlay CSV data."""
    global current_index, viewer
    if images:
        current_index = (current_index - 1) % len(images)  # Wrap around to the last image
        update_image()

def update_image():
    """Update the displayed image and overlay CSV data."""
    global viewer
    if images and csv_data is not None:
        # Clear existing layers and add the new image layer
        viewer.layers.clear()
        viewer.add_image(images[current_index], name=os.path.basename(image_files[current_index]))

        # Get the frame number corresponding to the current image
        frame_number = int(os.path.splitext(os.path.basename(image_files[current_index]))[0])

        # Filter CSV data for the current frame
        frame_data = csv_data[csv_data['tframe'] == frame_number]

        # Overlay the circles on the image
        overlay_points(frame_data)

def overlay_points(frame_data):
    """Overlay white circles of radius 20 pixels on the image for each (x, y) in the frame data."""
    global viewer
    if frame_data.empty:
        return

    points = np.array([frame_data['y'], frame_data['x']]).T

    if points.size > 0:
        viewer.add_points(
            points,
            size=20,
            face_color='transparent',  # Ensure transparency
            edge_color='white',
            name='Overlay Points'
        )
