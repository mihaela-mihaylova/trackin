import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import QFileDialog, QMessageBox, QWidget, QVBoxLayout, QPushButton
from qtpy.QtCore import Qt
import napari
import numpy as np

# Global variables to hold the state
viewer = None
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

    # Set up key bindings for the viewer
    setup_keybindings()

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

@magicgui(call_button="Choose Folder", auto_call=True)
def choose_folder():
    """Open a dialog to select a folder and load images from it."""
    global images, image_files, current_index, viewer, images_loaded
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder with Images")
    if folder_path:
        images, image_files = load_images_from_folder(folder_path)
        current_index = 0
        images_loaded = True
        if images:
            viewer.layers.clear()
            viewer.add_image(images[current_index], name=os.path.basename(image_files[current_index]))
            check_and_update_image()
            update_slider_max()

@magicgui(call_button="Load CSV", auto_call=True)
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

def next_image(event=None):
    """Display the next image in the sequence and overlay CSV data."""
    global current_index, viewer
    if images:
        current_index = (current_index + 1) % len(images)
        update_image()
        image_slider.image_index.value = current_index  # Sync slider value
    viewer.window.qt_viewer.setFocus()  # Keep focus on the viewer

def previous_image(event=None):
    """Display the previous image in the sequence and overlay CSV data."""
    global current_index, viewer
    if images:
        current_index = (current_index - 1) % len(images)
        update_image()
        image_slider.image_index.value = current_index  # Sync slider value
    viewer.window.qt_viewer.setFocus()  # Keep focus on the viewer

@magicgui(image_index={"widget_type": "Slider", "min": 0, "max": 0, "step": 1, "label": "Frame"}, auto_call=True)
def image_slider(image_index: int = 0):
    """Update the displayed image based on the slider value."""
    global current_index, viewer
    if image_index != current_index:
        current_index = image_index
        update_image()
        # Refocus the viewer after slider change
        viewer.window.qt_viewer.setFocus()


def update_slider_max():
    """Update the maximum value of the slider based on the number of images."""
    if images:
        image_slider.image_index.max = len(images) - 1
        image_slider.image_index.value = current_index
    else:
        image_slider.image_index.max = 0
        image_slider.image_index.value = 0

def update_image():
    """Update the displayed image and overlay CSV data."""
    global viewer
    if images:
        viewer.layers.clear()
        viewer.add_image(images[current_index], name=os.path.basename(image_files[current_index]))

        if csv_data is not None:
            frame_number = int(os.path.splitext(os.path.basename(image_files[current_index]))[0])
            frame_data = csv_data[csv_data['tframe'] == frame_number]
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
            face_color='transparent',
            border_color='white',
            name='Overlay Points'
        )

def setup_keybindings():
    """Set up key bindings for the viewer."""
    viewer.bind_key('Right', next_image)  # Right arrow key to move to the next image
    viewer.bind_key('Left', previous_image)  # Left arrow key to move to the previous image


def trackin_main():
    """Main function to show the plugin interface."""
    # Create a QWidget to act as the container
    container = QWidget()
    layout = QVBoxLayout()  # Create a vertical layout
    
    # Add buttons to the layout
    '''choose_folder_button = QPushButton("Choose Folder")
    choose_folder_button.clicked.connect(choose_folder)

    load_csv_button = QPushButton("Load CSV")
    load_csv_button.clicked.connect(load_csv)

    next_image_button = QPushButton("Next Image")
    next_image_button.clicked.connect(next_image)

    previous_image_button = QPushButton("Previous Image")
    previous_image_button.clicked.connect(previous_image)'''

    # Use the magicgui widgets and add them directly to the layout
    layout.addWidget(choose_folder.native)  # Add the magicgui widget's native Qt widget
    layout.addWidget(load_csv.native)
    #layout.addWidget(next_image.native)
    #layout.addWidget(previous_image.native)
    layout.addWidget(image_slider.native)  # Add the slider widget

    # Set layout and return the widget
    container.setLayout(layout)
    return container