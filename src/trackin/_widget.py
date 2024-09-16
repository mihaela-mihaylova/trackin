import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import QFileDialog, QMessageBox, QWidget, QVBoxLayout
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
container = None  # Global reference to the container

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


def generate_positions_list(df):
    DATA = []
    for _, dft in df.groupby('tframe'):
        positions = []
        for _, row in dft.iterrows():
            positions.append((row['y'], row['x'], row['displ_y'], row['displ_x']))
        DATA.append(positions)

    # Initialize the session.csv file
    with open("session.csv", "w") as f:
        f.write("")

    return DATA  # Return the generated positions list

def check_and_add_displ_cols(df):
    if 'displ_x' not in df.columns:
        df['displ_x'] = 0
    if 'displ_y' not in df.columns:
        df['displ_y'] = 0
    return df

@magicgui(call_button="Load CSV", auto_call=True)
@magicgui(call_button="Load CSV", auto_call=True)
def load_csv():
    """Open a dialog to select a CSV file and load its data."""
    global csv_data, csv_loaded, DATA
    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        try:
            csv_data = pd.read_csv(csv_path)
            # Check if displacement columns exist, if not, add them with value 0
            csv_data = check_and_add_displ_cols(csv_data)
            DATA = generate_positions_list(csv_data)  # Store the returned positions list in DATA
            print(f"CSV Data Loaded: {csv_path}")
            csv_loaded = True
            check_and_update_image()
        except Exception as e:
            print(f"Could not load CSV file: {e}")
            QMessageBox.critical(None, "CSV Load Error", f"Could not load CSV file: {e}")


def next_image(event=None):
    """Display the next image in the sequence and overlay CSV data."""
    global current_index, viewer, container
    if images:
        current_index = (current_index + 1) % len(images)
        update_image()
        image_slider.image_index.value = current_index  # Sync slider value
    
    # Refocus the container after key press
    if container:
        container.setFocus()

def previous_image(event=None):
    """Display the previous image in the sequence and overlay CSV data."""
    global current_index, viewer, container
    if images:
        current_index = (current_index - 1) % len(images)
        update_image()
        image_slider.image_index.value = current_index  # Sync slider value
    
    # Refocus the container after key press
    if container:
        container.setFocus()

@magicgui(image_index={"widget_type": "Slider", "min": 0, "max": 0, "step": 1, "label": "Frame"}, auto_call=True)
def image_slider(image_index: int = 0):
    """Update the displayed image based on the slider value."""
    global current_index, viewer, container
    if image_index != current_index:
        current_index = image_index
        update_image()
        # Refocus the container after slider change
        if container:
            container.setFocus()

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
    global viewer, DATA

    if images:
        # Clear previous layers and add the current image
        viewer.layers.clear()
        viewer.add_image(images[current_index], name=os.path.basename(image_files[current_index]))

        # Ensure DATA and csv_data are loaded
        if csv_data is not None and DATA is not None:
            # Get the frame number from the current image file name
            frame_number = int(os.path.splitext(os.path.basename(image_files[current_index]))[0])
            
            # Ensure the frame number is within the range of DATA
            if 0 <= frame_number < len(DATA):
                frame_data = DATA[frame_number]  # Access the corresponding frame data from the list
                overlay_points(pd.DataFrame(frame_data, columns=['y', 'x', 'displ_y', 'displ_x']))  # Convert list to DataFrame for easier handling


def overlay_points(frame_data):
    """Overlay white circles of radius 20 pixels on the image for each (x, y) in the frame data."""
    global viewer
    if frame_data.empty:
        return

    points = np.array([frame_data['y'], frame_data['x']]).T

    if points.size > 0:
        # Create the points layer
        points_layer = viewer.add_points(
            points,
            size=20,
            face_color='transparent',
            border_color='white',
            name='detections'
        )

        # Add mouse click event handler to the points layer
        points_layer.mouse_drag_callbacks.append(delete_detection)
        points_layer.mouse_drag_callbacks.append(add_detection)  # Add the shift+click handler


def delete_detection(layer, event):
    """Delete detection."""
    if event.button == 2:  # Right-click
        # Get the coordinates of the clicked point in world coordinates
        click_position = event.position

        # Find the index of the closest point to the click
        clicked_index = layer.get_value(event.position, world=True)

        if clicked_index is not None:
            # Retrieve the coordinates of the point at the clicked index
            point_coords = layer.data[clicked_index].copy()
            #print(f"Clicked on overlay circle at index: {clicked_index}, coordinates: {point_coords}")
            layer.data[clicked_index] = [-1000000, -1000000]
            DATA[current_index][clicked_index] = (-1000000, -1000000)
            layer.refresh()
            print(f'Removed point with coordinates {point_coords}')

def add_detection(layer, event):
    """Handle shift + left mouse click events to add a new point."""
    # Check if the Shift key is pressed and the left mouse button (button 1) is clicked
    if event.button == 1 and 'Shift' in event.modifiers:
        # Get the coordinates of the click position in world coordinates
        click_position = event.position
        # Append the new point to the layer's data
        new_point = [click_position[0], click_position[1]]  # [y, x] format
        layer.data = np.vstack([layer.data, new_point])  # Add the new point to the existing data
        DATA[current_index].append([click_position[0], click_position[1],0,0])
        # Refresh the layer to display the new point
        layer.refresh()

        print(f"Added new point at coordinates: {new_point}")


def setup_keybindings():
    """Set up key bindings for the viewer."""
    viewer.bind_key('Right', next_image)  # Right arrow key to move to the next image
    viewer.bind_key('Left', previous_image)  # Left arrow key to move to the previous image

def trackin_main():
    """Main function to show the plugin interface."""
    global container
    container = QWidget()
    layout = QVBoxLayout(container)  # Create a vertical layout
    
    # Use the magicgui widgets and add them directly to the layout
    layout.addWidget(choose_folder.native)  # Add the magicgui widget's native Qt widget
    layout.addWidget(load_csv.native)
    layout.addWidget(image_slider.native)  # Add the slider widget

    # Set layout to the container
    container.setLayout(layout)

    # Set focus policy and initially focus the container
    container.setFocusPolicy(Qt.StrongFocus)
    container.setFocus()

    return container
