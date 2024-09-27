import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import QFileDialog, QMessageBox, QWidget, QVBoxLayout
from qtpy.QtCore import Qt
import napari
import numpy as np
from napari.utils.events import EventEmitter
from .shared_state import shared_state
from datetime import datetime
from .tracking import find_node_by_attributes

viewer = None
current_index = 0
images = []
image_files = []
csv_data = None
images_loaded = False
csv_loaded = False
container = None  # Global reference to the container

# EVENT EMITTERS
csv_loaded_event = EventEmitter(source=None, type_name='csv_loaded')
data_updated_event = EventEmitter(source=None, type_name='data_updated')
remove_track_node_event = EventEmitter(source=None, type_name='remove_track_node')


def initialize_viewer(napari_viewer):
    """Initialize the Napari viewer object."""
    global viewer
    viewer = napari_viewer

    # Set up key bindings for the viewer
    setup_keybindings()


 # adds these displacement columns with value 0, in case they are not present
def check_and_add_displ_cols(df):
    if 'displ_x' not in df.columns:
        df['displ_x'] = 0
    if 'displ_y' not in df.columns:
        df['displ_y'] = 0
    return df

# takes a df and turns it into a list of lists, necessary for the way data is read in the tool
def generate_positions_list(df, folder_to_save):
    for _, dft in df.groupby('tframe'):
        positions = []
        for _, row in dft.iterrows():
            if not shared_state.TRACKED:
                positions.append((row['y'], row['x'], row['displ_y'], row['displ_x']))
            else:
                positions.append((row['y'], row['x'], row['displ_y'], row['displ_x'], row['track_no']))

        shared_state.DATA.append(positions)

    # generate the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # initialize the session.csv file, where tracks are saved
    with open(os.path.join(folder_to_save, f"session_{timestamp}.csv"), "w") as f:
        f.write("")
    #print(DATA)
    return shared_state.DATA 

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
            # Add white border to the image
            image_with_border = add_white_border(image, border_size=2)
            images.append(image_with_border)
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
    global csv_loaded, csv_data
    """Open a dialog to select a CSV file and load its data."""
    
    # Check if images are loaded
    if not images_loaded:
        # Show a popup window informing the user that images should be loaded first
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Load Images First")
        msg_box.setText("Please load the images before loading the CSV file.")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        return  # Exit the function without proceeding

    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        #try:
        csv_data = pd.read_csv(csv_path).astype(int)
        # Check if displacement columns exist, if not, add them with value 0
        csv_data = check_and_add_displ_cols(csv_data)
        shared_state.TRACKED = 'track_no' in csv_data.columns
        # Rearrange columns in df (in case we have tframe, y, x, track_no)
        if shared_state.TRACKED:
            csv_data = csv_data[['tframe', 'y', 'x', 'displ_y', 'displ_x', 'track_no']]
        folder_to_save = os.path.dirname(csv_path)
        shared_state.DATA = generate_positions_list(csv_data, folder_to_save)  # Store the returned positions list in DATA
        
        csv_loaded = True
        # Emit event to trigger track function in utils
        csv_loaded_event()  

        check_and_update_image()

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

def add_white_border(image, border_size=1):
    """Add a white border around the image."""
    # Create a new image with a white border around the original image
    new_image_shape = (
        image.shape[0] + 2 * border_size,
        image.shape[1] + 2 * border_size,
        3 if image.ndim == 3 else 1  # Handle both grayscale and RGB images
    )
    
    if image.ndim == 2:  # Grayscale image
        new_image = np.ones(new_image_shape[:2], dtype=image.dtype) * 255  # White border
        new_image[border_size:-border_size, border_size:-border_size] = image
    else:  # RGB image
        new_image = np.ones(new_image_shape, dtype=image.dtype) * 255  # White border
        new_image[border_size:-border_size, border_size:-border_size, :] = image
    
    return new_image

def update_image():
    """Update the displayed image and overlay CSV data without changing zoom level."""
    global viewer
    if images:
        # Get the current image and add a white border to it
        bordered_image = add_white_border(images[current_index], border_size=2)

        # Store the current camera settings (zoom and center)
        current_zoom = viewer.camera.zoom
        current_center = viewer.camera.center

        # Clear previous layers and add the current image with the white border
        viewer.layers.clear()
        viewer.add_image(bordered_image, name=os.path.basename(image_files[current_index]))

        # Restore the camera zoom and center position
        viewer.camera.zoom = current_zoom
        viewer.camera.center = current_center

        # Ensure DATA and csv_data are loaded
        if csv_data is not None and shared_state.DATA is not None:
            # Get the frame number from the current image file name
            frame_number = int(os.path.splitext(os.path.basename(image_files[current_index]))[0])
            
            # Ensure the frame number is within the range of DATA
            if 0 <= frame_number < len(shared_state.DATA):
                frame_data = shared_state.DATA[frame_number]  # Access the corresponding frame data from the list
                # Dynamically adjust column names based on whether `track_no` is present
                if not shared_state.TRACKED:
                    columns = ['y', 'x', 'displ_y', 'displ_x']  
                else:
                    columns = ['y', 'x', 'displ_y', 'displ_x', 'track_no']
                   
                # Create the DataFrame with the appropriate number of columns
                overlay_points(pd.DataFrame(frame_data, columns=columns))



def overlay_points(frame_data):
    """Overlay white circles on the image for each (x, y) in the frame data, with special styling for the track point."""
    global viewer

    print("Overlay function called")  # Debugging to ensure the function is called

    if frame_data.empty:
        print("No frame data provided")
        return

    # Extract (y, x) positions from the dataframe
    points = np.array([frame_data['y'], frame_data['x']]).T
    curr_track = shared_state.track

    # Prepare default attributes for all points
    border_colors = ['white'] * len(points)
    sizes = [20] * len(points)  # Default size
    border_widths = [1] * len(points)  # Default border width

    # Check if track contains detection in this frame and apply custom styles for the track point
    if curr_track[current_index] != -1:
        track_index = curr_track[current_index]
        border_colors[track_index] = 'yellow'  # Highlight the track point in yellow
        sizes[track_index] = 30  # Increase the size of the track point
        border_widths[track_index] = 3  # Ensure it's between 0 and 1 for relative mode

    if points.size > 0:
        # Create the points layer with custom attributes for the track point
        points_layer = viewer.add_points(
            points,
            size=sizes,  # Apply individual sizes
            face_color='transparent',
            border_color=border_colors,  # Apply individual border colors
            border_width=border_widths,  # Apply individual border widths
            border_width_is_relative=False,  # Set to relative mode
            name='detections'
        )

        # Function to switch focus to points_layer when clicked
        '''def switch_to_points_layer(layer, event):
            viewer.layers.selection.active = layer
            print("Switched to points layer")

        # Add the callback to switch focus to points layer
        points_layer.mouse_drag_callbacks.append(switch_to_points_layer)'''

        # Add mouse click event handlers to the points layer
        points_layer.mouse_drag_callbacks.append(delete_detection)
        points_layer.mouse_drag_callbacks.append(add_detection)  # Add the shift+click handler

    # Initially, set the points layer as active
    viewer.layers.selection.active = points_layer

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
            shared_state.DATA[current_index][clicked_index] = (-1000000, -1000000)
            layer.refresh()
            print(f'Removed point with coordinates {point_coords}')

def add_detection(layer, event):
    """Handle shift + left mouse click events to add a new point within image bounds."""
    if event.button == 1 and 'Shift' in event.modifiers:
        # Convert world coordinates to data coordinates (pixel coordinates)
        data_coords = layer.world_to_data(event.position)

        # Get the dimensions of the currently displayed image
        image_shape = images[current_index].shape  # This gets (height, width) for the image

        # Ensure the click is within the image bounds
        if (0 <= data_coords[0] < image_shape[0]) and (0 <= data_coords[1] < image_shape[1]):
            # Append the new point to the layer's data in data coordinates (pixels)
            new_point = [data_coords[0], data_coords[1]]  # [y, x] format

            # Add the new point to the existing points layer data
            layer.data = np.vstack([layer.data, new_point])

            # Determine the default point size (20) or get the minimum size of existing points
            if len(layer.size) > 0:
                point_size = 20
           
            # Create a new size array matching the updated data length
            new_size_array = np.full((len(layer.data),), point_size)

            # Ensure the size of the track detection in the current frame is changed to 30
            curr_track = shared_state.track
            if curr_track[current_index] != -1:
                track_index = curr_track[current_index]  # Get the correct track index
                new_size_array[track_index] = 30  # Set track point size to 30

            # Update the size array with the correct shape
            layer.size = new_size_array
            layer.border_width_is_relative = False  # Ensure the border width is absolute

            # Ensure the new point is added with the correct data in shared_state
            shared_state.DATA[current_index].append([data_coords[0], data_coords[1], 0, 0])

            # Refresh the layer to display the new point
            layer.refresh()

            print(f"Added new point at data coordinates: {new_point}, with size: {point_size}")
        else:
            print("Clicked outside the image bounds. Point not added.")

def delete_track_detection(layer):
    """Delete the track detection triggered by the 'D' key."""
    # Check if track contains detection in this frame
    curr_track = shared_state.track
    if curr_track[current_index] != -1:
        point_index = curr_track[current_index]
        point_coords = layer.data[point_index].copy()

        # Mark the track point as deleted (set it to a large out-of-bounds value)
        layer.data[point_index] = [-1000000, -1000000]
        shared_state.DATA[current_index][point_index] = (-1000000, -1000000)

        # Update the track for the current frame, marking it as deleted
        shared_state.track[current_index] = -1  # This explicitly updates the track index to -1
        node_to_remove = find_node_by_attributes(shared_state.G, time_point=current_index, idx=point_index)[0]
        shared_state.G.remove_node(node_to_remove)
        remove_track_node_event()
        update_image()
        layer.refresh()
        print(f"Removed track point with coordinates {point_coords}")
    else:
        print("No track point found to delete in the current frame.")

def setup_keybindings():
    """Set up key bindings for the viewer."""
    viewer.bind_key('Right', next_image)  # Right arrow key to move to the next image
    viewer.bind_key('Left', previous_image)  # Left arrow key to move to the previous image
    viewer.bind_key('D', lambda event: delete_track_detection(viewer.layers.selection.active))  # Bind 'D' key to delete track detection

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