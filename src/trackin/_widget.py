import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import QFileDialog, QMessageBox, QWidget, QVBoxLayout, QLabel  
from qtpy.QtCore import Qt, QTimer
import napari
import numpy as np
from napari.utils.events import EventEmitter
from .shared_state import shared_state
from datetime import datetime
from .tracking import add_node_with_dummy_edges
from napari.utils.notifications import show_info

viewer = None
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
accept_track_event = EventEmitter(source=None, type_name='accept_track')
delete_segment_event = EventEmitter(source=None, type_name='delete_segment')
save_segment_event = EventEmitter(source=None, type_name='save_segment')
delete_all_connections_event = EventEmitter(source=None, type_name='delete_all_connections')
graph_updated_event = EventEmitter(source=None, type_name='graph_updated')


# Create timers for handling continuous key press events
right_timer = QTimer()
left_timer = QTimer()


def initialize_viewer(napari_viewer):
    """Initialize the Napari viewer object."""
    global viewer
    viewer = napari_viewer

    # Set up key bindings for the viewer
    setup_keybindings()

    # Set up the timers for handling repeated key presses
    right_timer.timeout.connect(next_image)
    left_timer.timeout.connect(previous_image)


 # adds these displacement columns with value 0, in case they are not present
def check_and_add_displ_cols(df):
    """Add displ_x and add displ_y if these are not present in loaded df"""
    if 'displ_x' not in df.columns:
        df['displ_x'] = 0
    if 'displ_y' not in df.columns:
        df['displ_y'] = 0
    return df

def generate_positions_list(df, folder_to_save):
    """Load positions data into a list of lists, required by our tracking algorithm"""
    for _, dft in df.groupby('tframe'):
        positions = []
        for _, row in dft.iterrows():
            if not shared_state.TRACKED:
                positions.append((row['y'], row['x'], row['displ_y'], row['displ_x']))
            else:
                positions.append((row['y'], row['x'], row['displ_y'], row['displ_x'], row['track_no']))

        shared_state.DATA.append(positions)

    # initialize the session.csv file, where tracks are saved
    with open(os.path.join(folder_to_save, f"{shared_state.SESSION_FILE}"), "w") as f:
        f.write("")
    shared_state.NUM_DET_PER_FRAME = [len(sublist) for sublist in shared_state.DATA]
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

@magicgui(call_button="Load Images", auto_call=True)
def choose_folder():
    """Open a dialog to select a folder and load images from it."""
    global images, image_files, viewer, images_loaded
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder with Images")
    if folder_path:
        clear_detections_and_tracks()
        images, image_files = load_images_from_folder(folder_path)
        shared_state.current_index = 0
        images_loaded = True
        if images:
            viewer.layers.clear()
            viewer.add_image(images[shared_state.current_index], name=os.path.basename(image_files[shared_state.current_index]))
            check_and_update_image()
            update_slider_max()

@magicgui(call_button="Load Detections", auto_call=True)
def load_csv():
    global csv_loaded, csv_data
    """Open a dialog to select a CSV file and load its data."""
    print(csv_loaded)
    if not images_loaded:
        QMessageBox.information(None, "Load Images First", "Please load the images before loading the CSV file.")
        return
    
    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        show_info("Detections are being loaded...")
        if csv_loaded:
            clear_detections_and_tracks()
                
        csv_filename = csv_path.split('/')[-1].split('.')[0]
        # generate the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # name for file where generated tracks are saved
        shared_state.SESSION_FILE = f'track_session_{csv_filename}_{timestamp}.csv'
        # name for file where leftover positions are preserved
        shared_state.UPDATED_DATA_FILE = f'upd_{csv_filename}_{timestamp}.csv'

        csv_data = pd.read_csv(csv_path).astype(int)
        csv_data = check_and_add_displ_cols(csv_data)
        shared_state.TRACKED = 'track_no' in csv_data.columns
        if shared_state.TRACKED:
            csv_data = csv_data[['tframe', 'y', 'x', 'displ_y', 'displ_x', 'track_no']]
        shared_state.csv_folder_to_save = os.path.dirname(csv_path)
        shared_state.DATA = generate_positions_list(csv_data, shared_state.csv_folder_to_save)  
        csv_loaded = True
        csv_loaded_event()  # Trigger CSV loaded event

        # Compute track lines once when CSV is loaded
        graph_updated_event()

        compute_track_lines()

        check_and_update_image()

        
# loads a file with already accepted tracks and adds these and any subsequent accepted tracks to a new track file
@magicgui(call_button="Add Track File", auto_call=True)
def load_track_file():
    """Open a dialog to select a CSV file and load a csv with tracks generated in a previous session."""
    
    if not images_loaded or not csv_loaded:
        QMessageBox.information(None, "Load Images and Detections First.", "Please load the images and detections before loading the track file to add new tracks to.")
        return

    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        csv_filename = csv_path.split('/')[-1].split('.')[0]
        # generate the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # name for file where leftover positions are preserved
        shared_state.UPD_TRACK_FILE = f'with_new_tracks_added_{csv_filename}_{timestamp}.csv'

        try:
            track_df = pd.read_csv(csv_path).astype(int)
            expected_columns = ['tframe', 'y', 'x', 'displ_y', 'displ_x', 'track_no']
            
            # Check if the columns match the expected format
            if list(track_df.columns) != expected_columns:
                # Display error message if columns are incorrect
                QMessageBox.critical(None, "CSV Format Error", f"CSV file must have the following columns in order: {expected_columns}")
                return
            
            # Check if the file is empty or only has a header
            if track_df.empty or (len(track_df) == 0 and not track_df.columns.empty):
                QMessageBox.critical(None, "File Is Empty", "The loaded file is empty or only contains a header.")
                return
            
            # Load the current session tracks, if any
            # note that first part of the checked_session_file conditions is always met, added here because otherwise an error is thrown
            complete_path_to_session_file = os.path.join(shared_state.csv_folder_to_save, shared_state.SESSION_FILE)
            check_session_file = (os.path.getsize(complete_path_to_session_file) == 0)
            print(f'path exists: {os.path.exists(os.path.join(shared_state.csv_folder_to_save, shared_state.SESSION_FILE))}')
            print(f'size is 0: {os.path.getsize(os.path.join(shared_state.csv_folder_to_save, shared_state.SESSION_FILE)) == 0}')
            print(f'check_session_file:{check_session_file}')

            if check_session_file:
                # If session_df is empty, save track_df directly with the updated file name
                updated_track_df_path = os.path.join(shared_state.csv_folder_to_save, shared_state.UPD_TRACK_FILE)
                track_df.to_csv(updated_track_df_path, index=False)
                max_track_no = track_df['track_no'].max()
                show_info("Track File Added. \n The track file has been successfully added. New tracks will be added to the existing tracks.")
                
            else:
                session_df = pd.read_csv(os.path.join(shared_state.csv_folder_to_save, shared_state.SESSION_FILE))
                print(session_df)

                # If session_df has data, extract the max value of track_no from track_df
                max_track_no = track_df['track_no'].max()
                
                session_df['track_no'] = session_df['track_no'] + max_track_no
                print(session_df)
                
                # Concatenate the DataFrames
                updated_tracks_df = pd.concat([track_df, session_df], ignore_index=True)
                
                # Save the combined tracks to the new track file
                updated_tracks_df.to_csv(os.path.join(shared_state.csv_folder_to_save, f"{shared_state.UPD_TRACK_FILE}"), index=False)
                show_info("Track File Updated. \n The track file has been successfully updated with the new tracks.")
    
            shared_state.MAX_TRACK_ID = max_track_no

        except pd.errors.ParserError as e:
            QMessageBox.critical(None, "File Error", f"The selected file is not a valid CSV or is malformed: {e}")
            return
        except Exception as e:
            QMessageBox.critical(None, "Error", f"An unexpected error occurred: {e}")

# go to following image
def next_image(event=None):
    """Display the next image in the sequence and overlay CSV data."""
    global viewer, container
    if images:
        shared_state.current_index = (shared_state.current_index + 1) % len(images)
        update_image()
        image_slider.image_index.value = shared_state.current_index  # Sync slider value
    
    # Refocus the container after key press
    if container:
        container.setFocus()

# go to previous image
def previous_image(event=None):
    """Display the previous image in the sequence and overlay CSV data."""
    global viewer, container
    if images:
        shared_state.current_index = (shared_state.current_index - 1) % len(images)
        update_image()
        image_slider.image_index.value = shared_state.current_index  # Sync slider value
    
    # Refocus the container after key press
    if container:
        container.setFocus()

# responsible for the image slider which moves when images are changed
@magicgui(image_index={"widget_type": "Slider", "min": 0, "max": 0, "step": 1, "label": "Frame"}, auto_call=True)
def image_slider(image_index: int = 0):
    """Update the displayed image based on the slider value."""
    global viewer, container
    if image_index != shared_state.current_index:
        shared_state.current_index = image_index
        update_image()
        # Refocus the container after slider change
        if container:
            container.setFocus()

def update_slider_max():
    """Update the maximum value of the slider based on the number of images."""
    if images:
        image_slider.image_index.max = len(images) - 1
        image_slider.image_index.value = shared_state.current_index
    else:
        image_slider.image_index.max = 0
        image_slider.image_index.value = 0

# adds white border to images to distinct it from background
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

def overlay_points(frame_data):
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


def compute_track_lines():
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

def delete_detection(layer, event=None, use_key=False):
    """Delete detection or track detection based on right-click or 'D' key."""
    global clicked_index
    clicked_index = None
    # Determine the clicked index based on input type
    if not use_key:  # Called from mouse event
        if event.button == 2:  # Right-click
            click_position = event.position
            print(f'clicked position:{click_position}')
            clicked_index = layer.get_value(click_position, world=True)
            if clicked_index is None:
                return  # No valid point was clicked, exit
    else:  # Called using the 'D' key
        curr_track = shared_state.track
        if curr_track[shared_state.current_index] != -1:
            clicked_index = curr_track[shared_state.current_index]  # Use the tracked index if available
        else:
            print("No detection found to delete with 'D' key.")
            return

    # If a valid point is found, determine if it's part of a track or a standard detection
    if clicked_index is not None:
        curr_track = shared_state.track
        if curr_track[shared_state.current_index] == clicked_index:
            # Call the function for deleting track detection
            delete_det_by_key(layer)
        else:
            delete_det_by_mouse(layer, event)

        graph_updated_event()

    # Mark the event as handled if called from a mouse event, otherwise mouse gets locked
    if event is not None:
        event.handled = True

def delete_det_by_key(layer):
    """Delete track detection using the 'D' key."""
    point_coords = layer.data[clicked_index].copy()
    layer.data[clicked_index] = [-1000000, -1000000]

    if shared_state.TRACKED:
        shared_state.DATA[shared_state.current_index][clicked_index] = [-1000000, -1000000, 0, 0, None]
    else:
        shared_state.DATA[shared_state.current_index][clicked_index] = [-1000000, -1000000, 0, 0]

    node_name = f'D_{shared_state.current_index}_{clicked_index}'
    shared_state.G.remove_node(node_name)

    # Update the track state
    shared_state.track[shared_state.current_index] = -1
    remove_track_node_event()

    # Refresh the layer to update the visual display
    layer.refresh()
    update_image()

    print(f"Removed node {node_name} with coordinates {point_coords}.")


def delete_det_by_mouse(layer, event):
    """Delete track detection using right-click."""
    point_coords = layer.data[clicked_index].copy()
    layer.data[clicked_index] = [-1000000, -1000000]

    if shared_state.TRACKED:
        shared_state.DATA[shared_state.current_index][clicked_index] = [-1000000, -1000000, 0, 0, None]
    else:
        shared_state.DATA[shared_state.current_index][clicked_index] = [-1000000, -1000000, 0, 0]

    node_name = f'D_{shared_state.current_index}_{clicked_index}'
    shared_state.G.remove_node(node_name)

    layer.refresh()
    update_image()
    print(f"Removed node {node_name} with coordinates {point_coords}.")

    # Mark the event as handled to stop further processing
    event.handled = True
    return  # Exit here to stop any further mouse events

def on_click(layer, event):
    """Handle left mouse click and make clicked-on detection part of current track only if clicked directly on a point."""
    global new_point_added

    if new_point_added:
        # Ignore the click if a new point was just added
        return

    if event.button == 1:  # Check for left mouse button click
        click_position = event.position

        # Check if the click is on a valid point using get_value() method
        clicked_index = layer.get_value(click_position, world=True)

        if clicked_index is not None and clicked_index >= 0:
            # If a valid point was clicked, update the track and go to the next image
            shared_state.track[shared_state.current_index] = clicked_index
            next_image()
        else:
            print("Clicked outside of any detection point.")


new_point_added = False

def add_detection(layer, event):
    """Handle shift + left mouse click events to add a new point within image bounds."""
    global new_point_added

    if event.button == 1 and 'Shift' in event.modifiers:
        # Convert world coordinates to data coordinates (pixel coordinates)
        data_coords = layer.world_to_data(event.position)

        # Get the dimensions of the currently displayed image
        image_shape = images[shared_state.current_index].shape  # This gets (height, width) for the image

        # Ensure the click is within the image bounds
        if (0 <= data_coords[0] < image_shape[0]) and (0 <= data_coords[1] < image_shape[1]):
            # Append the new point to the layer's data in data coordinates (pixels)
            new_point = [int(data_coords[0]), int(data_coords[1])]  # [y, x] format

            # Add the new point to the existing points layer data
            layer.data = np.vstack([layer.data, new_point])

            # Determine the default point size (20) or get the minimum size of existing points
            if len(layer.size) > 0:
                point_size = 20

            # Create a new size array matching the updated data length
            new_size_array = np.full((len(layer.data),), point_size)

            # Ensure the size of the track detection in the current frame is changed to 30
            curr_track = shared_state.track
            if curr_track[shared_state.current_index] != -1:
                track_index = curr_track[shared_state.current_index]  # Get the correct track index
                new_size_array[track_index] = 30  # Set track point size to 30

            # Update the size array with the correct shape
            layer.size = new_size_array
            layer.border_width_is_relative = False  # Ensure the border width is absolute

            # Ensure the new point is added with the correct data in shared_state
            if shared_state.TRACKED:
                shared_state.DATA[shared_state.current_index].append([int(data_coords[0]), int(data_coords[1]), 0, 0, None])
            else:
                shared_state.DATA[shared_state.current_index].append([int(data_coords[0]), int(data_coords[1]), 0, 0])

            add_node_with_dummy_edges(node=f'D_{shared_state.current_index}_{shared_state.NUM_DET_PER_FRAME[shared_state.current_index]}',
                                      time_point=shared_state.current_index,
                                      idx=shared_state.NUM_DET_PER_FRAME[shared_state.current_index],
                                      y=int(data_coords[0]),
                                      x=int(data_coords[1]),
                                      displ_y=0,
                                      displ_x=0,
                                      G=shared_state.G,
                                      highest_frame_id=len(shared_state.DATA)-1,
                                      max_score=shared_state.MAX_SCORE)
            print(f'Added node D_{shared_state.current_index}_{shared_state.NUM_DET_PER_FRAME[shared_state.current_index]}.')
            shared_state.NUM_DET_PER_FRAME[shared_state.current_index] += 1

            # Refresh the layer to display the new point
            layer.refresh()

            print(f"Added new point at data coordinates: {new_point}, with size: {point_size}")

            # save up-to-date version of DATA
            write_updated_detections_to_file(shared_state.DATA, shared_state.UPDATED_DATA_FILE, shared_state.csv_folder_to_save)

            # Set the flag to indicate a new point was added
            new_point_added = True

            # Use a QTimer to reset the flag after 200 milliseconds
            QTimer.singleShot(100, reset_new_point_flag)
            graph_updated_event()
        else:
            print("Clicked outside the image bounds. Point not added.")

def acceptTrack(event=None):
    """Trigger the accept track event and move to frame 0."""
    accept_track_event()  # Trigger the event to handle track acceptance
    shared_state.current_index = 0  # Reset to the first frame (frame 0)
    update_image()  # Update the displayed image
    image_slider.image_index.value = shared_state.current_index  # Sync the slider value
    graph_updated_event()
    # Refocus the main container after handling the event
    if container:
        container.setFocus()

     
def saveSegment(event=None):
    print("saveSegment triggered!")  # Add this line to check if the function is called
    save_segment_event()
    shared_state.current_index = 0  # Reset to the first frame (frame 0)
    update_image()
    image_slider.image_index.value = shared_state.current_index  # Sync the slider value
    # Refocus the main container after handling the event
    if container:
        container.setFocus()

def deleteSegment(event=None):
    delete_segment_event()
    shared_state.current_index = 0  # Reset to the first frame (frame 0)
    update_image()  # Update the displayed image
    image_slider.image_index.value = shared_state.current_index  # Sync the slider value
    graph_updated_event()
    # Refocus the main container after handling the event
    if container:
        container.setFocus()

def deleteAllConnections(event=None):
    """Triggers the function which would delete all outgoing edges (apart from an X- or T-edge)"""
    delete_all_connections_event()
    update_image()
    graph_updated_event()
    # Refocus the main container after handling the event
    if container:
        container.setFocus()

def reset_new_point_flag():
    """Reset the flag indicating a new point was added."""
    global new_point_added
    new_point_added = False


def setup_keybindings():
    """Set up key bindings for the viewer."""
    viewer.bind_key('Right', handle_right)  # Right arrow key to move to the next image
    viewer.bind_key('Left', handle_left)  # Left arrow key to move to the previous image
    viewer.bind_key('D', lambda event: delete_detection(viewer.layers.selection.active, use_key=True))  # Trigger only for D key
    viewer.bind_key('Shift-Q', acceptTrack)  
    viewer.bind_key('W', saveSegment) # save correct segment
    viewer.bind_key('Shift-Z', deleteSegment) # delete a whole segment
    viewer.bind_key('X', deleteAllConnections) # delete all connections from a node, apart from the D-X edge

def trackin_main():
    """Main function to show the plugin interface."""
    global container, dets_label, conns_label
    container = QWidget()
    layout = QVBoxLayout(container)  # Create a vertical layout
    
    # Use the magicgui widgets and add them directly to the layout
    layout.addWidget(choose_folder.native)  # Add the magicgui widget's native Qt widget
    layout.addWidget(load_csv.native)
    layout.addWidget(load_track_file.native)
    layout.addWidget(image_slider.native)  # Add the slider widget
    
    # Set layout to the container
    container.setLayout(layout)

    # Set focus policy and initially focus the container
    container.setFocusPolicy(Qt.StrongFocus)
    container.setFocus()
    viewer.window.qt_viewer.dockLayerList.setVisible(False)
    viewer.window.qt_viewer.dockLayerControls.setVisible(False)

    return container


def clear_detections_and_tracks():
    """Explicitly clear detection points and track lines from the Napari viewer."""
    global viewer
    
    # Check and clear the 'detections' layer if it exists
    if 'detections' in viewer.layers:
        viewer.layers['detections'].data = np.empty((0, 2))
        viewer.layers['detections'].refresh()  # Ensure the layer refreshes
    
    # Check and clear the 'track_lines' layer if it exists
    if 'track_lines' in viewer.layers:
        viewer.layers['track_lines'].data = np.empty((0, 2, 2))
        viewer.layers['track_lines'].refresh()  # Ensure the layer refreshes
    
    # Reset shared state as needed
    shared_state.DATA = []  # Explicitly set to an empty list rather than None
    shared_state.track = []
    shared_state.track_lines = None
    shared_state.NUM_DET_PER_FRAME = []
    shared_state.TRACKED = False
    shared_state.MAX_TRACK_ID = None
    shared_state.G = None
    
    print("Cleared all detections and tracks.")

def write_updated_detections_to_file(data, updated_data_file, csv_path):
    # save up-to-date version of DATA

    f = open(os.path.join(csv_path, updated_data_file), "w")
    f.write('tframe,y,x,displ_y,displ_x\n')
    for i,_ in enumerate(data):
        for j,_ in enumerate(data[i]):
            # ensure that the detection that is deleted is not saved
            if (data[i][j][0]!=-1000000 and data[i][j][1]!=-1000000):
                f.write(f"{i},{data[i][j][0]},{data[i][j][1]},{data[i][j][2]},{data[i][j][3]}\n")
    f.close()

def handle_right(viewer):
    """Handle the Right arrow key press and release."""
    start_timer(right_timer)  # Start on key press
    yield  # Wait for key release
    stop_timer(right_timer)  # Stop on key release

def handle_left(viewer):
    """Handle the Left arrow key press and release."""
    start_timer(left_timer)  # Start on key press
    yield  # Wait for key release
    stop_timer(left_timer)  # Stop on key release

def start_timer(timer):
    """Start the timer to continuously update images."""
    if not timer.isActive():
        timer.start(50)  

def stop_timer(timer):
    """Stop the timer when the key is released."""
    if timer.isActive():
        timer.stop()

# Global references for the QLabel widgets
dets_label = None
conns_label = None

def update_labels():
    """Update the QLabel widgets with the current count of detections and connections."""
    global dets_label, conns_label  # Declare the variables as global to avoid NameError

    if shared_state.G is not None:
        # Compute the number of detections and connections
        shared_state.NUM_DETS = shared_state.G.number_of_nodes() - 2 - 2 * (len(shared_state.DATA) - 1)
        shared_state.NUM_CONN = shared_state.G.number_of_edges() - 2 * (len(shared_state.DATA) - 1)

        # Check if labels exist, if not create and add them to the layout
        if dets_label is None:
            dets_label = QLabel(f"Detections: {shared_state.NUM_DETS}")
            conns_label = QLabel(f"Connections: {shared_state.NUM_CONN}")
            container.layout().addWidget(dets_label)  # Add the labels to the layout dynamically
            container.layout().addWidget(conns_label)
        else:
            # If the labels already exist, just update the text
            dets_label.setText(f"Detections: {shared_state.NUM_DETS}")
            conns_label.setText(f"Connections: {shared_state.NUM_CONN}")
    else:
        # If G is None, remove or hide the labels (if necessary)
        if dets_label is not None and conns_label is not None:
            dets_label.setParent(None)  # Remove the label from the layout
            conns_label.setParent(None)
            dets_label = None  # Reset the label references
            conns_label = None

graph_updated_event.connect(update_labels)