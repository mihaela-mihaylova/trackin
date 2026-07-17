import os
import pandas as pd
from magicgui import magicgui
from skimage.io import imread
from qtpy.QtWidgets import QFileDialog, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog, QSlider, QApplication, QSizePolicy, QToolTip
from qtpy.QtCore import Qt, QTimer
import napari
import numpy as np
from napari.utils.events import EventEmitter
from .shared_state import shared_state
from datetime import datetime
from pathlib import PurePosixPath
from .tracking import add_node_with_dummy_edges
from napari.utils.notifications import show_info

viewer = None
images = []
image_files = []
csv_data = None
images_loaded = False
csv_loaded = False
container = None  # Global reference to the container

# napari's camera has no built-in zoom limits, so scrolling can otherwise
# zoom out until the view is a blank speck. These are recomputed relative to
# each dataset's fit-to-view zoom once images are loaded (see choose_folder).
MIN_ZOOM = 0.05
MAX_ZOOM = 100.0
ZOOM_STEP = 1.1  # multiplicative zoom change per wheel notch


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
REPEAT_INITIAL_DELAY = 350  # ms to hold a key before continuous scrolling kicks in
right_key_held = False
left_key_held = False


def clamp_camera_zoom(event=None):
    """Keep the camera zoom within [MIN_ZOOM, MAX_ZOOM]."""
    zoom = viewer.camera.zoom
    clamped = min(max(zoom, MIN_ZOOM), MAX_ZOOM)
    if clamped != zoom:
        viewer.camera.zoom = clamped

def on_mouse_wheel_zoom(viewer, event):
    """Zoom the camera on plain mouse scroll, replacing napari's default
    wheel-zoom so we control the direction and clamp range ourselves,
    regardless of the OS/trackpad scroll convention."""
    if event.modifiers:
        return  # leave modified scroll (e.g. Control = change frame) alone

    delta = event.delta[1]
    if event.native is not None and event.native.inverted():
        delta = -delta
    if delta == 0:
        return

    factor = ZOOM_STEP if delta > 0 else 1 / ZOOM_STEP
    new_zoom = viewer.camera.zoom * factor
    viewer.camera.zoom = min(max(new_zoom, MIN_ZOOM), MAX_ZOOM)
    event.handled = True

def initialize_viewer(napari_viewer):
    """Initialize the Napari viewer object."""
    global viewer
    viewer = napari_viewer

    # Set up key bindings for the viewer
    setup_keybindings()

    # Take over scroll-to-zoom so direction and range are under our control
    viewer.mouse_wheel_callbacks.append(on_mouse_wheel_zoom)

    # Backstop in case zoom changes through some other path (e.g. reset_view)
    viewer.camera.events.zoom.connect(clamp_camera_zoom)

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
    global images, image_files, viewer, images_loaded, MIN_ZOOM, MAX_ZOOM
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder with Images")
    if folder_path:
        clear_detections_and_tracks()
        images, image_files = load_images_from_folder(folder_path)
        shared_state.current_index = 0
        images_loaded = True
        if images:
            viewer.layers.clear()
            viewer.add_image(images[shared_state.current_index], name=os.path.basename(image_files[shared_state.current_index]))
            # napari fits the view to the image on add; use that as the
            # baseline for how far this dataset can be zoomed in/out
            fit_zoom = viewer.camera.zoom
            MIN_ZOOM = fit_zoom * 0.5
            MAX_ZOOM = fit_zoom * 30
            check_and_update_image()
            update_slider_max()

choose_folder.call_button.native.setToolTip(
    "Select a folder of image frames.\n"
    "Files must be named with numeric filenames (e.g. 0.png, 1.png, ...),\n"
    "one per frame, using a supported format: jpg, jpeg, png, tif, tiff, bmp, gif."
)

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

        # Start inspection from the beginning, regardless of the frame
        # the viewer happened to be on before these detections were loaded
        shared_state.current_index = 0
        image_slider.image_index.value = shared_state.current_index

        check_and_update_image()

        update_file_paths_display()

        # Move keyboard focus off the slider's editable readout (left focused,
        # and blinking its text cursor, after the file dialog closes) and back
        # onto the main panel so shortcuts like arrow keys work immediately
        if container:
            container.setFocus()

load_csv.call_button.native.setToolTip(
    "Select a CSV of detections for the loaded image frames.\n"
    "Required columns: tframe, y, x.\n"
    "Optional columns: displ_y, displ_x (default to 0 if omitted),\n"
    "and track_no (include it if these detections are already tracked)."
)

def generate_upd_track_filename(csv_path, timestamp):
    """Build the filename for a track file that continues a previous
    session's track numbering, from the selected CSV's path and a
    timestamp. Handles both / and \\ path separators, and keeps
    everything before the last dot (so "my.tracks.v2.csv" becomes
    "my.tracks.v2", not "my")."""
    csv_filename = PurePosixPath(csv_path.replace('\\', '/')).stem
    return f'with_new_tracks_added_{csv_filename}_{timestamp}.csv'

# loads a file with already accepted tracks and adds these and any subsequent accepted tracks to a new track file
@magicgui(call_button="Add Track File", auto_call=True)
def load_track_file():
    """Open a dialog to select a CSV file and load a csv with tracks generated in a previous session."""
    
    if not images_loaded or not csv_loaded:
        QMessageBox.information(None, "Load Images and Detections First.", "Please load the images and detections before loading the track file to add new tracks to.")
        return

    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        # generate the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # name for file where leftover positions are preserved
        shared_state.UPD_TRACK_FILE = generate_upd_track_filename(csv_path, timestamp)

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

            if track_file_label is not None:
                track_file_label.set_path(f"Track file loaded: {os.path.basename(csv_path)}")
                # set_path() defaults the tooltip to the same string as the
                # display text; override it with the actual full path instead
                track_file_label.setToolTip(csv_path)

            update_file_paths_display()

        except pd.errors.ParserError as e:
            QMessageBox.critical(None, "File Error", f"The selected file is not a valid CSV or is malformed: {e}")
            return
        except Exception as e:
            QMessageBox.critical(None, "Error", f"An unexpected error occurred: {e}")

load_track_file.call_button.native.setToolTip(
    "Select a CSV of tracks accepted in a previous session, so new tracks\n"
    "continue numbering from where that session left off.\n"
    "Must have exactly these columns, in this order:\n"
    "tframe, y, x, displ_y, displ_x, track_no."
)

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

def _frame_slider_wheel_event(qevent):
    """Step exactly one frame per wheel event, ignoring the reported scroll
    magnitude -- trackpads can report a delta worth more than one notch for
    a single gesture, which otherwise skips frames."""
    delta = qevent.angleDelta().y()
    if delta > 0:
        previous_image()
    elif delta < 0:
        next_image()
    qevent.accept()

for _qslider in image_slider.image_index.native.findChildren(QSlider):
    _qslider.wheelEvent = _frame_slider_wheel_event

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
            edge_color='white',
            edge_width=1,
            edge_width_is_relative=False,
            name='detections'
        )
        # Attach event handlers if necessary
        points_layer.mouse_drag_callbacks.append(delete_detection)
        points_layer.mouse_drag_callbacks.append(add_detection)
        points_layer.mouse_drag_callbacks.append(on_click)

    # Prepare attributes for the main points layer
    edge_colors = ['white'] * len(points)
    sizes = [20] * len(points)
    edge_widths = [1] * len(points)

    # Check if track contains detection in this frame and apply custom styles for the track point
    track_index = curr_track[shared_state.current_index]
    if track_index != -1:
        edge_colors[track_index] = 'yellow'
        sizes[track_index] = 35  # Increase the size of the track point
        edge_widths[track_index] = 3  # Thicker border for the track point

    # --- Apply updates only if attributes have changed ---
    if not np.array_equal(points_layer.size, sizes):
        points_layer.size = sizes
    if not np.array_equal(points_layer.edge_color, edge_colors):
        points_layer.edge_color = edge_colors
    if not np.array_equal(points_layer.edge_width, edge_widths):
        points_layer.edge_width = edge_widths

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
            edge_color='yellow',
            edge_width=3,
            edge_width_is_relative=False,
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
        if event.button != 2:  # Not a right-click: nothing to do here, let the
            return              # event fall through to napari's default pan/zoom.
        click_position = event.position
        print(f'clicked position:{click_position}')
        clicked_index = layer.get_value(click_position, world=True)
        if clicked_index is None:
            return  # No valid point was clicked, exit
    else:  # Called using the 'D' key
        curr_track = shared_state.track
        # e.g. right after images are reloaded, before a new CSV is loaded
        if not curr_track:
            print("No detections loaded to delete with 'D' key.")
            return
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

    # save up-to-date version of DATA
    write_updated_detections_to_file(shared_state.DATA, shared_state.UPDATED_DATA_FILE, shared_state.csv_folder_to_save)

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

    # save up-to-date version of DATA
    write_updated_detections_to_file(shared_state.DATA, shared_state.UPDATED_DATA_FILE, shared_state.csv_folder_to_save)

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
            layer.edge_width_is_relative = False  # Ensure the border width is absolute

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

def show_help():
    """Display a dialog listing the keyboard and mouse shortcuts."""

    def section(title, rows):
        row_html = "".join(
            f'<tr><td style="padding:2px 18px 2px 0; white-space:nowrap;">'
            f'<code style="background:#3a3a3a; color:#eee; padding:1px 6px; '
            f'border-radius:3px;">{key}</code></td>'
            f'<td style="padding:2px 0;">{desc}</td></tr>'
            for key, desc in rows
        )
        return (
            f'<p style="margin:14px 0 4px 0; font-size:12pt; font-weight:600;">{title}</p>'
            f'<table style="border-collapse:collapse;">{row_html}</table>'
        )

    help_html = (
        '<div style="font-size:10.5pt;">'
        + section(
            "Navigation",
            [
                ("Right Arrow", "Next frame (hold to scroll continuously)"),
                ("Left Arrow", "Previous frame (hold to scroll continuously)"),
                ("Slider", "Jump to a specific frame"),
            ],
        )
        + section(
            "Mouse (on the image)",
            [
                ("Left-click", "Add the clicked detection to the current track and advance to the next frame"),
                ("Shift + Left-click", "Add a new detection at that position"),
                ("Right-click", "Delete the clicked detection"),
            ],
        )
        + section(
            "Keyboard shortcuts",
            [
                ("D", "Delete the tracked detection in the current frame"),
                ("Shift+Q", "Accept the current track"),
                ("W", "Save the current segment (up to the current frame)"),
                ("Shift+Z", "Delete the current segment (up to the current frame)"),
                ("X", "Delete all outgoing connections from the current node"),
            ],
        )
        + "</div>"
    )

    dialog = QDialog(None)
    dialog.setWindowTitle("Trackin Help")
    dialog.setMinimumWidth(480)

    layout = QVBoxLayout(dialog)
    label = QLabel(help_html)
    label.setTextFormat(Qt.RichText)
    label.setWordWrap(True)
    layout.addWidget(label)

    close_button = QPushButton("Close")
    close_button.clicked.connect(dialog.accept)
    layout.addWidget(close_button)

    dialog.exec_()

track_file_label = None  # Shows which track file was last loaded, if any

# Persistent, copyable output-file-path rows (see create_path_row / update_file_paths_display)
leftover_row = None
leftover_path_field = None
session_row = None
session_path_field = None
upd_track_row = None
upd_track_path_field = None

# Collapsible "Session Files" card wrapping the three rows above
session_files_header = None
session_files_content = None

def set_session_files_expanded(expanded):
    """Show/hide the Session Files card's content and update its header arrow."""
    session_files_content.setVisible(expanded)
    arrow = "▾" if expanded else "▸"  # ▾ expanded, ▸ collapsed
    session_files_header.setText(f"{arrow} Session Files")

def toggle_session_files():
    set_session_files_expanded(not session_files_content.isVisible())

class ElidedPathLabel(QLabel):
    """A QLabel that displays a long path with an ellipsis instead of
    forcing its row (and so the sidebar panel) wider than the space
    actually available, while keeping the untruncated path (in
    .full_path, and as this label's tooltip) available for copying.
    Re-elides on resize so it stays correct if the panel is resized."""

    def __init__(self):
        super().__init__("")
        self.full_path = ""
        # QSizePolicy.Ignored means Qt's layout engine never lets this
        # label's text length dictate how wide its row/panel must be --
        # it always shrinks to whatever width it's actually given.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def set_path(self, path):
        self.full_path = path
        self.setToolTip(path)
        self._update_elided_text()

    def _update_elided_text(self):
        elided = self.fontMetrics().elidedText(self.full_path, Qt.ElideMiddle, self.width())
        super().setText(elided)

    def resizeEvent(self, event):
        self._update_elided_text()
        super().resizeEvent(event)

def show_field_explanation(button, explanation):
    """Show a click-triggered tooltip bubble explaining a field, positioned
    at the button that was clicked. Dismisses automatically on click-
    elsewhere or mouse-move, same as a normal hover tooltip would."""
    QToolTip.showText(button.mapToGlobal(button.rect().bottomLeft()), explanation, button)

def create_info_button(explanation):
    """Build a small pale-yellow 'i' button that shows explanation in a
    click-triggered tooltip bubble. Uses 'i' rather than '?' and a
    deliberately different color from the blue Help button, since that
    button's '?' is reserved for the keybinding cheatsheet -- this is a
    small, single-field hint, a different kind of help entirely."""
    info_button = QPushButton("i")
    info_button.setFixedSize(11, 11)
    info_button.setCursor(Qt.PointingHandCursor)
    info_button.setStyleSheet(
        "QPushButton {"
        "  background-color: #FFFFE0;"
        "  color: black;"
        "  border: 1px solid #d9c05a;"
        "  border-radius: 5px;"
        "  font-weight: bold;"
        "  font-style: italic;"
        "  font-size: 8px;"
        "  padding: 0px;"
        "}"
        "QPushButton:hover { background-color: #FFF8B0; }"
        "QPushButton:pressed { background-color: #FFEE99; }"
    )
    info_button.clicked.connect(lambda: show_field_explanation(info_button, explanation))
    return info_button

def create_path_row(description, explanation):
    """Build a labeled, copyable file-path row: a description label (with a
    small info button explaining what the file contains) above an
    ellipsis-truncated path (full path on hover) with a Copy button. Returns
    (row_widget, path_label) so callers can toggle the row's visibility and
    update its path later."""
    row_widget = QWidget()
    row_layout = QVBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(2)

    label_row_layout = QHBoxLayout()
    description_label = QLabel(description)
    description_label.setStyleSheet("font-weight: 600; font-size: 10pt;")
    label_row_layout.addWidget(description_label)
    label_row_layout.addWidget(create_info_button(explanation))
    label_row_layout.addStretch()
    row_layout.addLayout(label_row_layout)

    path_row_layout = QHBoxLayout()
    path_label = ElidedPathLabel()
    path_row_layout.addWidget(path_label, stretch=1)

    copy_button = QPushButton("Copy")
    copy_button.setFixedWidth(50)
    # Read the label's current full path at click time rather than capturing
    # one now, since this row's widget is reused and updated across loads
    copy_button.clicked.connect(lambda: QApplication.clipboard().setText(path_label.full_path))
    path_row_layout.addWidget(copy_button)

    row_layout.addLayout(path_row_layout)
    return row_widget, path_label

def update_file_paths_display():
    """Show/hide and refresh the output-file-path rows based on current state."""
    if csv_loaded:
        leftover_path_field.set_path(os.path.join(shared_state.csv_folder_to_save, shared_state.UPDATED_DATA_FILE))
        leftover_row.setVisible(True)
        session_path_field.set_path(os.path.join(shared_state.csv_folder_to_save, shared_state.SESSION_FILE))
        session_row.setVisible(True)
        # The whole Session Files card (header included) only makes sense
        # once there's a CSV loaded -- with just images loaded, there's
        # nothing to show yet.
        if session_files_header is not None:
            session_files_header.setVisible(True)
    else:
        leftover_row.setVisible(False)
        session_row.setVisible(False)
        if session_files_header is not None:
            session_files_header.setVisible(False)

    if shared_state.MAX_TRACK_ID is not None:
        upd_track_path_field.set_path(os.path.join(shared_state.csv_folder_to_save, shared_state.UPD_TRACK_FILE))
        upd_track_row.setVisible(True)
    else:
        upd_track_row.setVisible(False)

    # Auto-expand the card whenever a row's content actually changed, so a
    # newly added file (e.g. from "Add Track File") is immediately visible
    # rather than hidden behind a collapsed header the user has to think to
    # open. Only relevant once the header itself is showing (csv_loaded).
    if csv_loaded and session_files_content is not None:
        set_session_files_expanded(True)

def trackin_main():
    """Main function to show the plugin interface."""
    global container, dets_label, conns_label, track_file_label
    global leftover_row, leftover_path_field, session_row, session_path_field
    global upd_track_row, upd_track_path_field
    global session_files_header, session_files_content
    container = QWidget()
    layout = QVBoxLayout(container)  # Create a vertical layout

    # Small round help button, opens a dialog listing keyboard/mouse shortcuts
    help_button = QPushButton("?")
    help_button.setFixedSize(24, 24)
    help_button.setToolTip("Help")
    help_button.setCursor(Qt.PointingHandCursor)
    help_button.setStyleSheet(
        "QPushButton {"
        "  background-color: #2b7de9;"
        "  color: white;"
        "  border: none;"
        "  border-radius: 12px;"
        "  font-weight: bold;"
        "}"
        "QPushButton:hover { background-color: #4a90f0; }"
        "QPushButton:pressed { background-color: #1c5fc2; }"
    )
    help_button.clicked.connect(show_help)
    layout.addWidget(help_button, alignment=Qt.AlignRight)

    # Use the magicgui widgets and add them directly to the layout
    layout.addWidget(choose_folder.native)  # Add the magicgui widget's native Qt widget
    layout.addWidget(load_csv.native)
    layout.addWidget(load_track_file.native)

    # Shows the loaded track file's name once "Add Track File" succeeds,
    # since that success is otherwise only reported via a transient toast
    track_file_label = ElidedPathLabel()
    track_file_label.setStyleSheet("color: gray; font-style: italic;")
    layout.addWidget(track_file_label)

    layout.addWidget(image_slider.native)  # Add the slider widget

    # Collapsible "Session Files" card wrapping the copyable output-file-path
    # rows, shown/updated as state changes. Starts collapsed since there's
    # nothing to show until a CSV is loaded; update_file_paths_display()
    # auto-expands it whenever a row's content actually changes.
    session_files_header = QPushButton("▸ Session Files")
    session_files_header.setFlat(True)
    session_files_header.setStyleSheet(
        "QPushButton { text-align: left; font-weight: 600; border: none; padding: 2px 0; }"
    )
    session_files_header.setCursor(Qt.PointingHandCursor)
    session_files_header.clicked.connect(toggle_session_files)
    session_files_header.setVisible(False)  # nothing to show until a CSV is loaded
    layout.addWidget(session_files_header)

    session_files_content = QWidget()
    file_paths_layout = QVBoxLayout(session_files_content)
    file_paths_layout.setContentsMargins(0, 4, 0, 14)
    file_paths_layout.setSpacing(8)

    leftover_row, leftover_path_field = create_path_row(
        "Leftover detections",
        "Contains detections that are left after the ones in accepted tracks have been removed.",
    )
    session_row, session_path_field = create_path_row(
        "Tracks accepted this session",
        "Contains the detections included in the accepted tracks, together with the respective generated track ids.",
    )
    upd_track_row, upd_track_path_field = create_path_row(
        "Combined tracks",
        "Contains the tracks from the added track file, together with the accepted tracks in this session.",
    )

    for row in (leftover_row, session_row, upd_track_row):
        row.setVisible(False)
        file_paths_layout.addWidget(row)

    session_files_content.setVisible(False)
    layout.addWidget(session_files_content)

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
    shared_state.N_TRACKS = 0
    shared_state.NUM_DETS = None
    shared_state.NUM_CONN = None

    # Reset the filenames themselves too, not just MAX_TRACK_ID -- so that if
    # any future code path is ever reached with an empty/cleared DATA or
    # track, it fails loudly (e.g. IsADirectoryError from a blank path)
    # rather than silently writing into the *previous* dataset's files.
    shared_state.SESSION_FILE = ''
    shared_state.UPDATED_DATA_FILE = ''
    shared_state.UPD_TRACK_FILE = ''

    # Hide all three rows unconditionally rather than routing through
    # update_file_paths_display(): that function's leftover/session
    # visibility is gated on the module-level csv_loaded flag, which is
    # never reset to False here or in choose_folder() -- so if a CSV had
    # already been loaded once this session and images are reloaded
    # without a new CSV yet, csv_loaded is still stale True and would
    # re-show the *previous* dataset's file paths. Any caller that goes on
    # to load a fresh CSV (e.g. load_csv() reloading) will re-show the
    # rows correctly via its own update_file_paths_display() call right after.
    if leftover_row is not None:
        leftover_row.setVisible(False)
        session_row.setVisible(False)
        upd_track_row.setVisible(False)

    # Hide the whole Session Files card (not just collapse it) too, since
    # there's nothing left in it to show until a new CSV is loaded.
    if session_files_content is not None:
        set_session_files_expanded(False)
        session_files_header.setVisible(False)

    # Clear the "Track file loaded: ..." label too -- it previously wasn't
    # touched here at all, so it kept showing the *previous* dataset's
    # loaded track file even after images were reloaded.
    if track_file_label is not None:
        track_file_label.set_path("")

    # update_labels() (the Detections/Connections counts) only runs in
    # response to this event -- without firing it here, those labels kept
    # showing the *previous* dataset's stale counts until some unrelated
    # keypress happened to fire the event afterward.
    graph_updated_event()

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

def _maybe_start_repeat_right():
    """Only start auto-repeat if the key is still held after the initial delay."""
    if right_key_held:
        start_timer(right_timer)

def _maybe_start_repeat_left():
    """Only start auto-repeat if the key is still held after the initial delay."""
    if left_key_held:
        start_timer(left_timer)

def handle_right(viewer):
    """Handle the Right arrow key press and release."""
    global right_key_held
    right_key_held = True
    next_image()  # A single tap always advances exactly one frame
    QTimer.singleShot(REPEAT_INITIAL_DELAY, _maybe_start_repeat_right)  # Auto-repeat only if held
    yield  # Wait for key release
    right_key_held = False
    stop_timer(right_timer)  # Stop on key release

def handle_left(viewer):
    """Handle the Left arrow key press and release."""
    global left_key_held
    left_key_held = True
    previous_image()  # A single tap always advances exactly one frame
    QTimer.singleShot(REPEAT_INITIAL_DELAY, _maybe_start_repeat_left)  # Auto-repeat only if held
    yield  # Wait for key release
    left_key_held = False
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