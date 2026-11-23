"""The Load Images, Load Detections and Add Track File buttons: they ask
for the file or folder, let core/io.py read and check it, show any problem,
and put the result in the plugin's state (for detections, a new tracking
session and the names of the output files)."""

import os
from datetime import datetime

import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog, QMessageBox
from skimage.io import imread

from .core.io import (
    TrackFileError, check_and_add_displ_cols, check_detection_frames, check_image_numbering,
    combine_tracks, format_list, generate_upd_track_filename, list_image_files,
    read_detections_csv, read_track_file, tracks_table,
)
from ._interaction import show_messages, use_proposed_track
from ._layers import update_image
from .core.session import Session


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


def load_images_from_folder(folder_path):
    """Load images in numerical order from a given folder. Also returns the
    names of the images that could not be opened, so they can be reported
    together in one message."""
    image_files = sorted(
        list_image_files(folder_path),
        key=lambda x: int(os.path.splitext(x)[0])  # Sort by the numeric value of the filename
    )
    images = []
    unreadable = []
    for img in image_files:
        try:
            image = imread(os.path.join(folder_path, img))
            # Add white border to the image
            image_with_border = add_white_border(image, border_size=2)
            images.append(image_with_border)
        except Exception as e:
            print(f"Could not open image {img}: {e}")
            unreadable.append(img)
    return images, image_files, unreadable

def check_and_update_image(w):
    """Check if both images and CSV data are loaded, then update the image."""
    if w.state.images_loaded and w.state.csv_loaded:
        update_image()

def choose_folder(w):
    """Open a dialog to select a folder and load images from it."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    folder_path = QFileDialog.getExistingDirectory(None, "Select Folder with Images")
    if folder_path:
        # Check the image names before loading anything, so a folder that
        # breaks the rule leaves the current session untouched
        numbering_error = check_image_numbering(list_image_files(folder_path))
        if numbering_error:
            QMessageBox.critical(None, "Images Not Loaded", numbering_error)
            return
        new_images, new_image_files, unreadable = load_images_from_folder(folder_path)
        if unreadable:
            QMessageBox.critical(
                None, "Images Not Loaded",
                f"These images could not be opened: {format_list(unreadable)}. "
                "Check or replace these files.\n\nEvery frame needs an image that "
                "can be opened, otherwise the images would no longer match the "
                "frame numbers."
            )
            return
        _widget.clear_detections_and_tracks(w)
        w.state.images, w.state.image_files = new_images, new_image_files
        w.state.current_frame = 0
        w.state.images_loaded = True
        if w.state.images:
            w.viewer.layers.clear()
            w.viewer.add_image(w.state.images[w.state.current_frame], name=os.path.basename(w.state.image_files[w.state.current_frame]))
            # napari fits the view to the image on add; use that as the
            # baseline for how far this dataset can be zoomed in/out
            fit_zoom = w.viewer.camera.zoom
            w.state.min_zoom = fit_zoom * 0.5
            w.state.max_zoom = fit_zoom * 30
            check_and_update_image(w)
            w.navigator.update_slider_max()

def load_csv(w):
    """Open a dialog to select a CSV file and load its data."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    if not w.state.images_loaded:
        QMessageBox.information(None, "Load Images First", "Please load the images before loading the CSV file.")
        return
    
    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        show_info("Detections are being loaded...")
        # Check the detections before clearing anything, so a file that
        # breaks the rule leaves the current session untouched
        try:
            new_csv_data = read_detections_csv(csv_path)
        except (ValueError, OSError) as e:
            # e.g. a line with more values than the header has columns, or
            # a value that isn't a number in a tracking column
            QMessageBox.critical(None, "Detections Not Loaded", f"The detections file could not be read:\n\n{e}")
            return
        frames_error = check_detection_frames(new_csv_data, len(w.state.images))
        if frames_error:
            QMessageBox.critical(None, "Detections Not Loaded", frames_error)
            return
        if w.state.csv_loaded:
            _widget.clear_detections_and_tracks(w)

        csv_data = check_and_add_displ_cols(new_csv_data)
        w.state.session = Session.from_dataframe(csv_data, n_frames=len(w.state.images))

        csv_filename = csv_path.split('/')[-1].split('.')[0]
        # generate the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # name for file where generated tracks are saved
        w.state.session_file = f'track_session_{csv_filename}_{timestamp}.csv'
        # name for file where leftover positions are preserved
        w.state.leftover_file = f'upd_{csv_filename}_{timestamp}.csv'
        w.state.output_folder = os.path.dirname(csv_path)
        # create the empty session file, where accepted tracks are saved
        with open(os.path.join(w.state.output_folder, w.state.session_file), "w") as f:
            f.write("")
        w.state.csv_loaded = True
        # Build the graph and propose the first track
        if w.state.session.graph is None:
            show_info("Building graph...")
        show_messages(w.state.session.propose())
        use_proposed_track(w)

        w.refresh_stats()

        # Start inspection from the beginning, regardless of the frame
        # the viewer happened to be on before these detections were loaded
        w.state.current_frame = 0
        w.navigator.image_slider.image_index.value = w.state.current_frame

        check_and_update_image(w)

        _widget.update_file_paths_display()

        # Move keyboard focus off the slider's editable readout (left focused,
        # and blinking its text cursor, after the file dialog closes) and back
        # onto the widget so shortcuts like arrow keys work immediately
        w.focus()

# loads a file with already accepted tracks and adds these and any subsequent accepted tracks to a new track file
def load_track_file(w):
    """Open a dialog to select a CSV file and load a csv with tracks generated in a previous session."""
    # _widget.py imports this file, so it is imported here rather than at the top
    from . import _widget
    
    if not w.state.images_loaded or not w.state.csv_loaded:
        QMessageBox.information(None, "Load Images and Detections First.", "Please load the images and detections before loading the track file to add new tracks to.")
        return

    csv_path, _ = QFileDialog.getOpenFileName(None, "Select CSV File", "", "CSV Files (*.csv)")
    if csv_path:
        # generate the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # name for the file with both the loaded and the new tracks; only
        # used once the track file is accepted, so a refused file doesn't
        # redirect tracks accepted later into a file that was never written
        upd_track_file = generate_upd_track_filename(csv_path, timestamp)

        try:
            track_df = read_track_file(csv_path)
            max_track_no = track_df['track_no'].max()
            session = w.state.session
            # this session's accepted tracks, numbered after the track file's
            session_df = tracks_table(session.accepted, session.detections,
                                      session.extra_columns, first_track_no=max_track_no + 1)
            combined_df, combined_extra_columns = combine_tracks(track_df, session_df)
            combined_df.to_csv(os.path.join(w.state.output_folder, upd_track_file), index=False)
            if session.accepted:
                show_info("Track File Updated. \n The track file has been successfully updated with the new tracks.")
            else:
                show_info("Track File Added. \n The track file has been successfully added. New tracks will be added to the existing tracks.")

            w.state.combined_file = upd_track_file
            w.state.combined_extra_columns = combined_extra_columns
            w.state.max_prior_track_id = max_track_no

            if _widget.track_file_label is not None:
                _widget.track_file_label.set_path(f"Track file loaded: {os.path.basename(csv_path)}")
                # set_path() defaults the tooltip to the same string as the
                # display text; override it with the actual full path instead
                _widget.track_file_label.setToolTip(csv_path)

            _widget.update_file_paths_display()

        except TrackFileError as e:
            QMessageBox.critical(None, "Track File Not Loaded", str(e))
            return
        except Exception as e:
            QMessageBox.critical(None, "Error", f"An unexpected error occurred: {e}")

