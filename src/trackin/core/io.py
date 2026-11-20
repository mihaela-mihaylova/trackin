# Reading and checking the input files (detections files and image
# folders), and writing the output files. Used by the plugin's interface,
# which shows the messages the checks return and decides when to write.

import csv
import math
import os
from pathlib import PurePosixPath

import pandas as pd

# columns the tracking uses; any other column of a detections file is an
# extra column, kept as it is and written back out in the output files
TRACKING_COLUMNS = ['tframe', 'y', 'x', 'displ_y', 'displ_x', 'track_no']

def read_detections_csv(csv_path):
    """Read a detections file. The tracking columns are converted to whole
    numbers, as before; extra columns keep their values as they are (text,
    decimals, ...), so they come out the same in the output files."""
    # When the first line after the header has one value more than the
    # header has columns, pandas doesn't give an error, but shifts all
    # columns one place to the left (e.g. the y values end up in tframe).
    # So the values of that line are counted before pandas reads the file.
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
        lines = csv.reader(f)
        header = next(lines, [])
        first_line = next(lines, [])
    if len(first_line) > len(header):
        raise ValueError("Line 2 has more values than the header has columns.")
    df = pd.read_csv(csv_path)
    # A file separated by semicolons is read as one column (e.g. "tframe;y;x"),
    # so it fails this check too, and its columns show why
    missing = [c for c in ['tframe', 'y', 'x'] if c not in df.columns]
    if missing:
        raise ValueError(
            f"The file must have the columns tframe, y and x, but it has no {', '.join(missing)}.\n"
            f"Its columns are: {', '.join(str(c) for c in df.columns)}."
        )
    return df.astype({c: int for c in TRACKING_COLUMNS if c in df.columns})

 # adds these displacement columns with value 0, in case they are not present
def check_and_add_displ_cols(df):
    """Add displ_x and add displ_y if these are not present in loaded df"""
    if 'displ_x' not in df.columns:
        df['displ_x'] = 0
    if 'displ_y' not in df.columns:
        df['displ_y'] = 0
    return df

IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp', 'gif')
IMAGE_NAMING_RULE = (
    "Images must be named 0, 1, 2, ... (e.g. 0.png, 1.png, 2.png), "
    "one per frame, with no missing frames."
)

def list_image_files(folder_path):
    """Names of the image files in a folder, in no particular order."""
    return [f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)]

def format_list(items, max_listed=10):
    """Items as a comma-separated list for an error message. Lists at most
    `max_listed` of them, then says how many more there are, so a long list
    doesn't make the dialog unreadable."""
    items = [str(item) for item in items]
    if len(items) > max_listed:
        return ", ".join(items[:max_listed]) + f", and {len(items) - max_listed} more"
    return ", ".join(items)

def check_image_numbering(file_names):
    """Return an error message if the image files are not named 0, 1, 2, ...
    N-1 (one image per frame, no gaps, no duplicates), otherwise None. The
    message says what is wrong and how to fix it.

    Frame n is shown on the n-th image, so this is what makes every frame
    number (tframe) match the image with the same name."""
    if not file_names:
        return f"No images found in this folder. Supported formats: {', '.join(IMAGE_EXTENSIONS)}."

    not_numbered = sorted(f for f in file_names if not os.path.splitext(f)[0].isdigit())
    if not_numbered:
        problem = (
            f"These images are not named with only a frame number: {format_list(not_numbered)}. "
            "Rename them to just the number"
        )
        # suggest the new name for the first one, if its name contains a number
        name, extension = os.path.splitext(not_numbered[0])
        digits = "".join(c for c in name if c.isdigit())
        problem += f", e.g. {not_numbered[0]} → {int(digits)}{extension}." if digits else "."
        return problem + "\n\n" + IMAGE_NAMING_RULE

    numbers = [int(os.path.splitext(f)[0]) for f in file_names]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicates:
        def files_numbered(n):
            return ", ".join(sorted(f for f in file_names if int(os.path.splitext(f)[0]) == n))
        if len(duplicates) == 1:
            problem = f"More than one image has frame number {duplicates[0]}: {files_numbered(duplicates[0])}."
        else:
            clashes = [f"{n} ({files_numbered(n)})" for n in duplicates]
            problem = f"More than one image has the same frame number: {format_list(clashes)}."
        return problem + " Keep one image per frame number.\n\n" + IMAGE_NAMING_RULE

    first = min(numbers)
    if first > 0 and sorted(numbers) == list(range(first, first + len(numbers))):
        # consecutive, but not starting at 0 (e.g. 1, 2, 3, ...)
        return (
            f"The images are numbered from {first}, but must start at 0. "
            "Rename them so the first image is 0 (e.g. 0.png).\n\n" + IMAGE_NAMING_RULE
        )

    missing = sorted(set(range(max(numbers) + 1)) - set(numbers))
    if missing:
        if len(missing) == 1:
            problem = f"There is no image for frame {missing[0]}."
        else:
            problem = f"There is no image for {len(missing)} frames: {format_list(missing)}."
        return (
            problem + " Add the missing images, or renumber the images so there "
            "are no gaps.\n\n" + IMAGE_NAMING_RULE
        )
    return None

def check_detection_frames(df, n_images):
    """Return an error message if detections are in frames that have no
    image (tframe below 0, or n_images and higher), otherwise None.

    Every tframe has to be shown on the image with the same number, so a
    detections file with detections that have no image can't be loaded."""
    without_image = (df['tframe'] < 0) | (df['tframe'] >= n_images)
    if not without_image.any():
        return None
    frames = sorted(df.loc[without_image, 'tframe'].unique())
    return (
        f"{int(without_image.sum())} detection(s) are in frame(s) {format_list(frames)}, "
        f"but there are only images for frames 0 to {n_images - 1}.\n\n"
        "Only a detections file in which every detection can be shown on an "
        f"existing image can be loaded: every tframe must be between 0 and {n_images - 1}. "
        "To fix this, remove these detections from the file, or load an image "
        "folder that also contains these frames."
    )

def generate_upd_track_filename(csv_path, timestamp):
    """Build the filename for a track file that continues a previous
    session's track numbering, from the selected CSV's path and a
    timestamp. Handles both / and \\ path separators, and keeps
    everything before the last dot (so "my.tracks.v2.csv" becomes
    "my.tracks.v2", not "my")."""
    csv_filename = PurePosixPath(csv_path.replace('\\', '/')).stem
    return f'with_new_tracks_added_{csv_filename}_{timestamp}.csv'

# value written in the extra columns for a detection added by hand, which
# didn't come from the detections file and so has no values of its own there
NO_VALUE = -1

def csv_value(value):
    """A value as written in a CSV cell: empty for a missing value, and in
    quotes if it contains a comma, quote or line break, so it stays one cell."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if any(char in text for char in ',"\n'):
        text = '"' + text.replace('"', '""') + '"'
    return text

def write_leftover_file(path, detections, extra_columns):
    """Write the detections that are still available (not deleted, and not
    used in an accepted track) to the file at `path`: tframe, y, x,
    displ_y, displ_x and the extra columns. Tracking can be continued later
    from this file."""
    with open(path, "w") as f:
        f.write("tframe,y,x,displ_y,displ_x" + "".join("," + csv_value(c) for c in extra_columns) + "\n")
        for t, frame in enumerate(detections):
            for d in frame:
                if d.active:
                    f.write(f"{t},{d.y},{d.x},{d.displ_y},{d.displ_x}"
                            + "".join("," + csv_value(v) for v in d.extra) + "\n")

def append_to_session_file(path, track_no, track, detections, extra_columns):
    """Append accepted track number `track_no` to the session file at
    `path`: one row per detection (tframe, y, x, displ_y, displ_x, track_no
    and the extra columns), after a header row if the file is new or empty."""
    add_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if add_header:
            f.write("tframe,y,x,displ_y,displ_x,track_no"
                    + "".join("," + csv_value(c) for c in extra_columns) + "\n")
        for t, i in enumerate(track):
            if i >= 0:
                d = detections[t][i]
                f.write(f"{t},{d.y},{d.x},{d.displ_y},{d.displ_x},{track_no}"
                        + "".join("," + csv_value(v) for v in d.extra) + "\n")

def append_to_combined_file(path, track_id, track, detections, extra_columns, combined_extra_columns):
    """Append an accepted track to the combined file at `path` (written by
    Add Track File), numbered `track_id`, with the extra columns in the
    combined file's order. A column the detections file doesn't have gets
    NO_VALUE."""
    with open(path, "a") as f:
        for t, i in enumerate(track):
            if i >= 0:
                d = detections[t][i]
                values = dict(zip(extra_columns, d.extra))
                f.write(f"{t},{d.y},{d.x},{d.displ_y},{d.displ_x},{track_id}"
                        + "".join("," + csv_value(values.get(c, NO_VALUE)) for c in combined_extra_columns) + "\n")
