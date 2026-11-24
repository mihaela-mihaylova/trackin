"""The made-up movie the reference sessions are played on, and where their
recorded results are kept. Shared by the plugin's and the core's reference
tests."""

from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).parent / "data" / "reference"
N_FRAMES = 6

# A small made-up movie: three cells moving through 6 frames, with the
# predicted movement to the next frame (displ_y, displ_x). Cell B is not
# detected in frame 3, cell C appears in frame 2, there is one noise
# detection in frame 3, and a "label" column that the tracking doesn't use.
DETECTIONS = pd.DataFrame(
    [
        # tframe, y, x, displ_y, displ_x, label
        (0, 10, 10, 3, 2, "A"), (1, 13, 12, 3, 3, "A"), (2, 16, 15, 3, 2, "A"),
        (3, 19, 17, 3, 3, "A"), (4, 22, 20, 3, 2, "A"), (5, 25, 22, 0, 0, "A"),
        (0, 50, 50, 2, 4, "B"), (1, 52, 54, 3, 3, "B"), (2, 55, 57, 0, 0, "B"),
        (4, 60, 64, 2, 4, "B"), (5, 62, 68, 0, 0, "B"),
        (2, 80, 30, -2, 4, "C"), (3, 78, 34, -3, 3, "C"), (4, 75, 37, -2, 4, "C"),
        (5, 73, 41, 0, 0, "C"),
        (3, 35, 85, 0, 0, "noise"),
    ],
    columns=["tframe", "y", "x", "displ_y", "displ_x", "label"],
)

# The same movie as an already-tracked file (with track_no)
TRACKED_DETECTIONS = DETECTIONS.assign(
    track_no=DETECTIONS["label"].map({"A": 1, "B": 2, "C": 3, "noise": 4})
)
