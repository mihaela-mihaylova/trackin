"""One detection: a point in one frame that tracks are built from."""

from typing import NamedTuple


class Detection(NamedTuple):
    """A detection with named fields (d.y, d.x, ...). It can still be read
    by position like the plain tuples it replaces (d[0] is y, d[1] is x,
    ...), until all code reads it by name."""
    y: int
    x: int
    displ_y: int = 0
    displ_x: int = 0
    # track number from an earlier tracking run (the track_no column of an
    # already-tracked detections file); None if there is none
    prior_track: int | None = None
