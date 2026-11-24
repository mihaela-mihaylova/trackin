"""One detection: a point in one frame that tracks are built from."""

from typing import NamedTuple


class Detection(NamedTuple):
    """A detection in one frame, with named fields (d.y, d.x, ...)."""
    y: int
    x: int
    displ_y: int = 0
    displ_x: int = 0
    # track number from an earlier tracking run (the track_no column of an
    # already-tracked detections file); None if there is none
    prior_track: int | None = None
    # False once the detection is deleted or used in an accepted track;
    # its position is kept
    active: bool = True
    # the detection's values in the extra columns of the detections file
    # (e.g. label, area), in the order of the session's extra columns
    extra: tuple = ()
    # True for a detection added by hand rather than loaded from the file
    added: bool = False
