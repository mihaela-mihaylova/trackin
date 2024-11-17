# __init__.py

import napari

from ._widget import initialize_viewer, trackin_main
from .utils import build_graph

# Initialize the Napari viewer
# viewer = napari.Viewer()
viewer = napari.current_viewer()

# Pass the viewer object to initialize the viewer in widgets
initialize_viewer(viewer)
