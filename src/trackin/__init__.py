# __init__.py

import napari

from ._widget import initialize_viewer, trackin_main

# Initialize the Napari viewer
viewer = napari.Viewer()

# Pass the viewer object to initialize the viewer in widgets
initialize_viewer(viewer)
