# __init__.py

# trackin.py
import napari

from ._widget import choose_folder, next_image, previous_image, load_csv, initialize_viewer, trackin_main

# Initialize the Napari viewer
viewer = napari.Viewer()

# Pass the viewer object to initialize the viewer in widgets
initialize_viewer(viewer)

# Add the folder chooser, navigation buttons, and CSV loader to the viewer
#viewer.window.add_dock_widget(choose_folder, area='right')
#viewer.window.add_dock_widget(next_image, area='right')
#viewer.window.add_dock_widget(previous_image, area='right')
#viewer.window.add_dock_widget(load_csv, area='right')
viewer.window.add_dock_widget(trackin_main(), area='right')


napari.run()
