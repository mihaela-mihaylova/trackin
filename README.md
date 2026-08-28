# trackin

[![License GNU GPL v3.0](https://img.shields.io/pypi/l/trackin.svg?color=green)](https://github.com/mihaela-mihaylova/trackin/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/trackin.svg?color=green)](https://pypi.org/project/trackin)
[![Python Version](https://img.shields.io/pypi/pyversions/trackin.svg?color=green)](https://python.org)
[![tests](https://github.com/mihaela-mihaylova/trackin/workflows/tests/badge.svg)](https://github.com/mihaela-mihaylova/trackin/actions)
[![codecov](https://codecov.io/gh/mihaela-mihaylova/trackin/branch/main/graph/badge.svg)](https://codecov.io/gh/mihaela-mihaylova/trackin)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/trackin)](https://napari-hub.org/plugins/trackin)

A human-in-the-loop cell tracking tool.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `trackin` via [pip]:

    pip install trackin



To install latest development version :

    pip install git+https://github.com/mihaela-mihaylova/trackin.git


## Getting Started

1. Launch napari (`napari` from the command line) and open the plugin via
   **Plugins > trackin > Trackin**.
2. Click **Load Images** and select a folder of per-frame image files.
   Files must be named with numeric filenames (e.g. `0.png`, `1.png`, ...),
   one per frame, in a supported format: `jpg`, `jpeg`, `png`, `tif`, `tiff`,
   `bmp`, or `gif`.
3. Click **Load Detections** and select a CSV of detections for those
   frames. Required columns: `tframe`, `y`, `x`. Optional columns:
   `displ_y`, `displ_x` (default to `0` if omitted), and `track_no` (include
   it if the detections already belong to tracks from a prior run).
4. Step through frames with the arrow keys or the frame slider, and
   left-click a detection to add it to the current track (this also
   advances to the next frame). Shift-click on the image to add a detection
   that's missing; right-click, or **D**, to delete one.
5. Use **Shift+Q** to accept the current track, **W** to save a segment
   without ending it, **Shift+Z** to delete a segment, and **X** to remove
   all outgoing connections from the current node. The in-app **?** button
   lists the full set of mouse and keyboard shortcuts.
6. Optionally, click **Add Track File** to load a `track_session_*.csv`
   from a previous session so newly accepted tracks continue its track
   numbering instead of restarting from 1. It must have exactly these
   columns, in this order: `tframe, y, x, displ_y, displ_x, track_no`.

Accepted tracks are appended to a `track_session_*.csv` file saved next to
the detections CSV; any detections not yet assigned to a track are kept in
an `upd_*.csv` file in the same folder.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"trackin" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/mihaela-mihaylova/trackin/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
