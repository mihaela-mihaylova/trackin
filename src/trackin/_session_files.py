"""The widget's Session Files panel: one row per output file, with its
path shortened to fit, an "i" button explaining the file, and a Copy
button."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
                            QSizePolicy, QToolTip, QVBoxLayout, QWidget)


class ElidedPathLabel(QLabel):
    """A QLabel that displays a long path with an ellipsis instead of
    forcing its row (and so the widget) wider than the space
    actually available, while keeping the untruncated path (in
    .full_path, and as this label's tooltip) available for copying.
    Re-elides on resize so it stays correct if the widget is resized."""

    def __init__(self):
        super().__init__("")
        self.full_path = ""
        # QSizePolicy.Ignored means Qt's layout engine never lets this
        # label's text length dictate how wide its row/widget must be --
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


class SessionFilesPanel(QWidget):
    """The widget's Session Files panel: a header that opens and closes it,
    and one row per output file. Hidden until a detections file is loaded."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header = QPushButton("▸ Session Files")
        self.header.setFlat(True)
        self.header.setStyleSheet(
            "QPushButton { text-align: left; font-weight: 600; border: none; padding: 2px 0; }"
        )
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.clicked.connect(self.toggle)
        layout.addWidget(self.header)

        self.content = QWidget()
        file_paths_layout = QVBoxLayout(self.content)
        file_paths_layout.setContentsMargins(0, 4, 0, 14)
        file_paths_layout.setSpacing(8)

        self.leftover_row, self.leftover_path_field = create_path_row(
            "Leftover detections",
            "Contains the detections still available for tracking: detections in accepted tracks "
            "and detections you deleted (D, right-click or Shift-Z) are left out, and detections "
            "you added (Shift-click) are included.",
        )
        self.session_row, self.session_path_field = create_path_row(
            "Tracks accepted this session",
            "Contains the detections included in the accepted tracks, together with the respective generated track ids.",
        )
        self.combined_row, self.combined_path_field = create_path_row(
            "Combined tracks",
            "Contains the tracks from the added track file, together with the accepted tracks in this session.",
        )
        for row in (self.leftover_row, self.session_row, self.combined_row):
            file_paths_layout.addWidget(row)
        layout.addWidget(self.content)

        self.hide_all()  # nothing to show until a detections file is loaded

    def set_expanded(self, expanded):
        """Show/hide the panel's rows and update its header arrow."""
        self.content.setVisible(expanded)
        arrow = "▾" if expanded else "▸"  # ▾ expanded, ▸ collapsed
        self.header.setText(f"{arrow} Session Files")

    def toggle(self):
        self.set_expanded(not self.content.isVisible())

    def show_paths(self, leftover, session, combined=None):
        """Show the output files' paths (the combined file only if there is
        one) and open the panel, so a newly added file is visible straight
        away rather than hidden behind a closed header."""
        self.leftover_path_field.set_path(leftover)
        self.session_path_field.set_path(session)
        self.leftover_row.setVisible(True)
        self.session_row.setVisible(True)
        if combined is not None:
            self.combined_path_field.set_path(combined)
        self.combined_row.setVisible(combined is not None)
        self.setVisible(True)
        self.set_expanded(True)

    def hide_all(self):
        """Hide the rows and the whole panel: there are no output files to show."""
        for row in (self.leftover_row, self.session_row, self.combined_row):
            row.setVisible(False)
        self.set_expanded(False)
        self.setVisible(False)
