"""The help window opened by the widget's "?" button: the keyboard and
mouse shortcuts."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


def show_help():
    """Display a dialog listing the keyboard and mouse shortcuts."""

    def section(title, rows):
        row_html = "".join(
            f'<tr><td style="padding:2px 18px 2px 0; white-space:nowrap;">'
            f'<code style="background:#3a3a3a; color:#eee; padding:1px 6px; '
            f'border-radius:3px;">{key}</code></td>'
            f'<td style="padding:2px 0;">{desc}</td></tr>'
            for key, desc in rows
        )
        return (
            f'<p style="margin:14px 0 4px 0; font-size:12pt; font-weight:600;">{title}</p>'
            f'<table style="border-collapse:collapse;">{row_html}</table>'
        )

    help_html = (
        '<div style="font-size:10.5pt;">'
        + section(
            "Navigation",
            [
                ("Right Arrow", "Next frame (hold to scroll continuously)"),
                ("Left Arrow", "Previous frame (hold to scroll continuously)"),
                ("Slider", "Jump to a specific frame"),
            ],
        )
        + section(
            "Mouse (on the image)",
            [
                ("Left-click", "Add the clicked detection to the current track and advance to the next frame"),
                ("Shift + Left-click", "Add a new detection at that position"),
                ("Right-click", "Delete the clicked detection"),
            ],
        )
        + section(
            "Keyboard shortcuts",
            [
                ("D", "Delete the tracked detection in the current frame"),
                ("Shift+Q", "Accept the current track"),
                ("W", "Save the current segment (up to the current frame)"),
                ("Shift+Z", "Delete the current segment (up to the current frame)"),
                ("X", "Delete all outgoing connections from the current node"),
            ],
        )
        + "</div>"
    )

    dialog = QDialog(None)
    dialog.setWindowTitle("Trackin Help")
    dialog.setMinimumWidth(480)

    layout = QVBoxLayout(dialog)
    label = QLabel(help_html)
    label.setTextFormat(Qt.RichText)
    label.setWordWrap(True)
    layout.addWidget(label)

    close_button = QPushButton("Close")
    close_button.clicked.connect(dialog.accept)
    layout.addWidget(close_button)

    dialog.exec_()
