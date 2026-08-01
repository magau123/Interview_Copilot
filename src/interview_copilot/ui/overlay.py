"""Frameless content panels that float over the meeting window.

Each panel is its own top-level window so the interviewer's words and the answer
script can be parked wherever the candidate's eyes already are, independently of
each other and of the control console.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtWidgets import QAbstractScrollArea, QHBoxLayout, QSizeGrip, QVBoxLayout, QWidget


class OverlayWindow(QWidget):
    """Draggable, resizable, chrome-free window wrapping one content widget."""

    moved = Signal()
    contextMenuRequested = Signal(QPoint)

    MIN_WIDTH = 260
    MIN_HEIGHT = 120

    def __init__(self, content: QWidget, *, tooltip: str = "") -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setObjectName("overlayRoot")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self._drag_offset: QPoint | None = None
        self._content = content
        if tooltip:
            self.setToolTip(tooltip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 4)
        layout.setSpacing(2)
        layout.addWidget(content, 1)
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(grip_row)

        # Dragging must also work when the pointer is over the text itself.
        content.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        content.installEventFilter(self)
        if isinstance(content, QAbstractScrollArea):
            content.viewport().installEventFilter(self)

    @property
    def content(self) -> QWidget:
        return self._content

    def set_alert(self, alert: bool) -> None:
        """Highlight the frame, e.g. while a newer answer waits to be shown."""
        if self.property("alert") == alert:
            return
        self.setProperty("alert", alert)
        self.style().unpolish(self)
        self.style().polish(self)

    def geometry_values(self) -> list[int]:
        rect = self.geometry()
        return [rect.x(), rect.y(), rect.width(), rect.height()]

    def apply_geometry(self, values: list[int] | None) -> bool:
        if not values or len(values) != 4:
            return False
        x, y, width, height = (int(value) for value in values)
        self.setGeometry(
            x, y, max(width, self.MIN_WIDTH), max(height, self.MIN_HEIGHT)
        )
        return True

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt naming
        kind = event.type()
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._begin_drag(event.globalPosition().toPoint())
            return True
        if kind == QEvent.Type.MouseMove and self._drag_offset is not None:
            self._drag_to(event.globalPosition().toPoint())
            return True
        if kind == QEvent.Type.MouseButtonRelease and self._drag_offset is not None:
            self._end_drag()
            return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.MouseButton.LeftButton:
            self._begin_drag(event.globalPosition().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            self._drag_to(event.globalPosition().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            self._end_drag()
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.contextMenuRequested.emit(event.globalPos())
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self.moved.emit()

    def _begin_drag(self, global_position: QPoint) -> None:
        self._drag_offset = global_position - self.frameGeometry().topLeft()

    def _drag_to(self, global_position: QPoint) -> None:
        if self._drag_offset is not None:
            self.move(global_position - self._drag_offset)

    def _end_drag(self) -> None:
        self._drag_offset = None
        self.moved.emit()
