"""Teleprompter reading surface: the text moves, the eyes do not."""

from __future__ import annotations

import re
from time import monotonic

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QTextEdit

_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9'’\-]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# Reading one Chinese character takes roughly this fraction of an English word.
_CJK_WORD_RATIO = 0.6


def reading_units(text: str) -> float:
    """Speech length in English-word equivalents, so one speed fits both scripts."""
    return len(_LATIN_WORD_RE.findall(text)) + _CJK_WORD_RATIO * len(_CJK_RE.findall(text))


class TeleprompterView(QTextEdit):
    """Read-only view that scrolls itself at speaking pace.

    Scrolling is driven in pixels rather than lines so the text creeps upward
    smoothly instead of jumping, which is what keeps the reader's gaze parked on
    the guide line.
    """

    runningChanged = Signal(bool)

    TICK_MS = 33
    # Height fraction where the gaze should rest.
    GUIDE_RATIO = 0.34
    MIN_WPM = 60
    MAX_WPM = 240

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._wpm = 130
        self._units = 0.0
        self._tail = 0.0
        self._carry = 0.0
        self._running = False
        self._last_tick = 0.0
        self._guide = QFrame(self.viewport())
        self._guide.setObjectName("readingGuide")
        self._guide.setFixedHeight(2)
        self._guide.setVisible(False)
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def at_end(self) -> bool:
        scrollbar = self.verticalScrollBar()
        return scrollbar.value() >= scrollbar.maximum()

    def set_script(self, html: str, plain_text: str = "", *, preserve_scroll: bool = False) -> None:
        """Load content; rewind unless this is an in-place streaming update."""
        scrollbar = self.verticalScrollBar()
        previous = scrollbar.value()
        was_running = self._running
        if not preserve_scroll:
            self.set_running(False)
        self.setHtml(html)
        self._units = reading_units(plain_text or self.toPlainText())
        self._add_tail_space()
        if preserve_scroll:
            scrollbar.setValue(min(previous, scrollbar.maximum()))
            if was_running:
                self.set_running(True)
        else:
            self.rewind()

    def _add_tail_space(self) -> None:
        """Blank space below the text so the last line can still reach the guide."""
        self._tail = max(0.0, self.viewport().height() * (1 - self.GUIDE_RATIO))
        root = self.document().rootFrame()
        frame_format = root.frameFormat()
        frame_format.setBottomMargin(self._tail)
        root.setFrameFormat(frame_format)

    def rewind(self) -> None:
        self._carry = 0.0
        self.verticalScrollBar().setValue(0)

    def set_wpm(self, value: int) -> None:
        self._wpm = max(self.MIN_WPM, min(self.MAX_WPM, int(value)))

    def set_guide_visible(self, visible: bool) -> None:
        self._guide.setVisible(visible)
        self._position_guide()

    def set_running(self, running: bool) -> None:
        running = bool(running) and self.verticalScrollBar().maximum() > 0
        if running == self._running:
            return
        self._running = running
        if running:
            self._last_tick = monotonic()
            self._timer.start()
        else:
            self._timer.stop()
        self.runningChanged.emit(running)

    def toggle(self) -> None:
        self.set_running(not self._running)

    def nudge(self, pixels: int) -> None:
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.value() + pixels)

    def pixels_per_second(self) -> float:
        """Match scroll speed to speaking pace, whatever the font size is."""
        height = self.document().documentLayout().documentSize().height() - self._tail
        if height <= 0 or self._units <= 0:
            return 0.0
        seconds = self._units / self._wpm * 60
        if seconds <= 0:
            return 0.0
        return height / seconds

    def _tick(self) -> None:
        now = monotonic()
        elapsed = min(0.25, max(0.0, now - self._last_tick))
        self._last_tick = now
        speed = self.pixels_per_second()
        if speed <= 0:
            return
        self._carry += speed * elapsed
        step = int(self._carry)
        if step:
            self._carry -= step
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() + step)
        if self.at_end:
            self.set_running(False)

    def _position_guide(self) -> None:
        viewport = self.viewport()
        self._guide.setGeometry(
            0,
            int(viewport.height() * self.GUIDE_RATIO),
            viewport.width(),
            self._guide.height(),
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._position_guide()
