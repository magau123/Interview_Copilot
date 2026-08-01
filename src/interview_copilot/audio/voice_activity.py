"""Local "am I speaking right now" detector for the teleprompter.

Deliberately amplitude based: the teleprompter only needs to know whether the
candidate is talking, so there is no reason to spend a realtime ASR session (or
tolerate its latency) on it. Recognition accuracy is irrelevant here, which also
means a heavy accent never stalls the scroll.
"""

from __future__ import annotations

import logging
from time import monotonic

from interview_copilot.audio.capture import AudioCapture, AudioDevice, pcm16_rms

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 250.0
# Keep scrolling through the short gaps between words and sentences.
HANGOVER_SECONDS = 0.6
# Ignore clicks and keystrokes: speech must span at least two 100 ms chunks.
ATTACK_SECONDS = 0.1
# Exponential smoothing on 100 ms chunks; low enough to react within a syllable.
SMOOTHING = 0.45


class VoiceActivityMonitor:
    """Tracks smoothed microphone level and exposes a speaking flag."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        *,
        hangover_seconds: float = HANGOVER_SECONDS,
        attack_seconds: float = ATTACK_SECONDS,
    ) -> None:
        self.threshold = float(threshold)
        self.hangover_seconds = hangover_seconds
        self.attack_seconds = attack_seconds
        self._level = 0.0
        self._loud_since: float | None = None
        self._last_loud_at: float | None = None
        self._capture: AudioCapture | None = None
        self._error = ""

    @property
    def level(self) -> float:
        """Smoothed RMS amplitude of the latest audio (0..32767)."""
        return self._level

    @property
    def error(self) -> str:
        return self._error

    @property
    def running(self) -> bool:
        return bool(self._capture and self._capture.running)

    def start(self, device: AudioDevice) -> None:
        self.stop()
        self._error = ""
        self._capture = AudioCapture(device, self.feed, self._on_error)
        self._capture.start()

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self.reset()

    def reset(self) -> None:
        self._level = 0.0
        self._loud_since = None
        self._last_loud_at = None

    def feed(self, chunk: bytes, now: float | None = None) -> None:
        """Consume one PCM16 chunk; called from the capture thread."""
        moment = monotonic() if now is None else now
        self._level = (1 - SMOOTHING) * self._level + SMOOTHING * pcm16_rms(chunk)
        if self._level >= self.threshold:
            if self._loud_since is None:
                self._loud_since = moment
            self._last_loud_at = moment
        elif self._last_loud_at is None or moment - self._last_loud_at > self.hangover_seconds:
            self._loud_since = None

    def speaking(self, now: float | None = None) -> bool:
        moment = monotonic() if now is None else now
        if self._loud_since is None or self._last_loud_at is None:
            return False
        if moment - self._last_loud_at > self.hangover_seconds:
            return False
        return self._last_loud_at - self._loud_since >= self.attack_seconds

    def _on_error(self, message: str) -> None:
        self._error = message
        logger.warning("Voice activity capture failed: %s", message)
