from __future__ import annotations

import numpy as np

from interview_copilot.audio.voice_activity import VoiceActivityMonitor


def chunk(amplitude: int, samples: int = 1600) -> bytes:
    """One 100 ms PCM16 chunk at a constant amplitude."""
    return np.full(samples, amplitude, dtype=np.int16).tobytes()


def test_speech_starts_scrolling_after_attack_and_stops_after_hangover() -> None:
    monitor = VoiceActivityMonitor(threshold=250, hangover_seconds=0.6, attack_seconds=0.1)
    now = 0.0
    monitor.feed(chunk(3000), now)
    # A single loud chunk is not enough: keystrokes and clicks must not scroll.
    assert not monitor.speaking(now)

    now += 0.1
    monitor.feed(chunk(3000), now)
    assert monitor.speaking(now)

    # Short gap between words keeps scrolling.
    assert monitor.speaking(now + 0.4)
    # A real pause stops it.
    assert not monitor.speaking(now + 0.9)


def test_room_noise_never_counts_as_speech() -> None:
    monitor = VoiceActivityMonitor(threshold=250)
    now = 0.0
    for _ in range(20):
        monitor.feed(chunk(60), now)
        now += 0.1
    assert not monitor.speaking(now)
    assert monitor.level < 250


def test_reset_clears_state() -> None:
    monitor = VoiceActivityMonitor(threshold=250)
    monitor.feed(chunk(4000), 0.0)
    monitor.feed(chunk(4000), 0.2)
    assert monitor.speaking(0.2)
    monitor.reset()
    assert not monitor.speaking(0.2)
    assert monitor.level == 0.0
