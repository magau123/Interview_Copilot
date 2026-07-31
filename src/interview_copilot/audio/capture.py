from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio

logger = logging.getLogger(__name__)

TARGET_RATE = 16_000
CHUNK_MS = 100


@dataclass(frozen=True, slots=True)
class AudioDevice:
    index: int
    name: str
    sample_rate: int
    channels: int
    loopback: bool

    @property
    def label(self) -> str:
        suffix = "（系统声音）" if self.loopback else "（麦克风）"
        return f"{self.name} {suffix}"


def list_audio_devices() -> tuple[list[AudioDevice], list[AudioDevice]]:
    audio = pyaudio.PyAudio()
    try:
        loopbacks = [
            AudioDevice(
                index=int(info["index"]),
                name=str(info["name"]),
                sample_rate=int(info["defaultSampleRate"]),
                channels=max(1, int(info["maxInputChannels"])),
                loopback=True,
            )
            for info in audio.get_loopback_device_info_generator()
        ]
        microphones: list[AudioDevice] = []
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) <= 0 or info.get("isLoopbackDevice"):
                continue
            microphones.append(
                AudioDevice(
                    index=index,
                    name=str(info["name"]),
                    sample_rate=int(info["defaultSampleRate"]),
                    channels=int(info["maxInputChannels"]),
                    loopback=False,
                )
            )
        return loopbacks, microphones
    finally:
        audio.terminate()


class AudioCapture:
    """Blocking WASAPI capture isolated in a daemon thread."""

    def __init__(
        self,
        device: AudioDevice,
        on_audio: Callable[[bytes], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.device = device
        self.on_audio = on_audio
        self.on_error = on_error or (lambda message: logger.error(message))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-capture")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        audio = pyaudio.PyAudio()
        stream = None
        frames = max(1, int(self.device.sample_rate * CHUNK_MS / 1000))
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=self.device.channels,
                rate=self.device.sample_rate,
                input=True,
                input_device_index=self.device.index,
                frames_per_buffer=frames,
            )
            while not self._stop.is_set():
                raw = stream.read(frames, exception_on_overflow=False)
                self.on_audio(_to_pcm16_mono(raw, self.device.channels, self.device.sample_rate))
        except Exception as exc:
            self.on_error(f"音频设备已断开或无法读取：{exc}")
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()


def _to_pcm16_mono(raw: bytes, channels: int, source_rate: int) -> bytes:
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        usable = samples[: len(samples) - (len(samples) % channels)]
        samples = usable.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
    if source_rate == TARGET_RATE or len(samples) < 2:
        return samples.tobytes()
    target_length = max(1, round(len(samples) * TARGET_RATE / source_rate))
    source_axis = np.arange(len(samples), dtype=np.float64)
    target_axis = np.linspace(0, len(samples) - 1, target_length)
    resampled = np.interp(target_axis, source_axis, samples).astype(np.int16)
    return resampled.tobytes()
