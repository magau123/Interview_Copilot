from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

import websockets
from websockets.exceptions import InvalidStatus

from interview_copilot.config import Settings
from interview_copilot.models import SpeechEvent, SpeechEventType
from interview_copilot.ssl_util import create_ssl_context

EventHandler = Callable[[SpeechEvent], Awaitable[None] | None]
logger = logging.getLogger(__name__)


def classify_connection_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "certificate_verify_failed",
            "sslcertverificationerror",
            "certificate verify failed",
            "unable to get local issuer",
        )
    ):
        return (
            "SSL 证书校验失败。常见原因：公司代理/杀毒软件 HTTPS 解密、系统时间错误、"
            "或根证书不完整。可先关闭代理或把本程序加入杀毒白名单后重试。"
        )
    if any(
        token in lowered
        for token in (
            "connectionreset",
            "winerror 64",
            "connection refused",
            "timed out",
            "timeout",
            "network is unreachable",
            "name or service not known",
            "getaddrinfo failed",
        )
    ):
        return (
            "网络连接被中断或超时。设置看起来可以正常连通时，多为防火墙/代理/"
            "不稳定网络导致。可切换有线网络、关闭 VPN、或稍后重试。"
        )
    if "401" in text or "403" in text or "unauthorized" in lowered:
        return "鉴权失败：请检查 API Key 是否属于当前地域和业务空间。"
    if "model" in lowered and "missing" in lowered:
        return "请求缺少模型参数：请确认实时翻译模型名称填写正确。"
    return f"实时服务连接失败：{exc}"


class QwenRealtimeSpeech:
    """Qwen LiveTranslate duplex client with bounded audio buffering."""

    def __init__(self, settings: Settings, api_key: str, on_event: EventHandler) -> None:
        self.settings = settings
        self.api_key = api_key
        self.on_event = on_event
        self._audio: asyncio.Queue[bytes] = asyncio.Queue(maxsize=30)
        self._runner: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._ws = None
        self._active_url = ""
        # Confirmed translation text may arrive as cumulative snapshots or deltas.
        self._translation_confirmed = ""

    @property
    def running(self) -> bool:
        return bool(self._runner and not self._runner.done())

    def push_audio(self, chunk: bytes) -> None:
        if not chunk or self._stop.is_set():
            return
        if self._audio.full():
            with suppress(asyncio.QueueEmpty):
                self._audio.get_nowait()
        with suppress(asyncio.QueueFull):
            self._audio.put_nowait(chunk)

    async def start(self) -> None:
        if self.running:
            return
        if not self.api_key or not self.settings.realtime_urls:
            raise ValueError("请先配置 DashScope API Key，并填写 Workspace ID 或确认区域")
        self._stop.clear()
        self._runner = asyncio.create_task(self._run(), name="qwen-realtime")

    async def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            with suppress(Exception):
                await self._ws.send(json.dumps({"type": "session.finish"}))
            with suppress(Exception):
                await asyncio.wait_for(self._ws.close(), timeout=2)
        if self._runner:
            self._runner.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner
        self._runner = None

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            last_error: BaseException | None = None
            connected = False
            for url in self.settings.realtime_urls:
                if self._stop.is_set():
                    return
                try:
                    await self._emit(
                        SpeechEventType.STATUS, f"正在连接实时服务…\n{url.split('?')[0]}"
                    )
                    ssl_context = create_ssl_context()
                    async with websockets.connect(
                        url,
                        additional_headers={"Authorization": f"Bearer {self.api_key}"},
                        max_size=4 * 1024 * 1024,
                        ping_interval=20,
                        ping_timeout=20,
                        open_timeout=20,
                        ssl=ssl_context,
                    ) as ws:
                        self._ws = ws
                        self._active_url = url
                        await ws.send(json.dumps(self._session_config(), ensure_ascii=False))
                        await self._emit(
                            SpeechEventType.STATUS,
                            f"实时识别已连接（{url.split('://', 1)[-1].split('/')[0]}）",
                        )
                        connected = True
                        backoff = 1.0
                        sender = asyncio.create_task(self._send_audio(ws))
                        receiver = asyncio.create_task(self._receive(ws))
                        done, pending = await asyncio.wait(
                            {sender, receiver}, return_when=asyncio.FIRST_EXCEPTION
                        )
                        for task in pending:
                            task.cancel()
                        for task in done:
                            task.result()
                        break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    logger.warning("Realtime connect failed for %s: %s", url, exc)
                    continue
                finally:
                    self._ws = None

            if connected:
                continue
            if self._stop.is_set():
                break
            message = classify_connection_error(last_error or RuntimeError("未知连接错误"))
            await self._emit(SpeechEventType.ERROR, message)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)

    def _session_config(self) -> dict:
        return {
            "event_id": f"event_{int(time.time() * 1000)}",
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "pcm",
                "input_audio_transcription": {
                    "model": self.settings.asr_model,
                    "language": self.settings.source_language,
                },
                "translation": {"language": self.settings.target_language},
            },
        }

    async def _send_audio(self, ws) -> None:
        while not self._stop.is_set():
            chunk = await self._audio.get()
            await ws.send(
                json.dumps(
                    {
                        "event_id": f"event_{time.time_ns()}",
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )

    async def _receive(self, ws) -> None:
        async for raw in ws:
            event = json.loads(raw)
            kind = event.get("type", "")
            turn_id = str(event.get("item_id") or event.get("response_id") or "")
            if kind == "response.created":
                self._translation_confirmed = ""
            elif kind == "conversation.item.input_audio_transcription.text":
                # text = confirmed cumulative ASR, stash = tentative pending text
                await self._emit(
                    SpeechEventType.SOURCE_PARTIAL,
                    f"{event.get('text', '')}{event.get('stash', '')}",
                    turn_id,
                )
            elif kind == "conversation.item.input_audio_transcription.completed":
                await self._emit(
                    SpeechEventType.SOURCE_FINAL, event.get("transcript", ""), turn_id
                )
            elif kind in {"response.text.text", "response.audio_transcript.text"}:
                display = self._merge_translation_partial(
                    event.get("text", "") or "",
                    event.get("stash", "") or "",
                )
                await self._emit(SpeechEventType.TRANSLATION_PARTIAL, display, turn_id)
            elif kind in {"response.text.done", "response.audio_transcript.done"}:
                final = (
                    event.get("text")
                    or event.get("transcript")
                    or self._translation_confirmed
                    or ""
                )
                if len(final.strip()) < len(self._translation_confirmed.strip()):
                    final = self._translation_confirmed
                self._translation_confirmed = ""
                await self._emit(SpeechEventType.TRANSLATION_FINAL, final, turn_id)
            elif kind == "error":
                error = event.get("error", {})
                await self._emit(
                    SpeechEventType.ERROR,
                    str(error.get("message") or event.get("message") or "未知实时服务错误"),
                    turn_id,
                )

    def _merge_translation_partial(self, text: str, stash: str) -> str:
        """Accept either cumulative snapshots or incremental deltas from the API."""
        previous = self._translation_confirmed
        if not previous:
            self._translation_confirmed = text
        elif text.startswith(previous):
            self._translation_confirmed = text
        elif previous.startswith(text):
            # Server corrected / shortened the confirmed segment.
            self._translation_confirmed = text
        else:
            # Incremental chunk — append.
            self._translation_confirmed = previous + text
        return f"{self._translation_confirmed}{stash}"

    async def _emit(self, kind: SpeechEventType, text: str, turn_id: str = "") -> None:
        result = self.on_event(SpeechEvent(kind=kind, text=text, turn_id=turn_id))
        if inspect.isawaitable(result):
            await result


async def test_realtime_connection(settings: Settings, api_key: str) -> str:
    """One-shot connectivity check used by the Settings page."""
    if not api_key:
        raise ValueError("API Key 为空")
    if not settings.realtime_urls:
        raise ValueError("请填写 Workspace ID")
    last_error: BaseException | None = None
    for url in settings.realtime_urls:
        try:
            ssl_context = create_ssl_context()
            async with websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                open_timeout=15,
                close_timeout=5,
                ping_interval=None,
                ssl=ssl_context,
            ) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=8)
                payload = json.loads(msg)
                host = url.split("://", 1)[-1].split("/", 1)[0]
                return (
                    f"连接成功：{host}\n"
                    f"事件：{payload.get('type', 'unknown')}\n"
                    f"模型：{(payload.get('session') or {}).get('model', settings.realtime_model)}"
                )
        except InvalidStatus as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(classify_connection_error(last_error or RuntimeError("未知错误")))
