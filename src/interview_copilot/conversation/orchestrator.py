from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from time import monotonic

from interview_copilot.audio.capture import AudioCapture, AudioDevice
from interview_copilot.config import Settings
from interview_copilot.knowledge.service import KnowledgeService
from interview_copilot.memory.service import MemoryService
from interview_copilot.models import Answer, SpeechEvent, SpeechEventType
from interview_copilot.providers.qwen.application import stream_application_answer
from interview_copilot.providers.qwen.client import QwenClient
from interview_copilot.providers.qwen.realtime import QwenRealtimeSpeech
from interview_copilot.storage.database import Database

SpeechHandler = Callable[[SpeechEvent], Awaitable[None] | None]
AnswerHandler = Callable[[Answer], Awaitable[None] | None]
logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    def __init__(
        self,
        settings: Settings,
        api_key: str,
        database: Database,
        on_speech: SpeechHandler,
        on_answer: AnswerHandler,
    ) -> None:
        self.settings = settings
        self.api_key = api_key
        self.database = database
        self.on_speech = on_speech
        self.on_answer = on_answer
        self.qwen = QwenClient(settings, api_key)
        self.knowledge = KnowledgeService(database, self.qwen)
        self.memory = MemoryService(database, self.qwen)
        self.interviewer = QwenRealtimeSpeech(settings, api_key, self._on_interviewer_event)
        self.candidate = QwenRealtimeSpeech(settings, api_key, self._on_candidate_event)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._interviewer_capture: AudioCapture | None = None
        self._candidate_capture: AudioCapture | None = None
        self._session_id: int | None = None
        self._last_turn_id: int | None = None
        self._last_question = ""
        self._translation = ""
        self._answer_task: asyncio.Task[None] | None = None
        self._app_session_id = ""
        self._question_final_at = 0.0
        self._first_answer_recorded = False
        self.answer_enabled = settings.answer_enabled

    @property
    def running(self) -> bool:
        return self.interviewer.running

    async def start(
        self, loopback_device: AudioDevice, microphone_device: AudioDevice | None = None
    ) -> None:
        if self.running:
            return
        self._loop = asyncio.get_running_loop()
        self._session_id = self.database.start_session()
        self._app_session_id = ""
        await self.interviewer.start()
        if microphone_device is not None:
            await self.candidate.start()
        self._interviewer_capture = AudioCapture(
            loopback_device,
            lambda chunk: self._push_threadsafe(self.interviewer, chunk),
            self._audio_error,
        )
        self._interviewer_capture.start()
        if microphone_device is not None:
            self._candidate_capture = AudioCapture(
                microphone_device,
                lambda chunk: self._push_threadsafe(self.candidate, chunk),
                self._audio_error,
            )
            self._candidate_capture.start()

    async def stop(self) -> None:
        if self._interviewer_capture:
            self._interviewer_capture.stop()
        if self._candidate_capture:
            self._candidate_capture.stop()
        await asyncio.gather(self.interviewer.stop(), self.candidate.stop())
        for task in (self._answer_task,):
            if task and not task.done():
                task.cancel()
        if self._session_id is not None:
            summary = ""
            with suppress(TimeoutError, Exception):
                summary = await asyncio.wait_for(
                    self.memory.summarize_session(self._session_id), timeout=8
                )
            self.database.end_session(self._session_id, summary)
        self._session_id = None

    async def generate_for_text(self, question: str, translation: str = "") -> None:
        if not self.answer_enabled:
            return
        self._question_final_at = monotonic()
        self._first_answer_recorded = False
        await self._schedule_answer(question.strip(), translation.strip())

    def set_answer_enabled(self, enabled: bool) -> None:
        self.answer_enabled = enabled
        if not enabled and self._answer_task and not self._answer_task.done():
            self._answer_task.cancel()

    def _push_threadsafe(self, provider: QwenRealtimeSpeech, chunk: bytes) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(provider.push_audio, chunk)

    def _audio_error(self, message: str) -> None:
        if self._loop and not self._loop.is_closed():
            event = SpeechEvent(SpeechEventType.ERROR, message)
            self._loop.call_soon_threadsafe(asyncio.create_task, self._emit_speech(event))

    async def _on_interviewer_event(self, event: SpeechEvent) -> None:
        await self._emit_speech(event)
        if event.kind == SpeechEventType.TRANSLATION_FINAL:
            self._translation = event.text.strip()
            if self._question_final_at:
                logger.info(
                    "METRIC translation_after_final_ms %.1f",
                    (monotonic() - self._question_final_at) * 1000,
                )
            if self._last_turn_id and self._translation:
                self.database.update_turn_translation(self._last_turn_id, self._translation)
        elif event.kind == SpeechEventType.SOURCE_FINAL:
            question = event.text.strip()
            if len(question.split()) < 3:
                return
            self._last_question = question
            self._translation = ""
            self._question_final_at = monotonic()
            self._first_answer_recorded = False
            if self._session_id is not None:
                self._last_turn_id = self.database.add_turn(
                    self._session_id, "interviewer", question
                )
            if self.answer_enabled:
                await self._schedule_answer(question, self._translation)

    async def _on_candidate_event(self, event: SpeechEvent) -> None:
        if event.kind == SpeechEventType.SOURCE_FINAL and self._session_id is not None:
            text = event.text.strip()
            if text:
                self.database.add_turn(self._session_id, "candidate", text)

    async def _schedule_answer(self, question: str, translation: str) -> None:
        if not question:
            return
        if self._answer_task and not self._answer_task.done():
            self._answer_task.cancel()
        request_id = uuid.uuid4().hex
        self._answer_task = asyncio.create_task(
            self._answer(question, translation, request_id), name=f"answer-{request_id[:8]}"
        )

    async def _answer(self, question: str, translation: str, request_id: str) -> None:
        try:
            translation = self._translation if question == self._last_question else translation
            context = (
                self.database.recent_context(self._session_id, limit=4)
                if self._session_id is not None
                else ""
            )
            _answer, self._app_session_id = await stream_application_answer(
                self.settings,
                self.api_key,
                question,
                translation,
                context,
                self._emit_answer,
                session_id=self._app_session_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._emit_speech(
                SpeechEvent(SpeechEventType.ERROR, f"生成回答失败：{exc}", request_id)
            )

    async def _emit_speech(self, event: SpeechEvent) -> None:
        result = self.on_speech(event)
        if inspect.isawaitable(result):
            await result

    async def _emit_answer(self, answer: Answer) -> None:
        if not self._first_answer_recorded and (answer.english or answer.chinese):
            self._first_answer_recorded = True
            logger.info(
                "METRIC answer_first_text_ms %.1f",
                (monotonic() - self._question_final_at) * 1000,
            )
        result = self.on_answer(answer)
        if inspect.isawaitable(result):
            await result
