from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from interview_copilot.config import Settings
from interview_copilot.conversation.orchestrator import ConversationOrchestrator
from interview_copilot.models import Answer, SpeechEvent, SpeechEventType
from interview_copilot.storage.database import Database


@pytest.mark.asyncio
async def test_manual_generate_streams_question_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")
    speech_events: list[SpeechEvent] = []
    answers: list[Answer] = []
    seen: dict[str, str] = {}

    async def on_speech(event: SpeechEvent) -> None:
        speech_events.append(event)

    async def on_answer(answer: Answer) -> None:
        answers.append(answer)

    async def fake_stream(_settings, _api_key, question, on_update, *, session_id: str = ""):
        seen["question"] = question
        answer = Answer(english="I led the billing migration.")
        await on_update(answer)
        return answer, session_id

    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.stream_application_answer",
        fake_stream,
    )

    orchestrator = ConversationOrchestrator(
        settings, "sk-test", database, on_speech, on_answer
    )
    orchestrator._session_id = database.start_session()

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "Tell me about yourself please.")
    )
    assert orchestrator._answer_task is None
    assert seen == {}

    await orchestrator.generate_for_text("Tell me about yourself please.")
    assert orchestrator._answer_task is not None
    await orchestrator._answer_task

    assert seen == {"question": "Tell me about yourself please."}
    assert answers[-1].english == "I led the billing migration."
    assert any("正在生成回答" in event.text for event in speech_events)


@pytest.mark.asyncio
async def test_interjections_do_not_schedule_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")
    questions: list[str] = []
    speech_events: list[SpeechEvent] = []

    async def fake_stream(_settings, _api_key, question, on_update, *, session_id: str = ""):
        questions.append(question)
        answer = Answer(english="Sure.")
        await on_update(answer)
        return answer, session_id

    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.stream_application_answer",
        fake_stream,
    )

    async def on_speech(event: SpeechEvent) -> None:
        speech_events.append(event)

    orchestrator = ConversationOrchestrator(
        settings, "sk-test", database, on_speech, lambda _answer: None
    )
    orchestrator._session_id = database.start_session()

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "What was your role on that project?")
    )
    assert orchestrator._answer_task is None

    for noise in ("Okay.", "Mm-hmm", "Got it.", "..."):
        await orchestrator._on_interviewer_event(
            SpeechEvent(SpeechEventType.SOURCE_FINAL, noise)
        )
    assert questions == []
    assert any("已忽略插话" in event.text for event in speech_events)


@pytest.mark.asyncio
async def test_split_question_is_merged_for_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")

    orchestrator = ConversationOrchestrator(
        settings, "sk-test", database, lambda _event: None, lambda _answer: None
    )
    orchestrator._session_id = database.start_session()

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "Tell me about a hard project")
    )
    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "and your role in it")
    )

    assert orchestrator._last_question == (
        "Tell me about a hard project and your role in it"
    )


@pytest.mark.asyncio
async def test_repeated_final_does_not_reopen_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")

    orchestrator = ConversationOrchestrator(
        settings, "sk-test", database, lambda _event: None, lambda _answer: None
    )
    orchestrator._session_id = database.start_session()

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "Why did you leave that company?")
    )
    first_turn = orchestrator._last_turn_id
    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "Why did you leave that company")
    )

    assert orchestrator._last_turn_id == first_turn
