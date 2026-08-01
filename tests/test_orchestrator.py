from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from interview_copilot.config import Settings
from interview_copilot.conversation.orchestrator import ConversationOrchestrator
from interview_copilot.models import Answer, SpeechEvent, SpeechEventType
from interview_copilot.storage.database import Database


@pytest.mark.asyncio
async def test_answer_waits_for_translation_before_calling_app(
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

    async def fake_stream(
        _settings,
        _api_key,
        question,
        translation,
        _context,
        on_update,
        *,
        session_id: str = "",
    ):
        seen["question"] = question
        seen["translation"] = translation
        answer = Answer(chinese="完整中文回答")
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
    assert orchestrator._answer_task is not None
    await asyncio.sleep(0.05)
    assert seen == {}

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.TRANSLATION_FINAL, "请介绍一下你自己。")
    )
    await orchestrator._answer_task

    assert seen == {
        "question": "Tell me about yourself please.",
        "translation": "请介绍一下你自己。",
    }
    assert answers[-1].chinese == "完整中文回答"
    assert any("知识库" in event.text for event in speech_events)


@pytest.mark.asyncio
async def test_interjections_do_not_trigger_a_new_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")
    questions: list[str] = []
    speech_events: list[SpeechEvent] = []

    async def fake_stream(
        _settings,
        _api_key,
        question,
        _translation,
        _context,
        on_update,
        *,
        session_id: str = "",
    ):
        questions.append(question)
        answer = Answer(english="Sure.", chinese="好的。")
        await on_update(answer)
        return answer, session_id

    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.stream_application_answer",
        fake_stream,
    )
    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.TRANSLATION_WAIT_SECONDS", 0.01
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
    await orchestrator._answer_task

    for noise in ("Okay.", "Mm-hmm", "Got it.", "..."):
        await orchestrator._on_interviewer_event(
            SpeechEvent(SpeechEventType.SOURCE_FINAL, noise)
        )
    assert questions == ["What was your role on that project?"]
    assert any(event.kind == SpeechEventType.ANSWER_PENDING for event in speech_events)
    assert any("已忽略插话" in event.text for event in speech_events)


@pytest.mark.asyncio
async def test_split_question_is_merged_instead_of_replacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")
    questions: list[str] = []

    async def fake_stream(
        _settings,
        _api_key,
        question,
        _translation,
        _context,
        on_update,
        *,
        session_id: str = "",
    ):
        questions.append(question)
        answer = Answer(chinese="回答")
        await on_update(answer)
        return answer, session_id

    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.stream_application_answer",
        fake_stream,
    )
    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.TRANSLATION_WAIT_SECONDS", 0.01
    )

    orchestrator = ConversationOrchestrator(
        settings, "sk-test", database, lambda _event: None, lambda _answer: None
    )
    orchestrator._session_id = database.start_session()

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "Tell me about a hard project")
    )
    await orchestrator._answer_task
    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "and your role in it")
    )
    await orchestrator._answer_task

    assert questions == [
        "Tell me about a hard project",
        "Tell me about a hard project and your role in it",
    ]


@pytest.mark.asyncio
async def test_repeated_final_does_not_regenerate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")
    questions: list[str] = []

    async def fake_stream(
        _settings,
        _api_key,
        question,
        _translation,
        _context,
        on_update,
        *,
        session_id: str = "",
    ):
        questions.append(question)
        answer = Answer(chinese="回答")
        await on_update(answer)
        return answer, session_id

    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.stream_application_answer",
        fake_stream,
    )
    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.TRANSLATION_WAIT_SECONDS", 0.01
    )

    orchestrator = ConversationOrchestrator(
        settings, "sk-test", database, lambda _event: None, lambda _answer: None
    )
    orchestrator._session_id = database.start_session()

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "Why did you leave that company?")
    )
    await orchestrator._answer_task
    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "Why did you leave that company")
    )

    assert questions == ["Why did you leave that company?"]


@pytest.mark.asyncio
async def test_answer_falls_back_when_translation_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(answer_enabled=True, knowledge_app_id="aid-test")
    database = Database(tmp_path / "test.db")
    seen: dict[str, str] = {}

    async def fake_stream(
        _settings,
        _api_key,
        question,
        translation,
        _context,
        on_update,
        *,
        session_id: str = "",
    ):
        seen["question"] = question
        seen["translation"] = translation
        answer = Answer(chinese="timeout fallback")
        await on_update(answer)
        return answer, session_id

    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.stream_application_answer",
        fake_stream,
    )
    monkeypatch.setattr(
        "interview_copilot.conversation.orchestrator.TRANSLATION_WAIT_SECONDS",
        0.05,
    )

    orchestrator = ConversationOrchestrator(
        settings,
        "sk-test",
        database,
        lambda _event: None,
        lambda _answer: None,
    )
    orchestrator._session_id = database.start_session()

    await orchestrator._on_interviewer_event(
        SpeechEvent(SpeechEventType.SOURCE_FINAL, "What is your biggest strength today?")
    )
    assert orchestrator._answer_task is not None
    await orchestrator._answer_task

    assert seen["question"] == "What is your biggest strength today?"
    assert seen["translation"] == ""
