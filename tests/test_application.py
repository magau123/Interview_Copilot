from __future__ import annotations

import json

import httpx
import pytest

from interview_copilot.config import Settings
from interview_copilot.models import Answer
from interview_copilot.providers.qwen import application as application_mod
from interview_copilot.providers.qwen.application import (
    _assistant_delta,
    classify_application_error,
    completed_sentences,
    extract_english_answer,
    knowledge_chat_url,
    split_bilingual_answer,
    stream_application_answer,
)


def test_extract_english_answer_strips_markers() -> None:
    assert (
        extract_english_answer(
            "[EN]\nI led the billing migration.\nIt cut latency in half.\n"
            "[ZH]\n我主导了计费迁移。\n延迟降低了一半。"
        )
        == "I led the billing migration.\nIt cut latency in half."
    )


def test_extract_english_answer_accepts_plain_english() -> None:
    assert extract_english_answer("I come from the knowledge base.") == (
        "I come from the knowledge base."
    )
    assert extract_english_answer("   ") == ""


def test_split_bilingual_answer_keeps_english_field_only() -> None:
    assert split_bilingual_answer("英文：\nHello there.\n中文对照\n你好。") == (
        "Hello there.",
        "",
    )


def test_completed_sentences_holds_trailing_fragment() -> None:
    assert (
        completed_sentences("I led the migration.\nIt cut latency", final=False)
        == "I led the migration."
    )
    assert (
        completed_sentences("I led the migration.\nIt cut latency in half.", final=False)
        == "I led the migration.\nIt cut latency in half."
    )
    assert (
        completed_sentences("I led the migration.\nIt cut latency", final=True)
        == "I led the migration.\nIt cut latency"
    )


def test_knowledge_chat_url_uses_workspace_host() -> None:
    settings = Settings(
        workspace_id="llm-x0jmmfp7f7vu9rc5",
        region="cn-beijing",
        knowledge_app_id="aid-433c2467738a4ae1948488f117508609",
    )
    assert knowledge_chat_url(settings) == (
        "https://llm-x0jmmfp7f7vu9rc5.cn-beijing.maas.aliyuncs.com"
        "/api/v2/apps/knowledge/chat"
    )


def test_classify_access_denied() -> None:
    message = classify_application_error("App.AccessDenied", "App access denied.")
    assert "已在百炼控制台发布" in message
    assert "同一业务空间" in message


def test_assistant_delta_ignores_tool_events() -> None:
    tool_event = {
        "code": "200",
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "tool",
                        "content": "知识库检索结果……",
                        "extra": {"step": "tool_calling"},
                    }
                }
            ]
        },
    }
    assistant_event = {
        "code": "200",
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I am a data analyst.",
                        "extra": {"step": "generating"},
                    }
                }
            ]
        },
    }
    assert _assistant_delta(tool_event) == ""
    assert _assistant_delta(assistant_event) == "I am a data analyst."


@pytest.mark.asyncio
async def test_stream_application_answer_sends_question_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "code": "200",
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "tool",
                            "content": "检索结果不应展示",
                            "extra": {"step": "tool_calling"},
                        }
                    }
                ]
            },
        },
        {
            "code": "200",
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I come from the knowledge base.\nI",
                            "extra": {"step": "generating"},
                        }
                    }
                ]
            },
        },
        {
            "code": "200",
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": " focus on reliability.",
                            "extra": {"step": "generating"},
                        }
                    }
                ]
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        assert body["input"]["messages"] == [
            {"role": "user", "content": "Tell me about yourself."}
        ]
        payload = b"".join(
            f"data:{json.dumps(event, ensure_ascii=False)}\n\n".encode()
            for event in events
        )
        return httpx.Response(200, content=payload)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(application_mod.httpx, "AsyncClient", fake_client)

    settings = Settings(
        knowledge_app_id="aid-433c2467738a4ae1948488f117508609",
        region="cn-beijing",
        workspace_id="llm-x0jmmfp7f7vu9rc5",
    )
    updates: list[Answer] = []

    async def on_update(answer: Answer) -> None:
        updates.append(answer)

    answer, session_id = await stream_application_answer(
        settings,
        "sk-test",
        "Tell me about yourself.",
        on_update,
    )

    assert answer.english == (
        "I come from the knowledge base.\nI focus on reliability."
    )
    assert answer.chinese == ""
    assert session_id == ""
    assert len(updates) >= 2
    assert updates[0].english == "I come from the knowledge base."
    assert updates[-1].english == answer.english
    assert "检索结果不应展示" not in updates[-1].english
