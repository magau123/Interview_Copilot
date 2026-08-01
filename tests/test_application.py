from __future__ import annotations

import json

import httpx
import pytest

from interview_copilot.config import Settings
from interview_copilot.models import Answer
from interview_copilot.providers.qwen import application as application_mod
from interview_copilot.providers.qwen.application import (
    _assistant_delta,
    build_interview_prompt,
    classify_application_error,
    knowledge_chat_url,
    split_bilingual_answer,
    stream_application_answer,
)


def test_build_interview_prompt_asks_for_bilingual_sections() -> None:
    prompt = build_interview_prompt(
        "Tell me about yourself.",
        "请介绍一下你自己。",
        "interviewer: Hello",
    )
    assert "请介绍一下你自己。" in prompt
    assert "[EN]" in prompt
    assert "[ZH]" in prompt


def test_split_bilingual_answer_reads_marked_sections() -> None:
    english, chinese = split_bilingual_answer(
        "[EN]\nI led the billing migration.\nIt cut latency in half.\n"
        "[ZH]\n我主导了计费迁移。\n延迟降低了一半。"
    )
    assert english == "I led the billing migration.\nIt cut latency in half."
    assert chinese == "我主导了计费迁移。\n延迟降低了一半。"


def test_split_bilingual_answer_accepts_localised_headings() -> None:
    english, chinese = split_bilingual_answer("英文：\nHello there.\n中文对照\n你好。")
    assert english == "Hello there."
    assert chinese == "你好。"


def test_split_bilingual_answer_falls_back_to_script_detection() -> None:
    assert split_bilingual_answer("我来自知识库。") == ("", "我来自知识库。")
    assert split_bilingual_answer("I come from the knowledge base.") == (
        "I come from the knowledge base.",
        "",
    )
    assert split_bilingual_answer("   ") == ("", "")


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
                        "content": "我是数据分析师。",
                        "extra": {"step": "generating"},
                    }
                }
            ]
        },
    }
    assert _assistant_delta(tool_event) == ""
    assert _assistant_delta(assistant_event) == "我是数据分析师。"


@pytest.mark.asyncio
async def test_stream_application_answer_parses_sse_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "code": "200",
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "control",
                            "content": "",
                            "extra": {"step": "tool_calling"},
                            "tool_calls": [{"id": "1"}],
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
                            "content": "[EN]\nI come from the knowledge base.\n[ZH]\n我",
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
                            "content": "来自知识库。",
                            "extra": {"step": "generating"},
                        }
                    }
                ]
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "https://llm-x0jmmfp7f7vu9rc5.cn-beijing.maas.aliyuncs.com"
            "/api/v2/apps/knowledge/chat"
        )
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        assert body["parameters"]["agent_options"]["agent_id"] == (
            "aid-433c2467738a4ae1948488f117508609"
        )
        assert body["input"]["messages"][0]["role"] == "user"
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
        "请介绍一下你自己。",
        "",
        on_update,
    )

    assert answer.english == "I come from the knowledge base."
    assert answer.chinese == "我来自知识库。"
    assert session_id == ""
    assert len(updates) == 1
    assert updates[0].chinese == "我来自知识库。"
    assert "检索结果不应展示" not in updates[0].chinese
