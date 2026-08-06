from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable

import httpx

from interview_copilot.config import Settings
from interview_copilot.models import Answer

AnswerHandler = Callable[[Answer], Awaitable[None] | None]
logger = logging.getLogger(__name__)

_MARKER_LINE_RE = re.compile(
    r"(?im)^[\s>*#-]*[\[【(]?\s*(en|english|英文|英文回答|英文口述稿|zh|chinese|中文|中文回答|中文对照)"
    r"\s*[\]】)]?\s*[:：]?\s*$"
)
_INLINE_MARKER_RE = re.compile(
    r"(?i)\s*[\[【]\s*(en|english|zh|chinese|中文|英文)\s*[\]】]\s*"
)


def knowledge_chat_url(settings: Settings) -> str:
    workspace = settings.workspace_id.strip()
    if not workspace:
        raise ValueError("请先在设置中填写 Workspace ID")
    host = (
        f"{workspace}.cn-beijing.maas.aliyuncs.com"
        if settings.region == "cn-beijing"
        else f"{workspace}.ap-southeast-1.maas.aliyuncs.com"
    )
    return f"https://{host}/api/v2/apps/knowledge/chat"


def extract_english_answer(text: str) -> str:
    """Normalize model output into English-only answer text."""
    body = (text or "").strip()
    if not body:
        return ""
    # Drop leftover bilingual markers if the model still emits them.
    matches = list(_MARKER_LINE_RE.finditer(body))
    if matches:
        chunks: list[str] = []
        for index, match in enumerate(matches):
            label = match.group(1).strip().lower()
            if label in {"zh", "chinese", "中文", "中文回答", "中文对照"}:
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            chunk = body[match.end() : end].strip()
            if chunk:
                chunks.append(chunk)
        if chunks:
            body = "\n".join(chunks)
        else:
            # Only Chinese section markers — discard for English-only display.
            return ""
    body = _INLINE_MARKER_RE.sub("\n", body)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return "\n".join(lines)


def split_bilingual_answer(text: str) -> tuple[str, str]:
    """Compatibility helper: English-only answers live in the english field."""
    return extract_english_answer(text), ""


def completed_sentences(text: str, *, final: bool = False) -> str:
    """Return text safe to show while streaming.

    Finished lines / sentences appear immediately; a trailing unfinished fragment
    is held back until punctuation arrives or the stream ends.
    """
    cleaned = extract_english_answer(text)
    if not cleaned:
        return ""
    if final:
        return cleaned
    shown: list[str] = []
    lines = cleaned.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        is_last = index == len(lines) - 1
        if not is_last or re.search(r"[.!?]$", stripped):
            shown.append(stripped)
            continue
        # Mid-line: publish every finished sentence inside the current line.
        pieces = re.findall(r"[^.!?]*[.!?]+", stripped)
        shown.extend(piece.strip() for piece in pieces if piece.strip())
    return "\n".join(shown)


def classify_application_error(code: str, message: str) -> str:
    text = f"{code}: {message}".strip(": ")
    lowered = text.lower()
    if code in {"", "200", "success"} and not message:
        return "知识库应用调用失败：未知错误"
    if "app.accessdenied" in lowered or "app access denied" in lowered:
        return (
            "无权访问知识库应用（App.AccessDenied）。请确认："
            "1) 应用已在百炼控制台发布；"
            "2) API Key 与应用属于同一业务空间；"
            "3) 应用 ID 填写正确。"
        )
    if "invalidapikey" in lowered or "invalid api-key" in lowered:
        return "API Key 无效，请检查设置中的 DashScope API Key。"
    if "workspace.accessdenied" in lowered:
        return "无权访问业务空间。请使用该应用所在业务空间的 API Key，或检查 Workspace 权限。"
    return f"知识库应用调用失败：{text or '未知错误'}"


def _is_error_code(code: object) -> bool:
    if code is None or code == "":
        return False
    text = str(code).strip().lower()
    return text not in {"200", "success", "ok"}


def _assistant_delta(event: dict) -> str:
    """Extract assistant answer deltas; ignore tool/control RAG intermediate events."""
    output = event.get("output") or {}
    choices = output.get("choices") or []
    if not choices:
        return str(output.get("text") or "")
    message = choices[0].get("message") or {}
    role = str(message.get("role") or "")
    if role and role != "assistant":
        return ""
    extra = message.get("extra") or {}
    step = str(extra.get("step") or "")
    if step and step not in {"generating", "answer", ""}:
        return ""
    content = message.get("content")
    if content is None:
        return ""
    return str(content)


async def _publish(on_update: AnswerHandler, english: str) -> None:
    if not english.strip():
        return
    result = on_update(Answer(english=english.strip(), chinese=""))
    if inspect.isawaitable(result):
        await result


async def stream_application_answer(
    settings: Settings,
    api_key: str,
    question: str,
    on_update: AnswerHandler,
    *,
    session_id: str = "",
) -> tuple[Answer, str]:
    """Call workspace knowledge chat API with SSE streaming.

    Only the question text is sent. Prompting / style limits live in the Bailian app.
    """
    del session_id  # Official knowledge/chat uses messages; no session_id required.
    app_id = (settings.knowledge_app_id or "").strip()
    if not app_id:
        raise ValueError("请先在设置中填写阿里云知识库应用 ID（aid-...）")
    if not api_key:
        raise ValueError("请先配置 DashScope API Key")
    content = question.strip()
    if not content:
        raise ValueError("问题为空")

    url = knowledge_chat_url(settings)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "messages": [
                {"role": "user", "content": content},
            ]
        },
        "parameters": {
            "agent_options": {
                "agent_id": app_id,
            }
        },
        "stream": True,
    }

    raw = ""
    published = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                code, message = _extract_error(body)
                raise RuntimeError(classify_application_error(code, message))

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                code = event.get("code")
                if _is_error_code(code):
                    raise RuntimeError(
                        classify_application_error(
                            str(code or ""),
                            str(event.get("message") or ""),
                        )
                    )
                chunk = _assistant_delta(event)
                if not chunk:
                    continue
                raw += chunk
                visible = completed_sentences(raw, final=False)
                if visible and visible != published:
                    published = visible
                    await _publish(on_update, visible)

    english = extract_english_answer(raw)
    if english and english != published:
        await _publish(on_update, english)
    elif english and not published:
        await _publish(on_update, english)
    return Answer(english=english, chinese=""), ""


async def test_application_connection(settings: Settings, api_key: str) -> str:
    """One-shot connectivity check for the configured Bailian knowledge app."""
    chunks: list[str] = []

    async def _collect(answer: Answer) -> None:
        chunks.append(answer.english)

    answer, _session = await stream_application_answer(
        settings,
        api_key,
        question="Please introduce yourself briefly.",
        on_update=_collect,
    )
    preview = (answer.english or "".join(chunks)).strip()
    if not preview:
        raise RuntimeError("知识库应用返回为空，请检查应用是否已发布并挂载知识库。")
    return (
        f"知识库应用连通成功。\n"
        f"应用 ID：{settings.knowledge_app_id}\n"
        f"接口：{knowledge_chat_url(settings)}\n"
        f"预览：{preview[:180]}"
    )


def _extract_error(body: str) -> tuple[str, str]:
    text = body.strip()
    if "data:" in text:
        for line in reversed(text.splitlines()):
            if line.startswith("data:"):
                text = line[5:].strip()
                break
    try:
        payload = json.loads(text)
        return str(payload.get("code") or ""), str(payload.get("message") or text)
    except json.JSONDecodeError:
        return "", text
