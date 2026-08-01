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


def build_interview_prompt(question: str, translation: str = "", recent_context: str = "") -> str:
    chinese_question = (translation or "").strip()
    return f"""面试官问题（英文原文）:
{question.strip()}

面试官问题（中文翻译）:
{chinese_question or "（暂无）"}

近期对话上下文:
{recent_context.strip() or "（无）"}

请结合知识库输出两节内容，严格使用下面的分节标记，不要输出其他标题或解释：

[EN]
（面试时可以直接照读的英文回答；口语化，短句，每句不超过 18 个词，一句一行）
[ZH]
（同一份回答的中文，便于我快速确认意思；一句一行，与英文顺序一致）

要求：
- 两节内容表达同一份回答，不要在英文里夹中文，也不要在中文里夹英文
- 个人经历、项目、指标只能来自知识库，禁止编造
- 若知识库没有相关材料，明确说明依据不足，并给出可信的通用回答
- 按问题复杂度控制篇幅：简单题 1-3 句；经历/行为题约 45-60 秒口述长度
"""


_SECTION_LINE_RE = re.compile(
    r"(?im)^[\s>*#-]*[\[【(]?\s*(en|english|英文|英文回答|英文口述稿|zh|chinese|中文|中文回答|中文对照)"
    r"\s*[\]】)]?\s*[:：]?\s*$"
)
_SECTION_INLINE_RE = re.compile(r"(?i)[\[【]\s*(en|english|zh|chinese)\s*[\]】]")
_ENGLISH_LABELS = {"en", "english", "英文", "英文回答", "英文口述稿"}
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _looks_chinese(text: str) -> bool:
    stripped = "".join(text.split())
    if not stripped:
        return False
    return len(_CJK_RE.findall(stripped)) / len(stripped) > 0.15


def split_bilingual_answer(text: str) -> tuple[str, str]:
    """Split a '[EN] … [ZH] …' payload, tolerating missing or renamed markers."""
    body = text.strip()
    if not body:
        return "", ""
    matches = list(_SECTION_LINE_RE.finditer(body)) or list(_SECTION_INLINE_RE.finditer(body))
    sections: dict[str, list[str]] = {"en": [], "zh": []}
    if matches:
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            label = match.group(1).strip().lower()
            key = "en" if label in _ENGLISH_LABELS else "zh"
            sections[key].append(body[match.end() : end].strip())
        english = "\n".join(part for part in sections["en"] if part).strip()
        chinese = "\n".join(part for part in sections["zh"] if part).strip()
        if english or chinese:
            return english, chinese
    # No usable markers: keep the text in whichever field matches its script.
    return ("", body) if _looks_chinese(body) else (body, "")


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
        # Some payloads may still use flat text.
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


async def stream_application_answer(
    settings: Settings,
    api_key: str,
    question: str,
    translation: str,
    recent_context: str,
    on_update: AnswerHandler,
    *,
    session_id: str = "",
) -> tuple[Answer, str]:
    """Call workspace knowledge chat API with SSE streaming."""
    del session_id  # Official knowledge/chat uses messages; no session_id required.
    app_id = (settings.knowledge_app_id or "").strip()
    if not app_id:
        raise ValueError("请先在设置中填写阿里云知识库应用 ID（aid-...）")
    if not api_key:
        raise ValueError("请先配置 DashScope API Key")

    prompt = build_interview_prompt(question, translation, recent_context)
    url = knowledge_chat_url(settings)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "messages": [
                {"role": "user", "content": prompt},
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
                # Accumulate only; UI shows the complete answer once at the end.
                raw += chunk

    english, chinese = split_bilingual_answer(raw)
    answer = Answer(english=english, chinese=chinese)
    if english or chinese:
        result = on_update(answer)
        if inspect.isawaitable(result):
            await result
    return answer, ""


async def test_application_connection(settings: Settings, api_key: str) -> str:
    """One-shot connectivity check for the configured Bailian knowledge app."""
    chunks: list[str] = []

    async def _collect(answer: Answer) -> None:
        chunks.append(answer.chinese or answer.english)

    answer, _session = await stream_application_answer(
        settings,
        api_key,
        question="Please introduce yourself briefly for an interview assistant connectivity test.",
        translation="请用一两句话做自我介绍，用于面试助手连通性测试。",
        recent_context="",
        on_update=_collect,
    )
    preview = (answer.chinese or answer.english or "".join(chunks)).strip()
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
        # Prefer the last SSE data payload when present.
        for line in reversed(text.splitlines()):
            if line.startswith("data:"):
                text = line[5:].strip()
                break
    try:
        payload = json.loads(text)
        return str(payload.get("code") or ""), str(payload.get("message") or text)
    except json.JSONDecodeError:
        return "", text
