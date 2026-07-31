from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence

from openai import AsyncOpenAI

from interview_copilot.config import Settings
from interview_copilot.models import Answer, Source

AnswerHandler = Callable[[Answer], Awaitable[None] | None]


class QwenClient:
    def __init__(self, settings: Settings, api_key: str) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=api_key, base_url=settings.compatible_base_url)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 10):
            response = await self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=list(texts[start : start + 10]),
                dimensions=1024,
            )
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return vectors

    async def complete_text(self, system: str, prompt: str, max_tokens: int = 500) -> str:
        response = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_answer(
        self,
        question: str,
        translation: str,
        sources: list[Source],
        recent_context: str,
        on_update: AnswerHandler,
    ) -> Answer:
        evidence = "\n\n".join(
            f"[{source.document_name}]\n{source.text}" for source in sources
        )
        chinese_question = (translation or "").strip()
        prompt = f"""面试官问题（英文原文）:
{question}

面试官问题（中文翻译）:
{chinese_question or "（暂无）"}

近期对话上下文:
{recent_context or "（无）"}

知识库检索结果:
{evidence or "（未检索到相关资料）"}

请只输出一段适合面试作答的中文回复，不要输出英文，不要使用 [EN]/[ZH] 等标记。

长度要求:
- 问候、确认、定义、简单事实题：用 1-3 句简短回答
- 行为、项目、经历、动机、优劣势、冲突、领导力、情景题：给出约 45-60 秒口述长度的中文回答，尽量结合知识库中的具体经历；适合时用精简 STAR，但不要标注段落标题
- 技术解释题：给出约 30-45 秒的聚焦回答，除非问题明确要求极短定义

约束:
- 个人经历、公司、时间、指标、项目只能来自知识库检索结果，禁止编造
- 若知识库没有相关材料，明确说明依据不足，并给出可信的通用回答，不要编造细节
- 不要开场寒暄、不要重复收尾、不要多余铺垫
"""
        stream = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是面试助手。根据面试官问题和知识库检索结果，只输出中文回答。"
                        "回答要贴合问题，依据证据，禁止编造个人事实。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=self.settings.answer_max_tokens,
            stream=True,
            extra_body={"enable_thinking": False},
        )
        raw = ""
        answer = Answer(sources=sources)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            raw += delta
            answer = _parse_answer(raw, sources)
            result = on_update(answer)
            if inspect.isawaitable(result):
                await result
        return answer


def _parse_answer(raw: str, sources: list[Source]) -> Answer:
    text = (raw or "").strip()
    chinese = text
    # Tolerate legacy bilingual markers if the model still emits them.
    if "[ZH]" in text:
        chinese = text.split("[ZH]", 1)[1].strip()
    elif "[EN]" in text:
        chinese = text.split("[EN]", 1)[0].strip()
    return Answer(english="", chinese=chinese, sources=sources)
