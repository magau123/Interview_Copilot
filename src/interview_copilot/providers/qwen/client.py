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
        prompt = f"""Interview question (English):
{question}

Interview question (Chinese translation, for context only):
{chinese_question or "(none)"}

Recent dialogue context:
{recent_context or "(none)"}

Knowledge-base evidence:
{evidence or "(none)"}

Write an English interview answer I can read aloud.
Output English only. No Chinese. No [EN]/[ZH] markers.
Short spoken sentences, one per line, at most 18 words each.
Use knowledge-base facts only — do not invent personal details.
"""
        stream = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an interview assistant. Using the question and "
                        "knowledge-base evidence, output English-only spoken answers. "
                        "Do not invent personal facts."
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
    from interview_copilot.providers.qwen.application import extract_english_answer

    return Answer(english=extract_english_answer(raw), chinese="", sources=sources)
