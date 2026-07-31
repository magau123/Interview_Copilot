from __future__ import annotations

from interview_copilot.providers.qwen.client import QwenClient
from interview_copilot.storage.database import Database


class MemoryService:
    """Persists only user-confirmed facts; model summaries remain session-scoped."""

    def __init__(self, database: Database, qwen: QwenClient) -> None:
        self.database = database
        self.qwen = qwen

    async def add_confirmed_fact(self, text: str, source: str = "user") -> None:
        clean = text.strip()
        if not clean:
            return
        vector = (await self.qwen.embed([clean]))[0]
        self.database.add_memory(clean, source=source, confidence=1.0, vector=vector)

    async def summarize_session(self, session_id: int) -> str:
        transcript = self.database.recent_context(session_id, limit=50)
        if not transcript:
            return ""
        return await self.qwen.complete_text(
            "Summarize interview transcripts without inventing facts.",
            (
                "用中文简要总结本次面试的问题、候选人的回答和待改进项。"
                "不要把模型建议当作候选人的真实经历。\n\n" + transcript
            ),
            max_tokens=500,
        )
