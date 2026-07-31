from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from interview_copilot.models import Source
from interview_copilot.providers.qwen.client import QwenClient
from interview_copilot.storage.database import Database

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


class KnowledgeService:
    def __init__(self, database: Database, qwen: QwenClient) -> None:
        self.database = database
        self.qwen = qwen

    async def import_file(self, path: Path) -> int:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"不支持的文件类型：{path.suffix}")
        raw = await asyncio.to_thread(path.read_bytes)
        digest = hashlib.sha256(raw).hexdigest()
        text = await asyncio.to_thread(_extract_text, path)
        chunks = split_text(text)
        if not chunks:
            raise ValueError("文档中没有可索引的文本")
        vectors = await self.qwen.embed(chunks)
        document_id = self.database.upsert_document(path.name, str(path.resolve()), digest)
        self.database.replace_chunks(document_id, chunks, vectors)
        return len(chunks)

    async def search(self, query: str, limit: int = 5) -> list[Source]:
        if not query.strip():
            return []
        vector = (await self.qwen.embed([query]))[0]
        return self.database.search(vector, limit=limit)


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        return "\n\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    return path.read_text(encoding="utf-8", errors="replace")


def split_text(text: str, max_chars: int = 1200, overlap_chars: int = 120) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            pieces = [
                paragraph[index : index + max_chars]
                for index in range(0, len(paragraph), max_chars - overlap_chars)
            ]
        else:
            pieces = [paragraph]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            prefix = current[-overlap_chars:] if current else ""
            current = f"{prefix}\n{piece}".strip()
    if current:
        chunks.append(current)
    return chunks
