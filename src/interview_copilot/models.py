from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import monotonic


class SpeechEventType(StrEnum):
    SOURCE_PARTIAL = "source_partial"
    SOURCE_FINAL = "source_final"
    TRANSLATION_PARTIAL = "translation_partial"
    TRANSLATION_FINAL = "translation_final"
    # A question passed the interjection filter and an answer is being generated.
    ANSWER_PENDING = "answer_pending"
    ERROR = "error"
    STATUS = "status"


@dataclass(slots=True)
class SpeechEvent:
    kind: SpeechEventType
    text: str
    turn_id: str = ""
    received_at: float = field(default_factory=monotonic)


@dataclass(slots=True)
class Source:
    chunk_id: int
    document_name: str
    text: str
    score: float


@dataclass(slots=True)
class Answer:
    english: str = ""
    chinese: str = ""
    sources: list[Source] = field(default_factory=list)


@dataclass(slots=True)
class Turn:
    id: str
    source_text: str = ""
    translation: str = ""
    started_at: float = field(default_factory=monotonic)
