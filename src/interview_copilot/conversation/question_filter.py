"""Decide which interviewer utterances deserve a knowledge-base answer.

The realtime service finalises every detected utterance, including throat
clearing, backchannel ("okay", "got it") and half sentences produced when the
interviewer pauses mid-question. Answering those wastes a request and, worse,
replaces the answer the candidate is currently reading.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# Long enough to carry a question even without a question mark or cue word.
LONG_UTTERANCE_WORDS = 8
# Short utterances only pass when they start with an interrogative word.
MIN_LEAD_WORDS = 3

_WORD_RE = re.compile(r"[a-z0-9']+")
# Latin words plus single CJK characters, so length checks work in both scripts.
_TOKEN_RE = re.compile(r"[a-z0-9']+|[\u4e00-\u9fff]")
_NON_WORD_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")

FILLER_PHRASES = frozenset(
    {
        "a",
        "ah",
        "alright",
        "all right",
        "amazing",
        "and",
        "aha",
        "awesome",
        "bye",
        "cool",
        "correct",
        "exactly",
        "excellent",
        "fine",
        "good",
        "good good",
        "got it",
        "great",
        "great thanks",
        "hello",
        "hey",
        "hi",
        "hmm",
        "huh",
        "i see",
        "interesting",
        "makes sense",
        "mhm",
        "mm",
        "mm hmm",
        "next",
        "nice",
        "no",
        "no problem",
        "no worries",
        "of course",
        "oh",
        "ok",
        "okay",
        "okay okay",
        "okay thanks",
        "perfect",
        "please",
        "right",
        "sorry",
        "sounds good",
        "sure",
        "thank you",
        "thank you very much",
        "thanks",
        "uh",
        "uh huh",
        "um",
        "understood",
        "very good",
        "well",
        "wow",
        "yeah",
        "yep",
        "yes",
        "you know",
    }
)

CJK_FILLERS = frozenset(
    {
        "不错",
        "可以",
        "哦",
        "好",
        "好吧",
        "好的",
        "对",
        "对的",
        "明白",
        "明白了",
        "是",
        "是的",
        "没问题",
        "然后",
        "行",
        "谢谢",
        "嗯",
        "嗯嗯",
        "嗯好",
    }
)

# First word of a short question, e.g. "How do you handle that".
QUESTION_LEADS = frozenset(
    {
        "am",
        "any",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "how",
        "is",
        "may",
        "shall",
        "should",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whose",
        "why",
        "will",
        "would",
    }
)

# Imperative openings interviewers use instead of a literal question.
CUE_PHRASES = (
    "can you",
    "could you",
    "describe",
    "elaborate",
    "explain",
    "give me",
    "how about",
    "i would like to know",
    "introduce",
    "let's talk",
    "lets talk",
    "share",
    "suppose",
    "talk about",
    "tell me",
    "walk me",
    "what about",
    "would you",
)


def words(text: str) -> list[str]:
    """Latin word tokens; empty for pure noise, punctuation or Chinese text."""
    return _WORD_RE.findall(text.lower())


def tokens(text: str) -> list[str]:
    """Latin words and CJK characters, used for utterance length."""
    return _TOKEN_RE.findall(text.lower())


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def is_filler(text: str) -> bool:
    """True when the whole utterance is backchannel or greeting noise."""
    compact = _NON_WORD_RE.sub("", text.lower())
    if not compact:
        return True
    return " ".join(words(text)) in FILLER_PHRASES or compact in CJK_FILLERS


def is_answerable_question(text: str) -> bool:
    all_tokens = tokens(text)
    if not all_tokens or is_filler(text):
        return False
    if len(all_tokens) >= LONG_UTTERANCE_WORDS:
        return True
    if "?" in text or "？" in text:
        return True
    latin = words(text)
    normalized = " ".join(latin)
    if any(normalized.startswith(cue) for cue in CUE_PHRASES):
        return True
    return len(latin) >= MIN_LEAD_WORDS and latin[0] in QUESTION_LEADS


def is_continuation(text: str) -> bool:
    """A fragment that likely belongs to the question the interviewer just started."""
    if is_answerable_question(text):
        return False
    return len(tokens(text)) >= 2


def is_similar(first: str, second: str, threshold: float = 0.9) -> bool:
    """Guard against the service finalising the same sentence twice."""
    left, right = normalize(first), normalize(second)
    if not left or not right:
        return False
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= threshold


def merge_question(previous: str, addition: str) -> str:
    """Join a split question without duplicating the overlapping part."""
    head, tail = previous.strip(), addition.strip()
    if not head:
        return tail
    if not tail or tail.lower() in head.lower():
        return head
    return f"{head} {tail}"
