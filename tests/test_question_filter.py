from __future__ import annotations

import pytest

from interview_copilot.conversation.question_filter import (
    is_answerable_question,
    is_continuation,
    is_similar,
    merge_question,
)


@pytest.mark.parametrize(
    "text",
    [
        "Tell me about a time you disagreed with your manager.",
        "What is your biggest strength?",
        "Why?",
        "Can you walk me through the architecture",
        "How do you handle conflict",
        "描述一次你解决冲突的经历，并说明你的角色和结果。",
    ],
)
def test_real_questions_are_answerable(text: str) -> None:
    assert is_answerable_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Okay",
        "okay okay",
        "Got it.",
        "Mm-hmm",
        "Yeah",
        "Thank you very much",
        "Sounds good",
        "...",
        "嗯",
        "the the",
    ],
)
def test_backchannel_and_noise_are_ignored(text: str) -> None:
    assert not is_answerable_question(text)


def test_continuation_covers_sentence_fragments_only() -> None:
    assert is_continuation("and your role in it")
    assert not is_continuation("okay")
    assert not is_continuation("What did you learn?")


def test_similar_detects_refinalised_sentence() -> None:
    assert is_similar(
        "Tell me about yourself please",
        "Tell me about yourself, please.",
    )
    assert not is_similar(
        "Tell me about yourself",
        "What is your biggest weakness",
    )


def test_merge_question_joins_without_duplicating() -> None:
    assert merge_question("Tell me about a project", "and your role in it") == (
        "Tell me about a project and your role in it"
    )
    assert merge_question("Tell me about a project", "about a project") == (
        "Tell me about a project"
    )
    assert merge_question("", "and your role") == "and your role"
