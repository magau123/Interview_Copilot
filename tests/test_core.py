from __future__ import annotations

import numpy as np

from interview_copilot.audio.capture import _to_pcm16_mono
from interview_copilot.config import Settings
from interview_copilot.conversation.translation_ledger import TranslationLedger
from interview_copilot.knowledge.service import split_text
from interview_copilot.models import Source
from interview_copilot.providers.qwen.client import _parse_answer
from interview_copilot.providers.qwen.realtime import QwenRealtimeSpeech


def test_realtime_urls_follow_region() -> None:
    china = Settings(workspace_id="workspace", region="cn-beijing")
    intl = Settings(workspace_id="workspace", region="ap-southeast-1")
    assert "workspace.cn-beijing" in china.realtime_url
    assert "workspace.ap-southeast-1" in intl.realtime_url
    assert china.realtime_model in china.realtime_url
    assert china.realtime_urls[0].startswith("wss://workspace.cn-beijing")
    assert "dashscope.aliyuncs.com" in china.realtime_urls[1]
    assert "dashscope-intl.aliyuncs.com" in intl.realtime_urls[1]


def test_split_text_respects_size_and_overlap() -> None:
    chunks = split_text(("A" * 700) + "\n\n" + ("B" * 700), max_chars=800, overlap_chars=80)
    assert len(chunks) == 2
    assert all(len(chunk) <= 900 for chunk in chunks)
    assert chunks[1].startswith("A" * 80)


def test_answer_parser_handles_english_only_output() -> None:
    source = Source(1, "resume.pdf", "Built an API", 0.9)
    answer = _parse_answer("I built the service and kept it reliable.", [source])
    assert answer.english == "I built the service and kept it reliable."
    assert answer.chinese == ""
    assert answer.sources == [source]


def test_answer_parser_tolerates_legacy_markers() -> None:
    source = Source(1, "resume.pdf", "Built an API", 0.9)
    answer = _parse_answer(
        "[EN]\nI built the service.\n[ZH]\n我构建了该服务。",
        [source],
    )
    assert answer.english == "I built the service."
    assert answer.chinese == ""
    assert answer.sources == [source]


def test_audio_downmix_and_resample() -> None:
    stereo = np.column_stack(
        (np.arange(4800, dtype=np.int16), np.arange(4800, dtype=np.int16))
    ).ravel()
    output = np.frombuffer(_to_pcm16_mono(stereo.tobytes(), 2, 48_000), dtype=np.int16)
    assert 1590 <= len(output) <= 1610


def test_ledger_pairs_late_translation_with_previous_english() -> None:
    ledger = TranslationLedger()
    ledger.on_source_partial("Tell me about")
    ledger.on_source_final("Tell me about yourself.")
    ledger.on_translation_partial("请介绍")
    # Next English starts before Chinese finishes.
    ledger.on_source_partial("What is your")
    ledger.on_source_final("What is your strength?")
    # Late final translation belongs to the first sentence.
    ledger.on_translation_final("请介绍一下你自己。")
    ledger.on_translation_final("你的优势是什么？")

    assert ledger.history_pairs() == [
        ("Tell me about yourself.", "请介绍一下你自己。"),
        ("What is your strength?", "你的优势是什么？"),
    ]
    assert ledger.latest() == ("What is your strength?", "你的优势是什么？")


def test_ledger_updates_incomplete_history_when_translation_arrives() -> None:
    ledger = TranslationLedger()
    ledger.on_source_final("How do you handle conflict?")
    ledger.on_source_partial("Give an example.")
    assert ledger.history_pairs() == [("How do you handle conflict?", "")]
    ledger.on_translation_final("你如何处理冲突？")
    assert ledger.history_pairs() == [("How do you handle conflict?", "你如何处理冲突？")]
    assert ledger.latest() == ("Give an example.", "")


def test_completed_history_newest_first_for_display() -> None:
    ledger = TranslationLedger()
    ledger.on_source_final("First question.")
    ledger.on_translation_final("第一个问题。")
    ledger.on_source_final("Second question.")
    ledger.on_translation_final("第二个问题。")
    completed = [
        (english, chinese)
        for english, chinese in ledger.history_pairs()
        if english.strip() and chinese.strip()
    ]
    assert list(reversed(completed)) == [
        ("Second question.", "第二个问题。"),
        ("First question.", "第一个问题。"),
    ]


def test_translation_partial_supports_deltas_and_cumulative() -> None:
    client = QwenRealtimeSpeech(Settings(), "key", lambda _event: None)
    assert client._merge_translation_partial("你", "好") == "你好"
    assert client._merge_translation_partial("你好", "吗") == "你好吗"
    client._translation_confirmed = ""
    assert client._merge_translation_partial("你", "") == "你"
    assert client._merge_translation_partial("好吗", "") == "你好吗"
