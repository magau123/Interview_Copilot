from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Turn:
    english: str = ""
    chinese: str = ""
    english_final: bool = False
    chinese_final: bool = False
    history_index: int | None = None


@dataclass
class TranslationLedger:
    """Pairs English finals with later Chinese finals so history stays complete.

    LiveTranslate often starts the next English utterance before the previous
    Chinese translation finishes. Commit EN early, then fill/correct ZH in place.
    """

    history: list[list[str]] = field(default_factory=list)
    current: _Turn = field(default_factory=_Turn)
    _waiting: list[_Turn] = field(default_factory=list)
    max_history: int = 40

    def reset(self) -> None:
        self.history.clear()
        self.current = _Turn()
        self._waiting.clear()

    def on_source_partial(self, text: str) -> None:
        if self.current.english_final:
            self._park_current()
        self.current.english = text
        self.current.english_final = False

    def on_source_final(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self.current.english_final:
            same = cleaned == self.current.english.strip()
            if same and self.current.chinese_final:
                # Duplicate final for an already completed turn.
                return
            if (not same) or self.current.chinese_final:
                self._park_current()
        self.current.english = cleaned
        self.current.english_final = True
        if self.current.chinese_final:
            self._ensure_history_entry(self.current)

    def on_translation_partial(self, text: str) -> None:
        target = self._translation_target()
        target.chinese = text
        if target.history_index is not None:
            self._write_history(target)

    def on_translation_final(self, text: str) -> None:
        target = self._translation_target()
        # Prefer the longer of final event text vs best partial we already have.
        final = (text or "").strip()
        if len(final) < len(target.chinese.strip()):
            final = target.chinese.strip()
        target.chinese = final
        target.chinese_final = True
        self._ensure_history_entry(target)
        if target is not self.current and target in self._waiting:
            self._waiting.remove(target)

    def latest(self) -> tuple[str, str]:
        return self.current.english, self.current.chinese

    def history_pairs(self) -> list[tuple[str, str]]:
        return [(item[0], item[1]) for item in self.history]

    def _translation_target(self) -> _Turn:
        for turn in self._waiting:
            if not turn.chinese_final:
                return turn
        return self.current

    def _park_current(self) -> None:
        turn = self.current
        has_text = bool(turn.english.strip() or turn.chinese.strip())
        if has_text:
            self._ensure_history_entry(turn)
            if turn.english_final and not turn.chinese_final:
                self._waiting.append(turn)
        self.current = _Turn()

    def _ensure_history_entry(self, turn: _Turn) -> None:
        english = turn.english.strip()
        chinese = turn.chinese.strip()
        if not english and not chinese:
            return
        if turn.history_index is None:
            self.history.append([english, chinese])
            turn.history_index = len(self.history) - 1
            if len(self.history) > self.max_history:
                overflow = len(self.history) - self.max_history
                del self.history[:overflow]
                self._reindex_after_trim(overflow)
        else:
            self._write_history(turn)

    def _write_history(self, turn: _Turn) -> None:
        if turn.history_index is None:
            return
        if 0 <= turn.history_index < len(self.history):
            self.history[turn.history_index][0] = turn.english.strip()
            self.history[turn.history_index][1] = turn.chinese.strip()

    def _reindex_after_trim(self, overflow: int) -> None:
        for turn in (*self._waiting, self.current):
            if turn.history_index is None:
                continue
            turn.history_index -= overflow
            if turn.history_index < 0:
                turn.history_index = None
