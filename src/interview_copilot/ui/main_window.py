from __future__ import annotations

import asyncio
import html

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qasync import asyncSlot

from interview_copilot.audio.capture import AudioDevice, list_audio_devices
from interview_copilot.audio.voice_activity import VoiceActivityMonitor
from interview_copilot.config import SettingsStore
from interview_copilot.conversation.orchestrator import ConversationOrchestrator
from interview_copilot.conversation.translation_ledger import TranslationLedger
from interview_copilot.models import Answer, SpeechEvent, SpeechEventType
from interview_copilot.storage.database import Database
from interview_copilot.ui.overlay import OverlayWindow
from interview_copilot.ui.settings_dialog import SettingsDialog
from interview_copilot.ui.teleprompter import TeleprompterView

# Solid, high-contrast colors: see-through comes from window opacity, so the
# palette itself must stay crisp to survive the fade.
STYLESHEET = """
QMainWindow, QDialog, QWidget {
    background: #f2f5f7;
    color: #000000;
    font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei";
}
QWidget#overlayRoot {
    background: #ffffff;
    border: 1px solid #46525d;
}
QWidget#overlayRoot[alert="true"] {
    border: 2px solid #b8541b;
}
QLabel#panelHint {
    color: #2b343c;
    font-size: 12px;
}
QTextEdit#panelCard {
    background: transparent;
    border: none;
    padding: 0px;
}
QLineEdit, QComboBox, QSpinBox {
    background: #ffffff;
    border: 1px solid #5a6976;
    border-radius: 8px;
    padding: 6px 9px;
    color: #000000;
    selection-background-color: #87b7d3;
}
QPushButton {
    background: #e3e8ec;
    border: 1px solid #5a6976;
    border-radius: 8px;
    padding: 6px 12px;
    color: #000000;
}
QPushButton:hover { background: #ffffff; }
QPushButton:disabled { color: #77828c; background: #eef1f3; }
QPushButton#primary {
    background: #1f5f85;
    color: #ffffff;
    font-weight: 600;
    min-width: 84px;
}
QPushButton#primary:hover { background: #17506f; }
QPushButton#accent {
    background: #b8541b;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#accent:hover { background: #9c4715; }
QPushButton#accent:disabled { background: #e0d5cd; color: #8a7b70; }
QCheckBox { spacing: 6px; color: #000000; }
QFrame#readingGuide { background: #7fa8c2; }
QProgressBar#levelBar {
    background: #e3e8ec;
    border: 1px solid #5a6976;
    border-radius: 4px;
    padding: 0px;
}
QProgressBar#levelBar::chunk {
    background: #2f7d4f;
    border-radius: 3px;
}
QStatusBar {
    background: #e8edf0;
    color: #1b2228;
}
"""


class MainWindow(QMainWindow):
    """Control console; the interviewer text and the answer script float free."""

    # Two finished turns plus the live one = three visible conversation turns.
    CONTEXT_SENTENCES = 2
    CONTEXT_CHAR_LIMIT = 150
    # Above this the text becomes too faint to read during an interview.
    MAX_TRANSPARENCY = 85

    def __init__(self, store: SettingsStore) -> None:
        super().__init__()
        self.store = store
        self.settings = store.load()
        self.database = Database(self.settings.database_path)
        self.orchestrator: ConversationOrchestrator | None = None
        self.loopbacks: list[AudioDevice] = []
        self.microphones: list[AudioDevice] = []
        self._question_en = ""
        self._question_zh = ""
        self._script = Answer()
        self._pending_answer: Answer | None = None
        self._answer_pending = False
        self._rendered_script_html = ""
        self._ledger = TranslationLedger()
        self._voice = VoiceActivityMonitor(self.settings.teleprompter_threshold)
        # Created before the widgets: restoring a saved microphone can start it.
        self._voice_timer = QTimer(self)
        self._voice_timer.setInterval(50)
        self._voice_timer.timeout.connect(self._poll_voice)
        self.setWindowTitle("面试助手 · 控制台")
        self._migrate_settings()
        self._build_overlays()
        self._build_console()
        self._apply_style()
        self._render_translation()
        self._load_settings()
        self.refresh_devices()
        self._apply_window_transparency()
        # Devices are known only now, so honour the saved follow-voice preference.
        self._on_follow_toggled(self.follow_toggle.isChecked())

    def show(self) -> None:
        super().show()
        self.translation_overlay.show()
        self.answer_overlay.setVisible(self.answer_toggle.isChecked())

    def _build_overlays(self) -> None:
        self.translation_view = QTextEdit()
        self.translation_view.setObjectName("panelCard")
        self.translation_view.setReadOnly(True)
        # Fixed reading window: latest line stays pinned at the bottom and older
        # turns roll off the top instead of adding a scrollbar.
        self.translation_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.translation_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.answer_view = TeleprompterView()
        self.answer_view.setObjectName("panelCard")
        self.answer_view.set_guide_visible(True)
        self.answer_view.runningChanged.connect(self._on_scroll_running_changed)

        self.translation_overlay = OverlayWindow(
            self.translation_view, tooltip="面试官内容：按住可拖动，右下角可缩放"
        )
        self.translation_overlay.setWindowTitle("面试官")
        self.answer_overlay = OverlayWindow(
            self.answer_view, tooltip="提词器：按住可拖动，右键有更多操作"
        )
        self.answer_overlay.setWindowTitle("提词器")
        for overlay in (self.translation_overlay, self.answer_overlay):
            overlay.moved.connect(self._persist_overlay_geometry)
            overlay.contextMenuRequested.connect(self._show_overlay_menu)
        self._place_overlays()

    def _place_overlays(self) -> None:
        restored = self.translation_overlay.apply_geometry(self.settings.translation_geometry)
        restored &= self.answer_overlay.apply_geometry(self.settings.answer_geometry)
        if restored:
            return
        screen = QApplication.primaryScreen()
        area = screen.availableGeometry() if screen else None
        left = area.x() + 40 if area else 60
        top = area.y() + 40 if area else 60
        width = min(680, int(area.width() * 0.45)) if area else 680
        self.translation_overlay.setGeometry(left, top, width, 240)
        self.answer_overlay.setGeometry(left, top + 268, width, 430)

    def _build_console(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 10, 12, 6)
        layout.setSpacing(8)

        devices = QHBoxLayout()
        devices.setSpacing(8)
        self.loopback_combo = QComboBox()
        self.microphone_combo = QComboBox()
        self.microphone_combo.setToolTip("只在本机做音量检测，用于跟读滚动，不会上传或识别我的语音")
        self.microphone_combo.currentIndexChanged.connect(self._on_microphone_changed)
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button = QPushButton("开始")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle_session)
        devices.addWidget(QLabel("会议声音"))
        devices.addWidget(self.loopback_combo, 3)
        devices.addWidget(QLabel("我的麦克风"))
        devices.addWidget(self.microphone_combo, 2)
        devices.addWidget(self.refresh_button)
        devices.addWidget(self.start_button)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.answer_toggle = QCheckBox("回答建议")
        self.answer_toggle.setToolTip("开启后，问题结束后会从阿里云知识库应用拉取完整回答")
        self.answer_toggle.toggled.connect(self._on_answer_toggled)
        self.follow_toggle = QCheckBox("跟读滚动")
        self.follow_toggle.setToolTip("我一开口就滚动，停下就暂停（只用本地音量判断）")
        self.follow_toggle.toggled.connect(self._on_follow_toggled)
        self.pin_toggle = QCheckBox("置顶")
        self.pin_toggle.setToolTip("让两个显示框浮在 Teams 上方；关闭后会被其他窗口盖住")
        self.pin_toggle.toggled.connect(self._on_pin_toggled)
        self.switch_button = QPushButton("切换新回答（F4）")
        self.switch_button.setObjectName("accent")
        self.switch_button.setShortcut("F4")
        self.switch_button.setToolTip("面试官问了新问题；点击后才替换正在读的稿子")
        self.switch_button.clicked.connect(self.apply_pending_answer)
        self.switch_button.setEnabled(False)
        self.scroll_button = QPushButton("手动滚动（F2）")
        self.scroll_button.setShortcut("F2")
        self.scroll_button.clicked.connect(self._toggle_scroll)
        self.rewind_button = QPushButton("回到开头")
        self.rewind_button.clicked.connect(self._rewind_script)
        self.regenerate_button = QPushButton("重新生成")
        self.regenerate_button.clicked.connect(self.regenerate)
        self.level_bar = QProgressBar()
        self.level_bar.setObjectName("levelBar")
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(8)
        self.level_bar.setToolTip("麦克风音量：说话时应明显超过一半")
        self.transparency_label = QLabel("透明 45%")
        self.transparency_label.setObjectName("panelHint")
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setRange(0, self.MAX_TRANSPARENCY)
        self.transparency_slider.setFixedWidth(110)
        self.transparency_slider.setToolTip("两个显示框的整窗透明度，可透视到后面的 Teams")
        self.transparency_slider.valueChanged.connect(self._on_transparency_changed)
        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self.open_settings)
        actions.addWidget(self.answer_toggle)
        actions.addWidget(self.follow_toggle)
        actions.addWidget(self.pin_toggle)
        actions.addWidget(self.switch_button)
        actions.addWidget(self.scroll_button)
        actions.addWidget(self.rewind_button)
        actions.addWidget(self.regenerate_button)
        actions.addWidget(self.level_bar, 1)
        actions.addWidget(self.transparency_label)
        actions.addWidget(self.transparency_slider)
        actions.addWidget(self.settings_button)

        layout.addLayout(devices)
        layout.addLayout(actions)
        self.setCentralWidget(root)
        self.statusBar().showMessage("就绪 · 选择 Teams 正在播放的声音设备后点击开始")
        values = self.settings.console_geometry
        if values and len(values) == 4:
            x, y, width, height = (int(value) for value in values)
            self.setGeometry(x, y, max(860, width), max(120, height))
        else:
            self.resize(1020, 140)

    def _migrate_settings(self) -> None:
        if self.settings.llm_model == "qwen-plus":
            self.settings.llm_model = "qwen3.6-flash"
        if self.settings.answer_max_tokens <= 320:
            self.settings.answer_max_tokens = 800
        if not self.settings.knowledge_app_id:
            self.settings.knowledge_app_id = "aid-433c2467738a4ae1948488f117508609"
        if self.settings.answer_font_size < 14:
            # Pre-teleprompter default was tuned for a static paragraph.
            self.settings.answer_font_size = 20

    def _load_settings(self) -> None:
        self.answer_view.set_wpm(self.settings.teleprompter_wpm)
        self._voice.threshold = float(self.settings.teleprompter_threshold)
        self.follow_toggle.blockSignals(True)
        self.follow_toggle.setChecked(self.settings.teleprompter_follow_voice)
        self.follow_toggle.blockSignals(False)
        self.pin_toggle.blockSignals(True)
        self.pin_toggle.setChecked(self.settings.overlays_pinned)
        self.pin_toggle.blockSignals(False)
        self._apply_overlay_pinning()
        transparency = max(
            0, min(self.MAX_TRANSPARENCY, int(self.settings.background_transparency_percent))
        )
        self.settings.background_transparency_percent = transparency
        self.transparency_slider.blockSignals(True)
        self.transparency_slider.setValue(transparency)
        self.transparency_slider.blockSignals(False)
        self.transparency_label.setText(f"透明 {transparency}%")
        self.answer_toggle.setChecked(self.settings.answer_enabled)
        self._on_answer_toggled(self.settings.answer_enabled)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.store, self.settings, self)
        dialog.setStyleSheet(STYLESHEET)
        if dialog.exec() != int(SettingsDialog.DialogCode.Accepted):
            return
        was_running = bool(self.orchestrator and self.orchestrator.running)
        self.answer_view.set_wpm(self.settings.teleprompter_wpm)
        self._voice.threshold = float(self.settings.teleprompter_threshold)
        self._rendered_script_html = ""
        self._render_translation()
        self._render_answer()
        if self.orchestrator and not was_running:
            self.orchestrator = None
        self.statusBar().showMessage(
            "设置已保存；连接参数将在下次开始时生效" if was_running else "设置已保存",
            5000,
        )

    def _persist_overlay_geometry(self) -> None:
        self.settings.translation_geometry = self.translation_overlay.geometry_values()
        if self.answer_overlay.isVisible():
            self.settings.answer_geometry = self.answer_overlay.geometry_values()
        self.store.save(self.settings)

    def _on_pin_toggled(self, pinned: bool) -> None:
        self.settings.overlays_pinned = pinned
        self.store.save(self.settings)
        self._apply_overlay_pinning()
        self.statusBar().showMessage(
            "显示框已置顶" if pinned else "显示框可被其他窗口覆盖", 3000
        )

    def _apply_overlay_pinning(self) -> None:
        """Frameless panels have no taskbar entry, so pinning is also how they
        come back after another window buries them."""
        pinned = self.pin_toggle.isChecked()
        for overlay in (self.translation_overlay, self.answer_overlay):
            was_visible = overlay.isVisible()
            overlay.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, pinned)
            # Changing flags hides the window; restore whatever it was doing.
            if was_visible:
                overlay.show()

    def _show_overlay_menu(self, position: QPoint) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(STYLESHEET)
        if self._pending_answer is not None:
            menu.addAction("切换到新回答", self.apply_pending_answer)
        menu.addAction(
            "暂停滚动" if self.answer_view.running else "开始滚动", self._toggle_scroll
        )
        menu.addAction("回到开头", self._rewind_script)
        menu.addSeparator()
        pin_action = menu.addAction("置顶显示框", self.pin_toggle.setChecked)
        pin_action.setCheckable(True)
        pin_action.setChecked(self.pin_toggle.isChecked())
        menu.addAction("显示控制台", self._raise_console)
        menu.exec(position)

    def _raise_console(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def refresh_devices(self) -> None:
        try:
            self.loopbacks, self.microphones = list_audio_devices()
        except Exception as exc:
            self.statusBar().showMessage(f"读取音频设备失败：{exc}")
            return
        self.loopback_combo.clear()
        self.loopback_combo.addItems([device.label for device in self.loopbacks])
        self.microphone_combo.blockSignals(True)
        self.microphone_combo.clear()
        self.microphone_combo.addItem("不使用麦克风")
        self.microphone_combo.addItems([device.label for device in self.microphones])
        self.microphone_combo.blockSignals(False)
        self._restore_device_selection()

    def _restore_device_selection(self) -> None:
        if self.loopbacks:
            selected = 0
            saved = self.settings.audio_device_index
            if saved is not None:
                for index, device in enumerate(self.loopbacks):
                    if device.index == saved:
                        selected = index
                        break
            self.loopback_combo.setCurrentIndex(selected)
        selected_mic = 0
        saved_mic = self.settings.microphone_device_index
        if saved_mic is not None:
            for index, device in enumerate(self.microphones):
                if device.index == saved_mic:
                    selected_mic = index + 1
                    break
        self.microphone_combo.setCurrentIndex(selected_mic)

    def _selected_microphone(self) -> AudioDevice | None:
        index = self.microphone_combo.currentIndex() - 1
        if 0 <= index < len(self.microphones):
            return self.microphones[index]
        return None

    def _persist_selected_devices(self) -> None:
        if self.loopbacks and self.loopback_combo.currentIndex() >= 0:
            self.settings.audio_device_index = self.loopbacks[
                self.loopback_combo.currentIndex()
            ].index
        microphone = self._selected_microphone()
        self.settings.microphone_device_index = microphone.index if microphone else None
        self.store.save(self.settings)

    def _on_microphone_changed(self, _index: int) -> None:
        self.settings.microphone_device_index = (
            device.index if (device := self._selected_microphone()) else None
        )
        self.store.save(self.settings)
        if self.follow_toggle.isChecked():
            self._start_voice_monitor()

    def _ensure_orchestrator(self) -> ConversationOrchestrator:
        if self.orchestrator is None:
            api_key = self.store.get_api_key()
            if not api_key or not self.settings.workspace_id:
                raise ValueError("请先在设置中填写 API Key 和 Workspace ID")
            if self.answer_toggle.isChecked() and not self.settings.knowledge_app_id:
                raise ValueError("开启回答建议前，请先填写知识库应用 ID（aid-...）")
            self.orchestrator = ConversationOrchestrator(
                self.settings,
                api_key,
                self.database,
                self.on_speech,
                self.on_answer,
            )
        else:
            self.orchestrator.settings = self.settings
            self.orchestrator.api_key = self.store.get_api_key()
        self.orchestrator.set_answer_enabled(self.answer_toggle.isChecked())
        return self.orchestrator

    @asyncSlot()
    async def toggle_session(self) -> None:
        try:
            orchestrator = self._ensure_orchestrator()
            if orchestrator.running:
                self.start_button.setEnabled(False)
                await orchestrator.stop()
                self.start_button.setText("开始")
                self.statusBar().showMessage("会话已结束")
            else:
                if not self.loopbacks:
                    raise ValueError("没有检测到 WASAPI 系统声音设备")
                self._persist_selected_devices()
                self._reset_views()
                device = self.loopbacks[self.loopback_combo.currentIndex()]
                await orchestrator.start(device, None)
                self.start_button.setText("停止")
                self.statusBar().showMessage(f"正在听：{device.label}")
        except Exception as exc:
            QMessageBox.critical(self, "无法启动", str(exc))
        finally:
            self.start_button.setEnabled(True)

    @asyncSlot()
    async def regenerate(self) -> None:
        question = self._question_en.strip()
        if not question:
            return
        try:
            # An explicit request replaces the script, so drop the current one.
            self._script = Answer()
            self._pending_answer = None
            self._set_pending_alert(False)
            self._answer_pending = True
            self._render_answer()
            await self._ensure_orchestrator().generate_for_text(
                question, self._question_zh.strip()
            )
        except Exception as exc:
            self._answer_pending = False
            self._render_answer()
            QMessageBox.warning(self, "无法生成", str(exc))

    def on_speech(self, event: SpeechEvent) -> None:
        if event.kind == SpeechEventType.SOURCE_PARTIAL:
            self._ledger.on_source_partial(event.text)
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.SOURCE_FINAL:
            self._ledger.on_source_final(event.text)
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.TRANSLATION_PARTIAL:
            self._ledger.on_translation_partial(event.text)
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.TRANSLATION_FINAL:
            self._ledger.on_translation_final(event.text)
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.ANSWER_PENDING:
            self._answer_pending = True
            self._render_answer()
        elif event.kind == SpeechEventType.STATUS:
            self.statusBar().showMessage(event.text)
        elif event.kind == SpeechEventType.ERROR:
            self._answer_pending = False
            self._render_answer()
            self.statusBar().showMessage(event.text, 8000)

    def on_answer(self, answer: Answer) -> None:
        if not self.answer_toggle.isChecked():
            return
        self._answer_pending = False
        if self._has_script():
            # Never pull the script away from someone reading it out loud.
            self._pending_answer = answer
            self._set_pending_alert(True)
            self.statusBar().showMessage("新回答已就绪，按 F4 或在提词器上右键切换", 8000)
        else:
            self._apply_answer(answer)

    def apply_pending_answer(self) -> None:
        if self._pending_answer is None:
            return
        self._apply_answer(self._pending_answer)
        self.statusBar().showMessage("已切换到新回答", 3000)

    def _apply_answer(self, answer: Answer) -> None:
        self._script = answer
        self._pending_answer = None
        self._answer_pending = False
        self._set_pending_alert(False)
        self._render_answer()
        self.answer_view.rewind()

    def _set_pending_alert(self, pending: bool) -> None:
        """The console button and the panel frame both flag a waiting answer."""
        self.switch_button.setEnabled(pending)
        self.answer_overlay.set_alert(pending)

    def _has_script(self) -> bool:
        return bool(self._script.english.strip() or self._script.chinese.strip())

    def _reset_views(self) -> None:
        self._ledger.reset()
        self._question_en = ""
        self._question_zh = ""
        self._script = Answer()
        self._pending_answer = None
        self._answer_pending = False
        self._set_pending_alert(False)
        self._render_translation()
        self._render_answer()

    def _sync_from_ledger(self) -> None:
        self._question_en, self._question_zh = self._ledger.latest()
        self._render_translation()

    def _context_pairs(self) -> list[tuple[str, str]]:
        """Completed bilingual turns immediately before the active sentence."""
        pairs = self._ledger.history_pairs()
        active_index = self._ledger.current.history_index
        completed = [
            (english, chinese)
            for index, (english, chinese) in enumerate(pairs)
            if index != active_index and english.strip() and chinese.strip()
        ]
        return completed[-self.CONTEXT_SENTENCES :]

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        """Keep earlier turns compact so the live sentence always has room."""
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"

    def _render_translation(self) -> None:
        blocks = [
            self._bilingual_block(
                self._shorten(english, self.CONTEXT_CHAR_LIMIT),
                self._shorten(chinese, self.CONTEXT_CHAR_LIMIT),
                muted=True,
            )
            for english, chinese in self._context_pairs()
        ]
        blocks.append(
            self._bilingual_block(
                self._question_en or "Waiting for the interviewer…",
                self._question_zh or "等待面试官说话…",
                placeholder=not bool(self._question_en.strip()),
                emphasize=True,
            )
        )
        self._set_card_html(
            self.translation_view,
            "".join(blocks),
            font_size=self.settings.display_font_size,
        )
        self._pin_to_latest(self.translation_view)

    @staticmethod
    def _pin_to_latest(view: QTextEdit) -> None:
        cursor = view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        view.setTextCursor(cursor)
        view.ensureCursorVisible()
        scrollbar = view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _render_answer(self) -> None:
        if not self.answer_toggle.isChecked():
            body = self._plain_block(
                "已关闭回答建议。左侧仍会实时显示英文原文和中文翻译。",
                placeholder=True,
            )
        elif self._has_script():
            body = self._script_blocks()
        elif self._answer_pending:
            body = self._plain_block("正在从知识库生成回答…", placeholder=True)
        else:
            body = self._plain_block(
                "面试官问完一题后，可直接照读的英文稿会出现在这里。",
                placeholder=True,
            )
        html_body = self._card_html(body, font_size=self.settings.answer_font_size)
        if html_body == self._rendered_script_html:
            # Re-setting identical HTML would rewind the scroll mid-sentence.
            return
        self._rendered_script_html = html_body
        self.answer_view.set_script(html_body, self._script_plain_text())

    def _script_blocks(self) -> str:
        blocks = [self._script_lines(self._script.english, "scriptEn")]
        chinese = self._script.chinese.strip()
        if chinese:
            blocks.append('<div class="scriptDivider"></div>')
            blocks.append(self._script_lines(chinese, "scriptZh"))
        return "".join(block for block in blocks if block)

    @staticmethod
    def _script_lines(text: str, klass: str) -> str:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return "".join(f'<div class="{klass}">{html.escape(line)}</div>' for line in lines)

    def _script_plain_text(self) -> str:
        """Only the part that is read aloud sets the scrolling pace."""
        return self._script.english.strip() or self._script.chinese.strip()

    def _set_card_html(
        self,
        view: QTextEdit,
        block: str,
        *,
        font_size: int,
    ) -> None:
        view.setHtml(self._card_html(block, font_size=font_size))

    def _card_html(self, block: str, *, font_size: int) -> str:
        placeholder_size = max(12, font_size - 1)
        script_zh_size = max(13, int(font_size * 0.7))
        return f"""
            <style>
              body {{
                margin: 0;
                font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
              }}
              .turn {{
                margin: 0 0 12px 0;
              }}
              .en {{
                color: #000000;
                font-size: {font_size}px;
                line-height: 1.35;
              }}
              .zh {{
                color: #123c52;
                font-size: {font_size}px;
                line-height: 1.35;
                margin-top: 2px;
              }}
              .turn.muted .en {{ color: #5d6771; }}
              .turn.muted .zh {{ color: #5d6771; }}
              .turn.current .en {{ font-weight: 700; }}
              .turn.current .zh {{ font-weight: 700; }}
              .placeholder {{
                color: #59636c;
                font-size: {placeholder_size}px;
                line-height: 1.35;
              }}
              .answer {{
                border: 1px solid #8b96a1;
                border-radius: 10px;
                padding: 14px 16px;
                color: #080b0e;
                font-size: {font_size}px;
                line-height: 1.55;
              }}
              .answer.placeholder {{
                color: #59636c;
                font-size: {placeholder_size}px;
              }}
              .scriptEn {{
                color: #000000;
                font-size: {font_size}px;
                line-height: 1.5;
                margin: 0 0 10px 0;
              }}
              .scriptZh {{
                color: #33424f;
                font-size: {script_zh_size}px;
                line-height: 1.45;
                margin: 0 0 6px 0;
              }}
              .scriptDivider {{
                border-top: 1px solid #b3bcc5;
                margin: 18px 0 14px 0;
              }}
            </style>
            {block}
            """

    @staticmethod
    def _bilingual_block(
        english: str,
        chinese: str,
        *,
        placeholder: bool = False,
        muted: bool = False,
        emphasize: bool = False,
    ) -> str:
        english_text = html.escape(english).replace("\n", "<br>")
        chinese_text = html.escape(chinese).replace("\n", "<br>")
        en_class = "placeholder" if placeholder else "en"
        zh_class = "placeholder" if placeholder else "zh"
        classes = ["turn"]
        if muted:
            classes.append("muted")
        if emphasize:
            classes.append("current")
        return f"""
        <div class="{' '.join(classes)}">
          <div class="{en_class}">{english_text}</div>
          <div class="{zh_class}">{chinese_text}</div>
        </div>
        """

    @staticmethod
    def _plain_block(text: str, *, placeholder: bool) -> str:
        body = html.escape(text).replace("\n", "<br>")
        klass = "answer placeholder" if placeholder else "answer"
        return f'<div class="{klass}">{body}</div>'

    def _toggle_scroll(self) -> None:
        self.answer_view.toggle()

    def _rewind_script(self) -> None:
        self.answer_view.rewind()

    def _on_scroll_running_changed(self, running: bool) -> None:
        self.scroll_button.setText("暂停滚动（F2）" if running else "手动滚动（F2）")

    def _on_follow_toggled(self, enabled: bool) -> None:
        self.settings.teleprompter_follow_voice = enabled
        self.store.save(self.settings)
        if enabled:
            self._start_voice_monitor()
            return
        self._voice.stop()
        self._voice_timer.stop()
        self.level_bar.setValue(0)
        self.answer_view.set_running(False)

    def _start_voice_monitor(self) -> bool:
        device = self._selected_microphone()
        if device is None:
            self.statusBar().showMessage("请先选择“我的麦克风”，跟读滚动才能工作", 8000)
            self.follow_toggle.blockSignals(True)
            self.follow_toggle.setChecked(False)
            self.follow_toggle.blockSignals(False)
            return False
        self._voice.threshold = float(self.settings.teleprompter_threshold)
        self._voice.start(device)
        self._voice_timer.start()
        self.statusBar().showMessage(f"跟读滚动已开启：{device.label}", 4000)
        return True

    def _poll_voice(self) -> None:
        if self._voice.error:
            message = self._voice.error
            self._voice.stop()
            self.follow_toggle.setChecked(False)
            self.statusBar().showMessage(message, 8000)
            return
        # Full bar at twice the threshold keeps the useful range visible.
        ceiling = max(1.0, self._voice.threshold * 2)
        self.level_bar.setValue(int(min(100.0, self._voice.level / ceiling * 100)))
        if self.follow_toggle.isChecked():
            self.answer_view.set_running(self._voice.speaking())

    def _on_transparency_changed(self, value: int) -> None:
        value = max(0, min(self.MAX_TRANSPARENCY, int(value)))
        self.settings.background_transparency_percent = value
        self.transparency_label.setText(f"透明 {value}%")
        self._apply_window_transparency()
        self.store.save(self.settings)

    def _apply_window_transparency(self) -> None:
        """Fade the panels, frame and text included, to reveal the app behind."""
        transparency = max(
            0,
            min(self.MAX_TRANSPARENCY, int(self.settings.background_transparency_percent)),
        )
        opacity = (100 - transparency) / 100
        # The console stays opaque; it is meant to be parked outside the way.
        self.translation_overlay.setWindowOpacity(opacity)
        self.answer_overlay.setWindowOpacity(opacity)

    def _on_answer_toggled(self, enabled: bool) -> None:
        self.settings.answer_enabled = enabled
        self._script = Answer()
        self._pending_answer = None
        self._answer_pending = False
        self._set_pending_alert(False)
        self.regenerate_button.setEnabled(enabled)
        self.answer_overlay.setVisible(enabled and self.isVisible())
        if not enabled:
            self.answer_view.set_running(False)
        if self.orchestrator is not None:
            self.orchestrator.set_answer_enabled(enabled)
        self.store.save(self.settings)
        self._render_answer()
        self.statusBar().showMessage(
            "已开启回答建议" if enabled else "仅实时翻译",
            3000,
        )

    def _apply_style(self) -> None:
        for window in (self, self.translation_overlay, self.answer_overlay):
            window.setStyleSheet(STYLESHEET)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._voice_timer.stop()
        self._voice.stop()
        self.settings.console_geometry = [
            self.geometry().x(),
            self.geometry().y(),
            self.geometry().width(),
            self.geometry().height(),
        ]
        self._persist_overlay_geometry()
        self.translation_overlay.close()
        self.answer_overlay.close()
        if self.orchestrator and self.orchestrator.running:
            asyncio.create_task(self.orchestrator.stop())
        event.accept()
