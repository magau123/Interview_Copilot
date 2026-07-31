from __future__ import annotations

import asyncio
import html

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qasync import asyncSlot

from interview_copilot.audio.capture import AudioDevice, list_audio_devices
from interview_copilot.config import SettingsStore
from interview_copilot.conversation.orchestrator import ConversationOrchestrator
from interview_copilot.conversation.translation_ledger import TranslationLedger
from interview_copilot.models import Answer, SpeechEvent, SpeechEventType
from interview_copilot.storage.database import Database


class MainWindow(QMainWindow):
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
        self._answer_en = ""
        self._answer_zh = ""
        self._source_is_final = False
        self._ledger = TranslationLedger()
        self._follow_history = True
        self._updating_history = False
        self.setWindowTitle("实时英文面试助手")
        self.resize(1100, 760)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self._build_ui()
        self._load_settings()
        self.refresh_devices()
        self._apply_style()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_live_tab(), "实时助手")
        tabs.addTab(self._build_knowledge_tab(), "知识库")
        tabs.addTab(self._build_settings_tab(), "设置")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("就绪")

    def _build_live_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        controls = QHBoxLayout()
        self.loopback_combo = QComboBox()
        self.microphone_combo = QComboBox()
        self.refresh_button = QPushButton("刷新设备")
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button = QPushButton("开始")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle_session)
        self.answer_toggle = QCheckBox("AI 回答")
        self.answer_toggle.setToolTip("关闭时仅进行实时英文识别和中文翻译")
        self.answer_toggle.toggled.connect(self._on_answer_toggled)
        controls.addWidget(QLabel("会议声音"))
        controls.addWidget(self.loopback_combo, 2)
        controls.addWidget(QLabel("我的麦克风"))
        controls.addWidget(self.microphone_combo, 2)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.answer_toggle)
        controls.addWidget(self.start_button)
        layout.addLayout(controls)

        transcript_box = QGroupBox("实时翻译（历史最新在上，底部固定当前一句）")
        transcript_layout = QVBoxLayout(transcript_box)

        self.history_view = QTextEdit()
        self.history_view.setObjectName("bilingualCard")
        self.history_view.setReadOnly(True)
        self.history_view.setPlaceholderText("已完成的翻译显示在这里（最新在最上），向下滚动可看更早内容")
        history_bar = self.history_view.verticalScrollBar()
        history_bar.valueChanged.connect(self._on_history_scroll)
        history_bar.rangeChanged.connect(self._on_history_range_changed)
        transcript_layout.addWidget(self.history_view, 3)

        latest_label = QLabel("最新一句")
        latest_label.setObjectName("sectionHint")
        transcript_layout.addWidget(latest_label)

        self.question_view = QTextEdit()
        self.question_view.setObjectName("bilingualCard")
        self.question_view.setReadOnly(True)
        self.question_view.setMinimumHeight(140)
        self.question_view.setMaximumHeight(220)
        transcript_layout.addWidget(self.question_view, 1)

        self.answer_box = QGroupBox("当前回答（中文）")
        answer_layout = QVBoxLayout(self.answer_box)
        self.answer_view = QTextEdit()
        self.answer_view.setObjectName("bilingualCard")
        self.answer_view.setReadOnly(True)
        answer_layout.addWidget(self.answer_view)
        self.regenerate_button = QPushButton("为当前问题重新生成")
        self.regenerate_button.clicked.connect(self.regenerate)
        answer_layout.addWidget(self.regenerate_button, alignment=Qt.AlignmentFlag.AlignRight)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(transcript_box)
        splitter.addWidget(self.answer_box)
        splitter.setSizes([300, 380])
        layout.addWidget(splitter)
        self._render_history()
        self._render_question()
        self._render_answer()
        return root

    def _build_knowledge_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        help_label = QLabel(
            "知识库已改为调用阿里云百炼应用（应用内挂载的知识库）。"
            "请在「设置」中填写应用 ID（aid-...），并在百炼控制台维护与发布知识库。"
            "本机不再导入或索引文档。"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.knowledge_app_label = QLabel()
        self.knowledge_app_label.setWordWrap(True)
        layout.addWidget(self.knowledge_app_label)
        test_button = QPushButton("测试知识库应用")
        test_button.clicked.connect(self.test_knowledge_app)
        layout.addWidget(test_button)
        layout.addStretch(1)
        return root

    def _build_settings_tab(self) -> QWidget:
        root = QWidget()
        form = QFormLayout(root)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.workspace_edit = QLineEdit()
        self.region_combo = QComboBox()
        self.region_combo.addItem("中国（北京）", "cn-beijing")
        self.region_combo.addItem("国际（新加坡）", "ap-southeast-1")
        self.knowledge_app_edit = QLineEdit()
        self.knowledge_app_edit.setPlaceholderText("aid-xxxxxxxx")
        self.realtime_model_edit = QLineEdit()
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(12, 22)
        self.font_size_spin.setSuffix(" px")
        self.font_size_spin.valueChanged.connect(self._on_font_size_changed)
        self.answer_font_size_spin = QSpinBox()
        self.answer_font_size_spin.setRange(8, 18)
        self.answer_font_size_spin.setSuffix(" px")
        self.answer_font_size_spin.valueChanged.connect(
            self._on_answer_font_size_changed
        )
        form.addRow("DashScope API Key", self.api_key_edit)
        form.addRow("Workspace ID", self.workspace_edit)
        form.addRow("区域", self.region_combo)
        form.addRow("知识库应用 ID", self.knowledge_app_edit)
        form.addRow("实时翻译模型", self.realtime_model_edit)
        form.addRow("翻译字体大小", self.font_size_spin)
        form.addRow("AI 回复字体大小", self.answer_font_size_spin)
        buttons = QHBoxLayout()
        save_button = QPushButton("保存设置")
        save_button.setObjectName("primary")
        save_button.clicked.connect(self.save_settings)
        test_button = QPushButton("测试连接")
        test_button.clicked.connect(self.test_connection)
        buttons.addWidget(save_button)
        buttons.addWidget(test_button)
        form.addRow("", buttons)
        return root

    def _load_settings(self) -> None:
        # Migrate the original quality-first defaults to the requested low-latency profile.
        if self.settings.llm_model == "qwen-plus":
            self.settings.llm_model = "qwen3.6-flash"
        if self.settings.answer_max_tokens <= 320:
            self.settings.answer_max_tokens = 800
        if not self.settings.knowledge_app_id:
            self.settings.knowledge_app_id = "aid-433c2467738a4ae1948488f117508609"
        self.api_key_edit.setText(self.store.get_api_key())
        self.workspace_edit.setText(self.settings.workspace_id)
        index = self.region_combo.findData(self.settings.region)
        self.region_combo.setCurrentIndex(max(0, index))
        self.knowledge_app_edit.setText(self.settings.knowledge_app_id)
        self.realtime_model_edit.setText(self.settings.realtime_model)
        self.font_size_spin.setValue(self.settings.display_font_size)
        self.answer_font_size_spin.setValue(self.settings.answer_font_size)
        self.answer_toggle.setChecked(self.settings.answer_enabled)
        self._on_answer_toggled(self.settings.answer_enabled)
        self._refresh_knowledge_label()

    def save_settings(self) -> None:
        was_running = bool(self.orchestrator and self.orchestrator.running)
        self._apply_settings_from_form()
        if self.orchestrator and not was_running:
            self.orchestrator = None
        message = "设置已保存；连接参数将在下次开始时生效" if was_running else "设置已保存"
        self.statusBar().showMessage(message, 5000)

    def _apply_settings_from_form(self) -> None:
        self.settings.workspace_id = self.workspace_edit.text().strip()
        self.settings.region = str(self.region_combo.currentData())
        self.settings.knowledge_app_id = self.knowledge_app_edit.text().strip()
        self.settings.realtime_model = (
            self.realtime_model_edit.text().strip()
            or "qwen3.5-livetranslate-flash-realtime"
        )
        self.settings.display_font_size = self.font_size_spin.value()
        self.settings.answer_font_size = self.answer_font_size_spin.value()
        self.settings.answer_enabled = self.answer_toggle.isChecked()
        self.store.save(self.settings)
        self.store.set_api_key(self.api_key_edit.text().strip())
        self._refresh_knowledge_label()

    def _refresh_knowledge_label(self) -> None:
        app_id = self.settings.knowledge_app_id.strip() or "（未配置）"
        self.knowledge_app_label.setText(
            f"当前知识库应用 ID：{app_id}\n"
            f"调用接口：{self.settings.knowledge_chat_url or '（需先填写 Workspace ID）'}"
        )

    def refresh_devices(self) -> None:
        try:
            self.loopbacks, self.microphones = list_audio_devices()
        except Exception as exc:
            self.statusBar().showMessage(f"读取音频设备失败：{exc}")
            return
        self.loopback_combo.clear()
        self.loopback_combo.addItems([device.label for device in self.loopbacks])
        self.microphone_combo.clear()
        self.microphone_combo.addItem("不记录我的回答")
        self.microphone_combo.addItems([device.label for device in self.microphones])

    def _ensure_orchestrator(self) -> ConversationOrchestrator:
        self._apply_settings_from_form()
        if self.orchestrator is None:
            api_key = self.store.get_api_key()
            if not api_key or not self.settings.workspace_id:
                raise ValueError("请先在设置页面填写 API Key 和 Workspace ID")
            if self.answer_toggle.isChecked() and not self.settings.knowledge_app_id:
                raise ValueError("开启 AI 回答前，请先填写知识库应用 ID（aid-...）")
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
    async def test_connection(self) -> None:
        from interview_copilot.providers.qwen.application import test_application_connection
        from interview_copilot.providers.qwen.realtime import test_realtime_connection

        try:
            self._apply_settings_from_form()
            api_key = self.store.get_api_key()
            realtime = await test_realtime_connection(self.settings, api_key)
            messages = [realtime]
            if self.settings.knowledge_app_id.strip():
                try:
                    messages.append(
                        await test_application_connection(self.settings, api_key)
                    )
                except Exception as app_exc:
                    messages.append(f"知识库应用测试失败：{app_exc}")
            QMessageBox.information(self, "连接测试结果", "\n\n".join(messages))
            self.statusBar().showMessage("连接测试完成", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "连接测试失败", str(exc))
            self.statusBar().showMessage(f"连接测试失败：{exc}", 8000)

    @asyncSlot()
    async def test_knowledge_app(self) -> None:
        from interview_copilot.providers.qwen.application import test_application_connection

        try:
            self._apply_settings_from_form()
            api_key = self.store.get_api_key()
            if not api_key:
                raise ValueError("请先填写 DashScope API Key")
            if not self.settings.knowledge_app_id.strip():
                raise ValueError("请先填写知识库应用 ID")
            result = await test_application_connection(self.settings, api_key)
            QMessageBox.information(self, "知识库应用测试成功", result)
            self.statusBar().showMessage("知识库应用测试成功", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "知识库应用测试失败", str(exc))
            self.statusBar().showMessage(f"知识库应用测试失败：{exc}", 8000)

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
                microphone = None
                if self.microphone_combo.currentIndex() > 0:
                    microphone = self.microphones[self.microphone_combo.currentIndex() - 1]
                self._reset_translation_view()
                await orchestrator.start(
                    self.loopbacks[self.loopback_combo.currentIndex()], microphone
                )
                self.start_button.setText("停止")
                self.statusBar().showMessage("正在监听会议声音")
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
            await self._ensure_orchestrator().generate_for_text(
                question, self._question_zh.strip()
            )
        except Exception as exc:
            QMessageBox.warning(self, "无法生成", str(exc))

    def on_speech(self, event: SpeechEvent) -> None:
        if event.kind == SpeechEventType.SOURCE_PARTIAL:
            if self._source_is_final:
                self._answer_en = ""
                self._answer_zh = ""
                self._source_is_final = False
                self._render_answer()
            self._ledger.on_source_partial(event.text)
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.SOURCE_FINAL:
            self._ledger.on_source_final(event.text)
            self._source_is_final = True
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.TRANSLATION_PARTIAL:
            self._ledger.on_translation_partial(event.text)
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.TRANSLATION_FINAL:
            self._ledger.on_translation_final(event.text)
            self._sync_from_ledger()
        elif event.kind == SpeechEventType.STATUS:
            self.statusBar().showMessage(event.text)
        elif event.kind == SpeechEventType.ERROR:
            self.statusBar().showMessage(event.text, 8000)

    def on_answer(self, answer: Answer) -> None:
        if not self.answer_toggle.isChecked():
            return
        self._answer_en = ""
        self._answer_zh = answer.chinese
        self._render_answer()

    def _reset_translation_view(self) -> None:
        self._ledger.reset()
        self._question_en = ""
        self._question_zh = ""
        self._source_is_final = False
        self._follow_history = True
        self._render_history()
        self._render_question()

    def _sync_from_ledger(self) -> None:
        self._question_en, self._question_zh = self._ledger.latest()
        self._render_history()
        self._render_question()

    def _on_history_scroll(self, value: int) -> None:
        if self._updating_history:
            return
        # Newest history stays at the top; stick to top unless user scrolls down.
        self._follow_history = value <= 16

    def _on_history_range_changed(self, _minimum: int, _maximum: int) -> None:
        if self._follow_history:
            self._scroll_history_to_top()

    def _scroll_history_to_top(self) -> None:
        if not self._follow_history:
            return
        self._updating_history = True
        try:
            self.history_view.verticalScrollBar().setValue(0)
        finally:
            self._updating_history = False

    def _render_history(self) -> None:
        pairs = self._ledger.history_pairs()
        # Keep the active turn only in the pinned "latest" pane.
        active_index = self._ledger.current.history_index
        indexed = [
            (index, english, chinese)
            for index, (english, chinese) in enumerate(pairs, start=1)
            if index - 1 != active_index
        ]
        # Only show completed bilingual turns; skip empty/partial Chinese.
        indexed = [
            (index, english, chinese)
            for index, english, chinese in indexed
            if english.strip() and chinese.strip()
        ]
        if not indexed:
            self._updating_history = True
            try:
                self.history_view.clear()
            finally:
                self._updating_history = False
            return
        # Newest completed turn first so it is always visible at the top.
        blocks = [
            self._bilingual_block(
                english,
                chinese,
                f"SENTENCE {index}",
            )
            for index, english, chinese in reversed(indexed)
        ]
        self._updating_history = True
        try:
            self._set_card_html(
                self.history_view,
                "".join(blocks),
                follow_bottom=False,
            )
        finally:
            self._updating_history = False
        if self._follow_history:
            self._scroll_history_to_top()
            QTimer.singleShot(0, self._scroll_history_to_top)

    def _render_question(self) -> None:
        # Latest sentence is always pinned below history — never buried by scroll.
        self._set_card_html(
            self.question_view,
            self._bilingual_block(
                self._question_en or "Waiting for the interviewer…",
                self._question_zh or "等待面试官提问…",
                "LATEST",
                placeholder=not self._question_en,
            ),
            follow_bottom=True,
        )

    def _render_answer(self) -> None:
        if not self.answer_toggle.isChecked():
            chinese = "AI 回答已关闭，当前仅进行实时翻译。"
            placeholder = True
        else:
            chinese = self._answer_zh or "回答将在这里显示。"
            placeholder = not self._answer_zh
        self._set_card_html(
            self.answer_view,
            self._chinese_block(chinese, "当前回答", placeholder=placeholder),
            font_size=self.settings.answer_font_size,
            compact=True,
        )

    def _set_card_html(
        self,
        view: QTextEdit,
        block: str,
        *,
        follow_bottom: bool | None = None,
        font_size: int | None = None,
        compact: bool = False,
    ) -> None:
        scrollbar = view.verticalScrollBar()
        if follow_bottom is None:
            follow_bottom = scrollbar.value() >= scrollbar.maximum() - 8
        content_size = font_size or self.settings.display_font_size
        placeholder_size = max(8, content_size - 1)
        turn_margin = 8 if compact else 14
        turn_padding = 8 if compact else 12
        divider_margin = 7 if compact else 12
        line_height = 1.35 if compact else 1.55
        view.setHtml(
            f"""
            <style>
              body {{ margin: 4px; font-family: "Segoe UI", "Microsoft YaHei"; }}
              .turn {{ background: #0f172a; border: 1px solid #334155;
                       margin: 0 0 {turn_margin}px 0; padding: {turn_padding}px; }}
              .turn-title {{ color: #94a3b8; font-size: 10px; font-weight: 700;
                             letter-spacing: 1px; margin-bottom: 9px; }}
              .label {{ color: #60a5fa; font-size: 10px; font-weight: 700;
                        letter-spacing: 1px; margin-bottom: 4px; }}
              .content {{ color: #f8fafc; font-size: {content_size}px;
                          line-height: {line_height}; }}
              .placeholder {{ color: #64748b; font-size: {placeholder_size}px;
                              line-height: {line_height}; }}
              .divider {{ border-top: 1px solid #334155;
                          margin: {divider_margin}px 0; }}
              .zh {{ color: #34d399; }}
            </style>
            {block}
            """
        )
        if follow_bottom:
            scrollbar.setValue(scrollbar.maximum())

    @staticmethod
    def _bilingual_block(
        english: str,
        chinese: str,
        title: str,
        *,
        placeholder: bool = False,
    ) -> str:
        english_text = html.escape(english).replace("\n", "<br>")
        chinese_text = html.escape(chinese).replace("\n", "<br>")
        content_class = "placeholder" if placeholder else "content"
        return f"""
        <div class="turn">
          <div class="turn-title">{title}</div>
          <div class="label">ENGLISH</div>
          <div class="{content_class}">{english_text}</div>
          <div class="divider"></div>
          <div class="label zh">中文</div>
          <div class="{content_class}">{chinese_text}</div>
        </div>
        """

    @staticmethod
    def _chinese_block(
        chinese: str,
        title: str,
        *,
        placeholder: bool = False,
    ) -> str:
        chinese_text = html.escape(chinese).replace("\n", "<br>")
        content_class = "placeholder" if placeholder else "content"
        return f"""
        <div class="turn">
          <div class="turn-title">{title}</div>
          <div class="label zh">中文</div>
          <div class="{content_class}">{chinese_text}</div>
        </div>
        """

    def _on_font_size_changed(self, value: int) -> None:
        self.settings.display_font_size = value
        self._render_history()
        self._render_question()

    def _on_answer_font_size_changed(self, value: int) -> None:
        self.settings.answer_font_size = value
        self._render_answer()

    def _on_answer_toggled(self, enabled: bool) -> None:
        self.settings.answer_enabled = enabled
        self._answer_en = ""
        self._answer_zh = ""
        self.regenerate_button.setEnabled(enabled)
        self.answer_box.setVisible(enabled)
        self.answer_box.setTitle(
            "当前回答（中文）" if enabled else "当前回答（已关闭）"
        )
        if self.orchestrator is not None:
            self.orchestrator.set_answer_enabled(enabled)
        self.store.save(self.settings)
        self._render_answer()
        self.statusBar().showMessage(
            "AI 回答已开启" if enabled else "仅实时翻译，不生成回答",
            3000,
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111827; color: #e5e7eb; }
            QGroupBox { border: 1px solid #374151; border-radius: 8px; margin-top: 10px;
                        padding-top: 12px; font-weight: 600; }
            QTextEdit, QLineEdit, QComboBox, QSpinBox {
                background: #1f2937; border: 1px solid #374151; border-radius: 6px;
                padding: 7px; selection-background-color: #2563eb;
            }
            QPushButton { background: #374151; border: none; border-radius: 6px;
                          padding: 8px 14px; }
            QPushButton:hover { background: #4b5563; }
            QPushButton#primary { background: #2563eb; }
            QPushButton#primary:hover { background: #1d4ed8; }
            QTextEdit#bilingualCard {
                background: #0f172a; border: 1px solid #334155; border-radius: 10px;
                padding: 12px;
            }
            QLabel#sectionHint {
                color: #94a3b8; font-size: 11px; font-weight: 600;
                letter-spacing: 0.5px; margin-top: 4px;
            }
            QTabBar::tab { padding: 9px 18px; }
            QTabBar::tab:selected { border-bottom: 2px solid #60a5fa; }
            """
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.orchestrator and self.orchestrator.running:
            asyncio.create_task(self.orchestrator.stop())
        event.accept()
