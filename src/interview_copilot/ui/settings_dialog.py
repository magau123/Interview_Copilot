"""Settings live in a dialog so the floating panels stay pure content."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from qasync import asyncSlot

from interview_copilot.config import Settings, SettingsStore
from interview_copilot.ui.teleprompter import TeleprompterView

DEFAULT_REALTIME_MODEL = "qwen3.5-livetranslate-flash-realtime"


class SettingsDialog(QDialog):
    def __init__(self, store: SettingsStore, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.settings = settings
        self.setWindowTitle("设置")
        self.setMinimumWidth(460)

        self.api_key_edit = QLineEdit(store.get_api_key())
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.workspace_edit = QLineEdit(settings.workspace_id)
        self.region_combo = QComboBox()
        self.region_combo.addItem("中国（北京）", "cn-beijing")
        self.region_combo.addItem("国际（新加坡）", "ap-southeast-1")
        self.region_combo.setCurrentIndex(max(0, self.region_combo.findData(settings.region)))
        self.knowledge_app_edit = QLineEdit(settings.knowledge_app_id)
        self.knowledge_app_edit.setPlaceholderText("aid-xxxxxxxx")
        self.realtime_model_edit = QLineEdit(settings.realtime_model)
        self.font_size_spin = self._spin(13, 24, settings.display_font_size, " px")
        self.answer_font_size_spin = self._spin(14, 36, settings.answer_font_size, " px")
        self.wpm_spin = self._spin(
            TeleprompterView.MIN_WPM,
            TeleprompterView.MAX_WPM,
            settings.teleprompter_wpm,
            " 词/分",
        )
        self.wpm_spin.setToolTip("提词器基础速度；说得慢就调小，跟不上就调大")
        self.voice_threshold_spin = self._spin(
            50, 3000, settings.teleprompter_threshold, "", step=50
        )
        self.voice_threshold_spin.setToolTip(
            "麦克风判定为“正在说话”的音量门槛；环境吵就调大，说话时不滚动就调小"
        )

        form = QFormLayout()
        form.setSpacing(10)
        form.addRow("DashScope API Key", self.api_key_edit)
        form.addRow("Workspace ID", self.workspace_edit)
        form.addRow("区域", self.region_combo)
        form.addRow("知识库应用 ID", self.knowledge_app_edit)
        form.addRow("实时翻译模型", self.realtime_model_edit)
        form.addRow("翻译字号", self.font_size_spin)
        form.addRow("提词器字号", self.answer_font_size_spin)
        form.addRow("提词器速度", self.wpm_spin)
        form.addRow("麦克风灵敏度", self.voice_threshold_spin)

        hint = QLabel(
            "知识库请在阿里云百炼控制台维护并发布；本工具只调用应用 ID，不在本机导入文档。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("panelHint")

        buttons = QHBoxLayout()
        save_button = QPushButton("保存")
        save_button.setObjectName("primary")
        save_button.clicked.connect(self._save)
        test_button = QPushButton("测试连接")
        test_button.clicked.connect(self.test_connection)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(save_button)
        buttons.addWidget(test_button)
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addLayout(buttons)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, suffix: str, *, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(max(minimum, min(maximum, value)))
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def commit(self) -> None:
        """Copy the form into the shared Settings object and persist it."""
        self.settings.workspace_id = self.workspace_edit.text().strip()
        self.settings.region = str(self.region_combo.currentData())
        self.settings.knowledge_app_id = self.knowledge_app_edit.text().strip()
        self.settings.realtime_model = (
            self.realtime_model_edit.text().strip() or DEFAULT_REALTIME_MODEL
        )
        self.settings.display_font_size = self.font_size_spin.value()
        self.settings.answer_font_size = self.answer_font_size_spin.value()
        self.settings.teleprompter_wpm = self.wpm_spin.value()
        self.settings.teleprompter_threshold = self.voice_threshold_spin.value()
        self.store.save(self.settings)
        self.store.set_api_key(self.api_key_edit.text().strip())

    def _save(self) -> None:
        self.commit()
        self.accept()

    @asyncSlot()
    async def test_connection(self) -> None:
        from interview_copilot.providers.qwen.application import test_application_connection
        from interview_copilot.providers.qwen.realtime import test_realtime_connection

        try:
            self.commit()
            api_key = self.store.get_api_key()
            messages = [await test_realtime_connection(self.settings, api_key)]
            if self.settings.knowledge_app_id.strip():
                try:
                    messages.append(await test_application_connection(self.settings, api_key))
                except Exception as app_exc:
                    messages.append(f"知识库应用测试失败：{app_exc}")
            QMessageBox.information(self, "连接测试结果", "\n\n".join(messages))
        except Exception as exc:
            QMessageBox.critical(self, "连接测试失败", str(exc))
