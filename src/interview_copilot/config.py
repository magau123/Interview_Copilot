from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import keyring
from platformdirs import user_data_dir

APP_NAME = "InterviewCopilot"
KEYRING_SERVICE = "InterviewCopilot.DashScope"


@dataclass(slots=True)
class Settings:
    workspace_id: str = ""
    region: str = "cn-beijing"
    realtime_model: str = "qwen3.5-livetranslate-flash-realtime"
    asr_model: str = "qwen3-asr-flash-realtime"
    llm_model: str = "qwen3.6-flash"
    embedding_model: str = "text-embedding-v4"
    source_language: str = "en"
    target_language: str = "zh"
    answer_max_tokens: int = 800
    retrieval_limit: int = 3
    display_font_size: int = 16
    answer_font_size: int = 11
    answer_enabled: bool = False
    knowledge_app_id: str = "aid-433c2467738a4ae1948488f117508609"
    audio_device_index: int | None = None
    microphone_device_index: int | None = None

    @property
    def data_dir(self) -> Path:
        path = Path(user_data_dir(APP_NAME, appauthor=False))
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def database_path(self) -> Path:
        return self.data_dir / "interview_copilot.db"

    @property
    def knowledge_chat_url(self) -> str:
        workspace = self.workspace_id.strip()
        if not workspace:
            return ""
        host = (
            f"{workspace}.cn-beijing.maas.aliyuncs.com"
            if self.region == "cn-beijing"
            else f"{workspace}.ap-southeast-1.maas.aliyuncs.com"
        )
        return f"https://{host}/api/v2/apps/knowledge/chat"

    @property
    def realtime_url(self) -> str:
        urls = self.realtime_urls
        return urls[0] if urls else ""

    @property
    def realtime_urls(self) -> list[str]:
        """Preferred workspace endpoint first, then DashScope shared fallback."""
        model = self.realtime_model
        urls: list[str] = []
        if self.workspace_id:
            host = (
                f"{self.workspace_id}.cn-beijing.maas.aliyuncs.com"
                if self.region == "cn-beijing"
                else f"{self.workspace_id}.ap-southeast-1.maas.aliyuncs.com"
            )
            urls.append(f"wss://{host}/api-ws/v1/realtime?model={model}")
        if self.region == "cn-beijing":
            urls.append(f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={model}")
        else:
            urls.append(
                f"wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model={model}"
            )
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(urls))

    @property
    def compatible_base_url(self) -> str:
        if self.region == "cn-beijing":
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class SettingsStore:
    def __init__(self) -> None:
        self._path = Path(user_data_dir(APP_NAME, appauthor=False)) / "settings.json"

    def load(self) -> Settings:
        if not self._path.exists():
            return Settings()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            allowed = Settings.__dataclass_fields__.keys()
            return Settings(**{key: value for key, value in data.items() if key in allowed})
        except (OSError, ValueError, TypeError):
            return Settings()

    def save(self, settings: Settings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def get_api_key() -> str:
        return keyring.get_password(KEYRING_SERVICE, "default") or ""

    @staticmethod
    def set_api_key(value: str) -> None:
        if value:
            keyring.set_password(KEYRING_SERVICE, "default", value)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, "default")
            except keyring.errors.PasswordDeleteError:
                pass
