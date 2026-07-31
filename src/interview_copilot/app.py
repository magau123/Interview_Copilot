from __future__ import annotations

import asyncio
import logging
import sys

from PySide6.QtWidgets import QApplication
from qasync import QEventLoop

from interview_copilot.config import SettingsStore
from interview_copilot.ssl_util import configure_ssl
from interview_copilot.ui.main_window import MainWindow


def main() -> int:
    store = SettingsStore()
    settings = store.load()
    logging.basicConfig(
        filename=settings.data_dir / "application.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    configure_ssl()
    application = QApplication(sys.argv)
    application.setApplicationName("Interview Copilot")
    loop = QEventLoop(application)
    asyncio.set_event_loop(loop)
    window = MainWindow(store)
    window.show()
    application.aboutToQuit.connect(loop.stop)
    try:
        with loop:
            loop.run_forever()
    finally:
        window.database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
