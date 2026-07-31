from __future__ import annotations

import os
from pathlib import Path

from interview_copilot.ssl_util import create_ssl_context, sanitize_ssl_env


def test_sanitize_ssl_env_clears_cursor_ca(monkeypatch, tmp_path: Path) -> None:
    ca = tmp_path / "cursor-ca" / "cursor-node-ca-bundle-with-paloalto-intermediate.pem"
    ca.parent.mkdir(parents=True)
    ca.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    monkeypatch.setenv("CURL_CA_BUNDLE", str(ca))
    monkeypatch.setenv("OTHER_VAR", "keep-me")

    cleared = sanitize_ssl_env()

    assert set(cleared) == {"SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"}
    assert "SSL_CERT_FILE" not in os.environ
    assert "REQUESTS_CA_BUNDLE" not in os.environ
    assert "CURL_CA_BUNDLE" not in os.environ
    assert os.environ["OTHER_VAR"] == "keep-me"


def test_sanitize_ssl_env_keeps_non_cursor_ca(monkeypatch, tmp_path: Path) -> None:
    ca = tmp_path / "company-root.pem"
    ca.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)

    cleared = sanitize_ssl_env()

    assert cleared == []
    assert os.environ["SSL_CERT_FILE"] == str(ca)


def test_create_ssl_context_returns_context() -> None:
    context = create_ssl_context()
    assert context is not None
    assert context.check_hostname is True
