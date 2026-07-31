from __future__ import annotations

import logging
import os
import ssl
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_CA_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
_CURSOR_CA_MARKERS = ("cursor-ca", "cursor-node-ca")


def _is_cursor_ca_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return any(marker in normalized for marker in _CURSOR_CA_MARKERS)


def sanitize_ssl_env() -> list[str]:
    """Clear Cursor-injected CA env vars that break OpenSSL 3 verification.

    Cursor may set SSL_CERT_FILE / REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE to a
    corporate MITM intermediate PEM. That works for Cursor's Node stack but
    causes ``Missing Authority Key Identifier`` with Python's OpenSSL when the
    frozen GUI exe is launched from Explorer.
    """
    cleared: list[str] = []
    for name in _CA_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        if not _is_cursor_ca_path(value):
            continue
        del os.environ[name]
        cleared.append(name)
    return cleared


def configure_ssl() -> None:
    """Normalize SSL for GUI/frozen launches before any network I/O."""
    cleared = sanitize_ssl_env()
    if cleared:
        logger.info("Cleared Cursor CA env vars for SSL: %s", ", ".join(cleared))

    try:
        import truststore

        truststore.inject_into_ssl()
        logger.info("Using OS certificate store via truststore")
        return
    except Exception as exc:  # noqa: BLE001 - best-effort bootstrap
        logger.warning("truststore unavailable (%s); falling back to default SSL", exc)


def create_ssl_context() -> ssl.SSLContext:
    """Build an SSLContext that prefers the OS trust store on Windows."""
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # noqa: BLE001 - fall back to stdlib / certifi
        pass

    context = ssl.create_default_context()
    try:
        import certifi

        ca_path = Path(certifi.where())
        if ca_path.is_file():
            context.load_verify_locations(str(ca_path))
    except Exception:  # noqa: BLE001 - keep default context
        pass

    # Frozen apps should not inherit Cursor's broken CA override.
    if getattr(sys, "frozen", False):
        sanitize_ssl_env()
    return context
