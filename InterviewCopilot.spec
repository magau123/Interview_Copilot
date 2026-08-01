# -*- mode: python ; coding: utf-8 -*-
import pathlib
import sys

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['keyring.backends.Windows']

for package in (
    'pyaudiowpatch',
    'keyring',
    'truststore',
    'certifi',
    'PySide6',
    'shiboken6',
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# Conda ships its own MSVC runtime at the env root. If those DLLs land in
# _internal/, Windows loads them before PySide6's copies and QtWidgets fails
# with: "DLL load failed ... 找不到指定的程序" (ERROR_PROC_NOT_FOUND).
# Dropping every CRT copy and re-adding only PySide6's avoids depending on
# PyInstaller's de-duplication order, which can otherwise discard Qt's copies.
_CONFLICTING_CRT = {
    'msvcp140.dll',
    'msvcp140_1.dll',
    'msvcp140_2.dll',
    'msvcp140_codecvt_ids.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll',
    'concrt140.dll',
}


def _basename(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1].lower()


def _is_crt(entry):
    parts = entry if isinstance(entry, (tuple, list)) else (entry,)
    return any(_basename(part) in _CONFLICTING_CRT for part in parts)


def _pyside_crt_binaries():
    """Qt's own CRT, published both to _internal/ and next to the Qt DLLs."""
    import PySide6

    package_dir = pathlib.Path(PySide6.__file__).parent
    entries = []
    for dll in sorted(package_dir.glob('*.dll')):
        if dll.name.lower() not in _CONFLICTING_CRT:
            continue
        entries.append((dll.name, str(dll), 'BINARY'))
        entries.append((f'PySide6/{dll.name}', str(dll), 'BINARY'))
    if not entries:
        raise SystemExit('PySide6 MSVC runtime DLLs not found; cannot build a working exe.')
    return entries


# PySide6's Qt6Core imports the unversioned ICU exports that ship with Windows
# (System32\icuuc.dll). Dependency analysis walks PATH, so an unrelated Anaconda
# ICU 58 (whose exports carry a _58 suffix) can be collected instead; loading it
# from _internal then fails with ERROR_PROC_NOT_FOUND. Never bundle ICU.
def _is_icu(entry) -> bool:
    parts = entry if isinstance(entry, (tuple, list)) else (entry,)
    return any(_basename(part).startswith("icu") for part in parts)


def _conda_library_bin() -> pathlib.Path | None:
    for base in (sys.prefix, sys.base_prefix):
        candidate = pathlib.Path(base) / "Library" / "bin"
        if candidate.is_dir():
            return candidate
    return None


def _openssl_binaries() -> list[tuple[str, str, str]]:
    """_ssl.pyd needs these, and conda keeps them off PATH in Library/bin."""
    folder = _conda_library_bin()
    if folder is None:
        return []
    entries = []
    for pattern in ("libssl-*.dll", "libcrypto-*.dll"):
        entries += [(dll.name, str(dll), "BINARY") for dll in sorted(folder.glob(pattern))]
    return entries


binaries = [entry for entry in binaries if not _is_crt(entry) and not _is_icu(entry)]

a = Analysis(
    ['src\\interview_copilot\\app.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'boto3',
        'botocore',
        'matplotlib',
        'openpyxl',
        'pandas',
        'pyarrow',
        'scipy',
        'sqlalchemy',
    ],
    noarchive=False,
    optimize=0,
)

# Strip every CRT copy found by dependency analysis, then add back Qt's own.
a.binaries = [entry for entry in a.binaries if not _is_crt(entry) and not _is_icu(entry)]
a.binaries += _pyside_crt_binaries()

# Fail loudly instead of shipping an exe whose ssl module cannot load.
if not any("libcrypto" in _basename(entry[0]) for entry in a.binaries):
    openssl = _openssl_binaries()
    if not openssl:
        raise SystemExit("OpenSSL DLLs (libcrypto/libssl) not found; the exe would fail on import ssl.")
    a.binaries += openssl

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='InterviewCopilot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='InterviewCopilot',
)
