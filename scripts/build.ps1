$ErrorActionPreference = "Stop"
$env:PIP_USE_DEPRECATED = "legacy-certs"

python -m pip install --no-build-isolation -i "https://pypi.org/simple" -e ".[dev]"
python -m pytest
python -m ruff check .
python -m PyInstaller --noconfirm --clean "InterviewCopilot.spec"

Write-Host "Build complete: dist\InterviewCopilot\InterviewCopilot.exe"
