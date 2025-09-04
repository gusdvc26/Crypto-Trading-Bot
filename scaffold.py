# scaffold.py
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folders to create
DIRS = [
    "config",
    "data",
    "data/raw",
    "data/processed",
    "data/logs",
    "src",
    "src/ingestion",
    "src/signals",
    "src/backtesting",
    "src/execution",
    "src/utils",
    "tests",
]

# Files to create (with minimal placeholder content)
FILES: dict[str, str] = {
    "config/__init__.py": "",
    "config/settings.py": "# placeholder; paste the real settings.py here\n",
    "src/__init__.py": "",
    "src/ingestion/__init__.py": "",
    "src/ingestion/fetch_data.py": "# placeholder; paste the real fetch_data.py here\n",
    "src/signals/__init__.py": "",
    "src/signals/strategies.py": "# placeholder; add strategies here\n",
    "src/backtesting/__init__.py": "",
    "src/backtesting/backtest.py": "# placeholder; add backtest scaffolding here\n",
    "src/execution/__init__.py": "",
    "src/execution/trader.py": "# placeholder; add execution layer here\n",
    "src/utils/__init__.py": "",
    "src/utils/helpers.py": "# placeholder; paste the real helpers.py here\n",
    "tests/test_signals.py": "def test_placeholder():\n    assert True\n",
    # Optional quality-of-life files:
    ".gitignore": "\n".join([
        "# Python",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".Python",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".coverage",
        "htmlcov/",
        "",
        "# Environments",
        ".venv/",
        "venv/",
        "env/",
        "",
        "# Data & logs",
        "data/logs/*",
        "!data/logs/.gitkeep",
        "data/raw/*",
        "!data/raw/.gitkeep",
        "data/processed/*",
        "!data/processed/.gitkeep",
        "",
        "# OS",
        ".DS_Store",
        "Thumbs.db",
        ""
    ]),
    "README.md": "# crypto-signal-bot\n\nMinimal scaffold. Fill in modules step-by-step.\n",
    "requirements.txt": "# add deps as you go; e.g.:\n# aiohttp==3.9.*\n",
}

# Empty .gitkeep files to keep empty dirs in git
GITKEEPS = [
    "data/logs/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
]

def ensure_dirs():
    for d in DIRS:
        path = ROOT / d
        path.mkdir(parents=True, exist_ok=True)

def ensure_files():
    for f, content in FILES.items():
        fp = ROOT / f
        if not fp.exists():
            fp.write_text(content, encoding="utf-8")
    for g in GITKEEPS:
        gp = ROOT / g
        if not gp.exists():
            gp.parent.mkdir(parents=True, exist_ok=True)
            gp.write_text("", encoding="utf-8")

def main():
    ensure_dirs()
    ensure_files()
    print("Scaffold complete ✅")
    print("Next: paste in your real code for settings.py, helpers.py, and fetch_data.py.")

if __name__ == "__main__":
    main()
