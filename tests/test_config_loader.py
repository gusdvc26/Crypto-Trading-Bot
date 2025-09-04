import os
import pytest

from src.utils.helpers import load_config


def test_env_override_wins(monkeypatch):
    # risk.yml has tp_bps: 20 (stub). Override via env.
    monkeypatch.setenv("RISK_TP_BPS", "99")
    cfg = load_config("risk")
    assert cfg["tp_bps"] == 99


def test_missing_file_error():
    with pytest.raises(FileNotFoundError) as ei:
        load_config("does_not_exist")
    msg = str(ei.value)
    assert "does_not_exist" in msg
    assert "configs" in msg
    assert "assets" in msg and "risk" in msg and "exec" in msg

