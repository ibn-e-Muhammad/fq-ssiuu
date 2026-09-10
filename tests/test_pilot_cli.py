"""Legacy CLI regression checks require no model, token or external service."""

import sys

import pytest

from experiments.reproductions.QRA.historical.pilot_p1 import parse_args


def test_p1_rejects_historical_fp16_substitution(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pilot", "--open_unlearning_path", "/unused", "--skip_fp16"])
    with pytest.raises(SystemExit) as caught:
        parse_args()
    assert caught.value.code == 2
    assert "historical values cannot replace it" in capsys.readouterr().err


def test_p1_default_requires_current_run_fp16(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pilot", "--open_unlearning_path", "/unused"])
    assert parse_args().skip_fp16 is False
