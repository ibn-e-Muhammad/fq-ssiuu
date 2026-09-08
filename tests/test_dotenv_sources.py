"""Dotenv semantics with synthetic sources only; never access researcher secrets."""

import os
import sys
from types import SimpleNamespace

import fq_ssiuu.secrets as secrets
from fq_ssiuu.secrets import _try_dotenv as read_local_dotenv
from fq_ssiuu.secrets import _try_kaggle as read_kaggle_secret


def test_kaggle_dotenv_is_parsed_in_memory_without_interpolation(monkeypatch, capsys):
    calls = []

    def lookup(name):
        calls.append(name)
        if name == "FQ_SSIUU_DOTENV":
            return 'HF_TOKEN="fixture-${UNCHANGED}"\nGITHUB_TOKEN="another-fixture"\n'
        raise LookupError("Individual secret unavailable")

    monkeypatch.setattr(secrets, "_is_kaggle_runtime", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(UserSecretsClient=lambda: SimpleNamespace(get_secret=lookup)),
    )
    before = dict(os.environ)
    assert read_kaggle_secret("HF_TOKEN") == "fixture-${UNCHANGED}"
    assert calls == ["HF_TOKEN", "FQ_SSIUU_DOTENV"]
    assert dict(os.environ) == before
    assert capsys.readouterr().out == ""


def test_individual_kaggle_secret_takes_precedence(monkeypatch):
    calls = []

    def lookup(name):
        calls.append(name)
        return "individual-fixture"

    monkeypatch.setattr(secrets, "_is_kaggle_runtime", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(UserSecretsClient=lambda: SimpleNamespace(get_secret=lookup)),
    )
    assert read_kaggle_secret("GITHUB_TOKEN") == "individual-fixture"
    assert calls == ["GITHUB_TOKEN"]


def test_local_dotenv_reads_one_key_without_exporting_other_values(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("HF_TOKEN=local-fixture\nOTHER_PRIVATE_VALUE=unused-fixture\n")
    monkeypatch.setattr(secrets, "LOCAL_DOTENV_PATH", path)
    monkeypatch.setattr(secrets, "_is_kaggle_runtime", lambda: False)
    before = dict(os.environ)
    assert read_local_dotenv("HF_TOKEN") == "local-fixture"
    assert dict(os.environ) == before
    monkeypatch.setattr(secrets, "_is_kaggle_runtime", lambda: True)
    assert read_local_dotenv("HF_TOKEN") is None
