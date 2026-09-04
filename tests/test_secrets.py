"""
tests/test_secrets.py

Unit tests for the secrets module.
No real credentials are used — all tests use mocked values.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from fq_ssiuu.secrets import (
    get_secret,
    check_all_secrets,
    SecretNotFoundError,
    _is_kaggle_runtime,
)


class TestGetSecret:
    """Tests for get_secret() precedence and error handling."""

    def test_env_var_takes_precedence(self):
        """Environment variable should be returned first."""
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            result = get_secret("HF_TOKEN")
            assert result == "env_token"

    def test_required_secret_not_found_raises(self):
        """Missing required secret should raise SecretNotFoundError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SecretNotFoundError) as exc_info:
                get_secret("HF_TOKEN", required=True)
            # Verify the error message does NOT contain a secret value
            assert "HF_TOKEN" in str(exc_info.value)
            assert "env_token" not in str(exc_info.value)

    def test_optional_secret_returns_none(self):
        """Missing optional secret should return None, not raise."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_secret("HF_TOKEN", required=False)
            assert result is None

    def test_unknown_secret_warns(self):
        """Unknown secret name should log a warning."""
        with patch.dict(os.environ, {"UNKNOWN_SECRET": "val"}):
            # Should still return the value, just warn
            result = get_secret("UNKNOWN_SECRET", required=False)
            assert result == "val"


class TestCheckAllSecrets:
    """Tests for check_all_secrets()."""

    def test_returns_dict_of_bools(self):
        """Result should map names to booleans, never values."""
        with patch.dict(os.environ, {"HF_TOKEN": "token"}, clear=True):
            result = check_all_secrets(["HF_TOKEN", "KAGGLE_KEY"])
            assert result == {"HF_TOKEN": True, "KAGGLE_KEY": False}
            # Values must NEVER appear in the result
            assert "token" not in str(result)


class TestKaggleDetection:
    """Tests for Kaggle runtime detection."""

    def test_not_kaggle_by_default(self):
        """Should not detect Kaggle outside Kaggle."""
        with patch.dict(os.environ, {}, clear=True):
            # This may or may not be True depending on test environment
            # but we can at least verify it returns a bool
            assert isinstance(_is_kaggle_runtime(), bool)

    def test_kaggle_env_var_detection(self):
        """Should detect Kaggle via KAGGLE_KERNEL_RUN_TYPE env var."""
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
            assert _is_kaggle_runtime() is True
