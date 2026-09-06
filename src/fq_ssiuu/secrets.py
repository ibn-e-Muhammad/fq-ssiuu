"""
src/fq_ssiuu/secrets.py

Central secret-loading interface for the FQ-SSIUU project.

Precedence (first found wins):
    1. Environment variables (os.environ)
    2. Kaggle Secrets (kaggle_secrets.UserSecretsClient) — only inside Kaggle runtime
    3. Local .env file (python-dotenv, if installed)

Security:
    - Never logs, prints, or includes secret values in exceptions.
    - Exceptions redact the secret name but confirm which source was checked.
    - All access is through get_secret() with explicit name parameter.
"""

import logging
import os

logger = logging.getLogger(__name__)

_KNOWN_SECRETS = {
    "HF_TOKEN",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "GDRIVE_SERVICE_ACCOUNT_JSON",
    "GDRIVE_FOLDER_ID",
}


def _is_kaggle_runtime() -> bool:
    """Check if running inside a Kaggle notebook/script."""
    return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _try_env(name: str) -> str | None:
    """Try to load a secret from environment variables."""
    val = os.environ.get(name)
    if val:
        logger.debug("Secret '%s' found in environment variables.", name)
    return val


def _try_kaggle(name: str) -> str | None:
    """Try to load a secret from Kaggle Secrets."""
    if not _is_kaggle_runtime():
        return None
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        val = client.get_secret(name)
        if val:
            logger.debug("Secret '%s' found in Kaggle Secrets.", name)
            return val
    except Exception:
        logger.debug("Kaggle Secrets lookup failed for '%s'.", name)
    return None


def _try_dotenv(name: str) -> str | None:
    """Try to load a secret from .env file via python-dotenv."""
    try:
        from dotenv import load_dotenv

        load_dotenv()  # Idempotent — only loads once
        val = os.environ.get(name)
        if val:
            logger.debug("Secret '%s' found via .env file.", name)
            return val
    except ImportError:
        logger.debug("python-dotenv not installed; skipping .env lookup.")
    return None


def get_secret(name: str, required: bool = True) -> str | None:
    """Retrieve a secret by name.

    Checks sources in precedence order: env > Kaggle Secrets > .env file.

    Args:
        name: The secret name (e.g. 'HF_TOKEN').
        required: If True, raises SecretNotFoundError when not found.

    Returns:
        The secret value, or None if not found and not required.

    Raises:
        SecretNotFoundError: If required=True and the secret is not found.
    """
    if name not in _KNOWN_SECRETS:
        logger.warning(
            "Secret '%s' is not in the known secrets list. Consider adding it to _KNOWN_SECRETS.",
            name,
        )

    # Try each source in precedence order
    for source_fn in (_try_env, _try_kaggle, _try_dotenv):
        val = source_fn(name)
        if val:
            return val

    if required:
        raise SecretNotFoundError(
            f"Required secret '{name}' not found in any source "
            f"(checked: environment variables, Kaggle Secrets, .env file). "
            f"See .env.example for required variables."
        )

    logger.info("Optional secret '%s' not found; returning None.", name)
    return None


def check_all_secrets(required_names: list | None = None) -> dict:
    """Check availability of all known secrets.

    Args:
        required_names: If provided, only check these secrets.
            Defaults to all _KNOWN_SECRETS.

    Returns:
        Dict mapping secret name to bool (found or not).
        Never includes values.
    """
    names = required_names or sorted(_KNOWN_SECRETS)
    results = {}
    for name in names:
        try:
            val = get_secret(name, required=False)
            results[name] = val is not None
        except Exception:
            results[name] = False
    return results


class SecretNotFoundError(Exception):
    """Raised when a required secret cannot be found in any source."""

    pass
