"""
tests/conftest.py
Shared pytest configuration and custom mark registration.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_external_secret_providers(monkeypatch):
    """Tests use fixture secrets only, never Kaggle Secrets or the local .env."""
    import fq_ssiuu.secrets as secrets
    monkeypatch.setattr(secrets, "_try_kaggle", lambda name: None)
    monkeypatch.setattr(secrets, "_try_dotenv", lambda name: None)


def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line("markers", "cpu: marks tests that run on CPU only")
    config.addinivalue_line("markers", "requires_gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "requires_model: marks tests requiring model weights")
