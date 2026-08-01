"""
tests/conftest.py
Shared pytest configuration and custom mark registration.
"""
import pytest


def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line("markers", "cpu: marks tests that run on CPU only")
    config.addinivalue_line("markers", "requires_gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "requires_model: marks tests requiring model weights")
