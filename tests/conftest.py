import os

import pytest

os.environ.update(LLM_PROVIDER="mock", VECTOR_BACKEND="memory", CHECKPOINT_BACKEND="memory", API_AUTH_TOKEN="")

from resume_optimizer.config import Settings  # noqa: E402
from resume_optimizer.service import ResumeService  # noqa: E402


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def service(settings):
    return ResumeService(settings)
