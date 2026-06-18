"""Shared pytest fixtures for the echoflow voice pipeline test suite."""

import numpy as np
import pytest

from src.config import ASRBackend, LLMBackend, PipelineConfig, TTSBackend


@pytest.fixture
def mock_config() -> PipelineConfig:
    """PipelineConfig with all-mock backends for fast, offline tests."""
    config = PipelineConfig()
    config.asr.backend = ASRBackend.MOCK
    config.llm.backend = LLMBackend.MOCK
    config.tts.backend = TTSBackend.MOCK
    config.resilience.asr_timeout = 5.0
    config.resilience.llm_timeout = 10.0
    config.resilience.tts_timeout = 5.0
    return config


@pytest.fixture
def silent_audio() -> bytes:
    """1 second of silence at 16 kHz (PCM int16)."""
    return (np.zeros(16000, dtype=np.int16)).tobytes()


@pytest.fixture
def speech_audio() -> bytes:
    """1 second of loud noise to trigger VAD (PCM int16)."""
    return (np.random.randn(16000) * 8000).astype(np.int16).tobytes()
