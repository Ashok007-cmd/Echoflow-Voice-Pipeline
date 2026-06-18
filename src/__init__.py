"""Real-Time Multimodal Application.

A streaming voice assistant pipeline with:
  - Phase 1: Streaming ASR → LLM → TTS pipeline
  - Phase 2: End-to-end latency budget deconstruction and visualization
  - Phase 3: System resilience (timeouts, circuit breakers, graceful degradation)
"""

__version__ = "1.0.0"
