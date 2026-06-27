"""Tests for ASR, LLM, and TTS backends."""

import asyncio
import io
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Mock optional heavy dependencies before import so tests run without [local] extras installed
mock_pyttsx3 = MagicMock()
mock_pyttsx3.init.return_value = MagicMock()
sys.modules["pyttsx3"] = mock_pyttsx3

mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.float32 = "float32"
mock_torch.float16 = "float16"
sys.modules["torch"] = mock_torch

mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers

from src.config import ASRConfig, LLMConfig, TTSConfig, ASRBackend, LLMBackend, TTSBackend
from src.pipeline.asr import (
    create_asr_engine,
    GoogleASREngine,
    OpenAIASREngine,
    LocalWhisperASREngine,
)
from src.pipeline.llm_client import (
    create_llm_client,
    OpenAILLMClient,
    AnthropicLLMClient,
)
from src.pipeline.tts import (
    create_tts_engine,
    EdgeTTSEngine,
    Pyttsx3Engine,
)
from src.pipeline.models import AudioChunk, LLMResponse


# ═══════════════════════════════════════════════════════════════════
# ASR Backend Tests
# ═══════════════════════════════════════════════════════════════════

class TestGoogleASREngine:
    @pytest.fixture
    def config(self):
        return ASRConfig(backend=ASRBackend.GOOGLE)

    @pytest.fixture
    def engine(self, config):
        return create_asr_engine(config)

    @patch("speech_recognition.Recognizer")
    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_rec_class, engine):
        mock_rec = MagicMock()
        mock_rec.recognize_google.return_value = "hello test"
        mock_rec_class.return_value = mock_rec

        # Call transcribe
        result = await engine.transcribe(b"\x00" * 3200, 16000)
        assert result.error is None
        assert result.text == "hello test"
        assert result.latency_ms > 0

    @patch("speech_recognition.Recognizer")
    @pytest.mark.asyncio
    async def test_transcribe_unknown_value(self, mock_rec_class, engine):
        import speech_recognition as sr
        mock_rec = MagicMock()
        mock_rec.recognize_google.side_effect = sr.UnknownValueError("unknown")
        mock_rec_class.return_value = mock_rec

        result = await engine.transcribe(b"\x00" * 3200, 16000)
        assert result.text == ""
        assert result.error == "Could not understand audio"

    @patch("speech_recognition.Recognizer")
    @pytest.mark.asyncio
    async def test_transcribe_request_error(self, mock_rec_class, engine):
        import speech_recognition as sr
        mock_rec = MagicMock()
        mock_rec.recognize_google.side_effect = sr.RequestError("network error")
        mock_rec_class.return_value = mock_rec

        result = await engine.transcribe(b"\x00" * 3200, 16000)
        assert result.text == ""
        assert "Google API error" in result.error


class TestOpenAIASREngine:
    @pytest.fixture
    def config(self):
        return ASRConfig(backend=ASRBackend.OPENAI, openai_api_key="fake-key")

    @pytest.fixture
    def engine(self, config):
        return create_asr_engine(config)

    @patch("openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_transcribe_success(self, mock_openai_class, engine):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "openai text response"
        
        # AsyncOpenAI client structures calls as: client.audio.transcriptions.create
        mock_client.audio = MagicMock()
        mock_client.audio.transcriptions = MagicMock()
        
        # Make the create function an AsyncMock
        mock_create = AsyncMock(return_value=mock_response)
        mock_client.audio.transcriptions.create = mock_create
        
        mock_openai_class.return_value = mock_client

        result = await engine.transcribe(b"\x00" * 3200, 16000)
        assert result.text == "openai text response"
        assert result.error is None
        mock_create.assert_called_once()

    @patch("openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_transcribe_failure(self, mock_openai_class, engine):
        mock_client = MagicMock()
        mock_client.audio = MagicMock()
        mock_client.audio.transcriptions = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("API limit"))
        mock_openai_class.return_value = mock_client

        result = await engine.transcribe(b"\x00" * 3200, 16000)
        assert result.text == ""
        assert "API limit" in result.error


class TestLocalWhisperASREngine:
    @pytest.fixture
    def config(self):
        return ASRConfig(backend=ASRBackend.LOCAL_WHISPER)

    @patch("transformers.pipeline")
    @patch("transformers.AutoModelForSpeechSeq2Seq")
    @patch("transformers.AutoProcessor")
    @pytest.mark.asyncio
    async def test_transcribe_local_success(self, mock_processor, mock_model, mock_pipeline, config):
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": "local whisper output"}
        mock_pipeline.return_value = mock_pipe

        engine = create_asr_engine(config)
        # Call transcribe
        result = await engine.transcribe(b"\x00" * 3200, 16000)
        assert result.text == "local whisper output"
        assert result.error is None


# ═══════════════════════════════════════════════════════════════════
# LLM Backend Tests
# ═══════════════════════════════════════════════════════════════════

class TestOpenAILLMClient:
    @pytest.fixture
    def config(self):
        return LLMConfig(backend=LLMBackend.OPENAI, openai_api_key="fake-key")

    @pytest.fixture
    def client(self, config):
        return create_llm_client(config)

    @patch("openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_generate_stream_success(self, mock_openai_class, client):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock chunk structure for streaming: chunk.choices[0].delta.content & finish_reason
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content="Hello")
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock(content=" world")
        chunk2.choices[0].finish_reason = None

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock(content="")
        chunk3.choices[0].finish_reason = "stop"

        async def mock_async_iterator():
            yield chunk1
            yield chunk2
            yield chunk3

        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=mock_async_iterator())
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_completions

        responses = []
        async for resp in client.generate_stream("Hi"):
            responses.append(resp)

        assert len(responses) == 3
        assert responses[0].text == "Hello"
        assert responses[0].is_first_token is True
        assert responses[0].is_final is False

        assert responses[1].text == " world"
        assert responses[1].is_first_token is False
        assert responses[1].is_final is False

        assert responses[2].text == ""
        assert responses[2].is_final is True
        assert responses[2].finish_reason == "stop"


class TestAnthropicLLMClient:
    @pytest.fixture
    def config(self):
        return LLMConfig(backend=LLMBackend.ANTHROPIC, anthropic_api_key="fake-key")

    @pytest.fixture
    def client(self, config):
        return create_llm_client(config)

    @patch("anthropic.AsyncAnthropic")
    @pytest.mark.asyncio
    async def test_generate_stream_success(self, mock_anthropic_class, client):
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        class MockTextStream:
            async def __aiter__(self):
                yield "Claude"
                yield " response"

        class MockStreamContext:
            async def __aenter__(self):
                stream = MagicMock()
                stream.text_stream = MockTextStream()
                return stream
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=MockStreamContext())

        responses = []
        async for resp in client.generate_stream("Hi"):
            responses.append(resp)

        assert len(responses) == 3
        assert responses[0].text == "Claude"
        assert responses[0].is_first_token is True
        assert responses[0].is_final is False

        assert responses[1].text == " response"
        assert responses[1].is_first_token is False
        assert responses[1].is_final is False

        assert responses[2].text == ""
        assert responses[2].is_final is True
        assert responses[2].finish_reason == "stop"


# ═══════════════════════════════════════════════════════════════════
# TTS Backend Tests
# ═══════════════════════════════════════════════════════════════════

class TestEdgeTTSEngine:
    @pytest.fixture
    def config(self):
        return TTSConfig(backend=TTSBackend.EDGE_TTS)

    @pytest.fixture
    def engine(self, config):
        return create_tts_engine(config)

    @patch("edge_tts.Communicate")
    @pytest.mark.asyncio
    async def test_synthesize_success(self, mock_comm_class, engine):
        mock_comm = MagicMock()
        
        async def mock_stream():
            yield {"type": "audio", "data": b"mp3_data_chunk"}

        mock_comm.stream = mock_stream
        mock_comm_class.return_value = mock_comm

        result = await engine.synthesize("hello edge")
        assert result.audio_data == b"mp3_data_chunk"
        assert result.error is None
        assert result.is_final is True

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_play_fallback_and_cleanup(self, mock_exec, engine):
        # Trigger the fallback block by preventing soundfile/sounddevice imports
        with patch.dict("sys.modules", {"sounddevice": None, "soundfile": None}):
            # Set up mock subprocess wait
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            # Call play
            fake_audio = b"fake_mp3_audio_data"
            await engine.play(fake_audio)

            # Check if subprocess was launched with temporary file
            mock_exec.assert_called_once()
            args = mock_exec.call_args[0]
            assert args[0] == "ffplay"
            temp_file_path = args[3]
            assert temp_file_path.endswith(".mp3")

            # Check that temporary file is cleaned up after execution
            assert not os.path.exists(temp_file_path)


class TestPyttsx3Engine:
    @pytest.fixture
    def config(self):
        return TTSConfig(backend=TTSBackend.PYTTSX3)

    @pytest.fixture
    def engine(self, config):
        return create_tts_engine(config)

    @pytest.mark.asyncio
    async def test_synthesize_success(self, engine):
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        # Mock engine.save_to_file write action
        def mock_save_to_file(text, filename):
            with open(filename, "wb") as f:
                f.write(b"wav_audio_data")

        mock_engine.save_to_file = mock_save_to_file
        mock_engine.runAndWait = MagicMock()

        result = await engine.synthesize("hello pyttsx3")
        assert result.audio_data == b"wav_audio_data"
        assert result.error is None

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_play_fallback_and_cleanup(self, mock_exec, engine):
        with patch.dict("sys.modules", {"sounddevice": None, "soundfile": None}):
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            fake_audio = b"fake_wav_audio_data"
            await engine.play(fake_audio)

            mock_exec.assert_called_once()
            args = mock_exec.call_args[0]
            assert args[0] == "aplay"
            temp_file_path = args[1]
            assert temp_file_path.endswith(".wav")

            # Check cleanup
            assert not os.path.exists(temp_file_path)


# ═══════════════════════════════════════════════════════════════════
# Additional Edge Cases & Factory Tests
# ═══════════════════════════════════════════════════════════════════

def test_invalid_factories():
    with pytest.raises(ValueError):
        invalid_asr_config = ASRConfig(backend="invalid_backend")
        create_asr_engine(invalid_asr_config)

    with pytest.raises(ValueError):
        invalid_llm_config = LLMConfig(backend="invalid_backend")
        create_llm_client(invalid_llm_config)

    with pytest.raises(ValueError):
        invalid_tts_config = TTSConfig(backend="invalid_backend")
        create_tts_engine(invalid_tts_config)


# ═══════════════════════════════════════════════════════════════════
# Additional ASR, TTS, and LLM Streaming / Generate Tests
# ═══════════════════════════════════════════════════════════════════

class TestASRStreaming:
    @pytest.mark.asyncio
    async def test_transcribe_stream_success(self):
        from src.pipeline.asr import ASREngine
        from src.pipeline.models import TranscriptionResult
        
        class MockTranscribeASREngine(ASREngine):
            async def transcribe(self, audio_data: bytes, sample_rate: int):
                return TranscriptionResult(text="transcribed text", is_final=True, latency_ms=10.0)

        config = ASRConfig(backend=ASRBackend.MOCK, chunk_duration=0.1, silence_threshold=0.3)
        engine = MockTranscribeASREngine(config)
        
        speech_chunk = AudioChunk(data=b"\xff\x7f" * 100, timestamp=0.0, sample_rate=16000)
        silence_chunk = AudioChunk(data=b"\x00" * 200, timestamp=0.0, sample_rate=16000)
        
        async def mock_audio_stream():
            yield speech_chunk
            for _ in range(4):
                yield silence_chunk

        results = []
        async for res in engine.transcribe_stream(mock_audio_stream()):
            results.append(res)
            
        assert len(results) == 1
        assert results[0].text == "transcribed text"
        assert results[0].is_final is True
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_transcribe_stream_timeout(self):
        from src.pipeline.asr import ASREngine
        from src.pipeline.models import TranscriptionResult
        
        class TimeoutMockASREngine(ASREngine):
            async def transcribe(self, audio_data: bytes, sample_rate: int):
                await asyncio.sleep(2.0)
                return TranscriptionResult(text="too late", is_final=True)

        config = ASRConfig(
            backend=ASRBackend.MOCK,
            chunk_duration=0.1,
            silence_threshold=0.2,
            max_utterance_seconds=0.05
        )
        engine = TimeoutMockASREngine(config)
        
        speech_chunk = AudioChunk(data=b"\xff\x7f" * 100, timestamp=0.0, sample_rate=16000)
        silence_chunk = AudioChunk(data=b"\x00" * 200, timestamp=0.0, sample_rate=16000)
        
        async def mock_audio_stream():
            yield speech_chunk
            for _ in range(3):
                yield silence_chunk

        results = []
        async for res in engine.transcribe_stream(mock_audio_stream()):
            results.append(res)
            
        assert len(results) == 1
        assert results[0].text == ""
        assert results[0].is_final is True
        assert "timed out" in results[0].error

    @pytest.mark.asyncio
    async def test_transcribe_stream_exception(self):
        from src.pipeline.asr import ASREngine
        
        class ErrorMockASREngine(ASREngine):
            async def transcribe(self, audio_data: bytes, sample_rate: int):
                raise ValueError("unexpected transcribe failure")

        config = ASRConfig(backend=ASRBackend.MOCK, chunk_duration=0.1, silence_threshold=0.2)
        engine = ErrorMockASREngine(config)
        
        speech_chunk = AudioChunk(data=b"\xff\x7f" * 100, timestamp=0.0, sample_rate=16000)
        silence_chunk = AudioChunk(data=b"\x00" * 200, timestamp=0.0, sample_rate=16000)
        
        async def mock_audio_stream():
            yield speech_chunk
            for _ in range(3):
                yield silence_chunk

        results = []
        async for res in engine.transcribe_stream(mock_audio_stream()):
            results.append(res)
            
        assert len(results) == 1
        assert results[0].text == ""
        assert results[0].is_final is True
        assert "unexpected transcribe failure" in results[0].error
        
    def test_asr_engine_reset(self):
        from src.pipeline.asr import MockASREngine
        config = ASRConfig(backend=ASRBackend.MOCK)
        engine = MockASREngine(config)
        engine._buffer.append(b"hello")
        engine._silence_counter = 5
        engine._is_speaking = True
        
        engine.reset()
        assert len(engine._buffer) == 0
        assert engine._silence_counter == 0
        assert engine._is_speaking is False


class TestTTSStreaming:
    @pytest.mark.asyncio
    async def test_mock_tts_synthesize_stream(self):
        from src.pipeline.tts import MockTTSEngine
        config = TTSConfig(backend=TTSBackend.MOCK)
        engine = MockTTSEngine(config)
        
        async def mock_text_stream():
            yield LLMResponse(text="Hello", is_first_token=True, is_final=False)
            yield LLMResponse(text=" world", is_first_token=False, is_final=False)
            yield LLMResponse(text="", is_first_token=False, is_final=True)

        results = []
        async for res in engine.synthesize_stream(mock_text_stream()):
            results.append(res)
            
        assert len(results) == 3
        assert results[0].text == "Hello"
        assert results[0].is_final is False
        assert results[1].text == " world"
        assert results[1].is_final is False
        assert results[2].text == "Hello world"
        assert results[2].is_final is True

    @pytest.mark.asyncio
    @patch("edge_tts.Communicate")
    async def test_edge_tts_synthesize_stream(self, mock_comm_class):
        mock_comm = MagicMock()
        async def mock_stream():
            yield {"type": "audio", "data": b"chunk"}
        mock_comm.stream = mock_stream
        mock_comm_class.return_value = mock_comm

        config = TTSConfig(backend=TTSBackend.EDGE_TTS)
        engine = EdgeTTSEngine(config)

        async def mock_text_stream():
            yield LLMResponse(text="Hello.", is_first_token=True, is_final=False)
            yield LLMResponse(text="How", is_first_token=False, is_final=False)
            yield LLMResponse(text="are", is_first_token=False, is_final=False)
            yield LLMResponse(text="you", is_first_token=False, is_final=False)
            yield LLMResponse(text="doing", is_first_token=False, is_final=False)
            yield LLMResponse(text="today", is_first_token=False, is_final=False)
            yield LLMResponse(text="friend", is_first_token=False, is_final=False)
            yield LLMResponse(text="", is_first_token=False, is_final=True)

        results = []
        async for res in engine.synthesize_stream(mock_text_stream()):
            results.append(res)

        assert len(results) == 3
        assert results[0].text == "Hello."
        assert results[1].text == "Howareyoudoingtoday"
        assert results[2].text == "friend"

    @pytest.mark.asyncio
    async def test_pyttsx3_synthesize_stream(self):
        config = TTSConfig(backend=TTSBackend.PYTTSX3)
        engine = Pyttsx3Engine(config)

        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        def mock_save_to_file(text, filename):
            with open(filename, "wb") as f:
                f.write(b"wav_audio_data")

        mock_engine.save_to_file = mock_save_to_file
        mock_engine.runAndWait = MagicMock()

        async def mock_text_stream():
            yield LLMResponse(text="Offline", is_first_token=True, is_final=False)
            yield LLMResponse(text=" synthesis", is_first_token=False, is_final=False)
            yield LLMResponse(text="", is_first_token=False, is_final=True)

        results = []
        async for res in engine.synthesize_stream(mock_text_stream()):
            results.append(res)

        assert len(results) == 1
        assert results[0].text == "Offline synthesis"
        assert results[0].audio_data == b"wav_audio_data"
        assert results[0].is_final is True

    @pytest.mark.asyncio
    async def test_tts_engine_stop(self):
        config = TTSConfig(backend=TTSBackend.MOCK)
        engine = create_tts_engine(config)
        
        async def long_task():
            await asyncio.sleep(2.0)
            return "done"
            
        task = asyncio.create_task(long_task())
        engine._current_stream = task
        
        await engine.stop()
        assert task.cancelled() or task.done()


class TestLLMGenerate:
    @pytest.mark.asyncio
    async def test_mock_llm_generate(self):
        config = LLMConfig(backend=LLMBackend.MOCK)
        client = create_llm_client(config)
        client._mock_delay = 0.01
        
        res = await client.generate("Hello")
        assert "Hello" in res
        assert "mock LLM response" in res

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_openai_llm_generate(self, mock_openai_class):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock(content="OpenAI")
        chunk1.choices[0].finish_reason = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock(content=" answer")
        chunk2.choices[0].finish_reason = "stop"

        async def mock_async_iterator():
            yield chunk1
            yield chunk2

        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=mock_async_iterator())
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_completions

        config = LLMConfig(backend=LLMBackend.OPENAI, openai_api_key="fake")
        client = create_llm_client(config)
        res = await client.generate("test prompt")
        assert res == "OpenAI answer"

