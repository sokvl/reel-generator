"""
Tests for WhisperTranscriber — whisper module is patched at the attribute level
so no model download or GPU is needed.
"""
import unittest.mock as mock
import pytest

from editing.whisper_transcriber import WhisperTranscriber

PATCH_TARGET = "editing.whisper_transcriber.whisper"


@pytest.fixture
def transcriber():
    return WhisperTranscriber(model_size="base", language="en")


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

class TestWhisperTranscriberLazyLoad:
    def test_model_not_loaded_at_init(self, transcriber):
        assert transcriber._model is None

    def test_model_loaded_on_first_access(self, transcriber):
        with mock.patch(PATCH_TARGET) as stub:
            _ = transcriber.model
            stub.load_model.assert_called_once_with("base")

    def test_model_loaded_only_once(self, transcriber):
        with mock.patch(PATCH_TARGET) as stub:
            _ = transcriber.model
            _ = transcriber.model
            stub.load_model.assert_called_once()


# ---------------------------------------------------------------------------
# transcribe()
# ---------------------------------------------------------------------------

class TestWhisperTranscriberTranscribe:
    def _patch_model(self, transcriber, words):
        """words: list of (word_str, start, end)"""
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = {
            "segments": [{"words": [{"word": w, "start": s, "end": e} for w, s, e in words]}]
        }
        transcriber._model = fake_model
        return fake_model

    def test_returns_word_list(self, transcriber):
        self._patch_model(transcriber, [(" hello", 0.0, 0.5), (" world", 0.5, 1.0)])
        result = transcriber.transcribe("fake.wav")
        assert result == [
            {"word": "hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]

    def test_strips_whitespace_from_words(self, transcriber):
        self._patch_model(transcriber, [("  padded  ", 0.0, 1.0)])
        result = transcriber.transcribe("fake.wav")
        assert result[0]["word"] == "padded"

    def test_empty_segments_returns_empty_list(self, transcriber):
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = {"segments": []}
        transcriber._model = fake_model
        assert transcriber.transcribe("fake.wav") == []

    def test_multiple_segments_merged(self, transcriber):
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = {
            "segments": [
                {"words": [{"word": "a", "start": 0.0, "end": 0.5}]},
                {"words": [{"word": "b", "start": 0.5, "end": 1.0}]},
            ]
        }
        transcriber._model = fake_model
        result = transcriber.transcribe("fake.wav")
        assert len(result) == 2
        assert result[1]["word"] == "b"


# ---------------------------------------------------------------------------
# get_karaoke_lines()
# ---------------------------------------------------------------------------

class TestWhisperTranscriberGetKaraokeLines:
    def _patch_model(self, transcriber, words):
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = {
            "segments": [{"words": [{"word": w, "start": s, "end": e} for w, s, e in words]}]
        }
        transcriber._model = fake_model

    def test_single_line_produces_one_event_per_word(self, transcriber):
        words = [(f"w{i}", float(i), float(i + 1)) for i in range(3)]
        self._patch_model(transcriber, words)
        events = transcriber.get_karaoke_lines("x.wav", words_per_line=7)
        assert len(events) == 3

    def test_active_index_increments_within_line(self, transcriber):
        words = [(f"w{i}", float(i), float(i + 1)) for i in range(3)]
        self._patch_model(transcriber, words)
        events = transcriber.get_karaoke_lines("x.wav", words_per_line=7)
        assert [e["active_index"] for e in events] == [0, 1, 2]

    def test_line_wraps_every_n_words(self, transcriber):
        words = [(f"w{i}", float(i), float(i + 1)) for i in range(10)]
        self._patch_model(transcriber, words)
        events = transcriber.get_karaoke_lines("x.wav", words_per_line=4)
        assert len(events[0]["line_words"]) == 4
        assert events[4]["active_index"] == 0  # first word of second line

    def test_last_word_end_time_used_when_no_next(self, transcriber):
        self._patch_model(transcriber, [("only", 0.0, 2.5)])
        events = transcriber.get_karaoke_lines("x.wav", words_per_line=7)
        assert events[0]["end"] == 2.5

    def test_non_last_word_end_is_next_words_start(self, transcriber):
        self._patch_model(transcriber, [("a", 0.0, 0.4), ("b", 0.5, 1.0)])
        events = transcriber.get_karaoke_lines("x.wav", words_per_line=7)
        # First word ends when next word starts
        assert events[0]["end"] == 0.5

    def test_empty_audio_returns_empty_list(self, transcriber):
        fake_model = mock.MagicMock()
        fake_model.transcribe.return_value = {"segments": []}
        transcriber._model = fake_model
        assert transcriber.get_karaoke_lines("x.wav") == []


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------

class TestWhisperTranscriberUnload:
    def test_unload_clears_model(self, transcriber):
        transcriber._model = mock.MagicMock()
        with mock.patch("torch.cuda.is_available", return_value=False):
            transcriber.unload()
        assert transcriber._model is None

    def test_unload_is_safe_when_not_loaded(self, transcriber):
        with mock.patch("torch.cuda.is_available", return_value=False):
            transcriber.unload()  # should not raise

    def test_unload_clears_cuda_cache_when_available(self, transcriber):
        transcriber._model = mock.MagicMock()
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.empty_cache") as mock_cache:
            transcriber.unload()
        mock_cache.assert_called_once()
