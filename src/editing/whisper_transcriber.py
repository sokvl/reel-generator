import gc

import torch
import whisper


class WhisperTranscriber:
    """
    Lazy-loaded Whisper transcription service.
    The model is loaded only on first use (not during __init__).
    Can be injected as a dependency into VideoAssembler.
    """

    def __init__(self, model_size: str = "medium", language: str | None = None):
        self.model_size = model_size
        self.language = language
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"[WhisperTranscriber] Loading model '{self.model_size}'...")
            self._model = whisper.load_model(self.model_size)
            print("[WhisperTranscriber] Ready.")
        return self._model

    def transcribe(self, audio_path: str) -> list[dict]:
        """
        Returns word-level timestamps from Whisper.
        Each entry: {"word": str, "start": float, "end": float}
        """
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language=self.language,
        )

        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                })
        return words

    def unload(self):
        """Free Whisper model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[WhisperTranscriber] Model unloaded.")

    def get_karaoke_lines(
        self,
        audio_path: str,
        words_per_line: int = 7,
    ) -> list[dict]:
        """
        Groups words into lines and returns per-word highlight events.
        """
        words = self.transcribe(audio_path)
        if not words:
            return []

        events = []
        for line_start in range(0, len(words), words_per_line):
            line = words[line_start : line_start + words_per_line]
            line_texts = [w["word"] for w in line]

            for idx, word in enumerate(line):
                # The word is "active" until the next word begins.
                next_start = line[idx + 1]["start"] if idx + 1 < len(line) else word["end"]
                events.append({
                    "line_words": line_texts,
                    "active_index": idx,
                    "start": word["start"],
                    "end": next_start,
                })

        return events