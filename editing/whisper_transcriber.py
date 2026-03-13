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
        Returns a list of words with timestamps:
        [{"word": "hello", "start": 0.0, "end": 0.4}, ...]
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

    def group_words(self, words: list[dict], words_per_group: int = 4) -> list[dict]:
        """
        Groups words into chunks of N while preserving the group's start and end timestamps.
        Returns: [{"text": "...", "start": float, "end": float}, ...]
        """
        groups = []
        for i in range(0, len(words), words_per_group):
            chunk = words[i : i + words_per_group]
            groups.append({
                "text": " ".join(w["word"] for w in chunk),
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
            })
        return groups

    def get_caption_groups(
        self, audio_path: str, words_per_group: int = 4
    ) -> list[dict]:
        """Shortcut: transcribe and group immediately."""
        words = self.transcribe(audio_path)
        return self.group_words(words, words_per_group)