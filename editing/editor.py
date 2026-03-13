import os
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.MultiplySpeed import MultiplySpeed
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)

from config_file import FONT_PATH, WORD_FADE_DURATION
from editing.whisper_transcriber import WhisperTranscriber

ACTIVE_WORD_COLOUR = "yellow"


class VideoAssembler:
    def __init__(
        self,
        voiceovers_dir: str = "voiceovers/",
        videos_dir: str = "videos/",
        crossfade_sec: float = 0.5,
        fps: int = 60,
        out_path: str = "final_video.mp4",
        debug: bool = True,
        words_per_line: int = 7,
        transcriber: WhisperTranscriber | None = None,
    ):
        self.voiceovers_dir = voiceovers_dir
        self.videos_dir = videos_dir
        self.crossfade_sec = crossfade_sec
        self.fps = fps
        self.out_path = out_path
        self.debug = debug
        self.words_per_line = words_per_line

        self.transcriber = transcriber or WhisperTranscriber(
            model_size="base", language="en"
        )

        self.audio_paths: list[str] = []
        self.video_paths: list[str] = []
        self.clips: list = []
        self.audio: list = []
        self.scenes: list = []
        self.clips_with_transitions: list = []
        self.final = None

    def _load_paths_and_script(self):
        self.audio_paths = sorted(os.listdir(self.voiceovers_dir))
        self.video_paths = sorted(os.listdir(self.videos_dir))

    def _build_clips(self):
        self.clips = [
            VideoFileClip(f"{self.videos_dir}{p}") for p in self.video_paths
        ]
        self.clips = self.clips[: len(self.audio_paths)]
        self.audio = [
            AudioFileClip(f"{self.voiceovers_dir}{p}") for p in self.audio_paths
        ]

    def _render_karaoke_line(
        self,
        line_words: list[str],
        active_index: int,
        clip_width: int,
    ) -> TextClip:
        """
        Renders a full caption line where the active word is highlighted.
        MoviePy TextClip does not support per-word colour, so we split the
        line into three parts: prefix, active word, suffix — and stack them
        horizontally using CompositeVideoClip with calculated x offsets.
        """
        caption_width = int(clip_width * 0.75)
        full_text = " ".join(line_words)
        active_text = line_words[active_index]

        base = TextClip(
            text=full_text,
            font=FONT_PATH,
            font_size=38,
            color="#cccccc",
            stroke_color="black",
            stroke_width=3,
            method="caption",
            text_align="center",
            size=(caption_width, None),
        )

        highlight = TextClip(
            text=active_text,
            font=FONT_PATH,
            font_size=42,
            color=ACTIVE_WORD_COLOUR,
            stroke_color="black",
            stroke_width=3,
            method="label",
        )

        return base, highlight

    def _create_synced_captions(
        self, audio_path: str, clip_width: int, clip_height: int
    ) -> list:
        events = self.transcriber.get_karaoke_lines(
            audio_path, self.words_per_line
        )

        if not events:
            if self.debug:
                print(f"[Captions] No words found for: {audio_path}")
            return []

        text_clips = []

        for event in events:
            duration = max(0.05, event["end"] - event["start"])
            active_text = event["line_words"][event["active_index"]]
            word_clip = (
                TextClip(
                    text=active_text,
                    font=FONT_PATH,
                    font_size=52,
                    color=ACTIVE_WORD_COLOUR,
                    stroke_color="black",
                    stroke_width=4,
                    method="caption",
                    text_align="center",
                    size=(int(clip_width * 0.7), None),
                    margin=(20, 20),
                )
                .with_duration(duration)
                .with_start(event["start"])
                .with_position(("center", "center"))
            )
            text_clips.append(word_clip)

        if self.debug:
            print(
                f"[Captions] {len(events)} events for '{os.path.basename(audio_path)}'"
            )

        return text_clips

    def _build_scenes(self):
        scenes = []
        for i in range(len(self.clips)):
            clip = self.clips[i]
            audio_clip = self.audio[i]
            audio_path_full = f"{self.voiceovers_dir}{self.audio_paths[i]}"

            if abs(clip.duration - audio_clip.duration) > 1e-3:
                speed_factor = clip.duration / audio_clip.duration
                clip = MultiplySpeed(factor=speed_factor).apply(clip)

            text_clips = self._create_synced_captions(audio_path_full, clip.w, clip.h)
            if text_clips:
                clip = CompositeVideoClip([clip] + text_clips, size=clip.size)

            clip = clip.with_audio(audio_clip.with_duration(clip.duration))
            scenes.append(clip)

        self.scenes = scenes

    def _add_transitions(self):
        clips_with_transitions = []
        for idx, clip in enumerate(self.scenes):
            if idx == 0:
                clips_with_transitions.append(clip)
            else:
                clips_with_transitions.append(CrossFadeIn(self.crossfade_sec).apply(clip))

        if self.debug:
            for i, clip in enumerate(clips_with_transitions):
                print(f"Clip {i}: audio = {clip.audio is not None}")

        self.clips_with_transitions = clips_with_transitions

    def _concat_and_write(self):
        final = concatenate_videoclips(
            self.clips_with_transitions,
            method="compose",
            padding=-self.crossfade_sec,
        )
        final.write_videofile(
            self.out_path, fps=self.fps, codec="libx264", audio_codec="aac"
        )
        self.final = final

    def _cleanup(self):
        for c in self.scenes:
            c.close()
        if self.final is not None:
            self.final.close()

    def run(self):
        self._load_paths_and_script()
        self._build_clips()
        self._build_scenes()
        self._add_transitions()
        self._concat_and_write()
        self._cleanup()