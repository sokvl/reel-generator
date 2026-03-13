import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_PATH = os.environ.get(
    "FONT_PATH",
    os.path.join(
        BASE_DIR,
        "assets",
        "TikTok_Sans",
        "TikTokSans-VariableFont_opsz,slnt,wdth,wght.ttf"
    )
)

IMAGE_MODEL="black-forest-labs/FLUX.1-dev"
IMAGE_GUIDANCE_SCALE=3.5
IMAGE_INFERENCE_STEPS=20
IMAGE_MAX_SEQ_LEN=256

VIDEO_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
VIDEO_GUIDANCE_SCALE = 6.0
VIDEO_NUM_FRAMES = 81
VIDEO_FPS = 16
CAPTION_WORDS_PER_LINE = 7


#main.py
TTS_SAMPLE_RATE = 24000
FPS = 24
CROSSFADE_SEC = 0.5

#editor.py
WORD_MIN_DURATION=1.0
WORD_FADE_DURATION=0.3
