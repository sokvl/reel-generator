from pathlib import Path
import argparse
import gc
from config_file import TTS_SAMPLE_RATE
import soundfile as sf
from editing.editor import VideoAssembler
from helpers.load_prompt import load_prompt, load_json
from helpers.parse_scripts import parse_script_robust
from prompt.txt_prompt import prompt_factory
from script.scripter import LLMFacade
from archivist.archivist import Archivist
from pipelines.video_pipeline_t2i import VideoPipelineT2I
from kokoro import KPipeline


def get_run_paths(run_base: Path):
    run_base = run_base.expanduser().resolve()
    return {
        "base": run_base,
        "voiceovers": run_base / "voiceovers",
        "videos": run_base / "videos",
        "subs": run_base / "artifacts" / "subtitles.json",
        "out": run_base / "final_video.mp4",
    }

def ensure_run_inputs(paths: dict):
    if not paths["voiceovers"].is_dir():
        raise FileNotFoundError(f"Audio dir missing: {paths['voiceovers']}")
    if not paths["videos"].is_dir():
        raise FileNotFoundError(f"Video dir missing: {paths['videos']}")
    if not paths["subs"].is_file():
        raise FileNotFoundError(f"Subtitles file missing: {paths['subs']}")

def assemble(paths: dict, subtitles: dict, fps: int = 24, crossfade: float = 0.5):
    assembler = VideoAssembler(
        voiceovers_dir=str(paths["voiceovers"]) + "/", 
        videos_dir=str(paths["videos"]) + "/",
        subtitles=subtitles,
        crossfade_sec=crossfade,
        fps=fps,
        out_path=str(paths["out"]),
        debug=True,
    )
    assembler.run()
    return str(paths["out"])


def assemble_from_existing(run_base: Path) -> str:
    paths = get_run_paths(run_base)
    ensure_run_inputs(paths)
    subs = load_json(str(paths["subs"]))
    return assemble(paths, subs)

def generate_new_run(args) -> str:
    w, h, mode, topic, prompt_file = args.w, args.h, args.mode, args.topic, args.prompt_file
    seed = getattr(args, "seed", None)

    arch = Archivist(sig_seed=seed)
    arch.start_run(title=topic or "untitled", meta={"mode": mode, "w": w, "h": h, "prompt_file": prompt_file})

    llm = LLMFacade()
    if mode == "vid":
        prompt = load_prompt(prompt_file)
        arch.save_text("prompt_input_vid", prompt, subdir="prompts", ext=".txt")
    elif mode == "full":
        llm.set_prompts(system_prompt=str(prompt_factory("idea")), user_prompt=topic)
        idea = llm.run()
        arch.save_text("idea", idea, subdir="prompts", ext=".txt")
        llm.switch_task(system_prompt=str(prompt_factory("script")), user_prompt=idea)
        prompt = llm.run()
        arch.save_text("script_raw", prompt, subdir="prompts", ext=".txt")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    del llm; gc.collect()

    paths = get_run_paths(Path(arch.path("base")))
    voiceovers_dir = paths["voiceovers"]; voiceovers_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = paths["videos"]; videos_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_script_robust(prompt)
    video_prompt = prompt_factory("video")
    video_pipeline = VideoPipelineT2I(w=w, h=h, out_dir=str(videos_dir))
    tts = KPipeline(lang_code="a")

    subtitles = {}
    for i, scene in enumerate(parsed["scenes"].values(), start=1):
        visual = list(scene["visuals"].values())[0]
        narr = scene["narration"][0] if scene["narration"] else ""
        subtitles[i] = narr
        video_prompt.core = visual
        video_pipeline.diffuse(str(video_prompt), id=i)
        try:
            _, _, audio = next(tts(narr, voice="af_heart"))
            wav_path = voiceovers_dir / f"scene_{i}.wav"
            sf.write(str(wav_path), audio, TTS_SAMPLE_RATE)
            arch.record_video(kind="voiceover", video_path=str(wav_path), extra={"scene": i})
        except StopIteration:
            print(f"⚠️  Audio not found {i}")

    arch.save_json("subtitles", subtitles, subdir="artifacts")

    out_path = assemble(paths, subtitles)
    arch.finalize(final_video_path=out_path)
    return out_path

def parse_args():
    p = argparse.ArgumentParser("main.py")
    p.add_argument("--w", type=int, default=480)
    p.add_argument("--h", type=int, default=832)
    p.add_argument("--mode", choices=["full", "vid"], default="full")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--topic", type=str, default="A startup that uses AI to improve mental health.")
    p.add_argument("--assemble_from", type=str, default=None, help="runs/<...> to assemble only")
    return p.parse_args()

def validate_args(a):
    if a.assemble_from:
        return
    if a.w <= 0 or a.h <= 0:
        raise ValueError("Width and height must be positive.")
    if a.mode == "vid" and not a.prompt_file:
        raise ValueError("In 'vid' mode you must pass --prompt_file.")

if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    if args.assemble_from:
        print(assemble_from_existing(Path(args.assemble_from)))
    else:
        print(generate_new_run(args))