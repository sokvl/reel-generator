from pathlib import Path
import argparse
import gc
import torch
from config_file import TTS_SAMPLE_RATE, FONT_PATH
import soundfile as sf
from editing.editor import VideoAssembler
from helpers.load_prompt import load_prompt, load_json
from helpers.parse_scripts import parse_script_robust
from prompt.txt_prompt import prompt_factory
from script.scripter import LLMFacade
from archivist.archivist import Archivist
from pipelines.video_pipeline_t2i import VideoPipelineT2I


def _flush_gpu():
    """Force-free all GPU memory between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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

def _generate_script_with_fallback(llm, idea: str, arch, max_retries: int = 3) -> str:
    """Generate script with fallback retry on JSON parsing failure."""
    for attempt in range(max_retries):
        llm.switch_task(system_prompt=str(prompt_factory("script")), user_prompt=idea)
        prompt = llm.run()
        arch.save_text(f"script_raw_attempt_{attempt+1}", prompt, subdir="prompts", ext=".txt")
        
        try:
            # This will raise ValueError with "[RETRY_NEEDED]" if JSON is invalid but retriable
            parsed = parse_script_robust(prompt, attempt=attempt+1, max_attempts=max_retries)
            # If successful, save as the final script
            arch.save_text("script_raw", prompt, subdir="prompts", ext=".txt")
            return prompt
        except ValueError as e:
            error_msg = str(e)
            print(f"⚠️  Script generation failed: {error_msg}")
            if "[RETRY_NEEDED]" in error_msg and attempt < max_retries - 1:
                print(f"🔄  Retrying script generation (attempt {attempt + 2}/{max_retries})...")
            else:
                # Final attempt failed or max retries exhausted
                raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts") from e
    
    raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts")

def generate_new_run(args) -> str:
    w, h, mode, topic, prompt_file = args.w, args.h, args.mode, args.topic, args.prompt_file
    seed = getattr(args, "seed", None)

    arch = Archivist(sig_seed=seed)
    arch.start_run(title=topic or "untitled", meta={"mode": mode, "w": w, "h": h, "prompt_file": prompt_file})

    # ── Phase 1: LLM – idea & script generation ──────────────────────
    llm = LLMFacade()
    if mode == "vid":
        prompt = load_prompt(prompt_file)
        arch.save_text("prompt_input_vid", prompt, subdir="prompts", ext=".txt")
    elif mode == "full":
        llm.set_prompts(system_prompt=str(prompt_factory("idea")), user_prompt=topic)
        idea = llm.run()
        arch.save_text("idea", idea, subdir="prompts", ext=".txt")
        # Use fallback mechanism for script generation
        prompt = _generate_script_with_fallback(llm, idea, arch, max_retries=3)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    llm.unload()
    del llm
    _flush_gpu()
    print("[Phase 1/4] LLM complete – VRAM freed")

    paths = get_run_paths(Path(arch.path("base")))
    voiceovers_dir = paths["voiceovers"]; voiceovers_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = paths["videos"]; videos_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_script_robust(prompt)

    # Pre-extract scene data (CPU only, no model needed)
    scenes_data = []
    subtitles = {}
    video_prompt = prompt_factory("video")
    for i, scene in enumerate(parsed["scenes"].values(), start=1):
        visual = list(scene["visuals"].values())[0]
        narr = scene["narration"][0] if scene["narration"] else ""
        subtitles[i] = narr
        video_prompt.core = visual
        scenes_data.append({"id": i, "visual_prompt": str(video_prompt), "narration": narr})

    # ── Phase 2: Video generation (Wan model) ────────────────────────
    video_pipeline = VideoPipelineT2I(w=w, h=h, out_dir=str(videos_dir))
    for sd in scenes_data:
        video_pipeline.diffuse(sd["visual_prompt"], id=sd["id"])
    del video_pipeline
    _flush_gpu()
    print("[Phase 2/4] Video generation complete – VRAM freed")

    # ── Phase 3: TTS audio generation (Kokoro) ───────────────────────
    from kokoro import KPipeline
    tts = KPipeline(lang_code="a")
    for sd in scenes_data:
        try:
            _, _, audio = next(tts(sd["narration"], voice="af_heart"))
            wav_path = voiceovers_dir / f"scene_{sd['id']}.wav"
            sf.write(str(wav_path), audio, TTS_SAMPLE_RATE)
            arch.record_video(kind="voiceover", video_path=str(wav_path), extra={"scene": sd["id"]})
        except StopIteration:
            print(f"⚠️  Audio not found {sd['id']}")
    del tts
    _flush_gpu()
    print("[Phase 3/4] TTS complete – VRAM freed")

    # ── Phase 4: Assembly (Whisper + MoviePy) ─────────────────────────
    arch.save_json("subtitles", subtitles, subdir="artifacts")
    out_path = assemble(paths, subtitles)
    arch.finalize(final_video_path=out_path)
    print("[Phase 4/4] Assembly complete")
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
    if a.mode == "full" and not (a.topic or "").strip():
        raise ValueError("--topic must not be empty in 'full' mode.")

def preflight_checks(a):
    """Verify filesystem and environment before loading any model."""
    errors = []

    if not Path(FONT_PATH).is_file():
        errors.append(f"Font file not found: {FONT_PATH}")

    if a.assemble_from:
        p = Path(a.assemble_from).expanduser().resolve()
        if not p.is_dir():
            errors.append(f"--assemble_from path does not exist: {p}")
    else:
        if a.mode == "vid" and a.prompt_file and not Path(a.prompt_file).is_file():
            errors.append(f"--prompt_file not found: {a.prompt_file}")

        if not torch.cuda.is_available():
            print("WARNING: CUDA not available — models will run on CPU (very slow).")

    if errors:
        for e in errors:
            print(f"[preflight] ERROR: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    preflight_checks(args)
    if args.assemble_from:
        print(assemble_from_existing(Path(args.assemble_from)))
    else:
        print(generate_new_run(args))