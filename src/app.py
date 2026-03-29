"""
Streamlit interface for the AI video generation pipeline.

Usage:
    streamlit run app.py
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from config_file import (
    CROSSFADE_SEC,
    FPS,
    IMAGE_GUIDANCE_SCALE,
    IMAGE_INFERENCE_STEPS,
    IMAGE_MAX_SEQ_LEN,
    IMAGE_MODEL,
    TTS_SAMPLE_RATE,
    VIDEO_FPS,
    VIDEO_GUIDANCE_SCALE,
    VIDEO_MODEL,
    VIDEO_NUM_FRAMES,
    WORD_FADE_DURATION,
    WORD_MIN_DURATION,
)
from prompt.prompts.prompt_lib import (
    IDEA_BRAINSTORMING_PROMPT_HEADER,
    SCRIPT_GENERATION_PROMPT_HEAD,
    VIDEO_DESCRIPTION_PROMPT_TAIL,
)

log = logging.getLogger(__name__)

RUNS_DIR = Path("runs")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    path: Path
    manifest: dict

    @property
    def title(self) -> str:
        return self.manifest.get("title", self.path.name)

    @property
    def created_at(self) -> str:
        return self.manifest.get("created_at", "unknown")

    @property
    def signature(self) -> str:
        return self.manifest.get("signature", "")

    @property
    def meta(self) -> dict:
        return self.manifest.get("meta", {})

    @property
    def artifacts(self) -> list[dict]:
        return self.manifest.get("artifacts", [])

    def artifacts_by_kind(self, kind: str) -> list[dict]:
        return [a for a in self.artifacts if a.get("kind") == kind]

    def artifact_by_kind(self, kind: str) -> dict | None:
        matches = self.artifacts_by_kind(kind)
        return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_config() -> dict:
    return {
        "IMAGE_MODEL": IMAGE_MODEL,
        "IMAGE_GUIDANCE_SCALE": IMAGE_GUIDANCE_SCALE,
        "IMAGE_INFERENCE_STEPS": IMAGE_INFERENCE_STEPS,
        "IMAGE_MAX_SEQ_LEN": IMAGE_MAX_SEQ_LEN,
        "VIDEO_MODEL": VIDEO_MODEL,
        "VIDEO_GUIDANCE_SCALE": VIDEO_GUIDANCE_SCALE,
        "VIDEO_NUM_FRAMES": VIDEO_NUM_FRAMES,
        "VIDEO_FPS": VIDEO_FPS,
        "TTS_SAMPLE_RATE": TTS_SAMPLE_RATE,
        "FPS": FPS,
        "CROSSFADE_SEC": CROSSFADE_SEC,
        "WORD_MIN_DURATION": WORD_MIN_DURATION,
        "WORD_FADE_DURATION": WORD_FADE_DURATION,
    }


def _default_prompts() -> dict:
    return {
        "idea_header": IDEA_BRAINSTORMING_PROMPT_HEADER,
        "script_header": SCRIPT_GENERATION_PROMPT_HEAD,
        "video_tail": VIDEO_DESCRIPTION_PROMPT_TAIL,
    }


def _load_manifest(run_dir: Path) -> dict | None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_run(run_dir: Path) -> RunInfo | None:
    manifest = _load_manifest(run_dir)
    if manifest is None:
        return None
    return RunInfo(path=run_dir, manifest=manifest)


def _recent_runs(n: int = 20) -> list[RunInfo]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in RUNS_DIR.iterdir():
        if not p.is_dir():
            continue
        run = _load_run(p)
        if run is not None:
            runs.append(run)
    return sorted(runs, key=lambda r: r.created_at, reverse=True)[:n]


def _load_subtitles(run_dir: Path) -> dict | None:
    path = run_dir / "artifacts" / "subtitles.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse subtitles.json: {exc}")
        return None


def _read_text_artifact(artifact: dict) -> str | None:
    path = Path(artifact.get("path", ""))
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _build_pipeline_args(
    mode: str,
    topic: str,
    w: int,
    h: int,
    seed: int | None,
    prompt_file: str | None,
) -> object:
    """Build a plain namespace matching what generate_new_run() expects."""

    class _Args:
        pass

    args = _Args()
    args.mode = mode
    args.topic = topic
    args.w = w
    args.h = h
    args.seed = seed
    args.prompt_file = prompt_file
    return args


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Video Generator",
    layout="wide",
)

st.title("AI Video Generator")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "config" not in st.session_state:
    st.session_state.config = _default_config()

if "prompts" not in st.session_state:
    st.session_state.prompts = _default_prompts()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_run, tab_runs, tab_config, tab_prompts, tab_subtitles = st.tabs(
    ["Run", "Runs", "Config", "Prompts", "Subtitles"]
)

# ===========================================================================
# TAB: RUN
# ===========================================================================
with tab_run:
    st.header("Run pipeline")

    col_left, col_right = st.columns(2)

    with col_left:
        mode = st.selectbox(
            "Mode",
            ["full", "vid"],
            help=(
                "full — LLM generates idea and script from a topic.\n"
                "vid  — you supply a pre-written script file."
            ),
        )
        topic = st.text_area(
            "Topic",
            value="A startup that uses AI to improve mental health.",
            height=90,
            disabled=(mode == "vid"),
        )

    with col_right:
        w = st.number_input("Width (px)", value=480, step=32, min_value=64)
        h = st.number_input("Height (px)", value=832, step=32, min_value=64)
        raw_seed = st.number_input("Seed (0 = random)", value=0, min_value=0, step=1)
        seed = int(raw_seed) if raw_seed > 0 else None

    prompt_file_path: str | None = None
    if mode == "vid":
        uploaded = st.file_uploader("Script file (.txt or .json)", type=["txt", "json"])
        if uploaded is not None:
            tmp = Path("_tmp_uploads")
            tmp.mkdir(exist_ok=True)
            dest = tmp / uploaded.name
            dest.write_bytes(uploaded.read())
            prompt_file_path = str(dest)

    st.divider()

    assemble_only = st.checkbox("Assemble from existing run (skip generation)")
    selected_run_path: str | None = None

    if assemble_only:
        recent = _recent_runs()
        if recent:
            options = {f"{r.title}  —  {r.created_at}": str(r.path) for r in recent}
            chosen_name = st.selectbox(
                "Select run", list(options.keys()), key="run_tab_select_run"
            )
            selected_run_path = options[chosen_name]
        else:
            selected_run_path = st.text_input(
                "Run folder path", placeholder="runs/topic_abc123_130325"
            )

    st.divider()

    if st.button("Start", type="primary", use_container_width=True):
        if mode == "vid" and not prompt_file_path:
            st.error("Upload a script file to use 'vid' mode.")
            st.stop()
        if assemble_only and not selected_run_path:
            st.error("Specify a run folder to assemble from.")
            st.stop()

        from main import assemble_from_existing, generate_new_run

        try:
            if assemble_only:
                with st.spinner("Assembling video..."):
                    out = assemble_from_existing(Path(selected_run_path))
            else:
                args = _build_pipeline_args(
                    mode, topic, int(w), int(h), seed, prompt_file_path
                )
                with st.spinner("Running pipeline — this may take several minutes."):
                    out = generate_new_run(args)

            st.success(f"Done. Output: {out}")
            _, phone_col, _ = st.columns([2, 1, 2])
            with phone_col:
                st.video(out)

        except FileNotFoundError as exc:
            st.error(f"Missing file or directory: {exc}")
        except ValueError as exc:
            st.error(f"Validation error: {exc}")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.exception(exc)

# ===========================================================================
# TAB: RUNS
# ===========================================================================
with tab_runs:
    st.header("Run history")

    all_runs = _recent_runs()

    if not all_runs:
        st.info(f"No runs found in ./{RUNS_DIR}/")
    else:
        run_options = {
            f"{r.title}  —  {r.created_at}  [{r.signature}]": r
            for r in all_runs
        }
        chosen_label = st.selectbox(
            "Select run", list(run_options.keys()), key="runs_tab_select"
        )
        run = run_options[chosen_label]

        st.divider()

        # --- overview ---
        meta_col, video_col = st.columns([2, 1])

        with meta_col:
            st.subheader("Overview")
            m1, m2, m3 = st.columns(3)
            m1.metric("Title", run.title)
            m2.metric("Created", run.created_at)
            m3.metric("Signature", run.signature)

            if run.meta:
                st.caption("Run parameters")
                st.json(run.meta, expanded=True)

        with video_col:
            final = run.artifact_by_kind("final_video")
            if final:
                video_path = Path(final["path"])
                if video_path.is_file():
                    st.subheader("Final video")
                    st.video(str(video_path))
                else:
                    st.warning(f"Video file not found at: {video_path}")
            else:
                st.info("No final video recorded for this run.")

        st.divider()

        # --- prompts ---
        st.subheader("Prompts used")

        prompt_artifacts = [
            a for a in run.artifacts if a.get("kind") == "text:prompts"
        ]

        if not prompt_artifacts:
            st.info("No prompt files recorded for this run.")
        else:
            for artifact in prompt_artifacts:
                identity = artifact.get("extra", {}).get("identity", artifact["path"])
                content = _read_text_artifact(artifact)
                with st.expander(identity):
                    if content:
                        st.text(content)
                    else:
                        st.warning(f"File not found: {artifact['path']}")

        st.divider()

        # --- narration per scene ---
        st.subheader("Narration per scene")

        subs = _load_subtitles(run.path)
        if subs:
            for scene_key, text in subs.items():
                st.markdown(f"**Scene {scene_key}**")
                st.write(text)
        else:
            st.info("No subtitles found for this run.")

        st.divider()

        # --- full artifact log ---
        with st.expander("Full artifact log"):
            st.json(run.artifacts)

# ===========================================================================
# TAB: CONFIG
# ===========================================================================
with tab_config:
    st.header("Configuration")
    st.caption(
        "Changes apply to this session only. "
        "To make them permanent, update config_file.py directly."
    )

    cfg = st.session_state.config

    st.subheader("Image model")
    c1, c2 = st.columns(2)
    with c1:
        cfg["IMAGE_MODEL"] = st.text_input("Model ID", cfg["IMAGE_MODEL"])
        cfg["IMAGE_INFERENCE_STEPS"] = st.number_input(
            "Inference steps", value=cfg["IMAGE_INFERENCE_STEPS"], min_value=1, step=1
        )
    with c2:
        cfg["IMAGE_GUIDANCE_SCALE"] = st.number_input(
            "Guidance scale", value=cfg["IMAGE_GUIDANCE_SCALE"], step=0.1, format="%.1f"
        )
        cfg["IMAGE_MAX_SEQ_LEN"] = st.number_input(
            "Max sequence length", value=cfg["IMAGE_MAX_SEQ_LEN"], min_value=64, step=32
        )

    st.subheader("Video model")
    c1, c2 = st.columns(2)
    with c1:
        cfg["VIDEO_MODEL"] = st.text_input("Video model ID", cfg["VIDEO_MODEL"])
        cfg["VIDEO_NUM_FRAMES"] = st.number_input(
            "Num frames", value=cfg["VIDEO_NUM_FRAMES"], min_value=1, step=1
        )
    with c2:
        cfg["VIDEO_GUIDANCE_SCALE"] = st.number_input(
            "Video guidance scale", value=cfg["VIDEO_GUIDANCE_SCALE"], step=0.1, format="%.1f"
        )
        cfg["VIDEO_FPS"] = st.number_input(
            "Output FPS", value=cfg["VIDEO_FPS"], min_value=1, step=1
        )

    st.subheader("Audio and assembly")
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg["TTS_SAMPLE_RATE"] = st.number_input(
            "TTS sample rate (Hz)", value=cfg["TTS_SAMPLE_RATE"], step=1000
        )
        cfg["FPS"] = st.number_input(
            "Assembly FPS", value=cfg["FPS"], min_value=1, step=1
        )
    with c2:
        cfg["CROSSFADE_SEC"] = st.number_input(
            "Crossfade (s)", value=cfg["CROSSFADE_SEC"], step=0.05, format="%.2f"
        )
    with c3:
        cfg["WORD_MIN_DURATION"] = st.number_input(
            "Caption min duration (s)", value=cfg["WORD_MIN_DURATION"], step=0.1, format="%.1f"
        )
        cfg["WORD_FADE_DURATION"] = st.number_input(
            "Caption fade (s)", value=cfg["WORD_FADE_DURATION"], step=0.05, format="%.2f"
        )

    st.divider()

    col_reset, col_export = st.columns(2)
    with col_reset:
        if st.button("Reset to defaults", use_container_width=True):
            del st.session_state["config"]
            st.rerun()
    with col_export:
        st.download_button(
            "Export config as JSON",
            data=json.dumps(cfg, indent=2),
            file_name="config_override.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Current config (JSON)"):
        st.json(cfg)

# ===========================================================================
# TAB: PROMPTS
# ===========================================================================
with tab_prompts:
    st.header("Prompt editor")
    st.caption(
        "Edit the system prompts used by the LLM. "
        "Changes apply to this session. "
        "To persist, update prompt/prompts/prompt_lib.py directly."
    )

    p = st.session_state.prompts

    st.subheader("Idea generation — system prompt")
    p["idea_header"] = st.text_area(
        "Idea system prompt",
        value=p["idea_header"],
        height=260,
        label_visibility="collapsed",
    )

    st.subheader("Script generation — system prompt")
    p["script_header"] = st.text_area(
        "Script system prompt",
        value=p["script_header"],
        height=260,
        label_visibility="collapsed",
    )

    st.subheader("Video prompt — style tail")
    st.caption("Appended to every video diffusion prompt.")
    p["video_tail"] = st.text_area(
        "Video tail",
        value=p["video_tail"],
        height=80,
        label_visibility="collapsed",
    )

    st.divider()

    col_reset, col_export = st.columns(2)
    with col_reset:
        if st.button("Reset prompts to defaults", use_container_width=True):
            del st.session_state["prompts"]
            st.rerun()
    with col_export:
        st.download_button(
            "Export prompts as JSON",
            data=json.dumps(p, indent=2, ensure_ascii=False),
            file_name="prompts_override.json",
            mime="application/json",
            use_container_width=True,
        )

# ===========================================================================
# TAB: SUBTITLES
# ===========================================================================
with tab_subtitles:
    st.header("Subtitle viewer and editor")

    source = st.radio("Load from", ["Recent run", "Upload file"], horizontal=True)

    subs: dict | None = None

    if source == "Recent run":
        recent = _recent_runs()
        if not recent:
            st.info(f"No runs found in ./{RUNS_DIR}/")
        else:
            options = {f"{r.title}  —  {r.created_at}": r for r in recent}
            chosen_label = st.selectbox(
                "Select run", list(options.keys()), key="subs_tab_select_run"
            )
            subs = _load_subtitles(options[chosen_label].path)
            if subs is None:
                st.warning("No subtitles.json found in the selected run.")
    else:
        uploaded_subs = st.file_uploader("Upload subtitles.json", type=["json"])
        if uploaded_subs is not None:
            try:
                subs = json.loads(uploaded_subs.read().decode("utf-8"))
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")

    if subs is not None:
        st.divider()
        st.subheader("Edit")

        edited: dict = {}
        for key, text in subs.items():
            edited[key] = st.text_area(
                f"Scene {key}",
                value=str(text),
                height=80,
                key=f"sub_{key}",
            )

        st.divider()

        col_dl, col_preview = st.columns(2)
        with col_dl:
            st.download_button(
                "Export edited subtitles",
                data=json.dumps(edited, indent=2, ensure_ascii=False),
                file_name="subtitles_edited.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_preview:
            with st.expander("JSON preview"):
                st.json(edited)