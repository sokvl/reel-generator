# Reel Generator

A pipeline for turning text prompts into short-form videos (9:16) by coordinating AI models for scripting, video synthesis, and voiceovers.

## Goal
To automate the creation of social media content (Reels, TikToks) through a programmatic workflow, from initial idea generation to the final edited MP4.

## Pipeline Overview
The system runs through main.py in four distinct phases:

1. Phase 1: Content Generation (LLM)
   - Uses LiquidAI/LFM2-2.6B to generate topics and 5-scene scripts.

2. Phase 2: Video Generation (T2V)
   - Uses Wan-AI/Wan2.1-T2V-1.3B to generate 5-second clips per scene.

3. Phase 3: Audio Generation (TTS)
   - Uses Kokoro TTS for American English narration (24kHz).

4. Phase 4: Assembly & Subtitles
   - Uses Whisper for timing and MoviePy for editing.
   - Adds 0.5s crossfades and "karaoke-style" subtitles.

## Run Management
All generations ("runs") are stored in timestamped folders. This includes raw prompts, intermediate video/audio files, and final manifests, allowing for later analysis and performance tracking.

## Tech Stack
- Models: LiquidAI/LFM2-2.6B, Wan2.1-T2V, Kokoro, OpenAI Whisper.
- Libraries: PyTorch, Transformers, Diffusers, MoviePy.
- Infrastructure: CUDA acceleration and Archivist run management.

---

## Hardware & Scalability
Rendering diffusion-based video is resource-intensive. While this serves as a solid local foundation, standard home GPUs often face VRAM and speed bottlenecks. For better performance, the system could be integrated with external APIs (like Nano Banana 2) to offload rendering, enhance quality, and improve turnaround times.

---

## Final Output Demo

<p align="center">
  <video src="https://github.com/sokvl/reel-generator/raw/main/demo/neuroplasticity.mp4" width="320" controls>
    Your browser does not support the video tag.
  </video>
</p>

**Prompt:** "How does neuroplasticity work and what it is?"  
**Description:** A 60-second video explaining neural reorganization with synced narration and dynamic subtitles.