from .base_pipeline import BasePipeline
from pydantic import BaseModel
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from diffusers.schedulers import UniPCMultistepScheduler
import torch
import gc
from typing import Optional

class VideoPipelineT2I(BasePipeline):
    def __init__(
        self, 
        w: int, 
        h: int, 
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 
        out_dir: str = "videos/",
        flow_shift: float = 3.0 
    ):
        self.model_id = model_id
        self.w = w
        self.h = h
        self.out_dir = out_dir
        self.flow_shift = flow_shift
        self.pipe = None
        self.vae = None
        
    def _log_memory(self, stage: str):
        """Monitor VRAM usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[{stage}] VRAM - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    def _aggressive_cleanup(self):
        """Aggressively free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if self.vae is not None:
            del self.vae
            self.vae = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _load_pipeline(self):
        """Load pipeline with memory-efficient settings"""
        self._log_memory("Before load")
        
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=self.flow_shift
        )
        
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            vae=self.vae,
            torch_dtype=torch.bfloat16
        )
        self.pipe.scheduler = scheduler
        
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.enable_attention_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ xFormers enabled")
        except Exception as e:
            print(f"⚠ xFormers not available, using default attention: {e}")
        
        self._log_memory("After load")

    def diffuse(self, prompt: str, negative_prompt: Optional[str] = None, id: str = "output", 
                num_frames: int = 81, guidance_scale: float = 6.0, seed: Optional[int] = None, 
                fps: int = 16):
        """
        Generate video from text prompt
        
        Args:
            prompt: Text description of the video to generate
            negative_prompt: Things to avoid in generation
            id: Output filename (without extension)
            num_frames: Number of frames to generate
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            fps: Frames per second for output video
        """
        try:
            # Load pipeline
            self._aggressive_cleanup()
            self._load_pipeline()
            
            self._log_memory("Before generation")
            
            if negative_prompt is None:
                negative_prompt = """Bright tones, overexposed, static, blurred details, 
worst quality, low quality, ugly, deformed, contains text, contains letters, realistic, photo, photograph, photo-realistic"""
            
            generator = None
            if seed is not None and torch.cuda.is_available():
                generator = torch.Generator("cuda").manual_seed(seed)
                print(f"✓ Using seed: {seed}")
            
            with torch.inference_mode():
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=self.h,
                    width=self.w,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    generator=generator
                ).frames[0]
            
            self._log_memory("After generation")
            
            output_path = f"{self.out_dir}/{id}.mp4"
            export_to_video(output, output_path, fps=fps)
            
            print(f"✓ Video saved to {output_path}")
            return output_path
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA OOM Error: {e}")
            print("Try reducing: height/width, num_frames, or num_inference_steps")
            raise
        
        except Exception as e:
            print(f"❌ Error during video generation: {e}")
            raise
        
        finally:
            self._aggressive_cleanup()
            self._log_memory("After cleanup")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self._aggressive_cleanup()