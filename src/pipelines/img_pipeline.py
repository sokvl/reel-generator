from base_pipeline import BasePipeline

import torch
from diffusers import FluxPipeline

from config_file import IMAGE_MODEL, IMAGE_GUIDANCE_SCALE, IMAGE_INFERENCE_STEPS, IMAGE_MAX_SEQ_LEN

class ImgPipeline(BasePipeline):
    def __init__(self, w, h, model_id: str = IMAGE_MODEL, out_dir: str = "images/"):
        self.model_id = model_id
        self.w = w
        self.h = h
        self.out_dir = out_dir
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        self.pipe.enable_attention_slicing()
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

    def diffuse(self, prompt, id):
        result = self.pipe(          
            str(prompt),
            height=self.h,
            width=self.w,
            guidance_scale=IMAGE_GUIDANCE_SCALE,
            num_inference_steps=IMAGE_INFERENCE_STEPS,
            max_sequence_length=IMAGE_MAX_SEQ_LEN,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        path = f"{self.out_dir}img_{id}.png"
        result.save(path)
        return path