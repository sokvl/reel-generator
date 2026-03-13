import gc
from typing import List, Dict, Iterable, Protocol, Optional, Callable, Any
from dataclasses import dataclass, field

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from ..prompt.base_prompt import BasePrompt


@dataclass(frozen=True)
class Msg:
    role: str
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

def _validate_messages(msgs: Iterable[Msg]) -> None:
    allowed = {"system", "user", "assistant", "developer"}
    for m in msgs:
        if m.role not in allowed:
            raise ValueError(f"Unsupported role: {m.role}")
        if not isinstance(m.content, str) or not m.content:
            raise ValueError("Message content must be non-empty string.")



@dataclass
class LLMFacade:
    model_id: str = "LiquidAI/LFM2-2.6B"
    device_map: str | None = "auto"
    dtype: Any | None = torch.float16
    tokenizer: Optional[AutoTokenizer] = field(default=None, init=False, repr=False)
    model: Optional[AutoModelForCausalLM] = field(default=None, init=False, repr=False)
    messages: List[Msg] = field(default_factory=list)

    def ensure_loaded(self) -> None:
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map=self.device_map
            )


    def set_model(self, model):
        self.unload()
        self.model_id = model

    def set_prompts(self, system_prompt: BasePrompt, user_prompt: BasePrompt) -> "LLMFacade":
        msgs = [Msg("system", str(system_prompt)), Msg("user", str(user_prompt))]
        _validate_messages(msgs)
        self.messages = msgs
        return self

    def set_messages(self, msgs: Iterable[Msg]) -> "LLMFacade":
        msgs = list(msgs)
        _validate_messages(msgs)
        self.messages = msgs
        return self

    def add(self, role: str, content: str) -> "LLMFacade":
        m = Msg(role, content)
        _validate_messages([m])
        self.messages.append(m)
        return self

    def reset(self) -> "LLMFacade":
        self.messages = []
        return self


    def switch_task(
        self,
        system_prompt: BasePrompt,
        user_prompt: BasePrompt,
        keep_history: bool = False
    ) -> "LLMFacade":
        if keep_history:
            self.add("system", str(system_prompt))
            self.add("user", str(user_prompt))
        else:
            self.set_prompts(system_prompt, user_prompt)
        return self

    def run(
        self,
        gen_kwargs: Optional[dict] = None,
        return_text_only: bool = True
    ) -> str:
        self.ensure_loaded()
        gen_kwargs = gen_kwargs or {}

        # best parrams according to authors LFM2-2.6B
        defaults = dict(max_new_tokens=2048, temperature=0.3, min_p=0.15, repetition_penalty=1.05)
        for k, v in defaults.items():
            gen_kwargs.setdefault(k, v)

        inputs = self.tokenizer.apply_chat_template(
            [m.as_dict() for m in self.messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Zwracamy tylko nowo wygenerowany fragment
        text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return text if return_text_only else outputs

    def run_to_file(self, path: str, gen_kwargs: Optional[dict] = None) -> str:
        out = self.run(gen_kwargs=gen_kwargs, return_text_only=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)
        return path
    
    def unload(self) -> None:
        """Free model and tokenizer from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def __del__(self) -> None:
        self.unload()