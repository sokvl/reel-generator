from typing import Type

from .base_prompt import BasePrompt
from .prompt_repository import PromptRepository


class ScriptPrompt(BasePrompt):
    def __init__(self, core: str,
                 header: str, 
                 tail: str):
        super().__init__(core=core, header=header, tail=tail)

    def get_prompt(self) -> str:
        return str(self)


class IdeaPrompt(BasePrompt):
    """Prompt for generating startup/idea suggestions."""

    def __init__(self, 
                 core: str,
                 header: str, 
                 tail: str):
        
        super().__init__(core=core, header=header, tail=tail)

    def get_prompt(self) -> str:
        return str(self)

class VidPrompt(BasePrompt):
    def __init__(self, core: str,
                 header: str, 
                 tail: str):
        super().__init__(core=core, header=header, tail=tail)

    def get_prompt(self) -> str:
        return str(self)

def prompt_factory(prompt_type: str) -> BasePrompt:
    """Return a prompt instance by type.

    Args:
        prompt_type: One of "script" or "idea".

    Returns:
        Instance of a subclass of `BasePrompt`.

    Raises:
        ValueError: for unknown prompt types.
    """
    repo = PromptRepository()

    factories: dict[str, BasePrompt] = {
        "script": ScriptPrompt(*repo.script.values()),
        "idea": IdeaPrompt(*repo.idea.values()),
        "video": VidPrompt(*repo.video.values())
    }

    if prompt_type not in factories:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    instance = factories[prompt_type]
    
    return instance