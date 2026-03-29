from dataclasses import dataclass, field

from prompt.prompts.prompt_lib import (
                                    IDEA_BRAINSTORMING_PROMPT_HEADER,
                                    IDEA_BRAINSTORMING_PROMPT_CORE, 
                                    IDEA_BRAINSTORMING_PROMPT_TAIL,
                                    SCRIPT_GENERATION_PROMPT_HEAD,
                                    SCRIPT_GENERATION_PROMPT_CORE,
                                    SCRIPT_GENERATION_PROMPT_TAIL,
                                    VIDEO_DESCRIPTION_PROMPT_HEADER,
                                    VIDEO_DESCRIPTION_PROMPT_CORE,
                                    VIDEO_DESCRIPTION_PROMPT_TAIL
                                    )

@dataclass
class PromptRepository:    
    idea: dict[str, str] = field(default_factory=lambda: {
        "header": IDEA_BRAINSTORMING_PROMPT_HEADER,
        "core": IDEA_BRAINSTORMING_PROMPT_CORE,
        "tail": IDEA_BRAINSTORMING_PROMPT_TAIL
    })
    script: dict[str, str] = field(default_factory=lambda: {
        "header": SCRIPT_GENERATION_PROMPT_HEAD,
        "core": SCRIPT_GENERATION_PROMPT_CORE,
        "tail": SCRIPT_GENERATION_PROMPT_TAIL
    })
    video: dict[str, str] = field(default_factory=lambda: {
        "header": VIDEO_DESCRIPTION_PROMPT_HEADER,
        "core": VIDEO_DESCRIPTION_PROMPT_CORE,
        "tail": VIDEO_DESCRIPTION_PROMPT_TAIL
    })
