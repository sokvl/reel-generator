import pytest
from prompt.base_prompt import BasePrompt
from prompt.txt_prompt import prompt_factory, ScriptPrompt, IdeaPrompt, VidPrompt


# ---------------------------------------------------------------------------
# BasePrompt
# ---------------------------------------------------------------------------

class TestBasePrompt:
    def test_str_concatenates_parts(self):
        p = BasePrompt(core="CORE", header="HEAD", tail="TAIL")
        assert str(p) == "HEAD CORE TAIL"

    def test_str_without_tail(self):
        p = BasePrompt(core="C", header="H")
        assert str(p) == "H C "

    def test_core_setter_updates_prompt(self):
        p = BasePrompt(core="old", header="H", tail="T")
        p.core = "new"
        assert str(p) == "H new T"
        assert p.core == "new"

    def test_core_getter(self):
        p = BasePrompt(core="value", header="H")
        assert p.core == "value"


# ---------------------------------------------------------------------------
# prompt_factory
# ---------------------------------------------------------------------------

class TestPromptFactory:
    def test_script_returns_script_prompt(self):
        p = prompt_factory("script")
        assert isinstance(p, ScriptPrompt)

    def test_idea_returns_idea_prompt(self):
        p = prompt_factory("idea")
        assert isinstance(p, IdeaPrompt)

    def test_video_returns_vid_prompt(self):
        p = prompt_factory("video")
        assert isinstance(p, VidPrompt)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt type"):
            prompt_factory("bogus")

    def test_all_types_are_base_prompt_subclasses(self):
        for t in ("script", "idea", "video"):
            assert isinstance(prompt_factory(t), BasePrompt)

    def test_core_setter_works_on_factory_result(self):
        p = prompt_factory("video")
        p.core = "a cinematic shot"
        assert "a cinematic shot" in str(p)
