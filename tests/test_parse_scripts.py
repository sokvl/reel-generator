import json
import pytest
from helpers.parse_scripts import parse_script_robust, _repair_json


# ---------------------------------------------------------------------------
# _repair_json
# ---------------------------------------------------------------------------

def _make_scene(num, visual, narration):
    return {"scene_number": num, "visual": visual, "narration": narration}


def _valid_script(scenes=None):
    if scenes is None:
        scenes = [_make_scene(i, f"visual {i}", f"narration {i}") for i in range(1, 6)]
    return json.dumps({"script_title": "Test", "scenes": scenes})


class TestRepairJson:
    def test_trailing_comma_in_object(self):
        broken = '{"a": 1,}'
        result = json.loads(_repair_json(broken))
        assert result == {"a": 1}

    def test_trailing_comma_in_array(self):
        broken = '[1, 2, 3,]'
        result = json.loads(_repair_json(broken))
        assert result == [1, 2, 3]

    def test_unclosed_brace(self):
        broken = '{"a": 1'
        result = json.loads(_repair_json(broken))
        assert result == {"a": 1}

    def test_already_valid_passthrough(self):
        valid = '{"a": 1, "b": [1, 2]}'
        assert _repair_json(valid) == valid


# ---------------------------------------------------------------------------
# parse_script_robust
# ---------------------------------------------------------------------------

class TestParseScriptRobust:
    def test_valid_json(self):
        raw = _valid_script()
        parsed = parse_script_robust(raw)
        assert "scenes" in parsed
        assert len(parsed["scenes"]) == 5

    def test_strips_markdown_fences(self):
        raw = f"```json\n{_valid_script()}\n```"
        parsed = parse_script_robust(raw)
        assert len(parsed["scenes"]) == 5

    def test_json_embedded_in_prose(self):
        raw = f"Here is the script:\n{_valid_script()}\nDone."
        parsed = parse_script_robust(raw)
        assert len(parsed["scenes"]) == 5

    def test_script_title_extracted(self):
        raw = _valid_script()
        parsed = parse_script_robust(raw)
        assert parsed["script_title"] == "Test"

    def test_visual_as_list_fallback(self):
        scenes = [{"scene_number": 1, "visuals": [{"description": "a visual"}], "narration": "narr"}]
        raw = json.dumps({"script_title": "T", "scenes": scenes})
        parsed = parse_script_robust(raw)
        assert parsed["scenes"]["Scene 1"]["visuals"]["img_1"] == "a visual"

    def test_narration_missing_yields_empty_list(self):
        scenes = [{"scene_number": 1, "visual": "v", "narration": ""}]
        raw = json.dumps({"script_title": "T", "scenes": scenes})
        parsed = parse_script_robust(raw)
        assert parsed["scenes"]["Scene 1"]["narration"] == []

    def test_narration_present_wrapped_in_list(self):
        scenes = [{"scene_number": 1, "visual": "v", "narration": "hello world"}]
        raw = json.dumps({"script_title": "T", "scenes": scenes})
        parsed = parse_script_robust(raw)
        assert parsed["scenes"]["Scene 1"]["narration"] == ["hello world"]

    def test_scene_number_inferred_from_string(self):
        scenes = [{"scene": "Scene 3", "visual": "v", "narration": "n"}]
        raw = json.dumps({"script_title": "T", "scenes": scenes})
        parsed = parse_script_robust(raw)
        assert "Scene 3" in parsed["scenes"]

    def test_invalid_json_raises_retry_needed_on_first_attempt(self):
        with pytest.raises(ValueError, match=r"\[RETRY_NEEDED\]"):
            parse_script_robust("{invalid json", attempt=1, max_attempts=3)

    def test_invalid_json_raises_plain_error_on_last_attempt(self):
        with pytest.raises(ValueError) as exc_info:
            parse_script_robust("{invalid json", attempt=3, max_attempts=3)
        assert "[RETRY_NEEDED]" not in str(exc_info.value)

    def test_empty_scenes_dict(self):
        raw = json.dumps({"script_title": "Empty", "scenes": []})
        parsed = parse_script_robust(raw)
        assert parsed["scenes"] == {}
