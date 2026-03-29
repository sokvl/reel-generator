import json
import pytest
from helpers.load_prompt import load_prompt, load_json


class TestLoadPrompt:
    def test_reads_file_content(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("hello world")
        assert load_prompt(str(f)) == "hello world"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompt(str(tmp_path / "nonexistent.txt"))

    def test_empty_file_returns_empty_string(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert load_prompt(str(f)) == ""


class TestLoadJson:
    def test_reads_json(self, tmp_path):
        data = {"key": "value", "num": 42}
        f = tmp_path / "data.json"
        f.write_text(json.dumps(data))
        assert load_json(str(f)) == data

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(str(tmp_path / "missing.json"))

    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_json(str(f))
