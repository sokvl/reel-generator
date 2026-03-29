import json
import os
import pytest
from archivist.archivist import Archivist


@pytest.fixture
def arch(tmp_path, monkeypatch):
    """Archivist rooted in a temp directory."""
    monkeypatch.chdir(tmp_path)
    return Archivist(sig_seed=42)


class TestArchivistSanitize:
    def test_normal_name(self):
        assert Archivist._sanitize("hello world") == "hello_world"

    def test_special_chars_replaced(self):
        result = Archivist._sanitize("foo/bar:baz")
        assert "/" not in result
        assert ":" not in result

    def test_empty_string_returns_fallback(self):
        assert Archivist._sanitize("") == "untitled"

    def test_only_punctuation_returns_fallback(self):
        assert Archivist._sanitize("...") == "untitled"

    def test_custom_fallback(self):
        assert Archivist._sanitize("", fallback="default") == "default"

    def test_whitespace_collapsed(self):
        # multiple spaces → single underscore (re.sub collapses the run)
        assert Archivist._sanitize("a  b") == "a_b"


class TestArchivistStartRun:
    def test_creates_base_dir(self, arch):
        arch.start_run(title="test run")
        assert os.path.isdir(arch.paths["base"])

    def test_creates_subdirs(self, arch):
        arch.start_run(title="test")
        for sd in ("videos", "voiceovers", "prompts", "artifacts"):
            assert os.path.isdir(arch.paths[sd])

    def test_manifest_written(self, arch):
        arch.start_run(title="test")
        mpath = os.path.join(arch.paths["base"], "manifest.json")
        assert os.path.isfile(mpath)
        with open(mpath) as f:
            m = json.load(f)
        assert m["title"] == "test"

    def test_meta_saved_in_manifest(self, arch):
        arch.start_run(title="t", meta={"mode": "full"})
        mpath = os.path.join(arch.paths["base"], "manifest.json")
        with open(mpath) as f:
            m = json.load(f)
        assert m["meta"] == {"mode": "full"}

    def test_title_uses_first_word_lowercase(self, arch):
        base = arch.start_run(title="Hello World")
        assert "hello" in base

    def test_deterministic_signature_with_seed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a1 = Archivist(sig_seed=99)
        a2 = Archivist(sig_seed=99)
        assert a1.signature == a2.signature


class TestArchivistSaveText:
    def test_writes_file(self, arch):
        arch.start_run(title="t")
        path = arch.save_text("my_text", "hello", subdir="prompts", ext=".txt")
        assert os.path.isfile(path)
        with open(path) as f:
            assert f.read() == "hello"

    def test_recorded_in_manifest(self, arch):
        arch.start_run(title="t")
        arch.save_text("x", "content", subdir="prompts")
        assert any(a["kind"] == "text:prompts" for a in arch.manifest["artifacts"])

    def test_raises_before_start_run(self, arch):
        with pytest.raises(AssertionError):
            arch.save_text("x", "content")


class TestArchivistSaveJson:
    def test_writes_valid_json(self, arch):
        arch.start_run(title="t")
        path = arch.save_json("data", {"a": 1}, subdir="artifacts")
        with open(path) as f:
            assert json.load(f) == {"a": 1}

    def test_recorded_in_manifest(self, arch):
        arch.start_run(title="t")
        arch.save_json("x", {}, subdir="artifacts")
        assert any(a["kind"] == "json:artifacts" for a in arch.manifest["artifacts"])


class TestArchivistPath:
    def test_returns_subdir_path(self, arch):
        arch.start_run(title="t")
        p = arch.path("videos")
        assert p == arch.paths["videos"]

    def test_joins_extra_parts(self, arch):
        arch.start_run(title="t")
        p = arch.path("videos", "scene_1.mp4")
        assert p.endswith("scene_1.mp4")

    def test_raises_before_start_run(self, arch):
        with pytest.raises(AssertionError):
            arch.path("videos")


class TestArchivistFinalize:
    def test_records_final_video(self, arch):
        arch.start_run(title="t")
        arch.finalize(final_video_path="/tmp/out.mp4")
        assert any(a["kind"] == "final_video" for a in arch.manifest["artifacts"])

    def test_finalize_without_path_still_flushes(self, arch):
        arch.start_run(title="t")
        arch.finalize()
        mpath = os.path.join(arch.paths["base"], "manifest.json")
        assert os.path.isfile(mpath)
