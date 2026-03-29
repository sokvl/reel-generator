"""
Tests for main.py logic that does not require GPU or heavy models.
We test: validate_args, preflight_checks, get_run_paths, ensure_run_inputs.
"""
import argparse
from pathlib import Path
import pytest

# We patch FONT_PATH before importing main so tests are hermetic.
import unittest.mock as mock


def _make_args(**kwargs):
    defaults = dict(w=480, h=832, mode="full", prompt_file=None,
                    topic="some topic", assemble_from=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------

class TestValidateArgs:
    def _run(self, **kwargs):
        from main import validate_args
        validate_args(_make_args(**kwargs))

    def test_valid_full_mode(self):
        self._run()  # default args, should not raise

    def test_valid_vid_mode_with_prompt_file(self):
        self._run(mode="vid", prompt_file="prompt.txt")

    def test_vid_mode_without_prompt_file_raises(self):
        from main import validate_args
        with pytest.raises(ValueError, match="--prompt_file"):
            validate_args(_make_args(mode="vid", prompt_file=None))

    def test_negative_width_raises(self):
        from main import validate_args
        with pytest.raises(ValueError, match="positive"):
            validate_args(_make_args(w=-1))

    def test_zero_height_raises(self):
        from main import validate_args
        with pytest.raises(ValueError, match="positive"):
            validate_args(_make_args(h=0))

    def test_empty_topic_in_full_mode_raises(self):
        from main import validate_args
        with pytest.raises(ValueError, match="--topic"):
            validate_args(_make_args(topic="   "))

    def test_assemble_from_skips_all_other_checks(self):
        from main import validate_args
        # Even bad args are fine when assemble_from is set
        validate_args(_make_args(assemble_from="/some/path", w=-1, mode="vid", prompt_file=None))


# ---------------------------------------------------------------------------
# preflight_checks
# ---------------------------------------------------------------------------

class TestPreflightChecks:
    def _run(self, args, font_exists=True, extra_patches=None):
        patches = {"main.Path": mock.DEFAULT}
        if extra_patches:
            patches.update(extra_patches)

        with mock.patch("main.Path") as MockPath, \
             mock.patch("main.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True

            # Default: font file exists
            font_path_mock = mock.MagicMock()
            font_path_mock.is_file.return_value = font_exists
            font_path_mock.is_dir.return_value = False

            # Make Path(...) return font_path_mock by default
            MockPath.return_value = font_path_mock

            from main import preflight_checks
            preflight_checks(args)

    def test_all_good_does_not_raise(self, tmp_path):
        from main import preflight_checks
        args = _make_args()
        with mock.patch("main.Path") as MockPath, \
             mock.patch("main.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            m = mock.MagicMock()
            m.is_file.return_value = True
            m.is_dir.return_value = True
            m.expanduser.return_value = m
            m.resolve.return_value = m
            MockPath.return_value = m
            preflight_checks(args)

    def test_missing_font_raises_systemexit(self, tmp_path):
        from main import preflight_checks
        args = _make_args()
        with mock.patch("main.Path") as MockPath, \
             mock.patch("main.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            m = mock.MagicMock()
            m.is_file.return_value = False  # font missing
            m.is_dir.return_value = True
            m.expanduser.return_value = m
            m.resolve.return_value = m
            MockPath.return_value = m
            with pytest.raises(SystemExit):
                preflight_checks(args)

    def test_assemble_from_missing_dir_raises_systemexit(self):
        from main import preflight_checks
        args = _make_args(assemble_from="/no/such/dir")
        with mock.patch("main.Path") as MockPath, \
             mock.patch("main.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            font_mock = mock.MagicMock()
            font_mock.is_file.return_value = True

            dir_mock = mock.MagicMock()
            dir_mock.is_file.return_value = False
            dir_mock.is_dir.return_value = False
            dir_mock.expanduser.return_value = dir_mock
            dir_mock.resolve.return_value = dir_mock

            # First call → font check, second → assemble_from
            MockPath.side_effect = [font_mock, dir_mock]
            with pytest.raises(SystemExit):
                preflight_checks(args)

    def test_vid_mode_missing_prompt_file_raises_systemexit(self, tmp_path):
        from main import preflight_checks
        fake_prompt = str(tmp_path / "missing_prompt.txt")
        args = _make_args(mode="vid", prompt_file=fake_prompt)
        with mock.patch("main.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            # Use real Path so the file-existence check actually runs
            with mock.patch("main.FONT_PATH", str(tmp_path / "font.ttf")):
                # font doesn't exist either — but we only care about prompt_file here,
                # so create a dummy font
                (tmp_path / "font.ttf").write_bytes(b"")
                with pytest.raises(SystemExit):
                    preflight_checks(args)

    def test_no_cuda_prints_warning(self, capsys, tmp_path):
        from main import preflight_checks
        args = _make_args()
        (tmp_path / "font.ttf").write_bytes(b"")
        with mock.patch("main.FONT_PATH", str(tmp_path / "font.ttf")), \
             mock.patch("main.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            preflight_checks(args)
        out = capsys.readouterr().out
        assert "CUDA" in out


# ---------------------------------------------------------------------------
# get_run_paths
# ---------------------------------------------------------------------------

class TestGetRunPaths:
    def test_returns_expected_keys(self, tmp_path):
        from main import get_run_paths
        paths = get_run_paths(tmp_path)
        assert set(paths.keys()) == {"base", "voiceovers", "videos", "subs", "out"}

    def test_voiceovers_is_subdir_of_base(self, tmp_path):
        from main import get_run_paths
        paths = get_run_paths(tmp_path)
        assert paths["voiceovers"].parent == paths["base"]

    def test_subs_path(self, tmp_path):
        from main import get_run_paths
        paths = get_run_paths(tmp_path)
        assert paths["subs"].name == "subtitles.json"
        assert paths["subs"].parent.name == "artifacts"

    def test_out_is_mp4(self, tmp_path):
        from main import get_run_paths
        paths = get_run_paths(tmp_path)
        assert paths["out"].suffix == ".mp4"


# ---------------------------------------------------------------------------
# ensure_run_inputs
# ---------------------------------------------------------------------------

class TestEnsureRunInputs:
    def _make_paths(self, tmp_path, create_voiceovers=True, create_videos=True, create_subs=True):
        vo = tmp_path / "voiceovers"
        vid = tmp_path / "videos"
        subs = tmp_path / "artifacts" / "subtitles.json"
        if create_voiceovers:
            vo.mkdir()
        if create_videos:
            vid.mkdir()
        if create_subs:
            subs.parent.mkdir(parents=True, exist_ok=True)
            subs.write_text("{}")
        return {"voiceovers": vo, "videos": vid, "subs": subs}

    def test_all_present_does_not_raise(self, tmp_path):
        from main import ensure_run_inputs
        ensure_run_inputs(self._make_paths(tmp_path))

    def test_missing_voiceovers_raises(self, tmp_path):
        from main import ensure_run_inputs
        with pytest.raises(FileNotFoundError, match="Audio dir"):
            ensure_run_inputs(self._make_paths(tmp_path, create_voiceovers=False))

    def test_missing_videos_raises(self, tmp_path):
        from main import ensure_run_inputs
        with pytest.raises(FileNotFoundError, match="Video dir"):
            ensure_run_inputs(self._make_paths(tmp_path, create_videos=False))

    def test_missing_subs_raises(self, tmp_path):
        from main import ensure_run_inputs
        with pytest.raises(FileNotFoundError, match="Subtitles"):
            ensure_run_inputs(self._make_paths(tmp_path, create_subs=False))
