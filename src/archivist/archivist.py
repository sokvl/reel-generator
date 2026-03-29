import os
import re
import json
import string
import random
from datetime import datetime
from typing import Any, Mapping

class Archivist:
    def __init__(self, sig_seed: str | int | None = None):
        date_part = datetime.now().strftime("%d%m%y")
        random.seed(sig_seed if sig_seed is not None else date_part)
        id_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.signature = f"{id_part}_{date_part}"
        self.paths: dict[str, str] | None = None
        self.manifest: dict[str, Any] = {
            "signature": self.signature,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "title": None,
            "artifacts": []  
        }

    @staticmethod
    def _sanitize(name: str, fallback: str = "untitled") -> str:
        name = (name or "").strip()
        if not name:
            return fallback
        name = re.sub(r"\s+", "_", name)
        name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
        return name if name.strip("._-") else fallback

    def _setup_dirs(self, title: str) -> dict:
        safe_title = self._sanitize(title, "untitled")
        base = os.path.join("runs", f"{safe_title}_{self.signature}")
        subdirs = ["videos", "voiceovers", "prompts", "artifacts"]
        os.makedirs(base, exist_ok=True)
        paths = {"base": base}
        for sd in subdirs:
            p = os.path.join(base, sd)
            os.makedirs(p, exist_ok=True)
            paths[sd] = p
        return paths

    def start_run(self, title: str, meta: Mapping[str, Any] | None = None) -> str:
        title = title.split()[0].lower() if isinstance(title, str) and title.strip() else "untitled"
        self.paths = self._setup_dirs(title)
        self.manifest["title"] = title
        if meta:
            self.manifest["meta"] = dict(meta)

        self._flush_manifest()
        return self.paths["base"]

    def _flush_manifest(self):
        if not self.paths:
            return
        mpath = os.path.join(self.paths["base"], "manifest.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def _record(self, kind: str, path: str, extra: dict | None = None):
        self.manifest["artifacts"].append({
            "kind": kind, "path": path, "extra": extra or {},
            "timestamp": datetime.now().isoformat(timespec="seconds")
        })
        self._flush_manifest()

    def record_video(self, kind: str, video_path: str, extra: dict | None = None) -> str:
        return self._record(kind=kind, path=video_path, extra=extra)

    def save_text(self, identity: str, content: str, subdir: str = "prompts", ext: str = ".txt") -> str:
        assert self.paths, "Call start_run() first"
        fname = f"{self._sanitize(identity)}{ext}"
        p = os.path.join(self.paths[subdir], fname)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        self._record(kind=f"text:{subdir}", path=p, extra={"identity": identity})
        return p

    def save_json(self, identity: str, obj: Any, subdir: str = "prompts") -> str:
        assert self.paths, "Call start_run() first"
        fname = f"{self._sanitize(identity)}.json"
        p = os.path.join(self.paths[subdir], fname)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        self._record(kind=f"json:{subdir}", path=p, extra={"identity": identity})
        return p

    def save_binary(self, identity: str, data: bytes, subdir: str, ext: str) -> str:
        assert self.paths, "Call start_run() first"
        fname = f"{self._sanitize(identity)}{ext}"
        p = os.path.join(self.paths[subdir], fname)
        with open(p, "wb") as f:
            f.write(data)
        self._record(kind=f"bin:{subdir}", path=p, extra={"identity": identity})
        return p

    def path(self, subdir: str, *parts: str) -> str:
        assert self.paths, "Call start_run() first"
        return os.path.join(self.paths[subdir], *parts)

    def finalize(self, final_video_path: str | None = None):
        if final_video_path:
            self._record(kind="final_video", path=final_video_path, extra={})
        self._flush_manifest()
