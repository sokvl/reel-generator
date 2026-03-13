import json
import re
from typing import Dict, Any

def parse_script_robust(script_text: str) -> Dict[str, Any]:

    clean_text = "\n".join(
        line for line in script_text.splitlines() if not line.strip().startswith("```")
    ).strip()

    if "{" in clean_text and "}" in clean_text:
        start = clean_text.find("{")
        end = clean_text.rfind("}") + 1
        clean_text = clean_text[start:end]
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in script text")

    title = data.get("script_title", "").strip()
    scenes = data.get("scenes", [])

    parsed_scenes: Dict[str, Any] = {}

    for scene in scenes:
        num = scene.get("scene_number")
        if num is None:
            num_text = str(scene.get("scene", "")).strip()
            m = re.search(r"\d+", num_text)
            num = int(m.group()) if m else len(parsed_scenes) + 1

        scene_name = f"Scene {num}"
        img_id = f"img_{num}"

        if visual is None:
            visuals_raw = scene.get("visuals", [])
            if isinstance(visuals_raw, list) and visuals_raw:
                visual = visuals_raw[0].get("description", "")

        narration = scene.get("narration")
        if not isinstance(narration, str):
            narration = ""

        parsed_scenes[scene_name] = {
            "visuals": {img_id: visual.strip()},
            "narration": [narration.strip()] if narration else []
        }

    return {
        "script_title": title,
        "scenes": parsed_scenes
    }
