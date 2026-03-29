import json
import re
from typing import Dict, Any

def _repair_json(text: str) -> str:
    """Attempt to repair common JSON formatting issues."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix missing commas between values on different lines (common LLM issue)
    # e.g., "field": "value"\n"next_field" -> "field": "value",\n"next_field"
    text = re.sub(r'"\s*\n\s*"', '",\n"', text)
    
    # Fix missing commas after closing braces on same line
    # e.g., }\n{ -> },\n{
    text = re.sub(r'}\s*\n\s*(?=[{"])', '},\n', text)
    
    # If JSON appears incomplete, try to close it properly
    text = text.rstrip()
    
    # Check if the last string was properly closed
    quote_count = text.count('"') - text.count('\\"')
    if quote_count % 2 == 1:
        # Uneven number of quotes, need to close the string
        text += '"'
    
    # Track the stack of open braces/brackets to close them in the right order
    stack = []
    in_string = False
    escape_next = False
    
    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in '{[':
                stack.append(char)
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
    
    # Close any remaining open brackets/braces in reverse order
    while stack:
        opening = stack.pop()
        if opening == '{':
            text += '}'
        elif opening == '[':
            text += ']'
    
    return text

def parse_script_robust(script_text: str, attempt: int = 1, max_attempts: int = 3) -> Dict[str, Any]:

    clean_text = "\n".join(
        line for line in script_text.splitlines() if not line.strip().startswith("```")
    ).strip()

    if "{" in clean_text and "}" in clean_text:
        start = clean_text.find("{")
        end = clean_text.rfind("}") + 1
        clean_text = clean_text[start:end]
    
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as e:
        # Try to repair common JSON issues
        try:
            repaired_text = _repair_json(clean_text)
            data = json.loads(repaired_text)
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON in script text (attempt {attempt}/{max_attempts}): {str(e)}"
            if attempt < max_attempts:
                raise ValueError(error_msg + " [RETRY_NEEDED]")
            else:
                raise ValueError(error_msg)

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

        visual = scene.get("visual")
        if visual is None:
            visuals_raw = scene.get("visuals", [])
            if isinstance(visuals_raw, list) and visuals_raw:
                visual = visuals_raw[0].get("description", "")
        if not isinstance(visual, str):
            visual = ""

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