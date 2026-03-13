def load_prompt(path: str="prompt.txt") -> str:
    with open(path, "r") as f:
        return f.read()
    
def load_json(path: str="prompt.json") -> dict:
    import json
    with open(path, "r") as f:
        return json.load(f)