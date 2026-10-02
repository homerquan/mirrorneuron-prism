import json
from pathlib import Path


def load_models_from_dir(models_dir: str | Path) -> dict[str, dict]:
    models_dir = Path(models_dir)
    models = {}
    for f in models_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            # Simple extraction for demo
            models[f.name] = data
        except Exception as e:
            models[f.name] = {"error": str(e)}
    return models

if __name__ == "__main__":
    import sys
    dir_path = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps(load_models_from_dir(dir_path), indent=2))
