import json
from pathlib import Path
from typing import Dict, Union

def load_models_from_dir(models_dir: Union[str, Path]) -> Dict[str, dict]:
    models_dir = Path(models_dir)
    models = {}
    for f in models_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            provider = data.get("provider", {})
            # Simple extraction for demo
            models[f.name] = data
        except Exception as e:
            models[f.name] = {"error": str(e)}
    return models

if __name__ == "__main__":
    import sys
    dir_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/homer/Sandbox/GomokuBench/models"
    print(json.dumps(load_models_from_dir(dir_path), indent=2))
