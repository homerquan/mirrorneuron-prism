import json
from pathlib import Path
from typing import Dict, Union

def load_models_from_dir(models_dir: Union[str, Path]) -> Dict[str, dict]:
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
    import argparse

    parser = argparse.ArgumentParser(description="Load model configs from a directory")
    parser.add_argument("models_dir", help="Directory containing *.json model configs")
    args = parser.parse_args()
    print(json.dumps(load_models_from_dir(args.models_dir), indent=2))
