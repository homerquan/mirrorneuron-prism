import argparse
from pathlib import Path

from litellm_multicall.models_loader import load_models_from_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Load model configs from a directory")
    parser.add_argument("models_dir", nargs="?", default=str(Path(__file__).resolve().parents[1] / "models"))
    args = parser.parse_args()
    models = load_models_from_dir(args.models_dir)
    print(f"Loaded {len(models)} model configs")
    for name, cfg in list(models.items())[:5]:
        print(name, cfg.get("provider", {}))


if __name__ == "__main__":
    main()
