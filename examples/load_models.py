from litellm_multicall.models_loader import load_models_from_dir
import json

models = load_models_from_dir("/Users/homer/Sandbox/GomokuBench/models")
print(f"Loaded {len(models)} model configs")
for name, cfg in list(models.items())[:5]:
    print(name, cfg.get("provider", {}))
