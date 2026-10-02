import importlib.util
import shutil

from ..errors import PrismError
from .config_cmd import validate


def run(ctx):
    checks = []
    try:
        validate(ctx)
        checks.append({"name": "configuration", "ready": True})
    except (ValueError, OSError, PrismError) as exc:
        checks.append(
            {
                "name": "configuration",
                "ready": False,
                "remediation": str(exc).splitlines()[0],
            }
        )
    checks.append(
        {
            "name": "litellm_executable",
            "ready": bool(shutil.which("litellm")),
            "remediation": "install the proxy extra if serving is needed",
        }
    )
    checks.append(
        {
            "name": "inference_runtime",
            "ready": False,
            "remediation": "P3: generative execution is not implemented; provider fails closed",
        }
    )
    from ..classifier.artifacts import verify_controller

    for name, profile in ctx.project.controllers.items():
        deps = profile.backend == "rules" or all(
            importlib.util.find_spec(m) for m in ("torch", "transformers")
        )
        checks.append(
            {
                "name": f"{name}.dependencies",
                "ready": deps,
                "remediation": "install classifier-jevcpu with platform constraints",
            }
        )
        try:
            artifact = (
                verify_controller(ctx.artifact_lock, profile)
                if profile.backend != "rules"
                else None
            )
            checks.append(
                {
                    "name": f"{name}.artifacts",
                    "ready": True,
                    "weights_bytes": artifact["weights_bytes"] if artifact else 0,
                }
            )
        except (ValueError, OSError, PrismError) as exc:
            checks.append(
                {"name": f"{name}.artifacts", "ready": False, "remediation": str(exc)}
            )
    return {
        "checks": checks,
        "ready": all(c["ready"] for c in checks),
        "offline": True,
        "classifier_replicas": len(
            [p for p in ctx.project.controllers.values() if p.backend == "jev_cpu"]
        ),
        "replica_scope": "one per explicit controller session; each serving worker multiplies residency",
        "memory_note": "Qwen3-0.6B float32 parameters need about 2.4 GB per resident replica, plus runtime overhead",
        "cpu_affinity": "unsupported",
        "gpu_active_time": None,
        "energy_joules": None,
    }
