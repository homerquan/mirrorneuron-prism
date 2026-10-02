from ..storage import read_json, write_json
from .benchmark_cmd import run_path
from .output import envelope, redact, render


def list_runs(ctx):
    return {"runs": [read_json(p) for p in sorted(ctx.run_root.glob("*/status.json"))]}


def show(ctx):
    path = run_path(ctx, ctx.args.run_id)
    return redact(
        {
            "manifest": read_json(path / "manifest.json"),
            "status": read_json(path / "status.json"),
            "metrics": read_json(path / "metrics.json")
            if (path / "metrics.json").exists()
            else None,
        }
    )


def events(ctx):
    import json

    path = run_path(ctx, ctx.args.run_id) / "events.jsonl"
    records = []
    with path.open() as stream:
        for line in stream:
            record = redact(json.loads(line))
            if ctx.args.format == "jsonl":
                render(ctx.args, envelope("runs.events.event", record))
            else:
                records.append(record)
    return {"events": records if ctx.args.format != "jsonl" else None}


def export(ctx):
    data = show(ctx)
    # Export only manifest/settings/metrics, never model weights, datasets or traces.
    write_json(ctx.explicit(ctx.args.output), data)
    return {
        "run_id": ctx.args.run_id,
        "output": str(ctx.explicit(ctx.args.output)),
        "content_included": False,
    }
