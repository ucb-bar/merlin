"""Create and initialise an isolated run directory."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _git_head(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def git_root(start: Path) -> Path:
    """Discover the enclosing git repo root (the merlin *subject* checkout) from any depth.

    Historically this was ``root.parent`` (valid only when the project lived at the repo root).
    The project now lives at merlin/experiments/targetgen_evals/, so we ask git for the toplevel.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if out:
            return Path(out)
    except Exception:
        pass
    return start.parent


def materialize(
    root: Path,
    target: str,
    method: str,
    seed: int,
    force: bool,
    is_smoke_test: bool,
    budget: str,
    tracking_mode: str = "local",
    mlflow_tracking_uri: str | None = None,
    experiment_name: str | None = None,
    otel_endpoint: str | None = None,
) -> int:
    # Resolve method definition
    method_yaml = root / "methods" / method / "method.yaml"
    if not method_yaml.exists():
        print(f"ERROR: method definition not found: {method_yaml}", file=sys.stderr)
        return 1

    # Resolve target config
    target_yaml = root / "configs" / "targets" / f"{target}.yaml"
    if not target_yaml.exists():
        print(f"ERROR: target config not found: {target_yaml}", file=sys.stderr)
        return 1

    # Resolve budget config
    budget_yaml = root / "configs" / "budgets" / f"{budget}.yaml"
    if not budget_yaml.exists():
        print(f"ERROR: budget config not found: {budget_yaml}", file=sys.stderr)
        return 1

    # Compute run ID
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    run_id = f"{date_str}_{method}_seed{seed:03d}"

    # Runs live under the single out/ root (out/runs/targetgen-evals/…), never inside merlin/ or a
    # retired top-level runs/. root is the targetgen-evals dir (merlin/experiments/targetgen_evals),
    # so root.parents[2] == repo root. (The harness treats Merlin as an external subject and does not
    # import merlin.*, so it composes the out/runs path directly rather than via merlin.common.paths.)
    run_dir = root.parents[2] / "out" / "runs" / "targetgen-evals" / target / run_id

    if run_dir.exists():
        if force:
            import shutil
            shutil.rmtree(run_dir)
        else:
            print(
                f"ERROR: run directory already exists: {run_dir}\n"
                f"Use --force to overwrite.",
                file=sys.stderr,
            )
            return 1

    # Capture git state (the merlin subject checkout — discovered, not assumed to be root.parent)
    repo_root = git_root(root)
    git_hash = _git_head(repo_root)

    # Create directory structure
    generated_dir = run_dir / "generated" / f"{target}-mlir"
    generated_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "patches").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "contracts").mkdir()

    # Write run_manifest.yaml
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "target": target,
        "method": method,
        "seed": seed,
        "git_hash_at_init": git_hash,
        "is_smoke_test": is_smoke_test,
        "budget": budget,
        "created_at": now_iso,
        "promotion_flag": False,
        "observability": {
            "tracking_mode": tracking_mode,
            "mlflow": {
                "enabled": tracking_mode != "local",
                "tracking_uri": mlflow_tracking_uri,
                "experiment_name": experiment_name,
                "run_id": None,
            },
            "opentelemetry": {
                "enabled": tracking_mode in ("full", "debug"),
                "endpoint": otel_endpoint,
                "service_name": "targetgen-evals",
                "trace_id": None,
            },
            "capture_policy": {
                "capture_prompts": True,
                "capture_outputs": True,
                "capture_tool_results": True,
                "redact_secrets": True,
                "store_raw_llm_content": "local_only",
            },
        },
    }
    manifest_path = run_dir / "run_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=True, allow_unicode=True)

    # Write empty validation_report.json placeholder
    (run_dir / "validation_report.json").write_text(
        json.dumps({"status": "not_validated", "run_id": run_id}, indent=2) + "\n"
    )

    # Write summary.md placeholder
    smoke_note = " *(smoke test — not a real baseline)*" if is_smoke_test else ""
    (run_dir / "summary.md").write_text(
        f"# Run: {run_id}{smoke_note}\n\n"
        f"- target: `{target}`\n"
        f"- method: `{method}`\n"
        f"- seed: {seed}\n"
        f"- budget: `{budget}`\n"
        f"- git: `{git_hash}`\n"
        f"- created: {now_iso}\n\n"
        f"*Not yet validated. Run `python -m harness.cli validate {run_dir}` to validate.*\n"
    )

    # Start logger and record init event
    from harness.tracking import TargetGenRunLogger
    logger = TargetGenRunLogger.start(
        target=target,
        method=method,
        seed=seed,
        run_id=run_id,
        run_path=run_dir,
        tracking_mode=tracking_mode,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
        otel_endpoint=otel_endpoint,
    )
    logger.log_params({
        "target": target,
        "method": method,
        "seed": seed,
        "budget": budget,
        "is_smoke_test": is_smoke_test,
        "git_hash_at_init": git_hash,
    })
    logger.log_event("init_run.completed", {"run_id": run_id, "run_dir": str(run_dir)})
    logger.finish("success")
    logger.patch_manifest(manifest_path)
    logger.close()

    print(run_dir)
    print(f"  tracking mode: {tracking_mode}")
    print(f"  local logs:    {run_dir / 'logs'}")
    return 0
