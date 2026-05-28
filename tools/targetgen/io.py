"""TargetGen file I/O + view builders.

`_write_json` / `_write_yaml` are mkdir-safe one-liners. The view builders
(`_build_compile_view`, `_build_deployment_view`) are called from plan /
generate / stage-mutation to emit consistent artifact shape; the plan
writer (`_write_plan_artifacts`) orchestrates the trio.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_compile_view(capabilities, integration_styles: list[str]) -> dict:
    generic_flags = []
    if capabilities.platform.host_isa.startswith("riscv"):
        generic_flags.append(f"--iree-llvmcpu-target-triple=" f"{capabilities.platform.host_isa}-unknown-linux-gnu")
    features = ",".join(capabilities.isa.features)
    hw_key = (
        capabilities.deployment.compile_hw
        if capabilities.deployment and capabilities.deployment.compile_hw
        else "default"
    )
    target_name = (
        capabilities.deployment.compile_target
        if capabilities.deployment and capabilities.deployment.compile_target
        else capabilities.identity.name
    )
    compile_view = {
        "target_name": target_name,
        "default_hw": hw_key,
        "generic": generic_flags,
        "targets": {hw_key: [f"--target-isa-features={features}"] if features else []},
        "plugin_flags": [],
    }
    if "post_global_plugin" in integration_styles:
        compile_view["plugin_flags"] = [f"--iree-plugin={capabilities.identity.name.replace('_', '-')}"]
    return compile_view


def _build_deployment_view(capabilities) -> dict:
    assert capabilities.deployment is not None
    return {
        "name": capabilities.deployment.name,
        "mode": capabilities.deployment.mode,
        "build_profile": capabilities.deployment.build_profile,
        "compile_target": capabilities.deployment.compile_target,
        "compile_hw": capabilities.deployment.compile_hw,
        "hardware_recipe": capabilities.deployment.hardware_recipe,
        "chipyard": capabilities.deployment.chipyard,
        "runtime": capabilities.deployment.runtime,
        "extra": capabilities.deployment.extra,
    }


def _write_plan_artifacts(
    *,
    out_dir: Path,
    support_plan,
    task_graph: list,
    compile_view: dict,
    deployment_view: dict | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "support_plan.json", asdict(support_plan))
    _write_json(out_dir / "task_graph.json", [asdict(task) for task in task_graph])
    _write_json(
        out_dir / "verification_manifest.json",
        asdict(support_plan.verification_manifest),
    )
    _write_yaml(out_dir / "compile_view.yaml", compile_view)
    if deployment_view is not None:
        _write_yaml(out_dir / "deployment_view.yaml", deployment_view)
