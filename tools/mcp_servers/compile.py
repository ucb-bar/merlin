"""Compile MCP tools — typed wrappers around `./merlin compile`.

Depends on:
- tools/compile/cli.py (argparse — every flag exposed here must match)
- models/*.yaml schema (the yaml-parsing logic here reads `default_hw`,
  `targets`, `plugin_flags`, `generic` keys)
- build/compiled_models/<model>/<target>/ artifact layout (the
  inspection logic looks for `*.vmfb`, `phases/`, `benchmarks/` subdirs)

If `compile/cli.py`'s argparse gains/loses a flag, update the schema in
`_compile_model`'s `input_schema`. If the yaml schema changes, update
`_list_compile_targets` + `_resolve_target_devices`. If the artifact
layout changes, update `_inspect_artifact_dir`.

Tools here return parsed structured data (VMFB path + size, compile phases,
target devices) rather than raw stdout tails. The wrappers shell out to
`./merlin compile` for execution but parse outputs in Python so Claude
gets useful structure on every call instead of re-parsing logs each turn.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import yaml

_TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOLS_DIR))
import utils  # noqa: E402

REPO_ROOT = utils.REPO_ROOT
MERLIN = REPO_ROOT / "merlin"


class ToolError(RuntimeError):
    """Recoverable tool failure."""


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _require_str(args: dict[str, Any], key: str) -> str:
    v = args.get(key)
    if not isinstance(v, str) or not v:
        raise ToolError(f"missing or empty required arg: {key!r}")
    return v


# --- Tool: list compile targets ------------------------------------------


def _list_compile_targets(args: dict[str, Any]) -> dict[str, Any]:
    targets = []
    for yaml_path in sorted((REPO_ROOT / "models").glob("*.yaml")):
        try:
            cfg = yaml.safe_load(yaml_path.read_text())
        except Exception as e:
            targets.append({"name": yaml_path.stem, "error": str(e)})
            continue
        generic = cfg.get("generic") or []
        target_device = next(
            (
                f.split("=", 1)[1]
                for f in generic
                if isinstance(f, str)
                and (f.startswith("--iree-hal-target-device=") or f.startswith("--iree-hal-target-backends="))
            ),
            None,
        )
        targets.append(
            {
                "name": yaml_path.stem,
                "default_hw": cfg.get("default_hw"),
                "hw_variants": sorted((cfg.get("targets") or {}).keys()),
                "has_plugin_flags": bool(cfg.get("plugin_flags")),
                "target_device": target_device,
                "yaml_path": str(yaml_path.relative_to(REPO_ROOT)),
            }
        )
    return {"targets": targets, "count": len(targets)}


# --- Tool: list models ---------------------------------------------------


def _list_models(args: dict[str, Any]) -> dict[str, Any]:
    models_dir = REPO_ROOT / "models"
    found = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        mlirs = sorted(child.glob("*.mlir"))
        onnxs = sorted(child.glob("*.onnx"))
        if not mlirs and not onnxs:
            continue
        found.append(
            {
                "name": child.name,
                "mlir": [str(p.relative_to(REPO_ROOT)) for p in mlirs],
                "onnx": [str(p.relative_to(REPO_ROOT)) for p in onnxs],
                "readme": str((child / "README.md").relative_to(REPO_ROOT)) if (child / "README.md").exists() else None,
            }
        )
    return {"models": found, "count": len(found)}


# --- Tool: compile_model -------------------------------------------------


_PHASE_DUMP_RE = re.compile(r"^[0-9]+\.[A-Za-z][A-Za-z0-9_.\-]+\.mlir$")


def _inspect_artifact_dir(artifact_dir: pathlib.Path) -> dict[str, Any]:
    """Inspect the build/compiled_models/<model>/<target>/ directory."""
    if not artifact_dir.exists():
        return {"exists": False, "vmfbs": [], "vmfb_size_bytes": None, "phases": []}
    vmfbs = sorted(artifact_dir.glob("*.vmfb"))
    phases = []
    phases_dir = artifact_dir / "phases"
    if phases_dir.exists():
        for p in sorted(phases_dir.iterdir()):
            if _PHASE_DUMP_RE.match(p.name) or p.suffix == ".mlir":
                phases.append({"name": p.name, "size_bytes": p.stat().st_size})
    benchmarks = []
    bench_dir = artifact_dir / "benchmarks"
    if bench_dir.exists():
        benchmarks = [b.name for b in sorted(bench_dir.glob("*.mlir"))]
    primary_vmfb = vmfbs[0] if vmfbs else None
    return {
        "exists": True,
        "vmfbs": [str(v.relative_to(REPO_ROOT)) for v in vmfbs],
        "primary_vmfb": str(primary_vmfb.relative_to(REPO_ROOT)) if primary_vmfb else None,
        "vmfb_size_bytes": primary_vmfb.stat().st_size if primary_vmfb else None,
        "phase_dumps": phases,
        "per_dispatch_benchmarks": benchmarks,
    }


def _resolve_target_devices(target: str) -> list[str]:
    yaml_path = REPO_ROOT / "models" / f"{target}.yaml"
    if not yaml_path.exists():
        return []
    cfg = yaml.safe_load(yaml_path.read_text())
    devices: list[str] = []
    for flag_list in (cfg.get("generic") or [], cfg.get("plugin_flags") or []):
        for f in flag_list:
            if not isinstance(f, str):
                continue
            if f.startswith("--iree-hal-target-device="):
                devices.append(f.split("=", 1)[1])
            elif f.startswith("--iree-hal-target-backends="):
                devices.extend(f.split("=", 1)[1].split(","))
    return sorted(set(devices))


def _compile_model(args: dict[str, Any]) -> dict[str, Any]:
    """Compile a model via `./merlin compile` and return parsed structure.

    Return shape:
        {
          "passed": bool,
          "exit_code": int,
          "vmfb": {"path": str, "size_bytes": int} | null,
          "phase_dumps": [{"name": str, "size_bytes": int}, ...],
          "per_dispatch_benchmarks": [str, ...],
          "target_devices": [str, ...],
          "artifact_dir": str,
          "stderr_tail": [str, ...]   # only populated on failure
        }
    """
    input_path = _require_str(args, "input")
    target = _require_str(args, "target")
    hw = args.get("hw")
    quantized = bool(args.get("quantized", False))
    extra_flags = args.get("extra_flags") or []
    if not isinstance(extra_flags, list):
        raise ToolError("extra_flags must be a list of strings")

    cmd = [str(MERLIN), "compile", input_path, "--target", target]
    if hw:
        cmd.extend(["--hw", str(hw)])
    if quantized:
        cmd.append("--quantized")
    cmd.extend(str(f) for f in extra_flags)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)

    input_p = pathlib.Path(input_path)
    model_stem = input_p.stem.split(".")[0]
    artifact_dir = REPO_ROOT / "build" / "compiled_models" / model_stem / target
    art = _inspect_artifact_dir(artifact_dir)

    out: dict[str, Any] = {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "command": cmd,
        "vmfb": (
            {"path": art["primary_vmfb"], "size_bytes": art["vmfb_size_bytes"]} if art.get("primary_vmfb") else None
        ),
        "all_vmfbs": art.get("vmfbs", []),
        "phase_dumps": art.get("phase_dumps", []),
        "per_dispatch_benchmarks": art.get("per_dispatch_benchmarks", []),
        "target_devices": _resolve_target_devices(target),
        "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)) if artifact_dir.exists() else None,
    }
    if result.returncode != 0:
        out["stderr_tail"] = result.stderr.splitlines()[-30:]
        out["stdout_tail"] = result.stdout.splitlines()[-15:]
    return out


# --- Tool: compile_help --------------------------------------------------


def _compile_help(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(MERLIN), "compile", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {"exit_code": result.returncode, "help": result.stdout}


# --- Registry ------------------------------------------------------------


TOOL_REGISTRY: list[ToolDefinition] = [
    ToolDefinition(
        name="compile_list_targets",
        description=(
            "List every available compile target (one row per models/*.yaml). "
            "Returns each target's name, default_hw, hw_variants, target_device "
            "(parsed from the yaml), and whether plugin_flags are present."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_list_compile_targets,
    ),
    ToolDefinition(
        name="compile_list_models",
        description=(
            "List every model directory under models/ that has a .mlir or .onnx. "
            "Use to discover canonical model paths before calling compile_model."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_list_models,
    ),
    ToolDefinition(
        name="compile_model",
        description=(
            "Compile a model (.mlir or .onnx) for a specific target. Returns "
            "parsed structure: VMFB path + size, phase dumps, per-dispatch "
            "benchmark list, target_devices resolved from the yaml. On failure "
            "includes stderr_tail. Maps to any user request 'compile <model> "
            "for <target>'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Path to .mlir or .onnx (relative to repo root)."},
                "target": {"type": "string", "description": "Target name (one of models/*.yaml without .yaml)."},
                "hw": {"type": "string", "description": "Optional hw sub-target from the yaml's targets: section."},
                "quantized": {"type": "boolean", "description": "Apply the yaml's quantized: flag block."},
                "extra_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra raw flags forwarded to ./merlin compile.",
                },
            },
            "required": ["input", "target"],
            "additionalProperties": False,
        },
        handler=_compile_model,
    ),
    ToolDefinition(
        name="compile_help",
        description="Verbatim `./merlin compile --help`. Use when you need every available flag.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_compile_help,
    ),
]


def list_tool_definitions() -> list[ToolDefinition]:
    return list(TOOL_REGISTRY)


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    for tool in TOOL_REGISTRY:
        if tool.name == name:
            return tool.handler(arguments)
    raise ToolError(f"unknown tool: {name!r}; known: {[t.name for t in TOOL_REGISTRY]}")
