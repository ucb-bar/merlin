"""TargetGen spec loading — load_capability_spec + load_deployment_overlay wrappers.

Each TargetGen subcommand starts by loading a target's capability.yaml +
optional deployment overlay. The `_load_*` helpers here normalize that
pattern.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from targetgen import load_capability_spec, load_deployment_overlay


def _load_inputs(args: argparse.Namespace):
    capabilities = load_capability_spec(args.capability)
    if args.overlay:
        capabilities.deployment = load_deployment_overlay(args.overlay)
    return capabilities


def _load_execute_inputs(args: argparse.Namespace, target_dir: Path):
    if args.capability:
        capabilities = load_capability_spec(args.capability)
        if args.overlay:
            capabilities.deployment = load_deployment_overlay(args.overlay)
        return capabilities
    if (target_dir / "execution_bundle.json").exists():
        return None
    raise ValueError(
        "execute requires either a capability spec path or an existing " "--from-dir with execution artifacts"
    )


def _load_stage_inputs(args: argparse.Namespace, target_dir: Path):
    if args.capability:
        return _load_inputs(args)
    capability_snapshot = target_dir / "inputs" / "capability.yaml"
    if not capability_snapshot.exists():
        raise ValueError(
            "stage-mutation could not find inputs/capability.yaml in the target directory; "
            "run targetgen generate first or provide a capability spec path"
        )
    capabilities = load_capability_spec(str(capability_snapshot))
    overlay_snapshot = target_dir / "inputs" / "overlay.yaml"
    if overlay_snapshot.exists():
        capabilities.deployment = load_deployment_overlay(str(overlay_snapshot))
    return capabilities
