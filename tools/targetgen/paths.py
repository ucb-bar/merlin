"""TargetGen output-directory resolvers.

Centralizes the convention that artifacts land under
`<base>/<target_name>/...` and that subcommands can override either piece.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _target_out_dir(base: str, target_name: str) -> Path:
    return utils.REPO_ROOT / base / target_name


def _prompts_out_dir(base: str | None, target_out_dir: Path) -> Path:
    if not base:
        return target_out_dir / "prompts"
    path = Path(base)
    return path if path.is_absolute() else utils.REPO_ROOT / path


def _resolve_target_dir_arg(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else utils.REPO_ROOT / path


def _resolve_execute_target_dir(args: argparse.Namespace) -> Path:
    if args.from_dir:
        return _resolve_target_dir_arg(args.from_dir)
    if not args.capability:
        raise ValueError("execute requires either a capability spec path or --from-dir")
    capability_name = Path(args.capability).resolve().parent.name
    return _target_out_dir(args.out_dir, capability_name)


def _resolve_stage_target_dir(args: argparse.Namespace) -> Path:
    if args.from_dir:
        return _resolve_target_dir_arg(args.from_dir)
    if not args.capability:
        raise ValueError("stage-mutation requires either a capability spec path or --from-dir")
    capability_name = Path(args.capability).resolve().parent.name
    return _target_out_dir(args.out_dir, capability_name)
