"""Command line of ``python -m merlin.model_check`` (contract: ``docs/reference/model_check.md`` §2)."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from merlin.model_check import report as R
from merlin.model_check.models import ModelsFileError
from merlin.model_check.models import load as load_models

SPLITS = ("visible", "sealed", "all")
ENGINES = ("spike", "gsim")


def tree_sha256(root: Path) -> str:
    """Content digest of a directory tree: sorted relative paths and file bytes (symlinks by target)."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink()):
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode() + b"\0")
        if path.is_symlink():
            digest.update(b"L" + os.readlink(path).encode())
        else:
            digest.update(b"F" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _common(p: argparse.ArgumentParser, *, submission: bool = True) -> None:
    if submission:
        p.add_argument("--submission", type=Path, required=True, help="device package (has manifest.yaml)")
    p.add_argument("--models", type=Path, default=None, help="models file (default: the target's)")
    p.add_argument("--bundle", type=Path, default=None, help="frozen numerics bundle")
    p.add_argument("--work", type=Path, default=None, help="build directory (default under out/)")
    p.add_argument("--out", type=Path, required=True, help="report JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m merlin.model_check", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="build, replay kernels, run chains, judge the model")
    _common(check)
    check.add_argument("--model", required=True)
    check.add_argument("--program", default=None)
    check.add_argument("--split", choices=SPLITS, required=True)
    check.add_argument("--decode-steps", type=int, default=None)
    check.add_argument("--jobs", type=int, default=1)
    check.add_argument("--timeout-s", type=float, default=None)
    check.add_argument("--differential", action="store_true")
    check.add_argument("--agent-view", action="store_true")

    replay = sub.add_parser("replay-kernel", help="replay device kernels on frozen boundary inputs")
    _common(replay)
    replay.add_argument("--model", required=True)
    replay.add_argument("--program", required=True)
    group = replay.add_mutually_exclusive_group(required=True)
    group.add_argument("--group", dest="group", default=None, help="source_region_id of the group")
    group.add_argument("--all", dest="all_groups", action="store_true")
    replay.add_argument("--split", choices=SPLITS, required=True)
    replay.add_argument("--engine", choices=ENGINES, default="spike")
    replay.add_argument("--timeout-s", type=float, default=None)
    replay.add_argument("--agent-view", action="store_true")

    census = sub.add_parser("census", help="build only; classify every contraction")
    _common(census)
    census.add_argument("--model", required=True)
    census.add_argument("--program", default=None)

    binding = sub.add_parser("binding", help="Spike / RTL / FireSim binding receipt")
    _common(binding, submission=False)

    verify = sub.add_parser("bundle-verify", help="check every file of a numerics bundle")
    _common(verify, submission=False)
    return parser


def _not_implemented(args: argparse.Namespace, report: dict) -> dict:
    report["failures"].append(
        {"kind": "infrastructure", "gate": None, "detail": f"{args.command}: not implemented in this version"}
    )
    return report


_COMMANDS: dict[str, Callable[[argparse.Namespace, dict], dict]] = {}


def register(name: str):
    def wrap(fn: Callable[[argparse.Namespace, dict], dict]):
        _COMMANDS[name] = fn
        return fn

    return wrap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    submission = {}
    if getattr(args, "submission", None) is not None:
        root = args.submission
        submission = {"path": str(root), "tree_sha256": tree_sha256(root) if root.is_dir() else None}
    report = R.new_report(
        args.command, model=getattr(args, "model", None), split=getattr(args, "split", None), submission=submission
    )
    started = time.monotonic()
    try:
        if submission and submission["tree_sha256"] is None:
            raise ModelsFileError(f"submission is not a directory: {args.submission}")
        if getattr(args, "model", None) is not None:
            models = load_models(args.models)
            models.model(args.model)
        report = _COMMANDS.get(args.command, _not_implemented)(args, report)
    except ModelsFileError as error:
        report["failures"].append({"kind": "usage", "gate": None, "detail": str(error)})
    except Exception as error:  # noqa: BLE001 -- every failure must still produce a report
        report["failures"].append(
            {"kind": "infrastructure", "gate": None, "detail": f"{type(error).__name__}: {error}"}
        )
    report["timing"]["total_seconds"] = round(time.monotonic() - started, 3)
    R.settle(report)
    if getattr(args, "agent_view", False):
        report = R.agent_view(report)
    R.write(report, args.out)
    print(f"{args.command}: {report['status']} -> {args.out}", file=sys.stderr)
    return int(report["exit_code"])
