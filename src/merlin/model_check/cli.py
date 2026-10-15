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
MODES = ("fast", "full")


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


def _no_cache(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--no-cache", action="store_true", help="build every kernel and run every image (no kernel or replay cache)"
    )


def _engine(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host-package", type=Path, default=None, help="host package (default: the bundle's)")
    p.add_argument(
        "--contract", type=Path, default=None, help="target contract binding the simulators (default: the bundle's)"
    )
    p.add_argument("--rtl-facts", type=Path, default=None, help="RTL facts matching --contract (default: the bundle's)")
    p.add_argument("--dram-base", type=lambda v: int(v, 0), default=0x80000000)
    p.add_argument("--dram-bytes", type=lambda v: int(v, 0), default=16 << 30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m merlin.model_check", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="replay kernels and judge the model (fast); --mode full also runs chains")
    _common(check)
    check.add_argument("--model", required=True)
    check.add_argument("--program", default=None)
    check.add_argument("--split", choices=SPLITS, required=True)
    check.add_argument("--mode", choices=MODES, default="fast")
    check.add_argument("--decode-steps", type=int, default=None)
    check.add_argument("--jobs", type=int, default=1)
    check.add_argument("--timeout-s", type=float, default=None)
    check.add_argument("--differential", action="store_true")
    check.add_argument("--agent-view", action="store_true")
    check.add_argument("--engine", choices=ENGINES, default="spike")
    _engine(check)
    _no_cache(check)

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
    _engine(replay)
    _no_cache(replay)

    export = sub.add_parser("kernels-export", help="interface modules, call ABI and entries of a model's kernels")
    _common(export, submission=False)
    export.add_argument("--model", required=True)
    export.add_argument("--program", default=None)
    export.add_argument("--split", choices=("visible",), default="visible", help="only the visible split exports")
    export.add_argument("--out-dir", type=Path, required=True, help="directory to write <model>/ and index.md")

    census = sub.add_parser("census", help="build only; classify every contraction")
    _common(census)
    census.add_argument("--model", required=True)
    census.add_argument("--program", default=None)

    binding = sub.add_parser("binding", help="Spike / RTL / FireSim binding receipt")
    _common(binding, submission=False)
    _engine(binding)
    binding.add_argument("--board", default=None, help="board whose memory span Spike models (default: the spec's)")
    binding.add_argument("--spec", type=Path, default=None, help="binding spec (default: binding.yaml by --models)")
    binding.add_argument("--image", type=Path, action="append", default=[], help="replay image to check (repeatable)")

    verify = sub.add_parser("bundle-verify", help="check every file of a numerics bundle")
    _common(verify, submission=False)

    build = sub.add_parser("bundle-build", help="catalog, census and reference calls for a model's programs")
    _common(build, submission=False)
    build.add_argument("--captures", type=Path, required=True, help="sealed capture root (<model>/run/capture)")
    build.add_argument("--model", required=True)
    build.add_argument("--program", default=None)
    build.add_argument("--keep", choices=("all", "sample"), default="all", help="which device calls to record")
    build.add_argument("--host-package", type=Path, required=True, help="host package replays are built with")
    build.add_argument("--contract", type=Path, required=True, help="target contract binding the simulators")
    build.add_argument("--rtl-facts", type=Path, required=True, help="RTL facts matching --contract")

    variants = sub.add_parser("variants-build", help="seeded shape variants of every unique kernel (sealed cases)")
    _common(variants, submission=False)
    variants.add_argument("--model", required=True)
    variants.add_argument("--program", default=None)
    variants.add_argument("--seed", type=int, required=True, help="private rule seed (never shown to an agent)")

    prep = sub.add_parser("inputs-prepare", help="select and preprocess a model's evaluation/calibration inputs")
    _common(prep, submission=False)
    prep.add_argument("--model", required=True)
    prep.add_argument("--sources", type=Path, default=None, help="sources.yaml (default: the target's)")
    prep.add_argument("--data", type=Path, required=True, help="directory holding the pinned source files")
    prep.add_argument("--private", type=Path, required=True, help="private root (sealed and derived inputs)")
    prep.add_argument("--python", type=Path, required=True, help="the model's pinned environment interpreter")
    prep.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="model-specific inputs, e.g. tokenizer_path=..., rows_python=..., policy_path=...",
    )
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
    from merlin.model_check import commands as _commands  # noqa: F401 -- registers the subcommands

    args = build_parser().parse_args(argv)
    if getattr(args, "timeout_s", "unset") is None:
        args.timeout_s = 3600.0
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
