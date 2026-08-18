"""Which stages of the Triton example this machine can run, and the exact thing blocking the rest.

The stages split cleanly by cost: everything up to and including a numerically checked command buffer
needs only the repo and its venv, while certifying that command buffer on real RTL needs a simulator
someone had to build. A per-stage verdict up front is the difference between "the example is broken"
and "certify needs a Gemmini Verilator build".

Same rule as `examples/lib/preflight.py`: the VERDICT comes from the library's own guard
(`toolchain.probe()`, `gemmini.available()`), never from a re-derivation here. A preflight that drifts
from the code it guards blocks runs for reasons that stopped being true.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = REPO_ROOT / "out/artifacts/targets/gemmini/hand_v0"

# stage -> the components it needs. Ordered by cost, cheapest first.
STAGES = {
    "walk": ("merlin", "numpy", "triton", "target package"),
    "compile": ("merlin", "numpy", "triton", "target package"),
    "route": ("merlin", "numpy", "triton", "target package"),
    "converge": ("merlin", "numpy", "triton", "target package"),
    "certify-l1": ("merlin", "numpy", "triton", "target package", "spike-gemmini"),
    "certify-l2": ("merlin", "numpy", "triton", "target package", "gemmini verilator"),
}


def _components(package: Path) -> list[tuple[str, bool, str]]:
    """(label, present, what fixes it) for every prerequisite, itemised.

    Itemised because a single bool cannot be acted on: "the example needs a toolchain" sends a reader
    to fix whichever part they guess at, which is frequently the part that was already fine.
    """
    rows: list[tuple[str, bool, str]] = []
    try:
        import merlin  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return [(f"merlin importable ({type(exc).__name__})", False,
                 "uv sync --all-extras   (plain `python` is not on PATH; use .venv/bin/python)")]
    rows.append(("merlin", True, ""))

    try:
        import numpy  # noqa: F401
        rows.append(("numpy", True, ""))
    except Exception:  # noqa: BLE001
        rows.append(("numpy", False, "uv sync --all-extras"))

    from merlin.triton import toolchain
    probe = toolchain.probe()
    rows.append((f"triton (pinned {probe.pinned}, found {probe.installed})", probe.compatible,
                 "" if probe.compatible else f"uv pip install -e '.[triton]'   [{probe.reason}]"))
    for note in probe.notes:
        rows.append((f"  triton note: {note}", False, "reinstall triton from a real wheel"))

    ok = package.is_dir()
    rows.append((f"target package ({_rel(package)})", ok,
                 "" if ok else "pass --package <dir>, or see docs/guides/adding_a_target.md"))

    # The two simulators. Asked of the backend, which is what the certification itself asks.
    try:
        from merlin.runtime.backends import gemmini
        for label, runner, fix in (
            ("spike-gemmini", "spike", "set MERLIN_SPIKE / MERLIN_CHIPYARD (see examples/README.md)"),
            ("gemmini verilator", "verilator", "build the Gemmini Verilator sim in your chipyard checkout"),
        ):
            avail = gemmini.available(runner)
            rows.append((label, avail, "" if avail else fix))
    except Exception as exc:  # noqa: BLE001
        rows.append((f"gemmini backend ({type(exc).__name__}: {exc})", False, "see docs/"))
    return rows


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--require", metavar="STAGE",
                        help="exit non-zero unless this stage can run (used by run.sh)")
    args = parser.parse_args(argv)

    package = Path(args.package)
    if not package.is_absolute():
        package = REPO_ROOT / package
    rows = _components(package)
    present = {label.split(" (")[0].strip(): ok for label, ok, _ in rows}

    print("\n  components")
    for label, ok, fix in rows:
        mark = "yes" if ok else "NO "
        print(f"    [{mark}] {label}")
        if not ok and fix:
            print(f"          fix: {fix}")

    print("\n  stages")
    verdicts = {}
    for stage, needs in STAGES.items():
        missing = [n for n in needs if not present.get(n, False)]
        verdicts[stage] = not missing
        status = "can run" if not missing else f"blocked on {', '.join(missing)}"
        print(f"    {stage:12s} {status}")

    if args.require:
        if args.require not in STAGES:
            print(f"\n  unknown stage {args.require!r}; known: {', '.join(STAGES)}", file=sys.stderr)
            return 2
        if not verdicts[args.require]:
            missing = [n for n in STAGES[args.require] if not present.get(n, False)]
            print(f"\n  cannot run {args.require!r}: missing {', '.join(missing)}", file=sys.stderr)
            return 1
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
