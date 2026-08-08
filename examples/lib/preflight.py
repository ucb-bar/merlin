"""Report which stages of an example this machine can actually run, and what would fix the rest.

The examples deliberately span things anyone can do (grade a log) and things that need a toolchain, a
vendor SDK, or silicon nobody outside the lab has. Printing a definite verdict per stage up front is the
difference between "the example is broken" and "step 3 needs ZEPHYR_BASE".

Every check asks the library the same question the real code asks -- `spike.available()`,
`zephyr_model.available()` -- rather than re-deriving it here. A preflight that drifts from the code it
guards is worse than none, because it fails the run for a reason that is no longer true.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

STAGES = ("grade", "probe", "build", "package")


def _rows(board: str, sdk_dir: str | None):
    """(stage, ok, what it needs, how to get it) for each stage, cheapest first."""
    out = []

    have_numpy = _importable("numpy")
    out.append(("grade", have_numpy, "python + numpy",
                "pip install numpy — grading a returned console log needs nothing else"))

    try:
        from merlin.runtime.backends import spike
        sim = spike.available()
        detail = "" if sim else f" (looked for spike at {spike.spike_path()})"
    except Exception as exc:                                          # noqa: BLE001
        sim, detail = False, f" ({type(exc).__name__}: {exc})"
    out.append(("probe", sim, "spike + a riscv64 toolchain" + detail,
                "set MERLIN_SPIKE=/path/to/spike (or MERLIN_CHIPYARD to a chipyard checkout "
                "whose .conda-env provides riscv-tools)"))

    try:
        from merlin.runtime.backends import zephyr_model as zm
        zeph = zm.available()
        detail = "" if zeph else f" (ZEPHYR_BASE resolved to {zm._zephyr_base()})"
    except Exception as exc:                                          # noqa: BLE001
        zeph, detail = False, f" ({type(exc).__name__}: {exc})"
    out.append(("build", bool(zeph and sim), "the Zephyr tree this repo builds against" + detail,
                "set ZEPHYR_BASE to a zephyr-chipyard-sw checkout's zephyr_ws/zephyr"))

    # A UART-console board additionally needs its own SDK: the UART address and the clock rates its
    # baud divisor depends on are derived from that SDK's headers, never hardcoded. Asking for it here
    # rather than three hours into a build is the whole point of a preflight.
    needs_sdk = _board_needs_sdk(board)
    sdk_ok = (not needs_sdk) or bool(sdk_dir and Path(sdk_dir).is_dir())
    extra = ", plus this chip's own SDK checkout" if needs_sdk else ""
    fix = (f"pass --sdk-dir <checkout>, or export GEMMELOS_SDK, for {board}" if needs_sdk
           else f"{board} has a host-assisted console, so no vendor SDK is needed")
    out.append(("package", bool(zeph and sim and sdk_ok), "everything above" + extra, fix))
    return out


def _importable(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _board_needs_sdk(board: str) -> bool:
    try:
        from merlin.runtime import boards
        brd = boards.board(board)
        return brd.console != boards.CONSOLE_HTIF
    except Exception:                                                 # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", required=True)
    ap.add_argument("--sdk-dir", default=None)
    ap.add_argument("--require", default=None, choices=STAGES,
                    help="exit non-zero unless this stage is runnable")
    a = ap.parse_args(argv)

    rows = _rows(a.board, a.sdk_dir)
    width = max(len(s) for s, *_ in rows)
    print(f"preflight for {a.board}:")
    for stage, ok, needs, fix in rows:
        print(f"  [{'ok ' if ok else 'no '}] {stage:<{width}}  needs {needs}")
        if not ok:
            print(f"         -> {fix}")
    if not shutil.which("unzip"):
        print("  note: no `unzip` on PATH; the packaged zip can still be produced and inspected "
              "with python -m zipfile")
    if a.require:
        ok = dict((s, o) for s, o, *_ in rows)[a.require]
        if not ok:
            print(f"\nSTOP: this example's `{a.require}` stage cannot run here. The steps above it "
                  f"still can.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
