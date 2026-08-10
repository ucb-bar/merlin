"""Report which stages of an example this machine can run, and name the exact thing blocking the rest.

The examples span things anyone can do (grade a log) and things that need a toolchain, a vendor SDK, or
silicon nobody outside the lab has. A per-stage verdict up front is the difference between "the example is
broken" and "step 3 needs ZEPHYR_SDK_INSTALL_DIR".

Two rules this file exists to obey:

* the VERDICT comes from the library's own guard (`spike.available()`, `zephyr_model.available()`), never
  from a re-derivation here — a preflight that drifts from the code it guards fails runs for reasons that
  are no longer true;
* the EXPLANATION is itemised, because a single bool cannot be acted on. An earlier version reported a
  missing `cmake` as "needs the Zephyr tree this repo builds against", which sends a reader to fix the one
  thing that was already fine.

For the whole-repo version of this (every experiment capability, not just these two examples), see
`build_tools/scripts/check_repro_env.py`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

STAGES = ("grade", "probe", "build", "package")


def _components():
    """(label, present, env var that overrides it, resolved path) for each host prerequisite.

    Resolved through the library's own path helpers so the paths printed are the paths that will be used.
    """
    rows = []
    try:
        from merlin.runtime.backends import spike
        rows.append(("spike", spike.spike_path().is_file(), "MERLIN_SPIKE", spike.spike_path()))
        rows.append(("riscv64 gcc", spike.gcc_path().is_file(), "MERLIN_RISCV_GCC", spike.gcc_path()))
    except Exception as exc:                                          # noqa: BLE001
        rows.append((f"merlin importable ({type(exc).__name__})", False,
                     "see README: uv sync --all-extras", Path("-")))
        return rows
    # The compiler half. `zephyr_model.available()` deliberately reports only on the Zephyr build
    # environment, so a machine with cmake/ninja/SDK but no clang-23 passes that guard and then fails at
    # the first object. Itemise it here: clang-23 compiles every RISC-V object, mlir-translate lowers the
    # OpenMP IR every MULTI-hart image needs, and the lowering runner itself executes inside model2MLIR's
    # venv. See docs/guides/llvm_toolchain.md -- third_party/llvm-install is gitignored, so a fresh clone
    # has to build it.
    try:
        from merlin.llvmlower import toolchain as tc
        rows.append(("clang-23", tc.clang().is_file(), "MERLIN_CLANG "
                     "(see docs/guides/llvm_toolchain.md)", tc.clang()))
        rows.append(("mlir-translate", tc.mlir_translate().is_file(), "MERLIN_MLIR_TRANSLATE "
                     "(see docs/guides/llvm_toolchain.md)", tc.mlir_translate()))
        rows.append(("m2m venv", tc.m2m_python().is_file(), "MERLIN_M2M_DIR / MERLIN_M2M_VENV",
                     tc.m2m_python()))
    except Exception as exc:                                          # noqa: BLE001
        rows.append((f"llvm toolchain ({type(exc).__name__}: {exc})", False, "-", Path("-")))

    try:
        from merlin.runtime.backends import zephyr_model as zm
        for tool in ("cmake", "ninja"):
            found = zm.build_tool(tool)
            rows.append((tool, found is not None, "PATH, or MERLIN_CHIPYARD",
                         found or Path("(not on PATH)")))
        rows.append(("zephyr tree", zm._zephyr_base().is_dir(), "ZEPHYR_BASE", zm._zephyr_base()))
        rows.append(("zephyr sdk", zm._sdk_dir().is_dir(), "ZEPHYR_SDK_INSTALL_DIR", zm._sdk_dir()))
    except Exception as exc:                                          # noqa: BLE001
        rows.append((f"zephyr backend ({type(exc).__name__}: {exc})", False, "-", Path("-")))
    return rows


def _llvm_ok() -> bool:
    """Can this machine compile at all? `toolchain.available()` is the library's own guard (clang + the
    m2m venv the lowering runner executes in); mlir-translate is additionally required by any multi-hart
    image, and every example builds one."""
    try:
        from merlin.llvmlower import toolchain as tc
        return bool(tc.available()) and tc.mlir_translate().is_file()
    except Exception:                                                 # noqa: BLE001
        return False


def _rows(board: str, sdk_dir: str | None):
    """(stage, ok, one-line summary) per stage, cheapest first."""
    out = [("grade", _importable("numpy"), "python + numpy")]

    try:
        from merlin.runtime.backends import spike
        sim = spike.available()
    except Exception:                                                 # noqa: BLE001
        sim = False
    out.append(("probe", sim, "spike + a riscv64 toolchain + the bare-metal harness"))

    try:
        from merlin.runtime.backends import zephyr_model as zm
        zeph = bool(zm.available())
    except Exception:                                                 # noqa: BLE001
        zeph = False
    zeph = zeph and _llvm_ok()
    out.append(("build", zeph, "the above, plus clang-23/mlir-translate, cmake/ninja, a Zephyr tree "
                               "and the Zephyr SDK"))

    # A UART-console board additionally needs its own SDK: the UART address and the clock rates its baud
    # divisor depends on are derived from that SDK's headers, never hardcoded. Asking for it here rather
    # than three hours into a build is the point of a preflight.
    needs_sdk = _board_needs_sdk(board)
    sdk_ok = (not needs_sdk) or bool(sdk_dir and Path(sdk_dir).is_dir())
    extra = ", plus this chip's own SDK checkout" if needs_sdk else " (no vendor SDK needed)"
    out.append(("package", bool(zeph and sdk_ok), "everything above" + extra))
    return out, needs_sdk, sdk_ok


def _importable(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:                                                 # noqa: BLE001
        return False


def _board_needs_sdk(board: str) -> bool:
    try:
        from merlin.runtime import boards
        return boards.board(board).console != boards.CONSOLE_HTIF
    except Exception:                                                 # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", required=True)
    ap.add_argument("--sdk-dir", default=None)
    ap.add_argument("--require", default=None, choices=STAGES,
                    help="exit non-zero unless this stage is runnable")
    a = ap.parse_args(argv)

    rows, needs_sdk, sdk_ok = _rows(a.board, a.sdk_dir)
    width = max(len(s) for s, *_ in rows)
    print(f"preflight for {a.board}:")
    for stage, ok, summary in rows:
        print(f"  [{'ok ' if ok else 'no '}] {stage:<{width}}  {summary}")

    comps = _components()
    missing = [c for c in comps if not c[1]]
    if missing or a.require in ("build", "package"):
        print("\nhost prerequisites:")
        cw = max(len(c[0]) for c in comps)     # so a long label ('mlir-translate') keeps the column
        for label, present, envvar, path in comps:
            mark = "ok " if present else "no "
            print(f"  [{mark}] {label:<{cw}} {path}")
            if not present:
                print(f"         set {envvar}")
    if needs_sdk and not sdk_ok:
        print(f"\n  [no ] chip SDK      pass --sdk-dir <checkout> (or export it; see the example's "
              f"README)")
    if missing:
        print("\nSee examples/README.md ('Set up a machine from scratch') for where each of these comes "
              "from, and build_tools/scripts/check_repro_env.py for the whole-repo version of this "
              "report.")

    if a.require:
        ok = dict((s, o) for s, o, *_ in rows)[a.require]
        if not ok:
            sys.stdout.flush()      # or the STOP line lands above the report explaining it
            print(f"\nSTOP: this example's `{a.require}` stage cannot run here. The stages above it "
                  f"still can.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
