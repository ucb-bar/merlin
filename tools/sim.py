"""``./merlin sim`` — drive an mxGemmini fixture through the IREE plugin
pipeline, build the bare-metal bench ELF, and run it on the chipyard
VCS simulator. Sister of ``tools/spike.py``.

Pipeline::

    fixture.mlir
      └► ./merlin compile --target gemmini_mx_vcs[_fp4]
      └► ./merlin build --profile firesim --cmake-target bench_gemmini_mx_vcs_mlp_<fp8|fp4>
      └► make -C $CHIPYARD_ROOT/sims/<sim> run-binary-fast \
             CONFIG=<config> LOADMEM=1 BINARY=<elf>
      └► capture stdout, diff against --reference, exit PASS/FAIL.

VCS is the default simulator — Verilator can't compile the CVFPU IP that
``RadianceGemminiOnlyConfig`` pulls in (per the dev-blog 14.15 + 14.16
analysis).
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re
import subprocess
import sys

_LOG = logging.getLogger("merlin.sim")

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "build" / "sim"
_MERLIN_CLI = _REPO_ROOT / "merlin"

# Default mapping from (compile-target, --hw) → cmake bench target.
# Extend by passing `--bench-target <cmake_target>` to override for any
# (target, hw) pair not listed here. `None` matches the YAML's default_hw
# when --hw is omitted on the CLI.
_TARGET_TO_BENCH = {
    ("gemmini_mx_vcs", None): "bench_gemmini_mx_vcs_mlp_fp8",
    ("gemmini_mx_vcs", "VCS"): "bench_gemmini_mx_vcs_mlp_fp8",
    ("gemmini_mx_vcs", "VCS_FP4"): "bench_gemmini_mx_vcs_mlp_fp4",
}


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        type=pathlib.Path,
        help="Input .mlir fixture (any model — see tests/integration/<target>/fixtures/ for examples).",
    )
    parser.add_argument(
        "--target",
        default="gemmini_mx_vcs",
        help="Model YAML target (default: gemmini_mx_vcs). Any models/<target>.yaml is accepted; pass --bench-target if the cmake bench target name does not match the default mapping.",
    )
    parser.add_argument(
        "--hw",
        default=None,
        help="Hardware sub-target inside the YAML's `targets:` map (e.g. VCS, VCS_FP4). Defaults to the YAML's `default_hw`.",
    )
    parser.add_argument(
        "--bench-target",
        default=None,
        help="Explicit cmake bench target name. Overrides the default mapping in _TARGET_TO_BENCH; required when --target is not a key in the default mapping.",
    )
    parser.add_argument(
        "--simulator",
        default="vcs",
        choices=["vcs", "verilator"],
        help="Chipyard simulator backend (default: vcs)",
    )
    parser.add_argument(
        "--config",
        default="RadianceGemminiOnlyConfig",
        help="Chipyard CONFIG (default: RadianceGemminiOnlyConfig)",
    )
    parser.add_argument(
        "--reference",
        type=pathlib.Path,
        default=None,
        help="Path to expected output (one i32 per line). If set, run "
        "outputs are diffed against this and an exit code of 0/1 is "
        "returned.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help=f"Directory for produced artifacts (default: {_DEFAULT_OUTPUT_ROOT}/<fixture>)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep simulator working dir + log on success",
    )
    parser.add_argument(
        "--build-dir",
        default="host-merlin-release",
        help="Host build dir for iree-compile (default: host-merlin-release)",
    )
    parser.add_argument(
        "--firesim-build-dir",
        default="firesim-merlin-release",
        help="Firesim build dir holding the bench ELF (default: firesim-merlin-release)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip ./merlin build step (use a pre-built ELF)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip ./merlin compile step (use a pre-built VMFB)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Simulator wallclock timeout in seconds (default 900)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def _run(cmd: list[str], **kwargs) -> int:
    _LOG.info("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], **kwargs)


def _bench_elf_path(repo: pathlib.Path, build_dir: str, bench_name: str) -> pathlib.Path:
    return (
        repo
        / "build"
        / build_dir
        / "runtime"
        / "plugins"
        / "merlin-samples"
        / "Radiance"
        / "mxgemmini_vcs_runner"
        / bench_name
    )


_NUMERIC_RE = re.compile(r"^\s*-?\d+\s*$")


def _extract_numeric_lines(text: str) -> list[int]:
    """Pull integer-only lines out of the simulator stdout."""
    out: list[int] = []
    for line in text.splitlines():
        if _NUMERIC_RE.match(line):
            try:
                out.append(int(line.strip()))
            except ValueError:
                continue
    return out


def _diff_against_reference(sim_stdout: str, reference_path: pathlib.Path) -> tuple[bool, str]:
    expected = [int(s) for s in reference_path.read_text().split() if s.strip()]
    got = _extract_numeric_lines(sim_stdout)
    # The simulator may print other numeric lines (DRAM addresses,
    # cycle counts). The expected output is the *trailing* len(expected)
    # numeric lines — that's what the C runner prints just before
    # "[mxgemmini-vcs] PASS".
    if len(got) < len(expected):
        return (
            False,
            f"got fewer numeric lines than expected ({len(got)} < {len(expected)})",
        )
    tail = got[-len(expected) :]
    if tail == expected:
        return True, "all values match"
    diffs = [(i, e, g) for i, (e, g) in enumerate(zip(expected, tail)) if e != g]
    msg = f"{len(diffs)}/{len(expected)} values differ; " f"first 5: " + ", ".join(
        f"[{i}] expected={e} got={g}" for i, e, g in diffs[:5]
    )
    return False, msg


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[merlin-sim] %(message)s",
    )

    chipyard_root = os.environ.get("CHIPYARD_ROOT")
    if not chipyard_root:
        sys.stderr.write(
            "[merlin-sim] FAIL: CHIPYARD_ROOT not set. " "Source build_tools/firesim/setup_toolchain.sh first.\n"
        )
        return 2

    fixture: pathlib.Path = args.input.resolve()
    if not fixture.exists():
        sys.stderr.write(f"[merlin-sim] FAIL: input MLIR not found: {fixture}\n")
        return 2

    if args.bench_target is not None:
        bench_name = args.bench_target
    elif (args.target, args.hw) in _TARGET_TO_BENCH:
        bench_name = _TARGET_TO_BENCH[(args.target, args.hw)]
    else:
        sys.stderr.write(
            f"[merlin-sim] FAIL: --target {args.target!r} --hw {args.hw!r} has "
            f"no default cmake bench target; pass --bench-target <name> "
            f"explicitly. Defaults: {sorted(_TARGET_TO_BENCH.keys())}\n"
        )
        return 2
    name = fixture.stem
    out_dir = args.output_dir if args.output_dir is not None else _DEFAULT_OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: ./merlin compile ----
    if not args.skip_compile:
        compile_cmd = [
            _MERLIN_CLI,
            "compile",
            fixture,
            "--target",
            args.target,
            "--build-dir",
            args.build_dir,
            "--output-dir",
            out_dir,
        ]
        if args.hw is not None:
            compile_cmd.extend(["--hw", args.hw])
        rc = _run(compile_cmd)
        if rc != 0:
            sys.stderr.write(f"[merlin-sim] FAIL: ./merlin compile rc={rc}\n")
            return rc
        vmfbs = list(out_dir.glob("*.vmfb"))
        if not vmfbs:
            sys.stderr.write(f"[merlin-sim] FAIL: no .vmfb produced under {out_dir}\n")
            return 1
        _LOG.info("compile produced: %s", vmfbs[0])

    # ---- Step 2: ./merlin build ----
    if not args.skip_build:
        rc = _run(
            [
                _MERLIN_CLI,
                "build",
                "--profile",
                "firesim",
                "--cmake-target",
                bench_name,
            ]
        )
        if rc != 0:
            sys.stderr.write(f"[merlin-sim] FAIL: ./merlin build rc={rc}\n")
            return rc

    elf = _bench_elf_path(_REPO_ROOT, args.firesim_build_dir, bench_name)
    if not elf.exists():
        sys.stderr.write(
            f"[merlin-sim] FAIL: bench ELF not found: {elf}\n"
            f"        Run `./merlin build --profile firesim --cmake-target {bench_name}` "
            "or check --firesim-build-dir.\n"
        )
        return 1
    _LOG.info("bench ELF: %s", elf)

    # ---- Step 3: invoke the simulator ----
    sim_dir = pathlib.Path(chipyard_root) / "sims" / args.simulator
    if not (sim_dir / "Makefile").exists():
        sys.stderr.write(f"[merlin-sim] FAIL: chipyard sim dir has no Makefile: {sim_dir}\n")
        return 1
    log_path = out_dir / f"{bench_name}.simlog"
    cmd = [
        "make",
        "-C",
        str(sim_dir),
        "run-binary-fast",
        f"CONFIG={args.config}",
        "LOADMEM=1",
        f"BINARY={elf}",
    ]
    _LOG.info("$ %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as e:
        sys.stderr.write(f"[merlin-sim] FAIL: simulator timed out after {args.timeout}s\n")
        log_path.write_text((e.stdout or b"").decode("utf-8", errors="replace"))
        return 1

    sim_stdout = proc.stdout.decode("utf-8", errors="replace")
    log_path.write_text(sim_stdout)
    _LOG.info("simulator log: %s", log_path)

    if proc.returncode != 0:
        sys.stderr.write(
            f"[merlin-sim] FAIL: simulator exited with rc={proc.returncode}\n" f"        See log: {log_path}\n"
        )
        return proc.returncode

    # ---- Step 4: diff ----
    if args.reference is None:
        print("[merlin-sim] no --reference given; simulator stdout follows:")
        print("---")
        print(sim_stdout)
        print("---")
        print(f"[merlin-sim] PASS (no reference; log saved to {log_path})")
        return 0

    ref = args.reference.resolve()
    if not ref.exists():
        sys.stderr.write(f"[merlin-sim] FAIL: reference not found: {ref}\n")
        return 1
    ok, detail = _diff_against_reference(sim_stdout, ref)
    if ok:
        print(f"[merlin-sim] PASS — {detail} (log: {log_path})")
        return 0
    sys.stderr.write(f"[merlin-sim] FAIL — {detail}\n        log: {log_path}\n")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
