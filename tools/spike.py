"""`./merlin spike` — drive a Gemmini fixture through the real IREE plugin
pipeline and (optionally) run the resulting artifact on Spike.

This is a thin wrapper around `./merlin compile --target gemmini_spike`.
The previous incarnation of this script drove `iree-opt --pass-pipeline=...`
directly, bypassing the IREE plugin pipeline — that approach was wrong:
Merlin's whole purpose is to be an IREE plugin, so the gemmini compile
flow must go through `iree-compile` (i.e. `./merlin compile`) and let
IREE's standard DispatchCreation + bufferization + codegen handle the
downstream phases.

Pipeline:
  input.mlir (tensor-domain linalg)
    └► `./merlin compile --target gemmini_spike` (gemmini plugin runs at
       the post-global-optimization hook: linalg.matmul → gemmini.matmul →
       gemmini.matmul_tile, then optionally back to linalg for the IREE
       codegen-fallback path; see models/gemmini_spike.yaml).
    └► .vmfb with embedded RISC-V ELF.

Running the resulting .vmfb on Spike (extracting the ELF from the vmfb
and invoking `spike --extension=gemmini pk <elf>`) is handled by the
firesim sample flow at samples/SaturnOPU/simple_embedding_ukernel/ —
the same pattern saturn_opu_spike.yaml uses. Pointing this script at
that flow is captured as a follow-up; for now `./merlin spike` produces
the artifact and stops.

NOTE: this script requires `./merlin build --profile gemmini` to have
populated `build/host-merlin-debug/tools/iree-compile`.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import subprocess
import sys

_LOG = logging.getLogger("merlin.spike")

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "build" / "spike"
_MERLIN_CLI = _REPO_ROOT / "merlin"


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=pathlib.Path, help="Input .mlir fixture (tensor-domain)")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help=("Directory for the produced .vmfb (default: " "build/spike/<basename>)"),
    )
    parser.add_argument(
        "--target",
        default="gemmini_spike",
        help="Model YAML name (default: gemmini_spike)",
    )
    parser.add_argument(
        "--build-dir",
        default="host-merlin-debug",
        help=("Which build dir to use for iree-compile " "(default: host-merlin-debug)"),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[spike] %(message)s",
    )

    input_mlir: pathlib.Path = args.input.resolve()
    if not input_mlir.exists():
        raise SystemExit(f"Input MLIR not found: {input_mlir}")

    name = input_mlir.stem
    out_dir = args.output_dir if args.output_dir is not None else _DEFAULT_OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(_MERLIN_CLI),
        "compile",
        str(input_mlir),
        "--target",
        args.target,
        "--build-dir",
        args.build_dir,
        "--output-dir",
        str(out_dir),
    ]
    _LOG.info("$ %s", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.stderr.write(f"[spike] FAIL: ./merlin compile rc={rc}\n")
        return rc

    vmfbs = list(out_dir.glob("*.vmfb"))
    if not vmfbs:
        sys.stderr.write(f"[spike] FAIL: ./merlin compile succeeded but no .vmfb under " f"{out_dir}\n")
        return 1

    print(f"[spike] PASS: produced {vmfbs[0]}")
    print(
        "[spike] note: extracting an ELF from the .vmfb and running it on "
        "spike+pk is a follow-up; see the firesim sample at "
        "samples/SaturnOPU/simple_embedding_ukernel/ for the pattern."
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
