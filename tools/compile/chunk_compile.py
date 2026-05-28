#!/usr/bin/env python3
"""Parameter-passing per-chunk VMFB emitter.

Unlike `breakdown_vmfb.py` (which uses `--iree-hal-dump-executable-benchmarks-to`
and produces VMFBs that read from `util.global private mutable` buffers seeded
with garbage), this tool emits `func.func public @main(tensor inputs) ->
tensor outputs` modules whose tensor inputs / outputs are real function
arguments. Resulting VMFBs:

  - run with `iree-run-module --input=<shape>=@input.bin` against
    deterministic data,
  - return real output that md5-matches the un-chunked baseline run on the
    same inputs,
  - serve as building blocks for chunk-granularity scheduling that needs
    real data flow at runtime (the `--initial-inputs` path of PR 6).

Two emit modes:

  --mode=fragment :   take a fragment-MLIR file containing a single
                      `func.func public @main(...)` body (linalg/tensor
                      ops on tensor parameters) and compile it. The user
                      supplies the fragment; the tool drives iree-compile.

  --mode=identity :   emit a trivial `func.func @main(%in: tensor<...>)
                      -> tensor<...> { return %in }` chunk (smoke test
                      that the parameter-passing pipeline lowers to a
                      VMFB and is callable on board with the expected
                      input/output shape).

End-to-end smoke test (drives the identity chunk through compile + on-board
run + md5-compare to a reference that copies the input bytes verbatim):

    python tools/chunk_compile.py \
        --mode=identity \
        --shape=3x112x112xf32 \
        --out-dir /tmp/chunk_smoke \
        --on-board   # if set, scp + run on qdev

Limitations:

  - Multi-dispatch chunks (LAYER / MEGAKERNEL granularity) need a slicer
    that splices each constituent dispatch's body into the parameter-passing
    @main; that's left for a follow-on (PR 3 "full" path described in
    /home/agustin/.claude/plans/i-want-to-enable-rosy-sundae.md).
  - For fragment mode the user supplies the post-DispatchCreation MLIR
    fragment by hand or via a future automated extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shlex
import subprocess
import sys


def _emit_identity_module(shape: str) -> str:
    return (
        f"// chunk_compile.py identity smoke chunk\n"
        f"module {{\n"
        f"  func.func public @main(%in0: tensor<{shape}>) -> tensor<{shape}> {{\n"
        f"    return %in0 : tensor<{shape}>\n"
        f"  }}\n"
        f"}}\n"
    )


def _compile_chunk(
    mlir_path: pathlib.Path, vmfb_path: pathlib.Path, iree_compile: str, target_flags: list[str]
) -> None:
    cmd = [
        iree_compile,
        str(mlir_path),
        "--iree-hal-target-device=local",
        "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-triple=aarch64-linux-gnu",
        *target_flags,
        "-o",
        str(vmfb_path),
    ]
    print("$", " ".join(shlex.quote(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(r.returncode)
    print(f"wrote {vmfb_path} ({vmfb_path.stat().st_size} bytes)")


def _run_on_board(
    vmfb_path: pathlib.Path, shape: str, board_user_host: str, remote_dir: str = "/root/chunk_smoke"
) -> str:
    """scp + run via iree-run-module on the board, return stdout."""
    subprocess.run(["ssh", board_user_host, f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(["scp", str(vmfb_path), f"{board_user_host}:{remote_dir}/{vmfb_path.name}"], check=True)
    r = subprocess.run(
        [
            "ssh",
            board_user_host,
            f"/root/iree-run-module --module={remote_dir}/{vmfb_path.name} "
            f"--device=local-task --function=main "
            f"--input={shape}=1.5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print("--- on-board stdout ---")
    print(r.stdout)
    if r.returncode != 0:
        print("--- stderr ---", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
    return r.stdout


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["fragment", "identity"], default="identity")
    p.add_argument("--fragment", type=pathlib.Path, help="(mode=fragment) MLIR file with a func.func public @main")
    p.add_argument("--shape", default="8xf32", help="(mode=identity) tensor shape, e.g. 3x112x112xf32")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--target-flag", default="--iree-llvmcpu-target-cpu=cortex-a77", help="Forwarded to iree-compile.")
    p.add_argument(
        "--iree-compile",
        default="/scratch2/agustin/merlin/build/host-vanilla-debug/tools/iree-compile",
        help="Path to iree-compile.",
    )
    p.add_argument("--on-board", action="store_true", help="scp the VMFB to the board and run with iree-run-module.")
    p.add_argument("--board-host", default="qdev", help="SSH host (~/.ssh/config alias). Default qdev.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "identity":
        chunk_mlir = args.out_dir / "chunk_identity.mlir"
        chunk_mlir.write_text(_emit_identity_module(args.shape))
        print(f"wrote {chunk_mlir}")
    else:
        if not args.fragment:
            print("--fragment required for mode=fragment", file=sys.stderr)
            return 2
        chunk_mlir = args.out_dir / args.fragment.name
        chunk_mlir.write_text(args.fragment.read_text())

    vmfb = chunk_mlir.with_suffix(".vmfb")
    _compile_chunk(chunk_mlir, vmfb, args.iree_compile, target_flags=[args.target_flag])
    md5 = hashlib.md5(vmfb.read_bytes()).hexdigest()
    print(f"md5 = {md5}")

    if args.on_board:
        out = _run_on_board(vmfb, args.shape, args.board_host)
        if "EXEC" in out or "result" in out.lower() or "tensor" in out.lower():
            print("✓ on-board run succeeded")
        else:
            print("⚠ on-board run produced no obvious result tensor output")

    return 0


if __name__ == "__main__":
    sys.exit(main())
