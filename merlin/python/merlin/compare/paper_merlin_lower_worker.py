"""Sealed subprocess worker for one public Merlin MLIR program.

The parent producer supplies a closed environment, cwd, stdin, compiler, and
resource directory.  Keeping lowering in this subprocess makes those controls
transitive to the upstream MLIR subprocess launched by ``lower_model``.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

from merlin.llvmlower.codegen import compile_ll
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.session_bundle import rename_forward
from merlin.common.mlir_query import forward_signature


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--signature":
        inputs, outputs = forward_signature(Path(argv[2]).resolve())
        print(json.dumps({"inputs": inputs, "outputs": outputs},
                         sort_keys=True, separators=(",", ":")))
        return 0
    if len(argv) != 9:
        raise SystemExit(
            "usage: worker SOURCE ENTRYPOINT OUTPUT_ROOT RESOURCE_DIR TRIPLE MARCH MABI FEATURES")
    source_path = Path(argv[1]).resolve()
    entrypoint = argv[2]
    output_root = Path(argv[3]).resolve()
    resource_dir = Path(argv[4]).resolve()
    if tuple(argv[5:]) != ("riscv64-unknown-elf", "rv64gcv", "lp64d", "c,g,v"):
        raise SystemExit("worker target ABI differs from the closed K1 lowering recipe")
    source = source_path.read_text(encoding="utf-8")
    lowered = lower_model(
        rename_forward(source, entrypoint), output_root, targets=(), textual=True)
    compile_ll(
        lowered.ll_path, output_root / "model_host.o", target="x86",
        extra_flags=(f"-resource-dir={resource_dir}",))
    compile_ll(
        lowered.ll_path, output_root / "model_riscv.o", target="riscv",
        extra_flags=(f"-resource-dir={resource_dir}",))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
