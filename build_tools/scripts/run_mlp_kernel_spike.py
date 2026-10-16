#!/usr/bin/env python3
"""Build and run one MLP bundle on spike rv64gcv and compare it with the PyTorch golden.

The end-to-end example of external-kernel linking (docs/guides/triton_kernel_integration.md): the
bundle ``mlp_basic_triton`` calls ``triton_exp::fused_mlp2`` as a ``prov.kernel`` call, and the lowering
links the kernel's llvm-dialect body from the kernel table into the model before the MLIR pipeline.
``mlp_basic`` is the kernel-free twin.

The bundle is copied into the work directory first, because ``spike_model.build`` writes beside the
model it is given and the tracked bundles under ``merlin/tests/data/mlp_bundles/`` must stay clean.
Every intermediate (``lower/kernel_wrappers.mlir``, ``lower/model.upstream.mlir``, ``lower/model.ll``,
``model.elf``, ...) is left in ``<work>/build/``.

Needs spike, the riscv64 cross toolchain and clang (see docs/guides/rvv_e2e.md), with chipyard's env on
PATH (spike shells out to ``dtc``)::

    .venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py
    .venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --bundle mlp_basic
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BUNDLES = ("mlp_basic", "mlp_basic_triton")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", choices=BUNDLES, default="mlp_basic_triton")
    ap.add_argument("--work", type=Path, help="work directory (default: out/artifacts/probes/mlp_kernel/<bundle>)")
    args = ap.parse_args(argv)

    import numpy as np

    from merlin.common.paths import artifacts_dir, merlin_dir
    from merlin.runtime.backends import spike_model

    bundle = merlin_dir() / "tests" / "data" / "mlp_bundles" / args.bundle
    work = args.work or artifacts_dir() / "probes" / "mlp_kernel" / args.bundle
    if work.exists():
        shutil.rmtree(work)
    model = work / "model"
    shutil.copytree(bundle, model)

    build = spike_model.build(model, work / "build")
    run = spike_model.run(build["elf"], mem_bytes=build["mem_bytes"], vlen=build.get("vlen"))
    out = np.asarray(run["outputs"], dtype=np.float32)
    err = float(np.abs(out - np.load(bundle / "golden.npy").ravel()).max())
    print(f"bundle={args.bundle} kernels={build['kernels']} cycles={run['metrics']['cycles']} max_abs_err={err:.2e}")
    print(f"work={work}")
    return 0 if err < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
