#!/usr/bin/env python3
"""Build and run one MLP bundle on spike rv64gcv and compare it with the PyTorch golden.

The end-to-end example of external-kernel linking (docs/guides/triton_kernel_integration.md): the
bundle ``mlp_basic_triton`` calls ``triton_exp::fused_mlp2`` as a ``prov.kernel`` call, and
``--form`` picks which body from the kernel table gets linked into the model's LLVM IR (``llvm_ir``
or ``llvm_mlir``). ``mlp_basic`` is the kernel-free twin and takes no form.

The bundle is copied into the work directory first, because ``spike_model.build`` writes beside the
model it is given and the tracked bundles under ``merlin/tests/data/mlp_bundles/`` must stay clean.
Every intermediate (``lower/model.ll``, ``lower/kernel_shim.c``, ``lower/model.linked.ll``,
``model.elf``, ...) is left in ``<work>/build/``.

Needs spike, the riscv64 cross toolchain, clang and llvm-link (see docs/guides/rvv_e2e.md and
docs/guides/llvm_toolchain.md), with chipyard's env on PATH (spike shells out to ``dtc``)::

    .venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --form llvm_mlir
    .venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --form llvm_ir
    .venv/bin/python build_tools/scripts/run_mlp_kernel_spike.py --bundle mlp_basic
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BUNDLES = ("mlp_basic", "mlp_basic_triton")
FORMS = ("llvm_ir", "llvm_mlir")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", choices=BUNDLES, default="mlp_basic_triton")
    ap.add_argument(
        "--form", choices=FORMS, help="kernel body form (mlp_basic_triton only; default: the table's first)"
    )
    ap.add_argument("--work", type=Path, help="work directory (default: out/artifacts/probes/mlp_kernel/<variant>)")
    args = ap.parse_args(argv)
    if args.form and args.bundle != "mlp_basic_triton":
        ap.error("--form applies only to mlp_basic_triton, the bundle that calls a kernel")

    import numpy as np

    from merlin.common.paths import artifacts_dir, merlin_dir
    from merlin.runtime.backends import spike_model

    bundle = merlin_dir() / "tests" / "data" / "mlp_bundles" / args.bundle
    variant = f"{args.bundle}_{args.form}" if args.form else args.bundle
    work = args.work or artifacts_dir() / "probes" / "mlp_kernel" / variant
    if work.exists():
        shutil.rmtree(work)
    model = work / "model"
    shutil.copytree(bundle, model)

    build = spike_model.build(model, work / "build", kernel_form=args.form)
    run = spike_model.run(build["elf"], mem_bytes=build["mem_bytes"], vlen=build.get("vlen"))
    out = np.asarray(run["outputs"], dtype=np.float32)
    err = float(np.abs(out - np.load(bundle / "golden.npy").ravel()).max())
    print(f"variant={variant} kernels={build['kernels']} cycles={run['metrics']['cycles']} max_abs_err={err:.2e}")
    print(f"work={work}")
    return 0 if err < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
