#!/usr/bin/env python3
"""Regenerate the two tiny MLP model directories the external-kernel tests and reports run on.

Writes ``<out>/mlp_basic/`` and ``<out>/mlp_basic_triton/`` (normally ``merlin/tests/fixtures/``), each a
whole-model bundle in the layout ``llvmlower.c_runtime`` / ``runtime.backends.spike_model`` read:
``model.mlir``, ``weights.safetensors`` (+ ``.manifest.json``), ``inputs.npz`` (``in0``),
``input_order.json`` and ``golden.npy``. Absolute paths model2MLIR records in the module are made
relative to the model2MLIR checkout.

Both come from model2MLIR's checked-in captures (``workloads/<name>/<name>.mlir``; re-run
``workloads/capture.py`` there first if the importer changed). The two models share seed, weights and
input; ``mlp_basic_triton`` only differs in that fc1+relu+fc2+relu is one ``prov.kernel`` call.

**The golden comes from the kernel-free twin, run in PyTorch.** The Triton op cannot execute on a CPU
(that is the point of the placeholder), so ``mlp_basic``'s eager output is the oracle for BOTH bundles.
The script refuses to write if the two captures' weights differ, since that oracle would then be wrong
for one of them.

Run it with the model2MLIR venv (it needs torch and the m2m workloads, not merlin)::

    $MERLIN_M2M_VENV/bin/python build_tools/scripts/gen_mlp_kernel_fixtures.py \\
        --m2m $MERLIN_M2M_DIR --out merlin/tests/fixtures
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

WORKLOADS = ("mlp_basic", "mlp_basic_triton")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--m2m", required=True, type=Path, help="model2MLIR checkout")
    ap.add_argument("--out", required=True, type=Path, help="directory to write the bundles into")
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    src = args.m2m / "workloads"
    root = str(args.m2m.resolve()) + "/"
    payloads = {w: (src / w / f"{w}.safetensors").read_bytes() for w in WORKLOADS}
    if payloads["mlp_basic"] != payloads["mlp_basic_triton"]:
        print(
            "refusing: mlp_basic and mlp_basic_triton captured different weights; the shared "
            "golden would be wrong for one of them. Re-capture both.",
            file=sys.stderr,
        )
        return 1

    model, (x,) = _load(src / "mlp_basic" / "loader.py", "mlp_basic_loader").get_model_and_inputs()
    with torch.no_grad():
        golden = model(x).numpy().astype(np.float32)

    for w in WORKLOADS:
        dst = args.out / w
        dst.mkdir(parents=True, exist_ok=True)
        # model2MLIR records absolute paths (prov.weights_file, prov.kernel_source); a tracked fixture
        # must not name a personal directory, so they become model2MLIR-checkout-relative.
        text = (src / w / f"{w}.mlir").read_text(encoding="utf-8").replace(root, "")
        if root.rstrip("/") in text:
            print(f"refusing: {w}.mlir still names {root!r} after relativizing", file=sys.stderr)
            return 1
        (dst / "model.mlir").write_text(text, encoding="utf-8")
        shutil.copyfile(src / w / f"{w}.safetensors", dst / "weights.safetensors")
        shutil.copyfile(src / w / f"{w}.safetensors.manifest.json", dst / "weights.safetensors.manifest.json")
        np.savez(dst / "inputs.npz", in0=x.numpy().astype(np.float32))
        (dst / "input_order.json").write_text(json.dumps({"x": 0}) + "\n", encoding="utf-8")
        np.save(dst / "golden.npy", golden)
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
