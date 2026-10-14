#!/usr/bin/env python3
"""Regenerate the two tiny MLP model directories the external-kernel tests and reports run on.

Writes ``<out>/mlp_basic/`` and ``<out>/mlp_basic_triton/`` (normally ``merlin/tests/data/mlp_bundles/``),
each a whole-model bundle in the layout ``llvmlower.c_runtime`` / ``runtime.backends.spike_model`` read:
``model.mlir``, ``weights.safetensors`` (+ ``.manifest.json``), ``inputs.npz`` (``in0``),
``input_order.json`` and ``golden.npy``.

Both are captured FRESH from model2MLIR's workload loaders (``workloads/<name>/loader.py``) with
``m2m.convert(..., backend="fx_importer")``, never copied from a ``<name>.mlir`` left in that checkout:
a copy reproduces whatever happened to be captured last, so it can neither regenerate the bundles
faithfully nor tell ``merlin/tests/infra/test_mlp_bundles_fresh.py`` whether they are stale. The two
models share seed, weights and input; ``mlp_basic_triton`` only differs in that fc1+relu+fc2+relu is
one ``prov.kernel`` call.

``prov.weights_file`` is written as ``"weights.safetensors"`` -- the bundle-relative convention merlin's
own bundles use (``frontends.compile_inputs``) -- so a tracked bundle names no personal directory.

**The golden comes from the kernel-free twin, run in PyTorch.** The Triton op cannot execute on a CPU
(that is the point of the placeholder), so ``mlp_basic``'s eager output is the oracle for BOTH bundles.
The script refuses to write if the two captures' weights differ, since that oracle would then be wrong
for one of them.

Run it with the model2MLIR venv (it needs torch, triton and m2m, not merlin)::

    $MERLIN_M2M_VENV/bin/python build_tools/scripts/gen_mlp_kernel_fixtures.py \\
        --m2m $MERLIN_M2M_DIR --out merlin/tests/data/mlp_bundles
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

WORKLOADS = ("mlp_basic", "mlp_basic_triton")
WEIGHTS = "weights.safetensors"


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

    import m2m
    import numpy as np
    import torch

    src = args.m2m.resolve() / "workloads"
    loaders = {w: _load(src / w / "loader.py", f"{w}_loader") for w in WORKLOADS}

    inputs = None
    for w in WORKLOADS:
        dst = args.out / w
        dst.mkdir(parents=True, exist_ok=True)
        model, (x,) = loaders[w].get_model_and_inputs()
        weights = (dst / WEIGHTS).resolve()
        r = m2m.convert(model, (x,), backend="fx_importer", weights_path=str(weights))
        if not r.ok:
            print(f"refusing: model2MLIR could not capture {w}: {r.diagnostics}", file=sys.stderr)
            return 1
        text = r.mlir_text.replace(f'prov.weights_file = "{weights}"', f'prov.weights_file = "{WEIGHTS}"')
        for local in (str(args.m2m.resolve()), str(args.out.resolve())):
            if local in text:
                print(f"refusing: {w} model.mlir still names {local!r}", file=sys.stderr)
                return 1
        (dst / "model.mlir").write_text(text, encoding="utf-8")
        inputs = x if inputs is None else inputs
        np.savez(dst / "inputs.npz", in0=x.numpy().astype(np.float32))
        (dst / "input_order.json").write_text(json.dumps({"x": 0}) + "\n", encoding="utf-8")

    blobs = {w: (args.out / w / WEIGHTS).read_bytes() for w in WORKLOADS}
    if blobs["mlp_basic"] != blobs["mlp_basic_triton"]:
        print(
            "refusing: mlp_basic and mlp_basic_triton captured different weights; the shared "
            "golden would be wrong for one of them.",
            file=sys.stderr,
        )
        return 1

    # A fresh model for the oracle: capture may have moved or traced the one it was given.
    model, (x,) = loaders["mlp_basic"].get_model_and_inputs()
    with torch.no_grad():
        golden = model(x).numpy().astype(np.float32)
    for w in WORKLOADS:
        np.save(args.out / w / "golden.npy", golden)
        print(f"wrote {args.out / w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
