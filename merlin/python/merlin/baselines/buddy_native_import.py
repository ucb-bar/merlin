"""Buddy NATIVE torch importer helper — runs under the model2MLIR torch venv (NOT merlin's).

Phase-2 ingestion path: instead of consuming m2m's linalg ``model.mlir``, use buddy-mlir's OWN
``DynamoCompiler`` (``torch.export`` → buddy Graph → ``lower_to_top_level_ir`` → MLIR) so the arm
runs the SAME IR buddy's own users get. This produces DIFFERENT IR than m2m linalg and may bypass
the whole-model scalar-lowering SIGSEGV and the ``aten.select`` rank-mismatch the m2m path hit.

This module is invoked as a SUBPROCESS by :func:`merlin.baselines.buddy.native_import` under the
torch venv with ``PYTHONPATH`` pointing at buddy's built ``python_packages`` + the MLIR bindings —
because ``buddy.compiler.frontend`` needs torch + ``torch._dynamo`` + the ``buddy_mlir`` bindings,
none of which live in merlin's venv. It writes, into ``--out-dir``:

  * ``subgraph0.mlir`` — the buddy-lowered compute module (linalg/tosa on tensors)
  * ``forward.mlir``   — the main graph that calls the subgraph (buddy's ABI)
  * ``arg0.data``      — the flat float32 parameter blob (concatenated model params)
  * ``import_meta.json`` — input/param shapes + dtypes for the caller to build descriptors

Usage (from :mod:`buddy`)::

    <torch-venv-python> -m merlin.baselines.buddy_native_import \\
        --model tiny_llama --variant int8 --out-dir <dir> --loader <loader.py>

It is deliberately import-light at module top so a ``--help`` works even without buddy built; the
heavy imports happen inside :func:`main`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_loader(loader_path: str):
    """Import the m2m ``loader.py`` for a model as a module (it exposes get_model_and_inputs)."""
    spec = importlib.util.spec_from_file_location("m2m_loader", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Buddy native torch importer (runs under torch venv)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", default="int8")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--loader", required=True, help="path to the m2m workloads/<model>/loader.py")
    ap.add_argument("--registry", default="tosa", choices=["linalg", "tosa"],
                    help="buddy primary op registry (tosa is buddy's robust default; its linalg "
                         "registry has an expand_op bug on the LLM graph)")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Heavy imports (need the torch venv + buddy python_packages on PYTHONPATH).
    import numpy as np
    import torch
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph import GraphDriver
    from buddy.compiler.graph.transform import simply_fuse
    if args.registry == "tosa":
        from buddy.compiler.ops import tosa as _reg
    else:
        from buddy.compiler.ops import linalg as _reg
    try:
        from torch._inductor.decomposition import decompositions as inductor_decomp
    except Exception:  # noqa: BLE001
        inductor_decomp = {}

    # Load the REAL/native model + example inputs from the m2m loader (full-fidelity env is set by
    # the caller via os.environ before spawning us — the loader reads M2M_*_LAYERS etc.).
    loader = _load_loader(args.loader)
    model, inputs = loader.get_model_and_inputs()
    model = model.eval()

    dynamo_compiler = DynamoCompiler(
        primary_registry=_reg.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    # Prefer importer_by_export (torch.export -> stable arg order); fall back to the dynamo importer.
    with torch.no_grad():
        try:
            graphs = dynamo_compiler.importer_by_export(model, *inputs)
        except Exception as e:  # noqa: BLE001
            print(f"NOTE: importer_by_export failed ({str(e)[:120]}); using dynamo importer",
                  file=sys.stderr)
            torch._dynamo.reset()
            graphs = dynamo_compiler.importer(model, *inputs)
    if len(graphs) != 1:
        # graph breaks -> multiple subgraphs; buddy's GraphDriver still stitches them, but record it.
        print(f"NOTE: {len(graphs)} graphs (dynamo break)", file=sys.stderr)
    graph = graphs[0]
    params = dynamo_compiler.imported_params[graph]

    graph.fuse_ops([simply_fuse])
    driver = GraphDriver(graph)
    driver.subgraphs[0].lower_to_top_level_ir()
    (out / "subgraph0.mlir").write_text(str(driver.subgraphs[0]._imported_module))
    (out / "forward.mlir").write_text(str(driver.construct_main_graph(True)))

    # Flat float32 param blob (buddy's arg0.data convention), written by STREAMING each param to
    # disk in turn. The previous `np.concatenate([... for p in params])` materialized the ENTIRE
    # model as one contiguous fp32 array in RAM before writing — peak host memory ≈ 2× the blob
    # (the per-param fp32 copies + the concatenated result), which OOMs on the multi-GB VLAs
    # (rdt arg0.data is ~4.8 GB fp32; concatenating it needs ~10 GB transient). Writing param-by-
    # param to an open file keeps peak RAM at ONE param's fp32 copy, not the whole model, so the
    # blob is producible for any model that fits on disk. (The on-board harness already MMAPs this
    # blob read-only, so board-resident RAM is the demand-paged working set, not the whole file.)
    param_bytes = 0
    if params:
        with open(out / "arg0.data", "wb") as fh:
            for p in params:
                a = np.ascontiguousarray(p.detach().float().numpy().reshape([-1]))
                a.tofile(fh)
                param_bytes += a.nbytes
                del a  # free the per-param fp32 copy before the next param
    else:
        (out / "arg0.data").write_bytes(b"")

    meta = {
        "model": args.model,
        "variant": args.variant,
        "registry": args.registry,
        "n_graphs": len(graphs),
        "n_params": len(params),
        "param_bytes": int((out / "arg0.data").stat().st_size),
        "inputs": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in inputs],
    }
    (out / "import_meta.json").write_text(json.dumps(meta, indent=2))
    print("NATIVE_IMPORT_OK " + json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
