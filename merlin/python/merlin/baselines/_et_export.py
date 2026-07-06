"""AOT export helper for the ExecuTorch + XNNPACK baseline arm (runs UNDER the ExecuTorch venv).

Invoked as a subprocess by :mod:`merlin.baselines.executorch` using the dedicated ExecuTorch export
venv (``build/baselines/executorch/et-venv``), because ExecuTorch + its pinned torch are NOT in
merlin's ``.venv``. Dependency-light on merlin (everything arrives via argv) so it can run under a
foreign interpreter.

Flow (on OUR model + OUR captured input/golden, always through the XNNPACK partitioner):

    loader.get_model_and_inputs()      # HF model from model2MLIR/workloads/<model>/loader.py
      -> OVERRIDE example input with the captured inputs.npz (so it matches golden.npy)
      -> torch.export.export(model, (captured_input,))
      -> to_edge_transform_and_lower(partitioner=[XnnpackPartitioner()])
      -> to_executorch(config=external_constants=True)      # weights -> .ptd, program -> small .pte
      -> write model.pte (+ *.ptd next to it) + input0.bin (raw captured input bytes)

We externalize constants (``external_constants=True``) because a whole fp32 LLM's weights (~4 GB)
blow past flatbuffer's 2 GB program-buffer limit if embedded. The board-side stock executor_runner
then loads ``--model_path=model.pte --data_path=<ptd> --inputs=input0.bin`` and dumps the output;
the runner (in merlin's venv) compares that dump to golden.npy off-device — an honest same-input,
same-golden gate with no fabricated numbers.

Prints ``ET_EXPORT_JSON {...}`` with the pte/ptd/input paths, output shape/dtype, and how many
graph nodes XNNPACK delegated vs stayed portable (a coarse delegation signal complementing the
binary-level RVV audit).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_loader(loader_path: Path):
    spec = importlib.util.spec_from_file_location("_m2m_loader", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_TORCH_TO_NP = {}  # filled after torch import


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loader", required=True, help="path to model2MLIR workloads/<model>/loader.py")
    ap.add_argument("--inputs-npz", required=True, help="captured inputs.npz (seeds the golden)")
    ap.add_argument("--golden-npy", required=True, help="captured golden.npy (reference output)")
    ap.add_argument("--compute-golden", action="store_true",
                    help="compute the reference by running the eager torch model on the captured "
                         "input (writes it to --golden-npy). Use when the captured golden was made "
                         "with a DIFFERENT model config (e.g. a layer-reduced fit-on-board build), "
                         "so the correctness gate compares ExecuTorch vs eager-torch for THIS model.")
    ap.add_argument("--out", required=True, help="output .pte path (ptd + input0.bin go alongside)")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--no-xnnpack", action="store_true",
                    help="skip the XNNPACK partitioner (portable-kernel-only baseline)")
    ap.add_argument("--quantize", action="store_true",
                    help="PT2E W8A8 quantize before lowering (exercises XNNPACK's int8 qs8/qd8 RVV "
                         "ukernels). ExecuTorch does its OWN quantization, so pair with "
                         "--compute-golden (gate = eager-vs-ExecuTorch for THIS config).")
    ap.add_argument("--m2m-root", default="/scratch/agustin/projects/model2MLIR",
                    help="model2MLIR repo root (added to sys.path for its deps)")
    args = ap.parse_args()

    # This script sits in merlin/python/merlin/baselines/ alongside ``executorch.py``, which would
    # SHADOW the installed ``executorch`` package (Python puts the script dir on sys.path[0]).
    _self_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in ("", ".", _self_dir)]
    if args.m2m_root not in sys.path:
        sys.path.insert(0, args.m2m_root)

    import numpy as np
    import torch
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
    from torch.export import export

    # 1. Load model; substitute the CAPTURED input so the export trace matches the captured golden.
    loader = _load_loader(Path(args.loader))
    model, _example = loader.get_model_and_inputs()
    model = model.eval()

    npz = np.load(args.inputs_npz)
    keys = list(npz.keys())
    captured = tuple(torch.from_numpy(npz[k]) for k in keys)

    if args.compute_golden:
        with torch.no_grad():
            eager = model(*captured)
        eager_t = eager[0] if isinstance(eager, (tuple, list)) else eager
        golden_np = eager_t.detach().cpu().to(torch.float32).numpy()
        np.save(args.golden_npy, golden_np)
    else:
        golden_np = np.load(args.golden_npy)

    # 2. Optional PT2E W8A8 quantization (int8 path). Quantize the exported graph with the XNNPACK
    #    symmetric config, calibrate on the captured input, convert. This is ExecuTorch's OWN int8
    #    (distinct from merlin's int8 datapath), so it exercises XNNPACK's qs8/qd8 RVV ukernels.
    quantized = False
    if args.quantize:
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config)
            try:
                from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
            except Exception:  # noqa: BLE001
                from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

            cap = export(model, captured, strict=False).module()
            quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
            prepared = prepare_pt2e(cap, quantizer)
            with torch.no_grad():
                prepared(*captured)          # calibrate on the captured input
            model = convert_pt2e(prepared)   # now an int8 graph module
            quantized = True
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"PT2E int8 quantization failed: {e}") from e

    # 3. torch.export (non-strict first; HF causal-LMs trace cleanly non-strict).
    try:
        exported = export(model, captured, strict=False)
    except Exception:  # noqa: BLE001
        exported = export(model, captured, strict=True)

    # 4. XNNPACK partitioner (the RVV-microkernel delegate path).
    partitioners = []
    if not args.no_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        partitioners.append(XnnpackPartitioner())

    from executorch.exir import EdgeCompileConfig
    _cc = EdgeCompileConfig(_check_ir_validity=False) if quantized else None
    edge = to_edge_transform_and_lower(exported, partitioner=partitioners, compile_config=_cc)
    nodes = list(edge.exported_program().graph.nodes)
    delegated = sum(1 for n in nodes
                    if n.op == "call_function" and "call_delegate" in str(n.target))
    total_calls = sum(1 for n in nodes if n.op == "call_function")

    # 4. to_executorch with EXTERNAL constants: weights land in a .ptd next to the .pte, so the
    #    program flatbuffer stays under the 2 GB limit for the big fp32 LLMs.
    et_program = edge.to_executorch(
        config=ExecutorchBackendConfig(external_constants=True)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        et_program.write_to_file(fh)
    # .ptd tensor-data files (weights). write_tensor_data_to_file writes <key>.ptd into outdir.
    ptd_files: list[str] = []
    try:
        et_program.write_tensor_data_to_file(str(out.parent))
        ptd_files = sorted(str(p) for p in out.parent.glob("*.ptd"))
    except Exception as e:  # noqa: BLE001
        # If there were no external constants (small model), there is simply no ptd — not an error.
        ptd_files = []

    # 5. Save the captured input as raw bytes (the runner memcpy's the file into the input tensor).
    input_files = []
    for i, k in enumerate(keys):
        arr = np.ascontiguousarray(npz[k])
        p = out.parent / f"input{i}.bin"
        p.write_bytes(arr.tobytes())
        input_files.append({"path": str(p), "key": k, "dtype": str(arr.dtype),
                            "shape": list(arr.shape), "nbytes": arr.nbytes})

    summary = {
        "model": args.model_name,
        "pte": str(out),
        "pte_bytes": out.stat().st_size,
        "ptd_files": ptd_files,
        "input_files": input_files,
        "xnnpack": not args.no_xnnpack,
        "quantized": quantized,
        "delegated_nodes": delegated,
        "total_call_nodes": total_calls,
        "golden_shape": list(golden_np.shape),
        "golden_dtype": str(golden_np.dtype),
    }
    print("ET_EXPORT_JSON " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
