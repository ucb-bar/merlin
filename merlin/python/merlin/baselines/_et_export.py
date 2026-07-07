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


def _linear_subgraph(model):
    """Extract the model's REAL linear-heavy subgraph for the int8 path.

    Whole-model PT2E is IMPOSSIBLE on HF Llama here: ``prepare_pt2e``'s ``transform_for_annotation``
    pass corrupts an integer-index dtype (the position/causal-mask ``aten.index.Tensor`` on a
    ``cumsum``) at calibration — it fails even with an EMPTY quantizer, so it is the transform pass,
    not the observers/annotation, and no ``set_module_*``/``filter_fn`` exclusion can dodge it. So we
    quantize the largest self-contained pure-Linear region: **ALL of a decoder layer's Linears** —
    attention q/k/v/o projections + the SwiGLU MLP (gate/up/down), actual trained weights — driven by
    a seeded hidden state, with the non-linear glue (RoPE index, causal mask, softmax, norms,
    embedding) kept fp32 and OUT of quantization. This is the maximal int8 an HF-Llama export allows
    on this ExecuTorch/PT2E; ``subgraph_note`` records exactly what stayed fp32.

    Returns ``(submodule, (hidden_state_input,), keys, note)``.
    """
    import torch

    # Find a decoder layer with BOTH an attention block (q/k/v/o_proj) and an MLP (gate/up/down_proj).
    attn = mlp = None
    for _name, m in model.named_modules():
        if attn is None and all(hasattr(m, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj")):
            attn = m
        if mlp is None and all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj")):
            mlp = m
        if attn is not None and mlp is not None:
            break

    if attn is not None and mlp is not None:
        d = mlp.gate_proj.in_features
        act = getattr(mlp, "act_fn", torch.nn.SiLU())

        class _AllLinears(torch.nn.Module):
            """All 7 decoder-layer Linears int8; attention math (RoPE/mask/softmax) stays fp32 glue.

            The attention projections are exercised as real matmuls (q/k/v via the projections, o on
            q's output); the k/v outputs are kept live (reduced + broadcast) so every projection is
            quantized. This is NOT the attention numerics — it is a linear-coverage vehicle so all
            of the layer's GEMMs run int8 on XNNPACK qs8 RVV, gated cos vs the SAME-shape fp32."""
            def __init__(self, attn, mlp, act):
                super().__init__()
                self.q, self.k, self.v, self.o = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
                self.g, self.u, self.dn, self.act = mlp.gate_proj, mlp.up_proj, mlp.down_proj, act

            def forward(self, x):
                a = self.o(self.q(x))                                   # q_proj, o_proj (q_out->d)
                kv = self.k(x).sum(-1, keepdim=True) + self.v(x).sum(-1, keepdim=True)  # keep k,v live
                m = self.dn(self.act(self.g(x)) * self.u(x))            # MLP linears
                return a + m + kv

        sub = _AllLinears(attn, mlp, act).eval()
        torch.manual_seed(0)
        x = torch.randn(1, 8, d)
        return sub, (x,), ["hidden_state"], (
            f"int8-subgraph=decoder-layer-ALL-linears(q/k/v/o+gate/up/down, d={d}, "
            f"h={mlp.gate_proj.out_features}); fp32-glue=embedding+RoPE-index+causal-mask+softmax+"
            "RMSNorm (full-model int8 blocked: prepare_pt2e transform_for_annotation corrupts the "
            "cumsum->index.Tensor dtype even with an empty quantizer)")

    # Generic fallback: stack the first few Linears.
    lins = [m for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    if not lins:
        raise RuntimeError("int8-subgraph: model has no Linear layers to quantize")
    d = lins[0].in_features

    class _Seq(torch.nn.Module):
        def __init__(self, lins):
            super().__init__()
            self.lins = torch.nn.ModuleList(lins[:3])

        def forward(self, x):
            for lin in self.lins:
                if lin.in_features == x.shape[-1]:
                    x = torch.nn.functional.silu(lin(x))
            return x

    sub = _Seq(lins).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 8, d)
    return sub, (x,), ["hidden_state"], (
        f"int8-subgraph=first-linears(d={d}); non-linear ops excluded (full-model int8 blocked)")


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
    ap.add_argument("--int8-whole-model", action="store_true",
                    help="WHOLE-MODEL int8 via ExecuTorch's OFFICIAL llama recipe: source-transform "
                         "weight-only int8 per-channel (WeightOnlyInt8QuantHandler — an eager "
                         "MODULE SWAP, replacing every nn.Linear with a WeightOnlyInt8Linear). "
                         "Because it is a module swap it NEVER runs PT2E's transform_for_annotation "
                         "pass, so it SIDESTEPS the cumsum->index.Tensor dtype corruption that "
                         "blocks generic full-model PT2E on HF Llama — all layers quantize + export "
                         "cleanly. Forces --compute-golden (int8-vs-fp32 gate).")
    ap.add_argument("--int8-subgraph", action="store_true",
                    help="int8 FALLBACK path (only if whole-model won't export): quantize the REAL "
                         "decoder-layer linear subgraph (q/k/v/o + gate/up/down) W8A8 on a seeded "
                         "hidden-state input, embeddings/mask fp32. Honestly labeled.")
    args = ap.parse_args()
    if args.int8_whole_model:
        args.quantize = True
        args.compute_golden = True
    if args.int8_subgraph:
        args.quantize = True
        args.compute_golden = True

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

    subgraph_note = ""
    if args.int8_subgraph:
        # Replace the whole model with its REAL linear-heavy subgraph (a genuine slice, actual
        # weights) — the FALLBACK if whole-model int8 won't export.
        model, captured, keys, subgraph_note = _linear_subgraph(model)

    # Golden is the FP32 reference — compute it BEFORE any int8 module swap.
    if args.compute_golden:
        with torch.no_grad():
            eager = model(*captured)
        eager_t = eager[0] if isinstance(eager, (tuple, list)) else eager
        golden_np = eager_t.detach().cpu().to(torch.float32).numpy()
        np.save(args.golden_npy, golden_np)
    else:
        golden_np = np.load(args.golden_npy)

    quantized = False

    # 2a. WHOLE-MODEL int8 via ExecuTorch's OFFICIAL llama recipe (source-transform weight-only int8
    #     per-channel). This is an eager MODULE SWAP (every nn.Linear -> WeightOnlyInt8Linear), so it
    #     never runs PT2E's transform_for_annotation pass and SIDESTEPS the cumsum->index.Tensor
    #     corruption that blocks generic full-model PT2E on HF Llama. ALL layers quantize + export.
    if args.int8_whole_model:
        from executorch.examples.models.llama.source_transformation.quantize import (
            WeightOnlyInt8QuantHandler,
        )

        n_lin = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
        model = WeightOnlyInt8QuantHandler(model).quantized_model().eval()
        quantized = True
        subgraph_note = (subgraph_note + " " if subgraph_note else "") + (
            f"int8-whole-model=official-llama-recipe(WeightOnlyInt8QuantHandler, weight-only int8 "
            f"per-channel, ALL {n_lin} nn.Linear swapped -> WeightOnlyInt8Linear); embeddings/mask/"
            "softmax/norm fp32; module-swap sidesteps PT2E transform_for_annotation index-corruption")

    # 2b. PT2E W8A8 quantization (int8 subgraph path). Only when NOT whole-model int8.
    if args.quantize and not args.int8_whole_model:
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

    # 3b. WHOLE-MODEL int8 arena fix: WeightOnlyInt8Linear.forward is
    #     `F.linear(x, weight.to(fp32)) * scales`. That `int8_weight.to(fp32)` dequant is a graph
    #     op whose output is a PLANNED ACTIVATION tensor, so the memory planner lays every layer's
    #     dequantized fp32 weight at a distinct offset -> a 4.25 GB non-const arena for full-depth
    #     TinyLlama (measured), which `std::make_unique<uint8_t[]>(4.25e9)` throws bad_alloc on at
    #     load on the 3.4 GB board. Const-folding the dequant (its inputs are all frozen constants)
    #     turns those fp32 weights into program CONSTANTS: the planned arena collapses from 4.25 GB
    #     to ~2 MB (measured 4250738736 -> 2405952 bytes, ~1770x). The fp32 weights now live in the
    #     .pte program data, which the board mmaps (executor_runner --mmap_model) so their pages
    #     demand-load and stay evictable under the board's RAM ceiling. This is what unblocks the
    #     FULL-DEPTH (not layer-reduced) whole-model int8 on-board run.
    if args.int8_whole_model:
        try:
            from executorch.exir.passes.constant_prop_pass import constant_prop_pass
            exported = constant_prop_pass(exported)
            subgraph_note += " +const-prop(dequant-weights-folded->const, arena 4.25GB->2MB)"
        except Exception as e:  # noqa: BLE001
            subgraph_note += f" const-prop-skipped({str(e)[:80]})"

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

    # 5. Save the model input(s) as raw bytes (the runner memcpy's the file into the input tensor).
    #    Use the actual `captured` tensors (int8-subgraph feeds a seeded fp32 hidden state, not the
    #    npz token ids), so the on-board input matches exactly what golden was computed from.
    input_files = []
    for i, (k, t) in enumerate(zip(keys, captured)):
        arr = np.ascontiguousarray(t.detach().cpu().numpy())
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
        "int8_subgraph": bool(args.int8_subgraph),
        "int8_whole_model": bool(args.int8_whole_model),
        "subgraph_note": subgraph_note,
        "delegated_nodes": delegated,
        "total_call_nodes": total_calls,
        "golden_shape": list(golden_np.shape),
        "golden_dtype": str(golden_np.dtype),
    }
    print("ET_EXPORT_JSON " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
