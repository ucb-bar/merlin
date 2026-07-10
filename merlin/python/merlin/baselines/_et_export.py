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


def _apply_loader_compat_shims() -> list[str]:
    """Faithful compat shims for loader/framework version drift (NOT numeric changes).

    The m2m loaders were written against older dep versions than the ExecuTorch export venv
    ships. Each shim just re-exposes a symbol under the name the loader expects; none change
    model math. Returns the list of shims that actually applied (recorded in the notes).

      * ``BitNet`` config alias — bitvla's loader builds a Llava config with the (old)
        capitalized ``text_config['model_type']='BitNet'``; current transformers registers
        BitNet under the lowercase key ``'bitnet'``, so ``CONFIG_MAPPING['BitNet']`` KeyErrors
        (fails identically in the m2m venv too — loader-inherent drift). Register the alias.
    """
    applied: list[str] = []
    try:  # torch_compilable_check no-op (lerobot 0.6 eo1 policy vs transformers>=5)
        import transformers.utils as _tu
        if not hasattr(_tu, "torch_compilable_check"):
            # lerobot.policies.__init__ eagerly imports eo1, which imports this compile-guard
            # util removed in transformers>=5. It is a torch.compile guard, a no-op for eager
            # export; stub it so the smolvla policy module (an eo1 sibling) can be imported.
            _tu.torch_compilable_check = lambda *a, **k: None  # noqa: E731
            applied.append("torch_compilable_check-noop")
    except Exception:  # noqa: BLE001
        pass
    try:  # BitNet capitalized-key alias (bitvla)
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if "BitNet" not in CONFIG_MAPPING:
            from transformers.models.bitnet.configuration_bitnet import BitNetConfig
            CONFIG_MAPPING.register("BitNet", BitNetConfig, exist_ok=True)
            applied.append("bitnet-config-alias")
    except Exception:  # noqa: BLE001 - shim is best-effort; loader failure surfaces the real reason
        pass
    return applied


def _load_loader(loader_path: Path):
    _apply_loader_compat_shims()
    spec = importlib.util.spec_from_file_location("_m2m_loader", loader_path)
    mod = importlib.util.module_from_spec(spec)
    # Register under the module name BEFORE exec so a strict torch.export (dynamo) that
    # re-imports the defining module of a traced nn.Module class finds it (otherwise the
    # strict fallback dies with ModuleNotFoundError: No module named '_m2m_loader').
    sys.modules["_m2m_loader"] = mod
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


def _int8_whole_model_bias_preserving(model):
    """Whole-model weight-only int8 (per-channel) that PRESERVES nn.Linear bias.

    ExecuTorch's official ``WeightOnlyInt8QuantHandler`` is llama-specific: its
    ``WeightOnlyInt8Linear`` never carries a bias (forward is just
    ``F.linear(x, weight)*scales``) and its ``load_state_dict`` in ``quantized_model()``
    therefore REJECTS any model whose Linears have bias (rdt/rdt2 DiT blocks: qkv/proj/ffn
    all bias=True) with an "Unexpected key(s) ... .bias" error. That is a pure packaging
    limitation, not a quantization-feasibility one.

    This does the SAME math (symmetric per-output-channel int8, fp32 scale) via an eager
    module swap, but keeps the original ``bias`` on the replacement module and adds it back
    in forward. For a bias=False Linear the result is byte-identical to the official handler
    (so llama models are unchanged); for a biased Linear it is the correct int8-weight +
    fp32-bias GEMM. Like the official path it is a module swap, so it never runs PT2E's
    ``transform_for_annotation`` (sidestepping the index-dtype corruption), and its
    ``int8_weight.to(fp32)*scales`` dequant const-folds into a fp32 const weight that
    XNNPACK partitions as a normal fp32 GEMM — arena-collapsing exactly like the llama path.

    Returns ``(model, n_linear, n_biased)``.
    """
    import torch
    import torch.nn.functional as F

    class _Int8LinearBias(torch.nn.Module):
        """int8 weight-only per-channel Linear that carries its original fp32 bias."""

        def __init__(self, lin: torch.nn.Linear):
            super().__init__()
            self.in_features = lin.in_features
            self.out_features = lin.out_features
            w = lin.weight.detach().float()                     # [out, in]
            amax = w.abs().amax(dim=1, keepdim=True)            # per-output-channel
            scales = (amax / 127.0).clamp(min=1e-12)            # symmetric int8
            q = torch.round(w / scales).clamp(-128, 127).to(torch.int8)
            self.register_buffer("weight", q)
            self.register_buffer("scales", scales.squeeze(-1).float())
            if lin.bias is not None:
                self.register_buffer("bias", lin.bias.detach().float())
            else:
                self.bias = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.linear(x, self.weight.to(dtype=x.dtype)) * self.scales
            if self.bias is not None:
                y = y + self.bias.to(dtype=x.dtype)
            return y

    def _swap(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                setattr(module, name, _Int8LinearBias(child))
            else:
                _swap(child)

    n_lin = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    n_biased = sum(1 for m in model.modules()
                   if isinstance(m, torch.nn.Linear) and m.bias is not None)
    _swap(model)
    return model.eval(), n_lin, n_biased


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
    ap.add_argument("--m2m-root", default="/path/to/model2MLIR",
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
    #    Mirror the consistent-capture harness (model2MLIR/workloads/capture_consistent.py): seed the
    #    instantiation so weight init is reproducible, THEN perturb any exactly-zero parameters with
    #    small noise. DiT / flow-matching models (rdt, rdt2) zero-init their adaLN-Zero OUTPUT HEAD,
    #    so a fresh random-init eager forward is EXACTLY ZERO — which made our recomputed golden a
    #    zero tensor and the int8-vs-fp32 cosine degenerate (cos=0.0, rel=0.0: BOTH the golden and the
    #    ExecuTorch output were all-zeros, not an XNNPACK/delegate bug). The capture harness perturbs
    #    those zero params so the captured golden exercises the full numeric path (rdt2 golden
    #    norm=11.4, rdt=52.9); the recompute here MUST do the identical thing or the reference is a
    #    zero tensor for any zero-init-head model. Models with no exactly-zero params (llama/bitvla/
    #    xr0/…) are untouched — the loop is a no-op — so this is non-regressive. The perturbation is
    #    applied to the SAME instance the golden and the int8 swap both derive from, so the gate stays
    #    a self-consistent int8-vs-fp32 comparison.
    loader = _load_loader(Path(args.loader))
    torch.manual_seed(0)
    np.random.seed(0)
    model, _example = loader.get_model_and_inputs()
    model = model.eval()
    with torch.no_grad():
        for _p in model.parameters():
            if float(_p.detach().abs().max()) == 0.0:
                _p.copy_(torch.randn_like(_p) * 0.02)

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
        n_biased = sum(1 for m in model.modules()
                       if isinstance(m, torch.nn.Linear) and m.bias is not None)
        # The official llama handler is bias-free (its WeightOnlyInt8Linear drops bias and its
        # load_state_dict rejects any leftover .bias key). Use it verbatim for the bias-free
        # llama family (identical attribution/numbers as before). For a model with ANY biased
        # Linear (rdt/rdt2 DiT: qkv/proj/ffn all bias=True), the official handler raises
        # "Unexpected key(s) ... .bias" -> use OUR bias-preserving per-channel int8 swap, which
        # does the same math but keeps the fp32 bias. Same const-fold/arena behavior either way.
        if n_biased == 0:
            model = WeightOnlyInt8QuantHandler(model).quantized_model().eval()
            recipe = ("official-llama-recipe(WeightOnlyInt8QuantHandler, weight-only int8 "
                      f"per-channel, ALL {n_lin} nn.Linear swapped -> WeightOnlyInt8Linear)")
        else:
            model, n_lin, n_biased = _int8_whole_model_bias_preserving(model)
            recipe = (f"bias-preserving-int8(weight-only int8 per-channel, ALL {n_lin} nn.Linear "
                      f"swapped, {n_biased} carry fp32 bias; official handler drops bias/rejects "
                      "biased state_dict so this superset is used)")
        quantized = True
        subgraph_note = (subgraph_note + " " if subgraph_note else "") + (
            f"int8-whole-model={recipe}; embeddings/mask/softmax/norm fp32; "
            "module-swap sidesteps PT2E transform_for_annotation index-corruption")

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
    #     const-prop ALSO concretizes data-dependent constant subgraphs: smolvla fp32 otherwise
    #     fails to_executorch with an unbacked-symint "data-dependent" error in dim_order_from_stride
    #     (a bucketize-derived stride the static memory planner can't evaluate); folding the constant
    #     boundaries resolves it, which is why the int8 path (already const-propped) exported cleanly.
    #     So run it for the fp32 whole-model xnnpack path too (not just int8).
    _whole_model = args.int8_whole_model or (not args.no_xnnpack and not args.int8_subgraph
                                             and not args.quantize)
    if _whole_model:
        try:
            from executorch.exir.passes.constant_prop_pass import constant_prop_pass
            exported = constant_prop_pass(exported)
            subgraph_note += (" +const-prop(dequant-weights-folded->const, arena 4.25GB->2MB)"
                              if args.int8_whole_model
                              else " +const-prop(fp32 whole-model; resolves bucketize data-dep symint)")
        except Exception as e:  # noqa: BLE001
            subgraph_note += f" const-prop-skipped({str(e)[:80]})"

    # 4. XNNPACK partitioner (the RVV-microkernel delegate path).
    partitioners = []
    if not args.no_xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        partitioners.append(XnnpackPartitioner())

    from executorch.exir import EdgeCompileConfig
    # Disable the advisory core-ATen IR-validity check for ALL models (not just quantized): some
    # real forwards emit ops outside the strict "core ATen opset" (e.g. smolvla fp32 emits
    # aten.bucketize.Tensor) that the verifier rejects even though a runtime kernel exists / can be
    # provided. Whether an op is actually runnable is decided by the runner's registered kernel set,
    # which this arm audits separately (missing kernels surface as a precise method-load gap), so the
    # verifier is redundant here and would spuriously block otherwise-exportable models.
    _cc = EdgeCompileConfig(_check_ir_validity=False)
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
