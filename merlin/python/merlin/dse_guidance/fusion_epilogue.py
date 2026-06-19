"""Fusion / epilogue / accumulator contract — numerical structure beyond dtype capacity.

This extends the numerical contract from "which dtype fits" into *where numerical operations sit*:
the epilogue ops that follow each matmul (bias / activation / scale / clamp / cast), the accumulator
+ dequant/requant placement, and the fused-epilogue abstractions a future DSE could explore. It
detects only what the IR actually carries — the epilogue op-classes are read from the consuming
``linalg.generic`` bodies (``recovered_from_ir``); structure the flat dequantized capture erased
(dequant before matmul, requant, scale metadata, packed low-bit layout, sparsity) is reported
``lost`` / ``unavailable``, never invented. No speedup and no low-bit performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from merlin.design_pressure.ingest import mlir_m2m
from merlin.dse_guidance.design_envelope import E_IR, E_NA

_MAX_HOPS = 3   # local epilogue window: follow linalg.generic consumer chains this far

# value-preserving layout ops: a matmul output reshaped by these before any elementwise op has its
# downstream ops *reshape-separated* (not directly fused). We deliberately do NOT traverse them to
# grab the downstream op, because a reshape-distant addf/mulf is ambiguous (residual / rotary /
# gating / norm), not a fusible bias/scale — claiming it would over-state the fusion.
_SHAPE_OPS = {"tensor.expand_shape", "tensor.collapse_shape", "tensor.extract_slice",
              "linalg.transpose"}

# elementwise op-class -> the epilogue flag it sets
_BIAS = {"arith.addf"}
_SCALE = {"arith.mulf"}
_ACTIVATION = {"math.erf", "math.exp", "math.tanh", "math.gelu", "math.silu", "math.sigmoid"}
_CLAMP = {"arith.maximumf", "arith.minimumf"}
_CAST = {"arith.sitofp", "arith.fptosi", "arith.truncf", "arith.extf", "arith.fptrunc",
         "arith.uitofp"}

# accumulator / dequant / requant vocabularies (the verifier checks membership).
DEQ_BEFORE = "before_matmul"
DEQ_FUSED = "fused_candidate"
DEQ_AFTER_LOAD = "after_load"
DEQ_NA = "unavailable"
REQ_SEPARATE = "separate_op"
REQ_EPILOGUE = "epilogue_candidate"
REQ_NA = "unavailable"
ACC_MATERIALIZED = "materialized"
ACC_COMMITTED = "committed_directly"
ACC_NA = "unavailable"


def _body_op_names(op) -> set[str]:
    names: set[str] = set()
    for region in op.regions:
        for blk in region.blocks:
            for o in blk.ops:
                names.add(o.name)
    return names


@dataclass
class EpiloguePattern:
    index: int
    op_kind: str | None
    pattern: str                 # e.g. matmul->bias, matmul->bias->activation, matmul (none)
    has_bias: bool
    has_activation: bool
    has_scale: bool
    has_clamp: bool
    has_cast: bool
    epilogue_generic_count: int
    reshape_separated_epilogue: bool   # matmul output reshaped before any elementwise op
    evidence: str


def _pattern_label(flags: dict) -> str:
    parts = []
    if flags["bias"]:
        parts.append("bias")
    if flags["scale"]:
        parts.append("scale")
    if flags["activation"]:
        parts.append("activation")
    if flags["clamp"]:
        parts.append("clamp")
    if flags["cast"]:
        parts.append("cast")
    return "matmul" + ("->" + "->".join(parts) if parts else "")


@lru_cache(maxsize=None)
def epilogue_patterns(capture_dir: str) -> tuple[EpiloguePattern, ...]:
    """Per-matmul epilogue pattern, read from the consuming linalg.generic bodies (local window)."""
    path = f"{capture_dir}/model.mlir"
    try:
        module = mlir_m2m._parse_module(open(path, encoding="utf-8").read())
    except Exception:
        return ()
    matmuls = [o for o in module.walk() if o.name == "linalg.matmul"]
    out: list[EpiloguePattern] = []
    for i, mm in enumerate(matmuls):
        flags = {"bias": False, "scale": False, "activation": False, "clamp": False, "cast": False}
        op_kind = None
        v = mm.attributes.get("prov.op") or mm.attributes.get("m2m.op")
        if v is not None:
            op_kind = str(v).strip().strip('"')
        if op_kind == "addmm":
            flags["bias"] = True          # addmm = matmul + bias
        # directly-fused vs reshape-separated: is the immediate consumer an elementwise generic,
        # or a value-preserving layout op (reshape) that separates any downstream elementwise op?
        direct = [u.operation for r in mm.results for u in r.uses]
        direct_generic = any(o.name == "linalg.generic" for o in direct)
        reshape_separated = (not direct_generic) and any(o.name in _SHAPE_OPS for o in direct)
        n_generic = 0
        visited: set[int] = set()
        frontier = list(direct)
        hops = 0
        while frontier and hops < _MAX_HOPS:
            nxt = []
            for op in frontier:
                if id(op) in visited or op.name != "linalg.generic":
                    continue
                visited.add(id(op))
                n_generic += 1
                body = _body_op_names(op)
                if body & _BIAS:
                    flags["bias"] = True
                if body & _SCALE:
                    flags["scale"] = True
                if body & _ACTIVATION:
                    flags["activation"] = True
                if body & _CLAMP:
                    flags["clamp"] = True
                if body & _CAST:
                    flags["cast"] = True
                for r in op.results:
                    nxt.extend(u.operation for u in r.uses)
            frontier = nxt
            hops += 1
        out.append(EpiloguePattern(
            index=i, op_kind=op_kind, pattern=_pattern_label(flags),
            has_bias=flags["bias"], has_activation=flags["activation"], has_scale=flags["scale"],
            has_clamp=flags["clamp"], has_cast=flags["cast"], epilogue_generic_count=n_generic,
            reshape_separated_epilogue=reshape_separated, evidence=E_IR))
    return tuple(out)


# --------------------------------------------------------------- accumulator / dequant / requant

@dataclass
class AccumulatorContract:
    region: str
    op_count: int
    storage_dtype: str
    compute_dtype: str
    accumulator_dtype: str
    output_dtype: str
    scale_dtype: str             # "unavailable" — erased by a dequantized capture
    scale_granularity: str       # "unavailable"
    dequant_location: str
    requant_location: str
    accumulator_materialization: str
    evidence: str


def accumulator_contract(records, attribution, nc, patterns) -> list[AccumulatorContract]:
    """Per-region accumulator + dequant/requant placement contract (honest about erased fields)."""
    has_bias_by_idx = {p.index: p.has_bias for p in patterns}
    out: list[AccumulatorContract] = []
    if attribution is None:
        return out
    for r in attribution.regions:
        if r.attribution_status != "attributed":
            continue
        idxs = r.matmul_indices
        recs = [rec for rec in records if rec.index in idxs]
        if not recs:
            continue
        storage = compute = output = (recs[0].dtype or "f32")
        # dequant: no low-bit storage in the capture -> no dequant present (f32). Fused candidate
        # only when storage is actually low-bit (it is not here) -> unavailable.
        is_lowbit = storage.lower() in ("int8", "i8", "int4", "i4", "fp8", "f8")
        # accumulator is derived PER REGION from the region's own dtype (not the global-dominant
        # nc.accumulator_dtype, which disagrees for a mixed-dtype model such as bf16 smolvla):
        # integer low-bit storage accumulates in i32; float storage accumulates in its compute dtype.
        acc = "i32" if storage.lower() in ("int8", "i8", "int4", "i4") else compute
        dequant = DEQ_FUSED if is_lowbit else DEQ_NA
        # requant: an epilogue slot exists iff a bias epilogue was detected (it fuses like requant)
        any_bias = any(has_bias_by_idx.get(i, False) for i in idxs)
        requant = REQ_EPILOGUE if any_bias else REQ_NA
        # the matmul writes its output dtype directly; no separate accumulator op is materialized
        acc_mat = ACC_COMMITTED if acc == compute else ACC_MATERIALIZED
        out.append(AccumulatorContract(
            region=r.role, op_count=len(recs), storage_dtype=storage, compute_dtype=compute,
            accumulator_dtype=acc, output_dtype=output, scale_dtype="unavailable",
            scale_granularity="unavailable", dequant_location=dequant, requant_location=requant,
            accumulator_materialization=acc_mat, evidence=E_IR))
    return out


# --------------------------------------------------------------------------- certificates

@dataclass
class FusionCertificate:
    abstraction: str
    evidence: str
    required_compiler_proof: str
    required_hw_runtime_support: str
    dse_knobs_exposed: list
    accuracy_measurements_needed: str
    performance_measurements_needed_later: str
    what_is_not_claimed: str
    could_be_wrong_if: str


def certificates(patterns, accum) -> list[FusionCertificate]:
    """The fused-epilogue / numerical-placement abstraction candidates, with honest evidence."""
    any_bias = any(p.has_bias for p in patterns)
    any_act = any(p.has_activation for p in patterns)
    any_clamp = any(p.has_clamp for p in patterns)
    NOT = "no speedup, no low-bit performance, no throughput; structural candidate only"

    def cert(abstraction, present, proof, hw, knobs, acc_need, could_wrong):
        return FusionCertificate(
            abstraction=abstraction, evidence=(E_IR if present else E_NA),
            required_compiler_proof=proof, required_hw_runtime_support=hw,
            dse_knobs_exposed=knobs, accuracy_measurements_needed=acc_need,
            performance_measurements_needed_later="per-op latency / energy once a backend exists",
            what_is_not_claimed=NOT, could_be_wrong_if=could_wrong)

    return [
        cert("fused_requant_epilogue", any_bias,
             "the matmul output epilogue (bias detected via addf) can also host a requant; the "
             "epilogue slot is recovered_from_ir",
             "epilogue ALU on the matmul output path",
             ["epilogue_op_set", "requant_in_epilogue"],
             "int8 W8A8 accuracy (measured for gated workloads); fp8/int4 unavailable",
             "the detected addf is a residual add, not a fusible bias"),
        cert("accumulator_object", True,
             "the matmul accumulates in a wider type (i32 for int8 storage; derived)",
             "wide accumulator register/SRAM",
             ["accumulator_dtype", "accumulator_depth"],
             "int8 accuracy depends on i32 accumulation (measured); fp8/int4 unavailable",
             "the real datapath accumulates in a narrower type than assumed"),
        cert("accumulator_commit", any_bias,
             "the accumulator commits through the epilogue to the output dtype",
             "accumulator->epilogue->store path",
             ["commit_dtype", "commit_in_epilogue"],
             "output-dtype accuracy (int8 measured; others unavailable)",
             "a separate accumulator materialization op exists that the flat capture hid"),
        cert("activation_clamp_unit", any_act or any_clamp,
             "an activation/clamp op follows the matmul epilogue (erf/exp/maximumf detected)",
             "activation/clamp ALU on the output path",
             ["activation_kind", "clamp_bounds"],
             "activation in low precision needs accuracy measurement (int8 measured; others unavailable)",
             "the detected activation belongs to a separate op, not this matmul's epilogue"),
        cert("fused_dequant_matmul", False,
             "weights stored packed low-bit and dequantized on load fused into the matmul",
             "low-bit weight load + on-the-fly dequant",
             ["weight_format", "dequant_in_load"],
             "low-bit accuracy (int8 measured for gated workloads; fp8/int4 unavailable)",
             "the capture's f32 weights are the true storage (not a dequantized view)"),
        cert("scale_object", False,
             "per-tensor/-channel scale + zero-point carried alongside the packed weights",
             "scale sideband path",
             ["scale_granularity", "zero_point"],
             "quantization accuracy at the chosen granularity (unavailable — scales erased)",
             "scales are actually present in the source model but were folded away"),
        cert("packed_lowbit_tensor", False,
             "weights physically packed in a sub-byte layout",
             "packed-tensor load/unpack",
             ["pack_format", "pack_alignment"],
             "low-bit accuracy (int8 measured; fp8/int4 unavailable)",
             "the source model stores packed low-bit (the capture dequantized it)"),
        cert("resident_packed_weight_object", False,
             "the packed low-bit weights are made resident across the loop",
             "resident packed-weight store",
             ["resident_capacity", "pack_format"],
             "low-bit accuracy (int8 measured; fp8/int4 unavailable)",
             "packed residency is not legal for this weight layout"),
        cert("structured_sparsity_skip", False,
             "structured zeros in the weights allow skipping",
             "sparsity-aware datapath",
             ["sparsity_pattern", "skip_granularity"],
             "accuracy at the sparsity level (unavailable — no sparsity metadata)",
             "the weights are actually sparse (the dense f32 capture hid it)"),
    ]


# --------------------------------------------------------------------------- emitters

def epilogue_pattern_csv(pat_by_workload: dict) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for wl, pats in pat_by_workload.items():
        for p in pats:
            rows.append({"workload": wl, "op_index": p.index, "op_kind": p.op_kind or "matmul",
                         "pattern": p.pattern, "has_bias": p.has_bias,
                         "has_activation": p.has_activation, "has_scale": p.has_scale,
                         "has_clamp": p.has_clamp, "has_cast": p.has_cast,
                         "epilogue_generics": p.epilogue_generic_count,
                         "reshape_separated_epilogue": p.reshape_separated_epilogue,
                         "evidence": p.evidence})
    return _csv(rows, ["workload", "op_index", "op_kind", "pattern", "has_bias", "has_activation",
                       "has_scale", "has_clamp", "has_cast", "epilogue_generics",
                       "reshape_separated_epilogue", "evidence"])


def accumulator_contract_csv(acc_by_workload: dict) -> str:
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for wl, accs in acc_by_workload.items():
        for a in accs:
            rows.append({"workload": wl, "region": a.region, "op_count": a.op_count,
                         "storage_dtype": a.storage_dtype, "compute_dtype": a.compute_dtype,
                         "accumulator_dtype": a.accumulator_dtype, "output_dtype": a.output_dtype,
                         "scale_dtype": a.scale_dtype, "scale_granularity": a.scale_granularity,
                         "dequant_location": a.dequant_location,
                         "requant_location": a.requant_location,
                         "accumulator_materialization": a.accumulator_materialization,
                         "evidence": a.evidence})
    return _csv(rows, ["workload", "region", "op_count", "storage_dtype", "compute_dtype",
                       "accumulator_dtype", "output_dtype", "scale_dtype", "scale_granularity",
                       "dequant_location", "requant_location", "accumulator_materialization",
                       "evidence"])


def epilogue_candidates_yaml(cert_by_workload: dict) -> dict:
    # certificates are workload-independent in structure; emit one set keyed by abstraction with the
    # workloads whose IR shows the supporting pattern.
    by_abs: dict[str, dict] = {}
    for wl, certs in cert_by_workload.items():
        for c in certs:
            d = by_abs.setdefault(c.abstraction, {"cert": c, "present_in": []})
            if c.evidence == E_IR:
                d["present_in"].append(wl)
    return {"numerical_epilogue_candidates": {
        "note": "fused-epilogue / numerical-placement abstraction candidates. evidence is "
                "recovered_from_ir where the IR shows the supporting pattern, else unavailable "
                "(erased by a dequantized capture). int8 accuracy is measured for the gated "
                "workloads; fp8/int4 accuracy is unavailable. No speedup or low-bit performance "
                "is claimed.",
        "candidates": [
            {"abstraction": a, "evidence": d["cert"].evidence,
             "present_in_workloads": sorted(set(d["present_in"])),
             "required_compiler_proof": d["cert"].required_compiler_proof,
             "required_hw_runtime_support": d["cert"].required_hw_runtime_support,
             "dse_knobs_exposed": d["cert"].dse_knobs_exposed,
             "accuracy_measurements_needed": d["cert"].accuracy_measurements_needed,
             "performance_measurements_needed_later": d["cert"].performance_measurements_needed_later,
             "what_is_not_claimed": d["cert"].what_is_not_claimed,
             "could_be_wrong_if": d["cert"].could_be_wrong_if}
            for a, d in by_abs.items()]}}


def lost_contracts_csv(acc_by_workload: dict) -> str:
    """The numerical contracts a flat dequantized capture erased, per workload (honest accounting)."""
    from merlin.dse_guidance.case_study import _csv
    rows = []
    for wl, accs in acc_by_workload.items():
        storage = accs[0].storage_dtype if accs else "f32"
        lowbit = storage.lower() in ("int8", "i8", "int4", "i4", "fp8", "f8")
        for contract, status, reason in [
            ("low_bit_weight_storage", ("present" if lowbit else "lost"),
             "weights are f32 in the capture (dequantized) — native low-bit storage erased"
             if not lowbit else "low-bit storage visible"),
            ("scale_zero_point_metadata", "erased",
             "scales/zero-points folded away when the capture was dequantized"),
            ("packed_lowbit_layout", "lost",
             "weights stored in dense f32 layout — packed sub-byte layout erased"),
            ("dequant_op", "unavailable", "no dequant op in the capture (folded into f32 weights)"),
            ("requant_op", "unavailable", "no requant op in the capture (f32 datapath)"),
            ("sparsity_metadata", "unavailable", "weights dense — no structured-sparsity metadata"),
        ]:
            rows.append({"workload": wl, "contract": contract, "status": status,
                         "evidence": E_NA if status in ("lost", "erased", "unavailable") else E_IR,
                         "reason": reason})
    return _csv(rows, ["workload", "contract", "status", "evidence", "reason"])


def fusion_report_md(pat_by_workload: dict, acc_by_workload: dict, cert_by_workload: dict) -> str:
    from collections import Counter
    L = ["# Fusion / epilogue / numerical-contract report\n",
         "> Numerical structure beyond dtype capacity: the epilogue ops that follow each matmul, the "
         "accumulator/dequant/requant placement, and the fused abstractions structurally suggested. "
         "Patterns are `recovered_from_ir`; erased structure (dequant, requant, scale, packed layout, "
         "sparsity) is `unavailable`. **No speedup or low-bit performance is claimed.**\n"]
    # detected epilogue patterns
    pat = Counter()
    for pats in pat_by_workload.values():
        for p in pats:
            pat[p.pattern] += 1
    L.append("## Detected epilogue patterns (from IR)\n")
    L.append("| pattern | matmuls |")
    L.append("|---|---|")
    for p, n in pat.most_common():
        L.append(f"| {p} | {n} |")
    L.append("")
    # directly-fused vs reshape-separated
    n_reshape = sum(1 for pats in pat_by_workload.values() for p in pats
                    if p.reshape_separated_epilogue)
    n_plain = sum(1 for pats in pat_by_workload.values() for p in pats if p.pattern == "matmul")
    L.append("## Directly-fused vs reshape-separated\n")
    L.append(f"- Of the {n_plain} matmuls with **no directly-fused epilogue**, {n_reshape} are "
             "`reshape_separated_epilogue` — their output is reshaped (collapse/expand/transpose) "
             "before any elementwise op, so downstream ops (residual add, rotary, SiLU gating) are "
             "**not directly fused**. This is the bias-free LLaMA-style projection pattern. The "
             "reshape-distant ops are deliberately **not** labelled bias/scale: a reshape-distant "
             "addf/mulf is ambiguous (residual / rotary / gating / norm), so claiming it would "
             "over-state the fusion — it is reported as a layout-separated boundary instead.")
    L.append("")
    L.append("## Where numerical structure is preserved vs erased\n")
    L.append("- **Preserved (`recovered_from_ir`):** the matmul→bias epilogue (addmm / addf) and the "
             "activation/clamp ops (erf / exp / maximumf) that follow it — the epilogue slot is real.")
    L.append("- **Erased (`unavailable` / lost):** dequant-before-matmul, requant, scale/zero-point "
             "metadata, and the packed low-bit weight layout — the capture is dequantized f32, so "
             "native low-bit compute and its scales are gone (consistent with the cross-zoo "
             "numerical-fidelity finding).")
    L.append("")
    # which fused abstractions are suggested vs blocked
    sample = next(iter(cert_by_workload.values()), [])
    sup = [c.abstraction for c in sample if c.evidence == E_IR]
    blk = [c.abstraction for c in sample if c.evidence == E_NA]
    L.append("## Fused abstractions structurally suggested vs blocked\n")
    L.append(f"- **Suggested (IR-supported):** {', '.join('`'+a+'`' for a in sup) or '—'} — the "
             "epilogue slot, accumulator, and activation/clamp unit are backed by detected patterns.")
    L.append(f"- **Blocked (missing low-bit / scale / sparsity in the capture):** "
             f"{', '.join('`'+a+'`' for a in blk) or '—'} — these need a low-bit capture (packed "
             "weights + scales) or accuracy measurement before a DSE can use them.")
    L.append("")
    L.append("## How this changes future DSE search-space knobs\n")
    L.append("- adds an **epilogue-op-set** knob (which ops fuse onto the matmul output: bias / "
             "requant / activation / clamp) — the slot is proven present;")
    L.append("- adds an **accumulator-dtype** knob (i32 for int8 storage, derived);")
    L.append("- gates the **weight-format / dequant-in-load / scale-granularity** knobs behind a "
             "low-bit capture + accuracy measurement (int8 measured for the gated workloads; "
             "fp8/int4 unavailable).")
    L.append("\n**Caveat:** these are structural placement candidates with accuracy/performance "
             "measurements named per certificate. **No speedup and no low-bit performance is "
             "claimed.**\n")
    return "\n".join(L)
