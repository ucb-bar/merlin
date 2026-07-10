"""Numerical-contract fidelity audit — the precision counterpart to capture fidelity.

The temporal-contract work asks "which regions repeat / persist". This asks the orthogonal
question: **does the captured IR preserve the true numerical contract** — which tensors are
*stored*, *computed*, *accumulated*, *dequantized*, and *requantized* in which precision?

Read from the real capture (text-level, robust to xDSL parse gaps): the module quantization
scheme (`prov.quantization`), the matmul compute dtype (operand tensor types), and the
dequantize / quantize / requant op counts. The decisive, real cross-zoo finding this surfaces:
every int8/fp8 capture stores weights low-bit but runs **f32** matmuls (int8 captures carry
`dequantize` ops; none carry native i8 operands) — so native low-bit compute and the packed
layout are **absent from the capture**, a hidden DSE axis.

It emits **structural** numerical candidates with measurement plans — never a speedup, an
accuracy number, or a gap_closure. Lowering precision is a candidate to *measure*, not a claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance.design_envelope import E_DERIVED, E_IR, E_NA

_LOWBIT = {"int8_weight_only": "int8", "float8_weight_only_e4m3": "fp8",
           "int4_weight_only": "int4"}


@dataclass
class NumericalCandidate:
    axis: str
    evidence: dict
    required_accuracy_measurements: list[str]
    required_performance_measurements: list[str]
    required_hw_support: str
    could_be_wrong_if: list[str]
    legality: str = "structural"
    benefit: str = "unquantified"          # never a speedup
    quantification_blocked_by: str = "missing accuracy sweep + low-bit kernel calibration"


@dataclass
class NumericalContract:
    workload: str
    declared_quantization: str             # prov.quantization or "none"
    weight_storage_dtype: str              # int8 / fp8 / int4 / <compute dtype>
    compute_dtype: str                     # dominant matmul operand dtype (f32 / bf16 / i8)
    n_matmuls: int
    dequantize_ops: int
    requant_ops: int
    low_bit_storage: bool
    low_bit_compute_lost: bool             # low-bit stored but compute is wide
    packed_layout_visible: bool
    lost_structure: list[str] = field(default_factory=list)
    severity: str = "low"
    expected_contract: str | None = None
    mismatch: str | None = None
    candidates: list[NumericalCandidate] = field(default_factory=list)
    # Per-region dtype (joined from IR matmul records + role attribution), empty when unavailable.
    per_region_dtype: list[dict] = field(default_factory=list)
    # Honesty fields: what the capture does NOT carry is stated, never invented.
    accumulator_dtype: str = "?"
    accumulator_dtype_evidence: str = E_NA
    scale_metadata: str = "erased_or_unavailable"
    scale_metadata_evidence: str = E_NA
    sparsity_metadata: str = "erased_or_unavailable"
    sparsity_metadata_evidence: str = E_NA


def _read_text(capture_dir: str) -> str:
    try:
        return open(f"{capture_dir}/model.mlir", encoding="utf-8").read()
    except OSError:
        return ""


def extract_numerical_facts(capture_dir: str) -> dict:
    text = _read_text(capture_dir)
    if not text:
        return {"declared_quantization": "none", "n_matmuls": 0, "compute_dtype": "?",
                "dequantize_ops": 0, "quantize_ops": 0, "requant_ops": 0}
    from merlin.common import mlir_query

    module = mlir_query.parse(text)
    quant = "none"
    matmul_dtypes: list[str] = []
    dequantize = quantize = requant = 0
    for op in module.walk():
        if quant == "none":
            q = mlir_query.attr_str(op, "prov.quantization")
            if q is not None:
                quant = q
        if mlir_query.op_name(op) == "linalg.matmul":
            _, dt = mlir_query.type_shape_dtype(op.operands[0].type)  # first ins operand
            matmul_dtypes.append(dt)
        pv = mlir_query.attr_str(op, "prov.op")
        if pv == "dequantize":
            dequantize += 1
        elif pv == "quantize":
            quantize += 1
        elif pv in ("requant", "requantize"):
            requant += 1
    # dominant matmul operand dtype
    compute = max(set(matmul_dtypes), key=matmul_dtypes.count) if matmul_dtypes else "?"
    return {
        "declared_quantization": quant,
        "n_matmuls": len(matmul_dtypes),
        "compute_dtype": compute,
        "dequantize_ops": dequantize,
        "quantize_ops": quantize,
        "requant_ops": requant,
    }


def _wide(dtype: str) -> bool:
    return dtype in ("f32", "f64", "bf16", "f16", "fp16", "float32")


_LOWBIT_STORAGE = ("int8", "i8", "int4", "i4")


def per_region_dtype(records, attribution) -> list[dict]:
    """Per-region compute dtype, joining IR matmul records (dtype) with role attribution.

    Returns one entry per attributed region with at least one dtyped matmul; regions with no
    attributed/dtyped matmuls are omitted (never fabricated). Evidence is ``recovered_from_ir``.
    """
    if not records or attribution is None:
        return []
    by_index = {r.index: r for r in records}
    out: list[dict] = []
    for region in attribution.regions:
        if region.attribution_status != "attributed":
            continue
        dtypes = [by_index[i].dtype for i in region.matmul_indices
                  if i in by_index and by_index[i].dtype]
        if not dtypes:
            continue
        dominant = max(set(dtypes), key=dtypes.count)
        out.append({"region": region.role, "dtype": dominant, "n": len(dtypes), "evidence": E_IR})
    return out


def _accumulator_dtype(storage: str, compute: str) -> tuple[str, str]:
    """Inferred accumulator dtype + evidence. i32 for integer low-bit storage (derived), else the
    compute dtype as observed (recovered_from_ir)."""
    if storage.lower() in _LOWBIT_STORAGE:
        return "i32", E_DERIVED          # integer low-bit matmuls accumulate in i32 (inferred)
    return compute, E_IR


def audit(capture_dir: str, *, workload: str = "?", workload_class: str | None = None,
          repeated_head_weight_bytes: int | None = None,
          has_epilogue: bool = False, expected_contract: str | None = None,
          records=None, attribution=None) -> NumericalContract:
    """Audit one capture's numerical contract; attach structural candidates (no quantitative claims).

    When ``records`` (from :func:`attribution.extract_matmuls`) and ``attribution`` are supplied, a
    per-region dtype view is added by joining matmul dtypes to recovered roles.
    """
    f = extract_numerical_facts(capture_dir)
    quant = f["declared_quantization"]
    low_bit_storage = quant in _LOWBIT
    storage = _LOWBIT.get(quant, f["compute_dtype"])
    compute = f["compute_dtype"]
    low_bit_compute_lost = low_bit_storage and _wide(compute)
    # Weight-only quant dequantizes to a dense wide tensor before the matmul -> packed layout gone.
    packed_layout_visible = not (low_bit_storage and _wide(compute))

    lost: list[str] = []
    if low_bit_compute_lost:
        lost.append("native_low_bit_compute")
    if low_bit_storage and not packed_layout_visible:
        lost.append("packed_low_bit_layout")
    severity = "high" if lost else "low"

    mismatch = None
    if expected_contract and expected_contract.lower() not in (storage.lower(), quant.lower()):
        mismatch = f"expected {expected_contract}, captured {storage} storage / {compute} compute"
        if severity != "high":
            severity = "medium"

    acc_dtype, acc_evidence = _accumulator_dtype(storage, compute)
    contract = NumericalContract(
        workload=workload, declared_quantization=quant, weight_storage_dtype=storage,
        compute_dtype=compute, n_matmuls=f["n_matmuls"], dequantize_ops=f["dequantize_ops"],
        requant_ops=f["requant_ops"], low_bit_storage=low_bit_storage,
        low_bit_compute_lost=low_bit_compute_lost, packed_layout_visible=packed_layout_visible,
        lost_structure=lost, severity=severity, expected_contract=expected_contract,
        mismatch=mismatch,
        per_region_dtype=per_region_dtype(records, attribution),
        accumulator_dtype=acc_dtype, accumulator_dtype_evidence=acc_evidence,
    )
    contract.candidates = _candidates(contract, repeated_head_weight_bytes, has_epilogue,
                                      workload_class)
    return contract


def _candidates(c: NumericalContract, wbytes: int | None, has_epilogue: bool,
                workload_class: str | None) -> list[NumericalCandidate]:
    out: list[NumericalCandidate] = []

    # resident_packed_lowbit_weights: always relevant when there are reused head weights.
    ev = {"weight_storage_dtype": c.weight_storage_dtype, "compute_dtype": c.compute_dtype}
    if wbytes:
        ev["repeated_head_weight_bytes"] = wbytes
        ev["candidate_formats"] = ["fp8_w8a8", "int8_w8a8", "int4_weight_only",
                                   "int4_weight_fp8_activation"]
    out.append(NumericalCandidate(
        axis="resident_packed_lowbit_weights", evidence=ev,
        required_accuracy_measurements=["per-format accuracy vs fp32 (cos/argmax) sweep"],
        required_performance_measurements=["low-bit weight bytes", "dequant/pack cost",
                                           "resident capacity at each format"],
        required_hw_support="resident packed low-bit weight store + matching matmul datapath",
        could_be_wrong_if=["accuracy below the task gate at the chosen format",
                           "packed weights exceed resident capacity",
                           "dequant/pack cost outweighs the bandwidth saved"],
        quantification_blocked_by="missing accuracy sweep + low-bit kernel calibration + "
                                  "resident-capacity model"))

    # native_lowbit_compute: only when the capture proves low-bit storage but wide compute.
    if c.low_bit_compute_lost:
        out.append(NumericalCandidate(
            axis="native_lowbit_compute",
            evidence={"declared_quantization": c.declared_quantization,
                      "compute_dtype": c.compute_dtype, "dequantize_ops": c.dequantize_ops,
                      "note": "weights stored low-bit but matmul runs wide -> compute lost"},
            required_accuracy_measurements=["W8A8/W4A8 accuracy vs fp32 gate"],
            required_performance_measurements=["low-bit matmul cycles vs wide",
                                               "requant overhead per op"],
            required_hw_support="i8xi8->i32 (or fp8) matmul datapath + requant unit",
            could_be_wrong_if=["activation quant breaks accuracy",
                               "scalar core has no low-bit lane advantage"],
            quantification_blocked_by="missing low-bit-compute accuracy + kernel calibration"))

    # fused_dequant_matmul: when dequant runs as separate ops before the matmul.
    if c.dequantize_ops > 0:
        out.append(NumericalCandidate(
            axis="fused_dequant_matmul",
            evidence={"dequantize_ops": c.dequantize_ops},
            required_accuracy_measurements=["bit-exactness of fused vs separate dequant"],
            required_performance_measurements=["dequant materialization bytes",
                                               "extra dispatch count"],
            required_hw_support="dequant fused into the weight load / matmul prologue",
            could_be_wrong_if=["dequant is already hoisted", "fusion changes numerics"],
            quantification_blocked_by="missing measured dequant traffic / dispatch overhead"))

    # fused_requant_epilogue: when an epilogue keeps an accumulator live.
    if has_epilogue or c.requant_ops > 0:
        out.append(NumericalCandidate(
            axis="fused_requant_epilogue",
            evidence={"requant_ops": c.requant_ops, "has_epilogue": has_epilogue},
            required_accuracy_measurements=["fused requant/activation bit-exactness"],
            required_performance_measurements=["i32 intermediate bytes",
                                               "epilogue dispatch count"],
            required_hw_support="in-hardware accumulator commit + fused requant/activation",
            could_be_wrong_if=["epilogue already fused", "accumulator does not fit"],
            quantification_blocked_by="missing intermediate-materialization measurement"))

    # quantized_KV_cache: only for autoregressive / decode workloads.
    if workload_class and ("autoregress" in workload_class or "decode" in workload_class):
        out.append(NumericalCandidate(
            axis="quantized_KV_cache",
            evidence={"workload_class": workload_class},
            required_accuracy_measurements=["KV-quant accuracy vs fp16 KV"],
            required_performance_measurements=["KV bytes per step", "KV growth over decode"],
            required_hw_support="quantized resident KV-cache object",
            could_be_wrong_if=["short sequences make KV negligible",
                               "KV-quant degrades long-context accuracy"],
            quantification_blocked_by="missing KV-size profile + KV-quant accuracy sweep"))
    return out


def fidelity_report_md(contracts: list[NumericalContract]) -> str:
    """Cross-capture numerical-contract report (structural; no speedup/accuracy claims)."""
    L = ["# Numerical-contract fidelity\n"]
    L.append("> Does the capture preserve the numerical contract — what is stored, computed, "
             "accumulated, dequantized, and requantized, in which precision? These are structural "
             "observations from real captures; lowering precision is a candidate to **measure**, "
             "never a claimed speedup or accuracy.\n")
    L.append("| capture | declared quant | storage | compute | dequant ops | packed layout | "
             "lost | severity |")
    L.append("|---------|----------------|---------|---------|-------------|---------------|------|----------|")
    for c in contracts:
        L.append(f"| {c.workload} | {c.declared_quantization} | {c.weight_storage_dtype} | "
                 f"{c.compute_dtype} | {c.dequantize_ops} | "
                 f"{'visible' if c.packed_layout_visible else 'LOST'} | "
                 f"{', '.join(c.lost_structure) or '—'} | {c.severity} |")
    L.append("")
    lost = [c for c in contracts if c.low_bit_compute_lost]
    if lost:
        L.append(f"**Finding:** {len(lost)}/{len(contracts)} captures store weights low-bit but run "
                 "**wide (f32) matmuls** — native low-bit compute and the packed layout are absent "
                 "from the capture. The hidden DSE axes (`native_lowbit_compute`, "
                 "`resident_packed_lowbit_weights`, `fused_dequant_matmul`) are structural "
                 "candidates; ranking is blocked on accuracy sweeps + low-bit kernel calibration.\n")
    L.append("## Evidence labels\n")
    L.append("`recovered_from_ir` (dtypes, op counts, quantization) · `assumed_reference` "
             "(expected contract, if supplied) · candidates `uncalibrated` until measured.\n")
    return "\n".join(L)


def to_yaml_obj(c: NumericalContract) -> dict:
    return {
        "numerical_contract": {
            "workload": c.workload,
            "observed": {
                "declared_quantization": c.declared_quantization,
                "weight_storage_dtype": c.weight_storage_dtype,
                "compute_dtype": c.compute_dtype,
                "n_matmuls": c.n_matmuls,
                "dequantize_ops": c.dequantize_ops,
                "requant_ops": c.requant_ops,
                "packed_layout_visible": c.packed_layout_visible,
            },
            "per_region_dtype": c.per_region_dtype,
            "honesty": {
                "accumulator_dtype": {"value": c.accumulator_dtype,
                                      "evidence": c.accumulator_dtype_evidence},
                "scale_metadata": {"value": c.scale_metadata,
                                   "evidence": c.scale_metadata_evidence},
                "sparsity_metadata": {"value": c.sparsity_metadata,
                                      "evidence": c.sparsity_metadata_evidence},
                "note": "scale/zero-point/group-size and sparsity metadata are not present in a flat "
                        "dequantized capture; they are marked unavailable, never invented.",
            },
            "expected_contract": c.expected_contract,
            "mismatch": c.mismatch,
            "lost_structure": c.lost_structure,
            "severity": c.severity,
            "candidates": [
                {"axis": k.axis, "evidence": k.evidence,
                 "required_accuracy_measurements": k.required_accuracy_measurements,
                 "required_performance_measurements": k.required_performance_measurements,
                 "required_hw_support": k.required_hw_support,
                 "could_be_wrong_if": k.could_be_wrong_if,
                 "current_status": {"legality": k.legality, "benefit": k.benefit,
                                    "quantification_blocked_by": k.quantification_blocked_by}}
                for k in c.candidates
            ],
        }
    }


def torchao_integration_plan_md() -> str:
    """A PLAN (not a sweep): how a future DSE engine would wire TorchAO low-bit formats to the
    numerical abstraction candidates, and what each format needs measured before it is legal."""
    return (
        "# TorchAO integration plan — numerical-contract candidates\n\n"
        "> This is a **plan**, not a sweep. No quantization run is executed here, no speedup is "
        "claimed, and no accuracy number is asserted for any format that has not been measured. "
        "Only int8 (W8A8) has measured accuracy today (see `accuracy_gate_report.md`); every other "
        "format is `unavailable` until a gate is run.\n\n"
        "## What this connects\n"
        "The numerical-contract audit surfaces structural candidates "
        "(`resident_packed_lowbit_weights`, `native_lowbit_compute`, `fused_dequant_matmul`, "
        "`fused_requant_epilogue`). Each needs a real low-bit format to become concrete. TorchAO is "
        "the path to produce those formats and the metadata a compiler needs to keep them.\n\n"
        "## Order to try (cheapest-signal first) — accuracy status\n"
        "| format | TorchAO config | accuracy status | DSE candidate it informs |\n"
        "|--------|----------------|-----------------|--------------------------|\n"
        "| int8 W8A8 | `int8_dynamic_activation_int8_weight` | **measured: pass** (5/5, results.md) | "
        "`native_lowbit_compute`, `resident_packed_lowbit_weights` |\n"
        "| fp8 W8A8 | `float8_dynamic_activation_float8_weight` | unavailable (not measured) | "
        "`native_lowbit_compute` |\n"
        "| int4 weight-only | `int4_weight_only` | unavailable (not measured) | "
        "`resident_packed_lowbit_weights` |\n"
        "| int4 weight + fp8 act | (composed) | unavailable (not measured) | "
        "`resident_packed_lowbit_weights` |\n\n"
        "## What each format needs measured before it is DSE-legal\n"
        "- **accuracy gate** vs fp32 on the task metric (cos/argmax, and trajectory error for action "
        "outputs) — the only measurable-now leg; int8 done, others pending.\n"
        "- **packed-layout + scale metadata preserved** through capture — today the flat capture "
        "dequantizes to f32, so packed layout and scale/zero-point/group-size are erased "
        "(`scale_metadata: unavailable` in `numerical_contract.yaml`).\n"
        "- **low-bit kernel cost** — requires the proposed design (target_measured), not claimed here.\n\n"
        "## What compiler facts to preserve (so DSE can target the format)\n"
        "- packed weight layout as a dispatch-crossing object; per-tensor/-channel/-group scale "
        "objects; dequant/requant placement; i32 (or fp32) accumulator width and epilogue commit.\n\n"
        "## What is NOT claimed\n"
        "No speedup, no cycle/area/energy, no accuracy for any `unavailable` format, and no broad "
        "dtype sweep is run by this tool. This plan only states which formats to measure, in what "
        "order, and which abstraction each would unblock.\n")
