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

import re
from dataclasses import dataclass, field

_QUANT_RE = re.compile(r'prov\.quantization = "([^"]+)"')
_MATMUL_DTYPE_RE = re.compile(r'linalg\.matmul[^\n]*?ins\([^:]*:\s*tensor<[0-9x?]+x([a-z0-9]+)>')
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


def _read_text(capture_dir: str) -> str:
    try:
        return open(f"{capture_dir}/model.mlir", encoding="utf-8").read()
    except OSError:
        return ""


def extract_numerical_facts(capture_dir: str) -> dict:
    text = _read_text(capture_dir)
    quant_m = _QUANT_RE.search(text)
    quant = quant_m.group(1) if quant_m else "none"
    dtypes = _MATMUL_DTYPE_RE.findall(text)
    # dominant matmul operand dtype
    compute = max(set(dtypes), key=dtypes.count) if dtypes else "?"
    return {
        "declared_quantization": quant,
        "n_matmuls": text.count("linalg.matmul"),
        "compute_dtype": compute,
        "dequantize_ops": len(re.findall(r'prov\.op = "dequantize"', text)),
        "quantize_ops": len(re.findall(r'prov\.op = "quantize"', text)),
        "requant_ops": len(re.findall(r'prov\.op = "(?:requant|requantize)"', text)),
    }


def _wide(dtype: str) -> bool:
    return dtype in ("f32", "f64", "bf16", "f16", "fp16", "float32")


def audit(capture_dir: str, *, workload: str = "?", workload_class: str | None = None,
          repeated_head_weight_bytes: int | None = None,
          has_epilogue: bool = False, expected_contract: str | None = None) -> NumericalContract:
    """Audit one capture's numerical contract; attach structural candidates (no quantitative claims)."""
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

    contract = NumericalContract(
        workload=workload, declared_quantization=quant, weight_storage_dtype=storage,
        compute_dtype=compute, n_matmuls=f["n_matmuls"], dequantize_ops=f["dequantize_ops"],
        requant_ops=f["requant_ops"], low_bit_storage=low_bit_storage,
        low_bit_compute_lost=low_bit_compute_lost, packed_layout_visible=packed_layout_visible,
        lost_structure=lost, severity=severity, expected_contract=expected_contract,
        mismatch=mismatch,
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
