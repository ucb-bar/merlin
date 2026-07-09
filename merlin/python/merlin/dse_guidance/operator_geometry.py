"""Operator-geometry extraction — the first search-space-formation layer.

For every matmul-like op in every recaptured workload this records the real geometry the flat
capture carries: (M, N, K), MACs, per-operand bytes, aspect ratios, a deterministic geometric
shape class and (orthogonally) a semantic role recovered from ``prov.fqn``. This is what a future
DSE engine consumes to decide which compute-primitive shapes its search space must contain.

Honesty boundaries (verified against the captures):

* ``linalg.matmul`` is 2-D in these captures — there is **no batch dimension** in the IR, so
  ``batch_product = 1`` and ``batch_dims`` is ``unavailable`` (not invented).
* the extractor records one element dtype (the right-hand / weight operand). ``lhs_dtype`` and
  ``out_dtype`` are **not separately recoverable** from the record and are emitted ``unavailable``.
* **attention** is lowered into the matmul projections — no explicit attention op survives, so
  attention structure (heads, head_dim, kv_len, mask) is ``unavailable``; only the Q/K/V/O
  projection *matmuls* are visible (and are labelled semantically).
* **conv** ops: reported only if a ``linalg.conv*`` op is actually present (none are, in the
  current captures) — never fabricated.
* **epilogue**: basic detection only — an ``addmm`` carries a fused bias (``bias_addmm``);
  full fusion/activation/quant-epilogue analysis is a later phase.

Every field carries an evidence label. No field is a performance/cost claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.dse_guidance import shape_taxonomy as ST
from merlin.dse_guidance.attribution import MatmulRecord, role_from_fqn
from merlin.dse_guidance.design_envelope import E_FQN, E_IR, E_NA, ELEMENT_BYTES

UNAVAILABLE = "unavailable"


@dataclass
class OperatorShape:
    workload: str
    capture_id: str
    op_index: int
    region_id: str | None
    region_role: str
    prov_fqn: str | None
    op_kind: str | None
    epilogue: bool
    epilogue_hint: str           # "bias_addmm" | "none"
    M: int
    N: int
    K: int
    batch_product: int           # 1 (linalg.matmul is 2-D); batch_dims unavailable
    macs: int                    # M*N*K*batch_product
    lhs_bytes: int               # M*K*elem (activation in)
    rhs_weight_bytes: int        # K*N*elem (weight)
    output_bytes: int            # M*N*elem
    dtype: str | None            # recorded matmul element dtype (rhs/weight operand)
    aspect_ratio_MN: float
    aspect_ratio_MK: float
    aspect_ratio_NK: float
    is_tail_heavy: bool
    is_small_fragment: bool
    shape_class: str             # geometric (drives primitive coverage)
    semantic_class: str          # from prov.fqn (annotation)
    evidence_shape: str
    evidence_role: str
    evidence_dtype: str
    evidence_bytes: str


def _ratio(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def operator_shape(rec: MatmulRecord, workload: str, role: str | None) -> OperatorShape:
    elem = ELEMENT_BYTES.get(str(rec.dtype or "f32").strip().lower(), 4.0)
    role = role or "unknown"
    return OperatorShape(
        workload=workload, capture_id=workload, op_index=rec.index,
        region_id=rec.region_id, region_role=role, prov_fqn=rec.fqn,
        op_kind=rec.op, epilogue=rec.epilogue,
        epilogue_hint=("bias_addmm" if rec.epilogue else "none"),
        M=rec.M, N=rec.N, K=rec.K, batch_product=1, macs=rec.M * rec.N * rec.K,
        lhs_bytes=int(rec.M * rec.K * elem),
        rhs_weight_bytes=int(rec.K * rec.N * elem),
        output_bytes=int(rec.M * rec.N * elem),
        dtype=rec.dtype,
        aspect_ratio_MN=_ratio(rec.M, rec.N),
        aspect_ratio_MK=_ratio(rec.M, rec.K),
        aspect_ratio_NK=_ratio(rec.N, rec.K),
        is_tail_heavy=ST.is_tail_heavy(rec.M, rec.N),
        is_small_fragment=ST.is_small_fragment(rec.M, rec.N, rec.K),
        shape_class=ST.classify_geometry(rec.M, rec.N, rec.K),
        semantic_class=ST.classify_semantic(rec.fqn),
        evidence_shape=E_IR,
        evidence_role=(E_FQN if role != "unknown" else E_NA),
        evidence_dtype=(E_IR if rec.dtype else E_NA),
        evidence_bytes=E_IR)


def _role_map(attribution) -> dict[int, str]:
    """op_index -> attributed region role (explicit-mapping/prov_fqn), for region_role labelling."""
    out: dict[int, str] = {}
    if attribution is None:
        return out
    for r in attribution.regions:
        for idx in r.matmul_indices:
            out[idx] = r.role
    return out


def operator_shapes(records, workload: str, attribution=None) -> list[OperatorShape]:
    """Per-op geometry for one workload. Role comes from attribution, else prov.fqn, else unknown."""
    role_of = _role_map(attribution)
    out = []
    for rec in records:
        role = role_of.get(rec.index) or role_from_fqn(rec.fqn) or "unknown"
        out.append(operator_shape(rec, workload, role))
    return out


# ----------------------------------------------------------------- conv / attention visibility

def conv_ops_present(records_module_text: str) -> bool:
    """True if any linalg.conv* op appears in the raw IR text (else conv structure is absent)."""
    return "linalg.conv" in records_module_text


# --------------------------------------------------------------------------- emitters

_SHAPE_COLS = ["workload", "capture_id", "op_index", "region_id", "region_role", "prov_fqn",
               "op_kind", "epilogue", "epilogue_hint", "M", "N", "K", "batch_product", "macs",
               "lhs_bytes", "rhs_weight_bytes", "output_bytes", "dtype",
               "aspect_ratio_MN", "aspect_ratio_MK", "aspect_ratio_NK",
               "is_tail_heavy", "is_small_fragment", "shape_class", "semantic_class",
               "evidence_shape", "evidence_role", "evidence_dtype", "evidence_bytes"]


def operator_shape_csv(all_shapes: list[OperatorShape]) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = [{c: getattr(s, c) for c in _SHAPE_COLS} for s in all_shapes]
    return _csv(rows, _SHAPE_COLS)


def _histogram(shapes, attr: str) -> dict:
    from collections import Counter
    return dict(Counter(getattr(s, attr) for s in shapes))


def shape_summary_by_workload_csv(by_workload: dict[str, list[OperatorShape]]) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for w, shapes in by_workload.items():
        total_macs = sum(s.macs for s in shapes) or 1
        per_class: dict[str, list] = {}
        for s in shapes:
            per_class.setdefault(s.shape_class, []).append(s)
        for cls, ss in sorted(per_class.items()):
            macs = sum(s.macs for s in ss)
            rows.append({"workload": w, "shape_class": cls, "op_count": len(ss),
                         "macs_total": macs, "mac_fraction": round(macs / total_macs, 4)})
    return _csv(rows, ["workload", "shape_class", "op_count", "macs_total", "mac_fraction"])


def shape_summary_by_region_csv(by_workload: dict[str, list[OperatorShape]]) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for w, shapes in by_workload.items():
        groups: dict[tuple, list] = {}
        for s in shapes:
            groups.setdefault((s.region_role, s.shape_class), []).append(s)
        for (role, cls), ss in sorted(groups.items()):
            rows.append({"workload": w, "region_role": role, "shape_class": cls,
                         "op_count": len(ss), "macs_total": sum(s.macs for s in ss)})
    return _csv(rows, ["workload", "region_role", "shape_class", "op_count", "macs_total"])


def operator_cluster_csv(all_shapes: list[OperatorShape]) -> str:
    """Cross-workload clusters: one row per (geometry) shape_class with which workloads exhibit it."""
    from merlin.dse_guidance.corpus import _csv
    groups: dict[str, list] = {}
    for s in all_shapes:
        groups.setdefault(s.shape_class, []).append(s)
    total = sum(s.macs for s in all_shapes) or 1
    rows = []
    for cls, ss in groups.items():
        macs = sum(s.macs for s in ss)
        rows.append({"shape_class": cls, "op_count": len(ss), "macs_total": macs,
                     "mac_fraction": round(macs / total, 4),
                     "workloads": "; ".join(sorted({s.workload for s in ss})),
                     "semantic_roles": "; ".join(sorted({s.semantic_class for s in ss}))})
    rows.sort(key=lambda r: -r["macs_total"])
    return _csv(rows, ["shape_class", "op_count", "macs_total", "mac_fraction",
                       "workloads", "semantic_roles"])


def to_yaml_obj(by_workload: dict[str, list[OperatorShape]], conv_visible: bool) -> dict:
    workloads = []
    for w, shapes in by_workload.items():
        workloads.append({
            "workload": w,
            "n_operators": len(shapes),
            "shape_class_histogram": _histogram(shapes, "shape_class"),
            "semantic_class_histogram": _histogram(shapes, "semantic_class"),
            "n_tail_heavy": sum(1 for s in shapes if s.is_tail_heavy),
            "n_small_fragment": sum(1 for s in shapes if s.is_small_fragment),
            "operators": [
                {"op_index": s.op_index, "region_role": s.region_role, "prov_fqn": s.prov_fqn,
                 "M": s.M, "N": s.N, "K": s.K, "macs": s.macs,
                 "shape_class": s.shape_class, "semantic_class": s.semantic_class,
                 "is_tail_heavy": s.is_tail_heavy, "is_small_fragment": s.is_small_fragment,
                 "evidence": E_IR}
                for s in shapes],
        })
    return {"operator_geometry": {
        "note": "structural operator geometry recovered from the flat capture. M/N/K, MACs and "
                "bytes are recovered_from_ir; shape_class is a deterministic function of the IR "
                "shape; semantic_class is recovered_from_prov_fqn. No speedup or cost is claimed.",
        "attention_structure": {
            "value": UNAVAILABLE,
            "evidence": E_NA,
            "reason": "attention is lowered into matmul projections in the flat capture; heads / "
                      "head_dim / q_len / kv_len / mask are not recoverable. Only the Q/K/V/O "
                      "projection matmuls are visible (see semantic_class)."},
        "conv_structure": {
            "value": ("present_see_operators" if conv_visible else "none_visible"),
            "evidence": (E_IR if conv_visible else E_NA)},
        "workloads": workloads,
    }}


def report_md(by_workload: dict[str, list[OperatorShape]], all_shapes: list[OperatorShape]) -> str:
    from collections import Counter
    L = ["# Operator-geometry report\n",
         "> The first search-space-formation layer: what matmul-like operator shapes actually "
         "appear across the recaptured workloads, classified by deterministic geometry rules and "
         "(orthogonally) by semantic role from `prov.fqn`. **Structural geometry only — no speedup, "
         "no cycle-count, no performance claim.**\n"]
    L.append(f"Operators extracted: **{len(all_shapes)}** across "
             f"**{len(by_workload)}** workloads.\n")
    L.append("## Per-workload operator counts and dominant geometry\n")
    L.append("| workload | operators | dominant shape_class (by MACs) | tail-heavy | small fragments |")
    L.append("|---|---|---|---|---|")
    for w, shapes in by_workload.items():
        macs_by_cls = Counter()
        for s in shapes:
            macs_by_cls[s.shape_class] += s.macs
        dom = macs_by_cls.most_common(1)[0][0] if macs_by_cls else "—"
        L.append(f"| {w} | {len(shapes)} | {dom} | "
                 f"{sum(1 for s in shapes if s.is_tail_heavy)} | "
                 f"{sum(1 for s in shapes if s.is_small_fragment)} |")
    L.append("")
    # cross-workload shape clusters
    gc = Counter(); gm = Counter()
    sc = Counter()
    for s in all_shapes:
        gc[s.shape_class] += 1
        gm[s.shape_class] += s.macs
        sc[s.semantic_class] += 1
    total_macs = sum(gm.values()) or 1
    L.append("## Top shape classes by MAC count (geometry)\n")
    L.append("| shape_class | ops | MACs | MAC share |")
    L.append("|---|---|---|---|")
    for cls, m in gm.most_common():
        L.append(f"| {cls} | {gc[cls]} | {m:,} | {m/total_macs:.1%} |")
    L.append("")
    L.append("## Top shape classes by op count (geometry)\n")
    L.append("| shape_class | ops |")
    L.append("|---|---|")
    for cls, c in gc.most_common():
        L.append(f"| {cls} | {c} |")
    L.append("")
    L.append("## Semantic roles (from prov.fqn)\n")
    L.append("| semantic_class | ops |")
    L.append("|---|---|")
    for cls, c in sc.most_common():
        L.append(f"| {cls} | {c} |")
    L.append("")
    # irregularity findings
    tail = [s for s in all_shapes if s.is_tail_heavy]
    frag = [s for s in all_shapes if s.is_small_fragment]
    L.append("## Shape-irregularity findings\n")
    if tail:
        ex = tail[0]
        L.append(f"- **Tail-heavy operators:** {len(tail)} op(s) waste >10% against a 32×32 tile "
                 f"(e.g. `{ex.workload}` op {ex.op_index} {ex.M}×{ex.N}, "
                 f"{ST.tail_waste(ex.M, ex.N):.0%} waste).")
    if frag:
        ex = frag[0]
        L.append(f"- **Small dispatch fragments:** {len(frag)} op(s) below "
                 f"{ST.SMALL_FRAG_MACS:,} MACs (e.g. `{ex.workload}` op {ex.op_index}, "
                 f"{ex.macs:,} MACs) — dispatch-bound, not compute-bound.")
    if not tail and not frag:
        L.append("- No tail-heavy or small-fragment irregularities at the configured thresholds.")
    L.append("")
    L.append("## Not recovered (honest)\n")
    L.append("- **Attention structure** (heads / head_dim / kv_len / mask): `unavailable` — "
             "attention is lowered into the matmul projections; only Q/K/V/O projection matmuls "
             "are visible.")
    L.append("- **Conv structure**: no `linalg.conv*` ops present in the current captures.")
    L.append("- **Batch dims**: `linalg.matmul` is 2-D here; `batch_product = 1`.")
    L.append("\nSee `operator_shape_table.csv` for every operator and `operator_cluster_table.csv` "
             "for the cross-workload clusters. Thresholds are documented in `shape_taxonomy.py`.\n")
    return "\n".join(L)
