"""Corpus-level aggregation and the promotion ladder.

This is where per-kernel *observations* become *motifs* and, when they clear a promotion
threshold, *policy candidates* + *abstraction candidates*. The threshold encodes the
"appears across the corpus" test: a motif is promoted iff it appears in >=2 independent
sources OR in >=``min_kernels`` kernels. Evidence attached to emitted artifacts is always the
*real* set of kernel evidence-ids that fired the motif — never invented.

The CATALOG maps a motif to the abstraction/policy it justifies. ``promote`` only emits
catalog entries whose backing motif cleared the threshold, so nothing is asserted without
corpus support. Validation against the benchmark workloads (positive fires / negative control
silent) happens in :mod:`merlin.kernels` tests, not here.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from merlin.kernels.emit.abstraction_candidate import emit_abstraction_candidate
from merlin.kernels.emit.dialect_requirement import emit_dialect_requirement
from merlin.kernels.emit.interface_candidate import emit_interface_candidate
from merlin.kernels.emit.llvm_requirement import emit_llvm_requirement
from merlin.kernels.emit.policy_rule import emit_policy_rule
from merlin.kernels.emit.runtime_candidate import emit_runtime_candidate

# motif -> {abstraction: {...} | None, policy: {...}}
# Each policy `when` is symbolic (compiler-visible facts), never a single kernel's constants.
CATALOG: dict[str, dict] = {
    "packed_rhs": {
        "abstraction": {
            "name": "resident_packed_tensor",
            "kind": "memory_state",
            "motivation": "immutable RHS/weight is packed once and reused across a region; "
                          "keep it resident to avoid repeated pack/load.",
            "interface_features": ["resident_pack", "resident_tensor_type", "evict"],
        },
        "interface": {
            "name": "resident_packed_tensor",
            "interface_ops": ["resident_pack", "matmul_resident", "evict"],
            "interface_types": ["resident_tensor"],
            "compiler_must_prove": ["rhs_immutable", "reuse_count_above_threshold",
                                    "capacity_fit_or_eviction_inserted",
                                    "consumers_accept_packed_layout"],
            "hardware_must_provide": ["resident_storage", "packed_tensor_handle",
                                      "validity_until_eviction"],
            "runtime_must_provide": ["persistent_handle_lifetime", "command_ordering",
                                     "invalidation_protocol"],
        },
        "policy": {
            "policy": "packed_rhs_policy",
            # The discriminator is reuse + immutability (validated against benchmarks);
            # capacity is a separate regime dimension checked by the capacity sweep.
            "when": {"rhs_reuse_count": ">= 2", "rhs_mutable": "false"},
            "actions": ["preserve_packed_rhs_layout", "hoist_pack",
                        "consider_resident_packed_tensor"],
        },
    },
    "accumulator_commit": {
        "abstraction": {
            "name": "accumulator_commit",
            "kind": "memory_state",
            "motivation": "on a contraction op the accumulator stays live across a "
                          "bias/requant/activation epilogue; commit to memory only after the "
                          "epilogue to avoid extra writes.",
            "interface_features": ["accumulator_type", "commit", "keep_accumulator_live"],
        },
        "interface": {
            "name": "accumulator_commit",
            "interface_ops": ["accumulator", "commit"],
            "interface_types": ["accumulator"],
            "compiler_must_prove": ["epilogue_consumes_accumulator_immediately",
                                    "no_intervening_user_visible_materialization",
                                    "output_dtype_and_layout_known"],
            "hardware_must_provide": ["accumulator_state", "commit_epilogue_path"],
            "runtime_must_provide": ["command_ordering"],
        },
        "policy": {
            "policy": "accumulator_commit_policy",
            "when": {"op": "gemm|matmul|conv", "has_epilogue": "true",
                     "accumulator_live_across_epilogue": "true"},
            "actions": ["keep_accumulator_resident", "fuse_epilogue_before_commit",
                        "single_commit_store"],
        },
    },
    "vector_length_polymorphic": {
        "abstraction": None,  # schedule-level decision, not an HW/SW interface
        "policy": {
            "policy": "vl_agnostic_loop_policy",
            "when": {"target_has_scalable_vectors": "true"},
            "actions": ["emit_vl_agnostic_loop", "use_predicated_or_vl_tail",
                        "avoid_fixed_width_assumptions"],
        },
    },
    "double_buffering": {
        "abstraction": {
            "name": "async_pipeline",
            "kind": "async",
            "motivation": "data movement is double-buffered to overlap DMA with compute; "
                          "expose async copy + completion so the compiler can pipeline.",
            "interface_features": ["async_copy", "event_token", "double_buffer"],
        },
        "interface": {
            "name": "async_pipeline",
            "interface_ops": ["async_copy", "wait"],
            "interface_types": ["event_token"],
            "compiler_must_prove": ["operand_load_independent_of_current_compute",
                                    "double_buffer_capacity_available"],
            "hardware_must_provide": ["async_dma_engine", "completion_signal"],
            "runtime_must_provide": ["event_completion", "command_ordering"],
        },
        "policy": {
            "policy": "double_buffer_policy",
            "when": {"dma_compute_overlap_beneficial": "true", "staged_memory_target": "true"},
            "actions": ["double_buffer_operands", "overlap_dma_with_compute"],
        },
    },
    "weight_stationary_dataflow": {
        "abstraction": None,  # dataflow is a target capability, surfaced via contract
        "policy": {
            "policy": "weight_stationary_dataflow_policy",
            "when": {"target_dataflow": "weight_stationary", "op": "gemm|matmul|conv"},
            "actions": ["schedule_weight_stationary", "stage_weights_resident"],
        },
    },
    # --- RVV intrinsic decisions: schedule-level codegen knobs (abstraction None -> policy_rule
    # only). Each maps to an RVV target-package knob/lever via kernels/knobs.MOTIF_TO_KNOB,
    # which the tuning agent uses to propose forks. `when` is symbolic (compiler-visible facts),
    # never a kernel's literal LMUL/tile constants.
    "lmul_grouping": {
        "abstraction": None,
        "policy": {
            "policy": "lmul_grouping_policy",
            "when": {"target_has_scalable_vectors": "true", "op": "gemm|matmul|conv|dot",
                     "dtype": "f32|i8|bf16"},
            "actions": ["prefer_high_lmul", "set_vector_group_m4_or_m8"],
        },
    },
    "scalar_broadcast_fma": {
        "abstraction": None,
        "policy": {
            # Expert RVV GEMMs broadcast a scalar RHS into a FUSED multiply-add (vfmacc.vf);
            # our schedule's lower_contraction emits vfmul.vv+vfadd.vv separately (the measured
            # gap). This policy drives the contraction-lowering lever to recover fma.
            "policy": "fma_broadcast_policy",
            "when": {"op": "gemm|matmul", "rhs_reuse_count": ">= 1"},
            "actions": ["emit_scalar_broadcast_fma", "fuse_multiply_add", "register_block_rhs"],
        },
    },
    "int8_widening_mac": {
        "abstraction": None,
        "policy": {
            "policy": "int8_widening_policy",
            "when": {"dtype": "i8", "op": "gemm|matmul|conv"},
            "actions": ["use_vwmacc_widening", "i32_accumulator"],
        },
    },
    "vl_polymorphic_tail": {
        "abstraction": None,
        "policy": {
            "policy": "vl_tail_policy",
            "when": {"target_has_scalable_vectors": "true"},
            "actions": ["emit_vsetvl_loop", "vl_or_mask_tail"],
        },
    },
    "vector_reduction": {
        "abstraction": None,
        "policy": {
            "policy": "vector_reduction_policy",
            "when": {"op": "softmax|layernorm|rmsnorm|dot|reduce",
                     "target_has_scalable_vectors": "true"},
            "actions": ["emit_vector_reduction_tree", "use_vredsum_or_vfredusum"],
        },
    },
    "requant_narrowing": {
        "abstraction": None,
        "policy": {
            "policy": "requant_narrowing_policy",
            "when": {"dtype": "i8", "has_epilogue": "true"},
            "actions": ["fuse_requant_narrowing_store", "emit_vnclip_then_vse8"],
        },
    },
}


# motif -> runtime_candidate spec (L7). Promoted like CATALOG entries.
RUNTIME_CATALOG: dict[str, dict] = {
    "many_small_dispatches": {
        "name": "command_buffer_batching",
        "compiler_action": ["group_dispatches", "emit_command_buffer",
                            "amortize_repeated_config"],
        "runtime_requirement": ["batch_submit", "event_completion", "persistent_handles"],
    },
}


def _coerce(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return float(v) if ("." in v) else int(v)
        except ValueError:
            return v
    return v


def evaluate_when(when: dict, facts: dict) -> bool:
    """Evaluate a policy ``when`` clause against a workload's ``facts``.

    Supports ``>= <= > < ==`` numeric comparisons, ``a|b`` membership, and bare
    string/bool equality. Conditions whose key is absent from ``facts`` are skipped (not
    evaluable). Returns True iff at least one condition was evaluable and all evaluable
    conditions passed — so a policy never fires vacuously. Lets downstream consumers
    (Session 3, ``schedule`` dialect) test a rule against a concrete workload.
    """
    evaluated = 0
    for key, cond in when.items():
        if key not in facts:
            continue
        evaluated += 1
        fact = _coerce(facts[key])
        cond_s = str(cond).strip()
        ok = True
        for op in (">=", "<=", ">", "<", "=="):
            if cond_s.startswith(op):
                try:
                    rhs = _coerce(cond_s[len(op):].strip())
                    ok = {">=": fact >= rhs, "<=": fact <= rhs, ">": fact > rhs,
                          "<": fact < rhs, "==": fact == rhs}[op]
                except TypeError:
                    ok = False
                break
        else:
            if "|" in cond_s:
                ok = str(fact) in cond_s.split("|")
            else:
                ok = fact == _coerce(cond_s)
        if not ok:
            return False
    return evaluated > 0


@dataclass
class MotifStat:
    kernel_count: int = 0
    sources: set[str] = field(default_factory=set)
    targets: set[str] = field(default_factory=set)
    evidence_ids: set[str] = field(default_factory=set)
    example_paths: list[str] = field(default_factory=list)


def dedupe_records(records: list[dict]) -> tuple[list[dict], dict]:
    """Drop kernels whose text content was already counted under another source.

    Sibling corpora vendor files verbatim (triton-cpu ships the triton tutorials), which
    would otherwise inflate the cross-source signal. Keyed on ``meta.content_hash``; records
    without a hash (older indexes) are kept as-is. Returns ``(unique_records, diagnostic)``
    where the diagnostic names how many duplicates each source contributed.
    """
    seen: dict[str, str] = {}  # hash -> first source
    unique: list[dict] = []
    dup_by_source: dict[str, int] = {}
    for rec in records:
        h = (rec.get("meta") or {}).get("content_hash")
        if h:
            first = seen.get(h)
            if first is not None and first != rec.get("source"):
                src = rec.get("source", "?")
                dup_by_source[src] = dup_by_source.get(src, 0) + 1
                continue
            seen.setdefault(h, rec.get("source", "?"))
        unique.append(rec)
    diag = {"duplicates_skipped": sum(dup_by_source.values()), "by_source": dup_by_source}
    return unique, diag


def aggregate(records: list[dict]) -> dict[str, MotifStat]:
    """Count, per motif, how many kernels exhibit it and from which sources/targets."""
    stats: dict[str, MotifStat] = {}
    for rec in records:
        src = rec.get("source", "?")
        tgt = rec.get("target", "?")
        ev = rec.get("evidence", {}) or {}
        eid = ev.get("id", f"{src}_{tgt}_{rec.get('op','?')}")
        for motif in ev.get("motifs", []):
            st = stats.setdefault(motif, MotifStat())
            st.kernel_count += 1
            st.sources.add(src)
            st.targets.add(tgt)
            st.evidence_ids.add(eid)
            if len(st.example_paths) < 5:
                st.example_paths.append(rec.get("path", ""))
    return stats


def is_promotable(stat: MotifStat, min_kernels: int = 10) -> bool:
    """Promotion gate: >=2 sources OR >=min_kernels kernels."""
    return len(stat.sources) >= 2 or stat.kernel_count >= min_kernels


def _dispatch_observed(records: list[dict] | None) -> dict:
    """Corpus dispatch stats for the runtime candidate's ``observed`` block."""
    if not records:
        return {}
    counts = [r.get("features", {}).get("dispatch_metrics", {}).get("n_dispatches", 0)
              for r in records]
    counts = [c for c in counts if c >= 20]
    fracs = [r.get("features", {}).get("dispatch_metrics", {}).get("small_dispatch_fraction", 0)
             for r in records
             if r.get("features", {}).get("dispatch_metrics", {}).get("n_dispatches", 0) >= 20]
    if not counts:
        return {}
    return {"median_dispatches_per_kernel": int(statistics.median(counts)),
            "small_dispatch_fraction": round(statistics.mean(fracs), 3) if fracs else 0.0,
            "kernels_over_threshold": len(counts)}


@dataclass
class PromotionResult:
    candidates: list[dict] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)
    interfaces: list[dict] = field(default_factory=list)
    runtime_candidates: list[dict] = field(default_factory=list)
    dialect_requirements: list[dict] = field(default_factory=list)  # L6 (feeds TargetGen)
    llvm_requirements: list[dict] = field(default_factory=list)     # L8 (always: no fork yet)
    promoted: set[str] = field(default_factory=set)
    considered: dict[str, MotifStat] = field(default_factory=dict)


def promote(stats: dict[str, MotifStat], min_kernels: int = 10,
            records: list[dict] | None = None) -> PromotionResult:
    """Emit abstraction/interface/policy/runtime artifacts for promoted, cataloged motifs.

    ``records`` (optional) lets the runtime candidate carry observed dispatch statistics.
    """
    result = PromotionResult(considered=stats)
    for motif, entry in CATALOG.items():
        stat = stats.get(motif)
        if stat is None or not is_promotable(stat, min_kernels):
            continue
        result.promoted.add(motif)
        evidence = sorted(stat.evidence_ids)
        freq = {"kernels": stat.kernel_count, "sources": sorted(stat.sources)}
        spec = entry.get("abstraction")
        if spec:
            result.candidates.append(emit_abstraction_candidate(
                name=spec["name"], kind=spec["kind"], motivation=spec["motivation"],
                evidence=evidence, interface_features=spec["interface_features"],
                extra={"frequency": freq},
            ))
        iface = entry.get("interface")
        if iface:
            result.interfaces.append(emit_interface_candidate(
                name=iface["name"], interface_ops=iface["interface_ops"],
                interface_types=iface["interface_types"],
                justified_by={"motif": motif, "policies": [entry["policy"]["policy"]],
                              "frequency": freq},
                compiler_must_prove=iface.get("compiler_must_prove", ()),
                hardware_must_provide=iface.get("hardware_must_provide", ()),
                runtime_must_provide=iface.get("runtime_must_provide", ()),
            ))
        pol = entry["policy"]
        result.rules.append(emit_policy_rule(
            policy=pol["policy"], evidence=evidence, when=pol["when"], actions=pol["actions"],
            extra={"support": freq},
        ))
    # Runtime candidates (L7)
    for motif, spec in RUNTIME_CATALOG.items():
        stat = stats.get(motif)
        if stat is None or not is_promotable(stat, min_kernels):
            continue
        result.promoted.add(motif)
        result.runtime_candidates.append(emit_runtime_candidate(
            name=spec["name"], evidence=sorted(stat.evidence_ids),
            compiler_action=spec["compiler_action"],
            runtime_requirement=spec["runtime_requirement"],
            observed=_dispatch_observed(records),
        ))
    # de-dup candidates by name (two motifs may map to one abstraction/interface)
    for attr in ("candidates", "interfaces"):
        seen: dict[str, dict] = {}
        for c in getattr(result, attr):
            seen.setdefault(c["name"], c)
        setattr(result, attr, list(seen.values()))
    # L6/L8: each promoted interface yields one dialect requirement and one (always
    # fork-not-justified) LLVM requirement — completing the ladder without overclaiming.
    for iface in result.interfaces:
        result.dialect_requirements.append(emit_dialect_requirement(
            source_abstraction=iface["name"],
            required_ops=iface["interface_ops"],
            required_types=iface["interface_types"],
            extra={"justified_by": iface.get("justified_by", {})},
        ))
        result.llvm_requirements.append(emit_llvm_requirement(
            source_abstraction=iface["name"]))
    return result
