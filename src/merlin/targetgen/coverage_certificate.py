"""The per-compilation coverage certificate — the artifact that makes Acceleratable Region Recall
auditable.

For every region of a compiled workload it records BOTH sides of the ARR ratio:

- the **decision** the generated compiler actually made (``accelerator`` / ``in_contract`` /
  ``cpu_fallback``) — read from a :func:`merlin.targetgen.routing.route_plan` result (the *numerator*);
- the **eligibility** the hardware declares — from the independent oracle
  :func:`merlin.targetgen.eligibility.is_eligible` over the target's ``semantic_capabilities`` (the
  *denominator*).

The gap between the two is the compiler deficiency ARR is designed to surface: a region the hardware
*could* run (eligible) but the compiler left on the CPU lane (``false_fallback``), or — the integrity
direction — a region routed to the accelerator that the oracle says is *not* eligible
(``accelerated_ineligible``, an acceleration-precision miss).

This module consumes the routing output (numerator) and the eligibility oracle (denominator); it is NOT
the oracle, so it may legitimately reference both.

⚠️ BOTH SIDES OF THE RATIO ARE BUILT FROM THE ROUTING DEMANDS, so a contraction absent from those
demands is in neither. Earlier captures lost generic attention contractions this way; a generic op is
not, however, inherently unmatched. When the plan's contraction identities equal an independently
parsed inventory, ``denominator_completeness`` records that proof of inventory completeness and does
not charge the same generic work a second time. Without that check, the historical conservative
lower bound remains, and neither the upper recall nor its bound should be quoted alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from merlin.targetgen import eligibility as _el


def _decision_map(plan: dict) -> dict[int, str]:
    """Map each RouteResult (by identity) to the compiler's offload decision."""
    dec: dict[int, str] = {}
    for r in plan.get("mesh", []):
        dec[id(r)] = "accelerator"
    for r in plan.get("fallback", []):
        dec[id(r)] = "in_contract"
    for r in plan.get("scalar_rvv", []):
        dec[id(r)] = "cpu_fallback"
    return dec


def _flops(demand, family: str | None) -> int:
    """Best-effort work estimate. A contraction with known extents is ``2*M*K*N`` (MAC = mul+add);
    everything else is 0 until the region carries an element count (kept honest, never guessed)."""
    if family == "contraction" and demand.m and demand.k and demand.n:
        return 2 * int(demand.m) * int(demand.k) * int(demand.n)
    return 0


def _ratio(num: int, den: int):
    """num/den, or ``None`` when the denominator is empty (no eligible work to recall)."""
    return (num / den) if den else None


def _typed_source_witnesses(linalg_mlir: str | None, results: list) -> list[dict | None]:
    """Observe exact source value types, separately from routing-format admission.

    The pretty ``ins`` reader intentionally cannot assign one accelerator format to
    constants, shape operations, or mixed data/control operands.  Their parsed SSA
    types can nevertheless close the *source* precision census.  This witness
    says nothing about host lowering, numerical equivalence, or device support.
    An incomplete operation/order/type join yields no witness, never a guessed
    requested format.
    """
    absent = [None] * len(results)
    if not linalg_mlir or not results:
        return absent
    try:
        from xdsl.dialects import builtin

        from merlin.common import mlir_query as mq
        from merlin.common.ir_lock import IR_LOCK
        from merlin.targetgen.source_demand_inventory import indexed_demand_operations

        with IR_LOCK:
            module = mq.parse(linalg_mlir)
            funcs = [op for op in module.body.block.ops if mq.op_name(op) == "func.func"]
            if len(funcs) != 1 or len(funcs[0].body.blocks) != 1:
                return absent
            all_ops = [op for op in funcs[0].body.blocks[0].ops if mq.op_name(op) != "func.return"]
            ops = [op for _, op in indexed_demand_operations(all_ops)]
            if len(ops) != len(results):
                return absent

            def value_type(value):
                typ = value.type
                is_tensor = isinstance(typ, builtin.TensorType)
                element = typ.element_type if is_tensor else typ
                if not isinstance(element, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)):
                    return None
                if is_tensor:
                    shape = typ.get_shape()
                    if shape is None or any(type(dim) is not int or dim < 0 for dim in shape):
                        return None
                return str(typ), str(element), is_tensor

            witnesses: list[dict | None] = []
            for op, result in zip(ops, results, strict=True):
                demand = result.demand
                name = mq.op_name(op)
                tag = mq.attr_str(op, "prov.op")
                region = mq.attr_str(op, "prov.region_id")
                if (
                    demand.carrier_op != name
                    or demand.region_id != region
                    or tag is None
                    or (demand.op != tag if name.startswith("linalg.") else demand.op != name)
                ):
                    return absent
                operands = [value_type(value) for value in op.operands]
                outputs = [value_type(value) for value in op.results]
                if not outputs or any(row is None for row in (*operands, *outputs)):
                    witnesses.append(None)
                    continue
                typed_operands = [row for row in operands if row is not None]
                typed_outputs = [row for row in outputs if row is not None]
                witnesses.append(
                    {
                        "status": "typed_source"
                        if any(row[2] for row in (*typed_operands, *typed_outputs))
                        else "not_applicable",
                        "operand_types": [row[0] for row in typed_operands],
                        "result_types": [row[0] for row in typed_outputs],
                        "operand_element_types": [row[1] for row in typed_operands],
                        "result_element_types": [row[1] for row in typed_outputs],
                        "operand_is_tensor": [row[2] for row in typed_operands],
                        "result_is_tensor": [row[2] for row in typed_outputs],
                        "scope": "parsed source types only; no host/device support or emitted-code equivalence",
                    }
                )
            return witnesses
    except Exception:  # noqa: BLE001 -- a source type parse failure remains unknown
        return absent


def denominator_completeness(linalg_mlir: str | None, *, demands: list | None = None) -> dict | None:
    """Price the contraction work the ROUTING DEMANDS could not see, from the module itself.

    ``None`` when there is no module to read. On a module that will not parse this returns an ``error``
    entry rather than nothing: a certificate that silently omits the completeness block is
    indistinguishable from one whose denominator is provably complete, which is the direction that
    flatters us.
    """
    if not linalg_mlir:
        return None
    try:
        from merlin.common import mlir_query as _mq
        from merlin.common.ir_lock import IR_LOCK
        from merlin.xdsl_dialects.lowering.contraction_coverage import contraction_coverage

        with IR_LOCK:  # xDSL's parser is not thread-safe; see common.ir_lock
            rep = contraction_coverage(_mq.parse(linalg_mlir))
    except Exception as exc:  # noqa: BLE001 — advisory, but the gap must stay visible
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "note": "denominator completeness UNKNOWN — the module could not be priced, so the "
            "recall below is an upper bound with no stated floor",
        }

    # A generic contraction is not intrinsically unmatched: the demand reader
    # now recognizes structurally shaped generics.  Only an exact, independently
    # checked correspondence with the actual route plan may discharge the old
    # pessimistic bound.  A missing tag or an incomplete plan keeps that bound.
    inventory_verified = False
    incomplete_formats: list[str] = []
    if demands is not None:
        from collections import Counter

        from merlin.targetgen.capsule_source import ModelDemandIncomplete, model_op_demands_checked

        try:
            fmt = demands[0].in_fmt if demands else "unknown"
            checked = model_op_demands_checked(linalg_mlir, fmt)
            def identity(d):
                return d.op, d.batch, d.m, d.k, d.n, d.captured_input_formats
            inventory_verified = Counter(identity(d) for d in checked if d.family == "contraction") == Counter(
                identity(d) for d in demands if d.family == "contraction"
            )
            incomplete_formats = [
                d.site or d.op for d in checked if d.family == "contraction" and not d.source_formats_complete
            ]
        except ModelDemandIncomplete:
            pass

    caveats: list[str] = []
    if inventory_verified:
        return {
            "matched_contraction_macs": rep.total_macs,
            "unmatched_contraction_macs": 0,
            "unmatched_contraction_share": 0.0,
            "n_unmatched_contractions": 0,
            "n_unpriceable_contractions": len(rep.unpriceable),
            "unpriceable_result_types": list(rep.unpriceable),
            "unmatched": [],
            "generic_labels": dict(rep.labels),
            "inventory_status": (
                "verified_structure_operand_formats_incomplete"
                if incomplete_formats
                else "verified_against_parsed_contractions"
            ),
            "unverified_operand_format_contractions": incomplete_formats[:8],
            "caveats": (
                [f"{len(rep.unpriceable)} contraction(s) could not be priced from loop extents"]
                if rep.unpriceable
                else []
            )
            + (
                [
                    f"{len(incomplete_formats)} contraction(s) have incomplete captured operand formats; "
                    "they are excluded from the eligible denominator"
                ]
                if incomplete_formats
                else []
            ),
        }
    if rep.unlowered:
        caveats.append(
            f"{len(rep.unlowered)} generic contraction(s) worth {rep.unlowered_macs} MAC "
            f"({rep.unlowered_share:.1%} of all contraction MACs) cannot be certified as matched "
            "to this route plan; the lower bound charges them as unmatched"
        )
    if rep.unpriceable:
        caveats.append(
            f"{len(rep.unpriceable)} contraction(s) could not be priced (no derivable loop extents), so "
            f"even the lower bound below is optimistic — they are counted as ops, never as work"
        )
    return {
        "matched_contraction_macs": rep.lowered_macs,
        "unmatched_contraction_macs": rep.unlowered_macs,
        "unmatched_contraction_share": rep.unlowered_share,
        "n_unmatched_contractions": len(rep.unlowered),
        "n_unpriceable_contractions": len(rep.unpriceable),
        "unpriceable_result_types": list(rep.unpriceable),
        "unmatched": [
            {"result_type": u.result_type, "loop_extents": {str(d): e for d, e in u.loop_extents}, "macs": u.macs}
            for u in rep.unlowered
        ],
        "generic_labels": dict(rep.labels),
        "inventory_status": "not_verified_against_route_plan",
        "caveats": caveats,
    }


def executed_false_fallbacks(execution: dict | None) -> dict:
    """Which kernels the router ASSIGNED to the accelerator did not execute there.

    The execution-evidenced counterpart of ``false_fallback_count``, and it is computable for a reason
    worth stating: both halves are keyed by the SAME thing, the kernel symbol. ``mesh_route_symbols``
    is what the router assigned; the dispatch ledger records, per completed call, which lane it went to.
    No join to the plan's per-op demands is required -- that join does not exist, and this does not need
    it, so the measurable half of the question is answerable without inventing the unmeasurable half.

    ``status`` is ``not_measured`` when either side is absent, never an empty list of fallbacks: "no
    symbols fell back" and "nobody recorded which symbols ran" are opposite conclusions.
    """
    ex = execution or {}
    routed = ex.get("mesh_route_symbols")
    ledger = ex.get("dispatch_ledger")
    if not isinstance(routed, (list, tuple)) or not isinstance(ledger, list):
        return {
            "status": "not_measured",
            "detail": "the run recorded no route symbols or no dispatch ledger, so which assigned "
            "kernel executed where was never observed",
        }
    on_accel = {
        str(e.get("symbol"))
        for e in ledger
        if isinstance(e, dict) and e.get("status") == "pass" and e.get("lane") == "on_mesh"
    }
    assigned = [str(x) for x in routed]
    fell_back = sorted(sym for sym in assigned if sym not in on_accel)
    return {
        "status": "measured",
        "n_routed": len(assigned),
        "n_executed_on_accelerator": len([s for s in assigned if s in on_accel]),
        "n_false_fallback": len(fell_back),
        # Named, not just counted: "4 kernels fell back" gives a reader nothing to act on.
        "false_fallback_symbols": fell_back[:64],
        "detail": "kernels the router assigned to the accelerator that no completed call placed "
        "there; joined on the kernel symbol, which both records carry",
    }


def source_region_execution(regions: list[dict], execution: dict | None) -> dict:
    """Join eligible source regions to completed runtime calls by provenance identity.

    A plan and a module census are both predictions. An outlined kernel's
    ``prov.region_id`` is encoded in its symbol by the core outliner, and the
    runtime records that exact symbol only after the call completes. The
    static outline inventory must also account for every captured linalg
    operation in an eligible provenance region; one completed operation cannot
    certify its unexecuted siblings. Unknown, failed, mixed, or missing calls
    cannot be credited as accelerator execution. This is operation accounting
    and a lane observation, not a proof of numerical equivalence.
    """
    from collections import Counter, defaultdict

    from merlin.xdsl_dialects.lowering.outline import region_id_of_symbol

    wanted = {r["region_id"] for r in regions if r["target_eligible"] and r["region_id"]}
    unattributed = sum(1 for r in regions if r["target_eligible"] and not r["region_id"])
    ledger = (execution or {}).get("dispatch_ledger")
    if not isinstance(ledger, list):
        return {
            "status": "not_measured",
            "n_eligible_source_regions": len(wanted),
            "unattributed_demands": unattributed,
        }

    # A region id is NOT an operation id: one frontend operation can yield
    # several linalg roots with the same provenance. Match the complete source
    # operation multiset to the exact runtime outline before crediting a lane.
    # Non-linalg eligible work is not outlined by this runner and therefore
    # remains an unverified obligation, not a free accelerator success.
    source_keys = Counter(
        (r["region_id"], r.get("carrier_op"), r["op"])
        for r in regions
        if r["region_id"] in wanted and str(r.get("carrier_op") or "").startswith("linalg.")
    )
    unoutlined_eligible = sum(
        1
        for r in regions
        if r["target_eligible"]
        and r["region_id"] in wanted
        and not str(r.get("carrier_op") or "").startswith("linalg.")
    )
    manifest = (execution or {}).get("outlined_dispatches")

    def content_hash(value: object) -> str:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(canonical).hexdigest()

    direct_keys: Counter[tuple[str, str, str]] = Counter()
    contraction_keys: Counter[tuple[str, str, str]] = Counter()
    requant_keys: Counter[tuple[str, str, str]] = Counter()
    outlined_symbols: set[str] = set()
    primary_symbols: dict[str, set[str]] = defaultdict(set)
    auxiliary_symbols: set[str] = set()
    invalid_outlined_rows = 0
    source_families: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    source_eligibility: dict[tuple[str, str, str], set[bool]] = defaultdict(set)
    for row in regions:
        key = (row["region_id"], row.get("carrier_op"), row["op"])
        if key in source_keys:
            source_families[key].add(row["semantic_family"])
            source_eligibility[key].add(bool(row["target_eligible"]))
    if isinstance(manifest, list):
        for row in manifest:
            if not isinstance(row, dict):
                invalid_outlined_rows += 1
                continue
            sym, rid, root, op = (row.get(key) for key in ("symbol", "region_id", "root_op", "prov_op"))
            if not isinstance(sym, str) or not sym or sym in outlined_symbols:
                invalid_outlined_rows += 1
                continue
            outlined_symbols.add(sym)
            if rid not in wanted:
                continue
            if (
                not isinstance(rid, str)
                or region_id_of_symbol(sym) != rid
                or not isinstance(root, str)
                or not isinstance(op, str)
            ):
                invalid_outlined_rows += 1
                continue
            key = (rid, root, op)
            role = row.get("prov_role")
            if role is None:
                direct_keys[key] += 1
                primary_symbols[rid].add(sym)
            elif role == "contraction":
                contraction_keys[key] += 1
                primary_symbols[rid].add(sym)
            elif role == "requant":
                requant_keys[key] += 1
                auxiliary_symbols.add(sym)
            else:
                invalid_outlined_rows += 1

    def named_counts(counts: Counter) -> dict[str, int]:
        return {"|".join(str(part) for part in key): count for key, count in sorted(counts.items())}

    primary_keys = direct_keys + contraction_keys
    unmatched = source_keys - primary_keys
    unexpected = primary_keys - source_keys
    # The integer quantization rewrite expands one captured contraction into
    # a contraction and its requantization. Both children are mandatory, but
    # only the contraction is an accelerator-placement obligation. An unknown
    # or partial expansion cannot make the source operation appear covered.
    invalid_split_keys = sorted(
        key
        for key in set(contraction_keys) | set(requant_keys)
        if contraction_keys[key] != requant_keys[key] or source_families.get(key) != {"contraction"}
    )
    ambiguous_source_keys = sorted(key for key, values in source_eligibility.items() if len(values) != 1)
    completed_symbols = {
        entry["symbol"]
        for entry in ledger
        if isinstance(entry, dict) and isinstance(entry.get("symbol"), str) and entry.get("status") == "pass"
    }
    expected_symbols = (
        {
            row["symbol"]
            for row in manifest
            if isinstance(row, dict) and row.get("region_id") in wanted and isinstance(row.get("symbol"), str)
        }
        if isinstance(manifest, list)
        else set()
    )
    unexecuted = sorted(expected_symbols - completed_symbols)

    host_lanes = frozenset(("native_cpu", "xnnpack_host", "scalar_rvv_lane", "host_fallback"))
    observed: dict[str, list[str]] = defaultdict(list)
    auxiliary_host_symbols: set[str] = set()
    for entry in ledger:
        if not isinstance(entry, dict) or not isinstance(entry.get("symbol"), str):
            continue
        sym = entry["symbol"]
        rid = region_id_of_symbol(sym)
        if rid not in wanted or sym not in outlined_symbols:
            continue
        lane = entry.get("lane") if entry.get("status") == "pass" else None
        if sym in auxiliary_symbols:
            if lane in host_lanes:
                auxiliary_host_symbols.add(sym)
            continue
        if sym in primary_symbols[rid]:
            observed[rid].append("accelerator" if lane == "on_mesh" else "host" if lane in host_lanes else "unverified")

    unobserved = sorted(wanted - observed.keys())
    host = sorted(rid for rid, lanes in observed.items() if lanes and all(lane == "host" for lane in lanes))
    accelerator = sorted(
        rid for rid, lanes in observed.items() if lanes and all(lane == "accelerator" for lane in lanes)
    )
    mixed = sorted(rid for rid, lanes in observed.items() if "accelerator" in lanes and "host" in lanes)
    unresolved = sorted(wanted - set(host) - set(accelerator) - set(mixed) - set(unobserved))
    outline_complete = isinstance(manifest, list) and not (
        invalid_outlined_rows
        or unmatched
        or unexpected
        or invalid_split_keys
        or ambiguous_source_keys
        or unoutlined_eligible
        or unexecuted
    )
    return {
        "status": "measured" if outline_complete and not (unattributed or unobserved or unresolved) else "incomplete",
        "n_eligible_source_regions": len(wanted),
        "n_source_operations_in_eligible_regions": sum(source_keys.values()),
        "n_eligible_executed_on_accelerator": len(accelerator),
        "eligible_accelerator_region_ids": accelerator,
        "eligible_host_region_ids": host,
        "eligible_mixed_region_ids": mixed,
        "unobserved_region_ids": unobserved,
        "unresolved_region_ids": unresolved,
        "unattributed_demands": unattributed,
        "unoutlined_eligible_demands": unoutlined_eligible,
        "unmatched_source_operation_counts": named_counts(unmatched),
        "unexpected_outlined_operation_counts": named_counts(unexpected),
        "invalid_split_source_keys": ["|".join(key) for key in invalid_split_keys],
        "ambiguous_source_keys": ["|".join(key) for key in ambiguous_source_keys],
        "auxiliary_host_symbols": sorted(auxiliary_host_symbols),
        "unexecuted_outlined_symbols": unexecuted,
        "invalid_outlined_rows": invalid_outlined_rows,
        "outline_inventory_status": "matched" if outline_complete else "incomplete",
        "outlined_dispatches_sha256": content_hash(manifest) if isinstance(manifest, list) else None,
        "dispatch_ledger_sha256": content_hash(ledger),
        "evidence": "source_operation_inventory_to_runtime_outline_to_completed_dispatch_ledger",
        "scope": ("static operation accounting and lane execution only; "
                  "no arithmetic or transformation equivalence proof"),
    }


def build(
    plan: dict,
    cap_map: dict,
    *,
    target: str | None = None,
    linalg_mlir: str | None = None,
    execution: dict | None = None,
) -> dict:
    """Build the coverage certificate from a ``route_plan`` result and a capability map.

    ``plan`` is the dict returned by :func:`merlin.targetgen.routing.route_plan_on` / ``route_plan``
    (needs ``results`` + the ``mesh``/``fallback``/``scalar_rvv`` buckets). ``cap_map`` is the
    independent denominator from :func:`merlin.targetgen.eligibility.capability_map_for_target`.

    ``execution`` is the run's ``mesh_execution`` record, when one exists. EVERY NUMBER BELOW IS
    DERIVED FROM THE PLAN, which lists what the router ASSIGNED and not what ran -- exactly the
    conflation already fixed on the lane side, where one submission assigned 15 matmuls to the mesh and
    fell back on all 15 at run time. The certificate therefore states its own evidence in ``arr_evidence``
    and, when execution accounting exists, carries it beside the plan numbers in ``execution_crosscheck``
    with ``agrees`` set. A disagreement means the recalls above describe intent that did not happen.

    ``linalg_mlir`` is the module those demands were derived FROM. Pass it and the certificate also
    reports what the demands missed; omit it and the completeness block is absent, which is itself
    reported (``denominator_completeness: null``) rather than read as "nothing was missed".
    """
    dec = _decision_map(plan)
    regions: list[dict] = []
    n_eligible = n_accelerated = n_eligible_accelerated = 0
    eligible_flops = accelerated_eligible_flops = 0
    accelerated_ineligible = 0

    results = list(plan.get("results", []))
    typed_witnesses = _typed_source_witnesses(linalg_mlir, results)
    for r, source_witness in zip(results, typed_witnesses, strict=True):
        d = r.demand
        observed = d.captured_input_formats
        captured_input = observed[0] if observed else d.elem_fmt
        captured_weight = observed[1] if observed is not None and len(observed) > 1 else None
        # A region tag can name an enclosing convolution while this carrier is
        # a nonreducing gather. A deliberately absent operation family must
        # not be recovered from that discarded enclosing-region name.
        semantic_op = d.carrier_op if d.family is None and d.carrier_op is not None else d.op
        result_elements = source_witness["result_element_types"] if source_witness is not None else []
        captured_output = result_elements[0] if result_elements and len(set(result_elements)) == 1 else None
        desc = _el.RegionDescriptor(
            source=d.site or d.op,
            op=semantic_op,
            family=d.family,
            in_dtype=d.admission_input_fmt,
            weight_dtype=d.admission_weight_fmt,
            m=d.m,
            k=d.k,
            n=d.n,
            rank=d.rank,
            batch=d.batch or 1,
            form=d.form,
            out_dtype=captured_output,
        )
        verdict = (
            _el.is_eligible(desc, cap_map)
            if d.source_formats_complete
            else _el.EligibilityVerdict(
                False,
                desc.resolved_family(),
                "captured operand formats incomplete; eligibility unverified",
                undetermined=True,
                refusal="input_dtype",
            )
        )
        family = verdict.family
        decision = dec.get(id(r), "cpu_fallback")
        accelerated = decision == "accelerator"
        flops = _flops(d, family)

        precision_transform_required = (
            None
            if (observed is not None and not d.source_formats_complete)
            or (observed is None and d.elem_fmt is None and str(d.carrier_op or "").startswith("linalg."))
            else bool(captured_input is not None and not _el._dtype_ok(captured_input, (d.admission_input_fmt,)))
        )
        # Typed source-only CPU operations do not request a hidden conversion
        # into an accelerator format. A missing operand format may not, however,
        # hide a potentially eligible contraction or other hardware operation.
        # Recheck the selected capability map without claiming source admission;
        # only a definitive hardware refusal permits this source census.
        possible_hardware = (
            _el.is_eligible(replace(desc, captured_input_formats=d.captured_input_formats), cap_map)
            if source_witness is not None
            else None
        )
        hardware_dtype_refusal = False
        if possible_hardware is not None and possible_hardware.undetermined:
            tensor_elements = [
                element
                for element, is_tensor in zip(
                    source_witness["operand_element_types"], source_witness["operand_is_tensor"], strict=True
                )
                if is_tensor
            ]
            # A mixed index/data operation has no single routing format. Ask
            # the existing selected-capability oracle about *each* actual
            # tensor operand as a possible data input, without guessing roles.
            # Only a definite input-dtype refusal for *every* assignment is
            # enough. A form/shape refusal with missing source roles may itself
            # depend on absent metadata. Contractions with one compatible
            # operand cannot be excused by an incompatible weight or index.
            hardware_dtype_refusal = bool(
                tensor_elements
                and all(
                    not candidate.eligible and not candidate.undetermined and candidate.refusal == "input_dtype"
                    for candidate in (
                        _el.is_eligible(
                            replace(desc, in_dtype=element, captured_input_formats=None), cap_map
                        )
                        for element in tensor_elements
                    )
                )
            )
        source_witness_applied = (
            precision_transform_required is None
            and source_witness is not None
            and possible_hardware is not None
            and not possible_hardware.eligible
            and (not possible_hardware.undetermined or hardware_dtype_refusal)
            and decision == "cpu_fallback"
        )
        if source_witness_applied:
            precision_transform_required = False
        row = {
            "source": d.site or d.op,
            "op": d.op,
            "region_id": d.region_id,
            "source_family": d.source_family,
            "carrier_op": d.carrier_op,
            "form": d.form,
            "semantic_family": family,
            "requested_input_format": d.in_fmt,
            "captured_input_format": captured_input,
            "captured_weight_format": captured_weight,
            "captured_result_format": captured_output,
            "eligibility_input_format": d.admission_input_fmt,
            "eligibility_weight_format": d.admission_weight_fmt,
            "requested_format_mismatch": bool(
                (captured_input is not None and not _el._dtype_ok(captured_input, (d.in_fmt,)))
                or (
                    captured_weight is not None
                    and d.weight_fmt is not None
                    and not _el._dtype_ok(captured_weight, (d.weight_fmt,))
                )
            ),
            "precision_transform_required": precision_transform_required,
            "target_eligible": verdict.eligible,
            "eligibility_reason": verdict.reason,
            "eligibility_undetermined": verdict.undetermined,
            "decision": decision,
            "unit": r.unit,
            "gap": r.gap,
            "estimated_work_flops": flops,
        }
        if source_witness_applied:
            row["source_precision_witness"] = source_witness
        regions.append(row)

        if verdict.eligible:
            n_eligible += 1
            eligible_flops += flops
            if accelerated:
                n_eligible_accelerated += 1
                accelerated_eligible_flops += flops
        if accelerated:
            n_accelerated += 1
            if not verdict.eligible:
                accelerated_ineligible += 1

    false_fallback = sum(1 for reg in regions if reg["target_eligible"] and reg["decision"] != "accelerator")

    # Work the matcher never turned into a demand. A MAC is a multiply AND an add, so it is 2 flops on
    # the same scale `_flops` uses -- mixing the two units would understate the correction by half.
    completeness = denominator_completeness(linalg_mlir, demands=[r.demand for r in plan.get("results", [])])
    unmatched_flops = 2 * int(completeness.get("unmatched_contraction_macs") or 0) if completeness else 0
    unmatched_regions = int(completeness.get("n_unmatched_contractions") or 0) if completeness else 0

    # THE EVIDENCE, said out loud. A recall that reads as a measurement of the compiler when it is a
    # summary of the router is the same defect `lane_report` was hardened against, and this surface had
    # never been told about it.
    xcheck = None
    if execution:

        def _n(v):
            return None if v is None or isinstance(v, str) else int(v)

        routed, ran = _n(execution.get("matmul_layers_routed")), _n(execution.get("matmul_layers_on_mesh"))
        fell = _n(execution.get("matmul_layers_host_fallback"))
        xcheck = {
            "matmul_layers_routed": routed,
            "matmul_layers_on_mesh": ran,
            "matmul_layers_host_fallback": fell,
            # None, never True: "nobody could tell" is not "they agree".
            "agrees": None if (routed is None or ran is None) else (routed == ran),
            "why": (
                "the plan assigned `routed` contraction layers to the accelerator and `on_mesh` "
                "of them executed there; when these differ, the recalls in this certificate "
                "describe an intent the run did not carry out"
            ),
        }

    # The execution-evidenced half of the same question, reported BESIDE the plan-derived recalls rather
    # than replacing them: the recalls are per-region and this is per-kernel-symbol, so they are not the
    # same denominator and collapsing them would invent a number neither record supports.
    executed = executed_false_fallbacks(execution)
    source_execution = source_region_execution(regions, execution)

    return {
        "target": target,
        "source_mlir_sha256": (
            hashlib.sha256(linalg_mlir.encode("utf-8")).hexdigest() if linalg_mlir is not None else None
        ),
        "denominator_source": "semantic_capabilities (independent eligibility oracle)",
        "arr_evidence": "routing_plan",
        "execution_crosscheck": xcheck,
        "executed_false_fallbacks": executed,
        "source_region_execution": source_execution,
        "n_precision_transform_obligations": sum(r["precision_transform_required"] is True for r in regions),
        "n_requested_format_mismatches": sum(r["requested_format_mismatch"] for r in regions),
        "n_unknown_capture_formats": sum(r["precision_transform_required"] is None for r in regions),
        "precision_transform_verification": {
            "status": (
                "not_verified"
                if any(r["precision_transform_required"] is True for r in regions)
                else "unknown_capture_format"
                if any(r["precision_transform_required"] is None for r in regions)
                else "not_required"
            ),
            "scope": ("capture operand format versus effective routing format; "
                      "no conversion proof is supplied by placement"),
        },
        "n_regions": len(regions),
        "n_eligible": n_eligible,
        "n_accelerated": n_accelerated,
        "n_eligible_accelerated": n_eligible_accelerated,
        "false_fallback_count": false_fallback,
        "accelerated_ineligible_count": accelerated_ineligible,
        "eligible_flops": eligible_flops,
        "accelerated_eligible_flops": accelerated_eligible_flops,
        "unmatched_contraction_flops": unmatched_flops,
        "unmatched_contraction_regions": unmatched_regions,
        "denominator_completeness": completeness,
        "metrics": {
            # the headline ARR numbers for this single compilation (None == no eligible work).
            # The two recalls are computed over the regions the matcher PRODUCED, which is why each
            # carries a lower bound beside it -- see the module docstring. Quote a recall WITH its
            # bound or not at all: alone, the upper one reads as a measurement when it is a ceiling.
            "acceleratable_region_recall": _ratio(n_eligible_accelerated, n_eligible),
            "acceleratable_flop_recall": _ratio(accelerated_eligible_flops, eligible_flops),
            # REGION recall's floor, by the same construction as the flop floor below. This is the one
            # people actually quote -- it is the plain "how many of the regions it could accelerate did
            # it" number -- and it was the only headline metric here with no floor beside it, so the
            # single most-cited figure was also the single least-bracketed one. Charging every unmatched
            # contraction to the denominator is deliberately the unflattering assumption: each is counted
            # as eligible and unaccelerated.
            "acceleratable_region_recall_lower_bound": _ratio(n_eligible_accelerated, n_eligible + unmatched_regions),
            # The same recall with every unmatched contraction charged to the denominator: the floor
            # under the number above, on the assumption (deliberately the unflattering one) that all of
            # that work was eligible and none of it was accelerated. True recall lies between the two;
            # they coincide exactly when the matcher missed nothing.
            "acceleratable_flop_recall_lower_bound": _ratio(
                accelerated_eligible_flops, eligible_flops + unmatched_flops
            ),
            "acceleration_precision": _ratio(n_accelerated - accelerated_ineligible, n_accelerated),
        },
        "regions": regions,
    }


def for_target(plan: dict, target: str, *, linalg_mlir: str | None = None, execution: dict | None = None) -> dict:
    """Convenience: load the target's declared capability map and build the certificate."""
    cap_map = _el.capability_map_for_target(target)
    return build(plan, cap_map, target=target, linalg_mlir=linalg_mlir, execution=execution)
