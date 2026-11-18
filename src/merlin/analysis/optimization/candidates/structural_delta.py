"""Compare emitted whole-model work without turning partial counters into cycle verdicts.

Static instruction counts, dynamic host IR work, and declared transfer payload are
separate dimensions. Their units must never be added or silently cancelled. This
brief directs the next compiler question; it does not select a timing winner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

# ---- structural_delta --------------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value >= 0 and math.isfinite(value) else None


def _metrics(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    diag = _mapping(analysis.get("diagnostics"))
    plan = _mapping(diag.get("verified_global_plan_emission"))
    host = _mapping(plan.get("host_activity"))
    host_known = host.get("status") == "derived"
    dynamic = _mapping(host.get("dynamic_operations"))
    artifact = _mapping(_mapping(diag.get("target_artifact_activity")).get("candidate"))
    issued = _mapping(artifact.get("issued")) if artifact.get("status") == "decoded" else {}
    arm = _mapping(_mapping(diag.get("arms")).get("candidate"))
    movement = _mapping(arm.get("movement"))
    placement = _mapping(_mapping(diag.get("model_contraction_placement")).get("candidate"))
    machine = _mapping(_mapping(diag.get("machine_artifact_activity")).get("candidate"))
    emitted_digest = _mapping(analysis.get("emission")).get("candidate_lowered_sha256")
    machine_known = (
        machine.get("status") == "compiled" and bool(emitted_digest) and machine.get("source_sha256") == emitted_digest
    )
    sites = _mapping(machine.get("instruction_sites")) if machine_known else {}
    result: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any, scope: str, axis: str) -> None:
        result[name] = {"value": _count(value), "scope": scope, "cca_axis": axis}

    add(
        "full_model_contraction_macs",
        placement.get("total_contraction_macs"),
        "captured contraction arithmetic, host and device",
        "compute.contraction_form",
    )
    add(
        "dispatches",
        plan.get("emitted_dispatches", plan.get("tasks")),
        "verified emitted tasks, not runtime cycles",
        "envelope.calls_in_loop",
    )
    add(
        "command_buffer_declared_movement_bytes",
        movement.get("exact_bytes") if movement.get("is_lower_bound") is False else None,
        "command-buffer declared payload, not all issued movement or DRAM",
        "communication.intermediate_materialized",
    )
    for field, axis in (
        ("compute_instructions", "compute.contraction_form"),
        ("configuration_instructions", "dispatch.descriptor_reuse"),
        ("loop_descriptor_instructions", "dispatch.loop_offloaded"),
        ("movement_instructions", "communication.resident_across_calls"),
        ("synchronization_instructions", "communication.fences"),
    ):
        add(
            "issued_" + field,
            issued.get(field),
            "classified emitted instruction sites; excludes unclassified sites and descriptor-expanded work",
            axis,
        )
    for field in ("load_payload_bytes", "store_payload_bytes", "static_allocation_payload_bytes"):
        add(
            "host_" + field,
            host.get(field) if host_known else None,
            "pre-optimization LLVM payload, not DRAM or machine instructions",
            "communication.intermediate_materialized",
        )
    for field, axis in (
        ("integer_arithmetic", "compute.contraction_form"),
        ("floating_arithmetic", "compute.activation_vectorization"),
        ("conversion", "layout.transpose_materialized"),
    ):
        add(
            "host_dynamic_" + field,
            dynamic.get(field, 0) if host_known else None,
            "pre-optimization LLVM operation executions, not CPU cycles",
            axis,
        )
    add(
        "machine_object_bytes",
        machine.get("object_bytes") if machine_known else None,
        "compiled relocatable object size, including metadata; not runtime memory",
        "compute.contraction_form",
    )
    for field in ("total", "vector", "vsetvl", "scalar_int", "scalar_float", "undecoded"):
        value = sites.get(field)
        if field == "total" and sites.get("schema") != "encoded_instruction_sites_v1":
            value = None  # Older totals exclude undecoded sites and are not commensurate.
        add(
            "machine_instruction_sites_" + field,
            value,
            "post-LLVM static instruction sites, not dynamic executions or cycles",
            "compute.contraction_form",
        )
    return result


def _bound_machine_object(analysis: Mapping[str, Any], *, arm: str = "candidate") -> str | None:
    machine = _mapping(_mapping(_mapping(analysis.get("diagnostics")).get("machine_artifact_activity")).get(arm))
    source = _mapping(analysis.get("emission")).get(arm + "_lowered_sha256")
    digest = machine.get("object_sha256")
    if (
        machine.get("status") != "compiled"
        or not source
        or machine.get("source_sha256") != source
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        return None
    return digest


def compare_machine_objects(
    before: Mapping[str, Any], after: Mapping[str, Any], *, before_arm: str = "candidate", after_arm: str = "candidate"
) -> dict[str, Any]:
    """Compare source-bound object bytes, not graph equivalence or execution cost.

    Arm selection is explicit: an unchanged last edit must not be confused with
    an unchanged optimization baseline. Missing arms never fall back to another.
    """
    left = _bound_machine_object(before, arm=before_arm)
    right = _bound_machine_object(after, arm=after_arm)
    return {
        "status": ("UNKNOWN" if left is None or right is None else "identical" if left == right else "different"),
        "before_sha256": left,
        "after_sha256": right,
        "scope": "source-bound relocatable kernel object bytes only; not the linked full program",
    }


def _instruction_accounting(analysis: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(
        _mapping(_mapping(analysis.get("diagnostics")).get("target_artifact_activity")).get("candidate")
    )
    decoded = artifact.get("status") == "decoded"
    issued = _mapping(artifact.get("issued")) if decoded else {}
    resolution = _mapping(artifact.get("encoding_resolution")) if decoded else {}
    unknown = resolution.get("unknown_instruction_indices")
    unknown_count = (
        len(set(unknown))
        if isinstance(unknown, list) and all(type(index) is int and index >= 0 for index in unknown)
        else None
    )
    return {
        "loop_descriptor_sites": _count(issued.get("loop_descriptor_instructions")),
        "classification_status": resolution.get("status", "UNKNOWN"),
        "unclassified_instruction_sites": unknown_count,
        "named_without_roles": resolution.get("named_without_role"),
        "dynamic_compute_work": "UNKNOWN",
        "physical_moved_bytes": None,
        "descriptor_expanded_work": "UNKNOWN",
        "scope": "static classified issue sites, not dynamic accelerator work or physical traffic",
    }


def compare_full_model_structure(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Expose changed work and confounders for two verified emissions of one graph.

    Missing metrics are unresolved even when both sides lack them. A structural
    decrease can be worth pursuing without licensing a global performance claim.
    Numerical qualification is a separate requirement, including inherited waivers.
    """
    problems: list[str] = []
    graphs = []
    for label, analysis in (("before", before), ("after", after)):
        diag = _mapping(analysis.get("diagnostics"))
        graph = _mapping(diag.get("captured_logical_graph"))
        digest = graph.get("logical_dispatch_digest")
        if graph.get("status") != "verified" or not isinstance(digest, str) or not digest:
            problems.append(label + " complete graph is unverified")
        graphs.append(digest)
        if _mapping(diag.get("verified_global_plan_emission")).get("status") != "verified":
            problems.append(label + " plan emission is unverified")
    if graphs[0] != graphs[1]:
        problems.append("logical model graphs differ")
    result: dict[str, Any] = {
        "schema": "full_model_structural_delta_v1",
        "status": "refused" if problems else "compared",
        "problems": problems,
        "graph_digest": graphs[1],
        "metrics": {},
        "decreased": [],
        "increased": [],
        "unknown": [],
        "changed_cca_axes": [],
        "structural_relation": "unresolved",
        "cycle_selection": "UNMEASURED",
        "licence": "structural work accounting only; not numerical equivalence or a timing winner",
        "required_next_evidence": [],
        "machine_object_comparison": {"status": "UNKNOWN"},
    }
    if problems:
        return result
    accounting = {arm: _instruction_accounting(analysis) for arm, analysis in (("before", before), ("after", after))}
    result["instruction_accounting"] = accounting
    if any(
        row["loop_descriptor_sites"] is not None and row["loop_descriptor_sites"] > 0 for row in accounting.values()
    ):
        result["required_next_evidence"].append(
            "loop descriptors can issue compute and movement internally: fewer standalone sites are "
            "not work deletion; verify the selected task's emitted route, descriptor semantics and "
            "matched short context before attributing a performance gain"
        )
    if any(row["classification_status"] != "complete" for row in accounting.values()):
        result["required_next_evidence"].append(
            "instruction classification is incomplete or unknown: zero classified compute/movement "
            "sites is not proof that the accelerator performs no such work"
        )
    result["machine_object_comparison"] = compare_machine_objects(before, after)
    identical_object = result["machine_object_comparison"]["status"] == "identical"
    if identical_object:
        result["required_next_evidence"].insert(
            0,
            "kernel object is byte-identical: pre-LLVM count reductions do not demonstrate "
            "emitted-kernel work deletion; inspect any separate ABI/runtime changes before a timing claim",
        )
    left, right = _metrics(before), _metrics(after)
    for name, row in right.items():
        a, b = left[name]["value"], row["value"]
        delta = None if a is None or b is None else b - a
        result["metrics"][name] = {**row, "before": a, "after": b, "delta": delta}
        result["metrics"][name].pop("value")
        if delta is None:
            result["unknown"].append(name)
        elif delta != 0:
            result["decreased" if delta < 0 else "increased"].append(name)
            result["changed_cca_axes"].append(row["cca_axis"])
    result["changed_cca_axes"] = sorted(set(result["changed_cca_axes"]))
    down, up, unknown = result["decreased"], result["increased"], result["unknown"]
    if down and up:
        relation = "mixed_work_tradeoff"
    elif down:
        relation = "decrease_in_observed_metrics"
    elif up:
        relation = "increase_in_observed_metrics"
    else:
        relation = "no_observed_change"
    result["structural_relation"] = relation
    if unknown:
        result["required_next_evidence"].append(
            "resolve only missing work dimensions relevant to this edit; unknown work does not cancel"
        )
    if up and down:
        result["required_next_evidence"].append(
            "price the changed competing mechanisms with matched short probes before choosing by cost"
        )
    if up or down:
        result["required_next_evidence"].append(
            "qualify changed semantics; probe only uncertain changed costs, not unrelated compute"
        )
    result["required_next_evidence"].append(
        "occupancy, contention and cache effects remain unknown without emitted context and cost evidence"
    )
    return result


# ---- structural_levels -------------------------------------------------------------------------
# Inefficiencies visible in an emitted command buffer, tagged by the optimisation level they live at.
#
# WHY THIS EXISTS. A corpus can only measure the levels it has capsules for. On this target that is the
# tile and intra-layer rungs, with two inter-layer members and nothing at the boundary, fusion or
# global rungs -- so an agent optimising against measured cycles alone is steered, silently, to ignore
# whole classes of inefficiency. It is not asked about them and it cannot be scored on them, so it does
# not look.
#
# The command buffer already carries enough structure to SEE several of those inefficiencies without
# running anything: it names its tensors, says which op produced each value and which consumed it,
# records where a value was drained to memory, and declares what each op fused into its epilogue. That
# is enough to point at a round trip through memory, a weight staged twice, or an epilogue left
# unfused -- for free, before any oracle time is spent.
#
# WHAT A FINDING IS, AND WHAT IT IS EMPHATICALLY NOT. Every finding here is a STRUCTURAL OBSERVATION:
# this program does a thing that costs something. None of them is a cycle count, an estimate, or a
# prediction, and nothing here may be read as one. A finding says an inefficiency is present; only a
# measurement says removing it was worth anything. That asymmetry is the same one the rest of this
# package runs on -- a screen may eliminate, it may never certify -- and it is why these are reported
# as findings with a level and a reason rather than as a score to optimise.
#
# A buffer this cannot read yields UNKNOWN and says why. An empty finding list means "none of the
# patterns below is present", never "this program is efficient".

#: The optimisation ladder, coarsest last. These are the level names the capsule corpus itself
#: declares, so a finding can be joined to the family that would measure it -- where one exists.
#:
#: This is a MIRROR of the corpus generator's own level vocabulary, and mirrors drift: it shipped
#: without ``L1_separation_floor``, so a finding could not be tagged at that rung and
#: :func:`level_summary` reported a ladder with a hole in it -- silently, because a rung that is never
#: counted is indistinguishable from a rung with nothing on it, which is exactly the failure mode this
#: module exists to argue against. The vocabulary is NOT imported from the generator that owns it
#: (``experiments/corpora/generate_corpus.py``): that is a contract script and this is library
#: code, and a library reaching into a script to learn its own constants inverts the layering. It is
#: pinned instead by a test that FAILS when the two lists differ
#: (``tests/dse/test_structural_levels.py``) -- the same discipline as
#: :func:`merlin.analysis.perf.bounds.attribution.buckets_match_reference`, where a mirror is kept honest by a check
#: that can fail rather than by a comment asking the next editor to remember.
LEVELS = ("L1_tile", "L1_separation_floor", "L2_intra_layer", "L3_inter_layer", "L4_boundary", "L5_fusion", "L6_global")

#: ABI vocabulary. These are the emitted program's own opcode names, carried by the buffer rather
#: than assumed about any device: a backend speaking this ABI emits them whatever its ISA.
#: An opcode outside this vocabulary is not an error -- it simply matches no pattern below.
COMMIT_OPCODES = ("COMMIT",)
#: Ops that place a value into on-chip residency, and ops that give that residency back.
STAGE_OPCODES = ("RES_PACK",)
RELEASE_OPCODES = ("EVICT",)
#: The attribute an op carries listing the work it folded into its own output stage.
EPILOGUE_ATTRIBUTE = "epilogue"

UNKNOWN = "UNKNOWN"


def _commands(buffer: object) -> "list[Mapping[str, Any]] | None":
    if not isinstance(buffer, Mapping):
        return None
    rows = buffer.get("commands")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    return [r for r in rows if isinstance(r, Mapping)]


def _operands(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("operands")
    return value if isinstance(value, Mapping) else {}


def _attributes(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("attributes")
    return value if isinstance(value, Mapping) else {}


def _produced(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Value names this command writes. The ABI names a destination ``dst``."""
    dst = _operands(row).get("dst")
    return (str(dst),) if isinstance(dst, str) and dst else ()


def _consumed(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Value names this command reads: every named operand that is not the destination."""
    return tuple(str(v) for k, v in sorted(_operands(row).items()) if k != "dst" and isinstance(v, str) and v)


def findings(buffer: object) -> dict[str, Any]:
    """Structural inefficiencies in one emitted program, each tagged with its optimisation level."""
    rows = _commands(buffer)
    if rows is None:
        return {"status": UNKNOWN, "reason": "the command buffer declares no command list"}
    if not rows:
        return {"status": UNKNOWN, "reason": "the command buffer declares no commands"}

    found: list[dict[str, Any]] = []

    # ---- L2: a value given back to on-chip residency and then staged again -------------------
    # Staging a weight, releasing it, and staging the same weight again means the program paid the
    # residency cost twice for one value. Whether that was necessary depends on what else needed the
    # space, which this cannot see -- so it is reported, not condemned.
    staged: dict[str, int] = {}
    released: set[str] = set()
    for index, row in enumerate(rows):
        opcode = str(row.get("opcode") or "")
        if opcode in STAGE_OPCODES:
            for name in _consumed(row):
                if name in released and name in staged:
                    found.append(
                        {
                            "level": "L2_intra_layer",
                            "kind": "residency_restaged",
                            "value": name,
                            "at_command": index,
                            "first_staged_at": staged[name],
                            "detail": (
                                f"{name!r} was staged for on-chip residency, released, and "
                                f"staged again; the program paid to place the same value twice"
                            ),
                        }
                    )
                staged.setdefault(name, index)
        if opcode in RELEASE_OPCODES:
            released.update(_consumed(row))

    # ---- L3: a value drained to memory and then read back ------------------------------------
    # A value that is committed and later consumed made a round trip the program could have avoided
    # by keeping it. This is the inter-layer inefficiency the corpus has only two members for.
    committed: dict[str, int] = {}
    for index, row in enumerate(rows):
        opcode = str(row.get("opcode") or "")
        if opcode in COMMIT_OPCODES:
            for name in _consumed(row):
                committed.setdefault(name, index)
            continue
        for name in _consumed(row):
            if name in committed:
                found.append(
                    {
                        "level": "L3_inter_layer",
                        "kind": "memory_round_trip",
                        "value": name,
                        "at_command": index,
                        "committed_at": committed[name],
                        "detail": (
                            f"{name!r} was drained to memory and then read back by a later "
                            f"command; the value made a round trip instead of staying resident"
                        ),
                    }
                )

    # ---- L5: a producer with an empty epilogue whose only consumer is the next command --------
    # An op that declares no epilogue and whose result is consumed immediately is the shape a fused
    # epilogue would collapse. Whether this backend CAN fuse that pair is not decidable here, which
    # is exactly why this is a finding and not an instruction.
    consumers: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        for name in _consumed(row):
            consumers.setdefault(name, []).append(index)
    for index, row in enumerate(rows):
        epilogue = _attributes(row).get(EPILOGUE_ATTRIBUTE)
        if not isinstance(epilogue, Sequence) or isinstance(epilogue, (str, bytes)) or epilogue:
            continue
        for name in _produced(row):
            uses = consumers.get(name) or []
            if len(uses) == 1 and uses[0] == index + 1:
                found.append(
                    {
                        "level": "L5_fusion",
                        "kind": "unfused_single_consumer",
                        "value": name,
                        "at_command": index,
                        "consumed_at": uses[0],
                        "detail": (
                            f"command {index} declares an empty {EPILOGUE_ATTRIBUTE} and its "
                            f"result {name!r} is read by exactly one command, the next one; "
                            f"that is the shape a fused epilogue would collapse"
                        ),
                    }
                )

    return {
        "status": "read",
        "commands": len(rows),
        "findings": found,
        "by_level": level_summary(found),
        "basis": (
            "structural patterns in the emitted command buffer; each names an inefficiency "
            "that is PRESENT, and none of them is a cycle count, an estimate, or a claim "
            "that removing it would be worth anything"
        ),
        "not_covered": (
            "nothing here inspects the host boundary or the choice of operand encoding, "
            "so a finding list without those levels is silence about them, not a pass"
        ),
    }


def level_summary(found: "Sequence[Mapping[str, Any]]") -> dict[str, int]:
    """How many findings sit at each level, including the levels with none."""
    counts = {level: 0 for level in LEVELS}
    for row in found:
        level = str(row.get("level") or "")
        if level in counts:
            counts[level] += 1
    return counts


__all__ = [
    "COMMIT_OPCODES",
    "EPILOGUE_ATTRIBUTE",
    "LEVELS",
    "RELEASE_OPCODES",
    "STAGE_OPCODES",
    "findings",
    "level_summary",
]
