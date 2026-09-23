"""Host-owned emission diagnostics from explicit artifacts and resource declarations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes as _sha256

from .broker_evidence import _is_sha256
from .contracts import StageGateError

ORDERING_REFUSED = "refused_unqualified_ordering"


BARRIER_UNKNOWN = "UNKNOWN"


def _demand_lower_bound(buffer: Mapping[str, Any], peak_macs_per_cycle: int | None) -> dict[str, Any]:
    """Cycles this arm cannot beat, from its own declared work and operands.

    A bound, not a prediction. Compute demand is the priced MAC count over the structural peak;
    movement demand is the operand bytes the buffer itself declares. Both are floors -- a spilling
    schedule re-fetches, so real movement is only ever larger -- which keeps the result honestly a
    lower bound rather than an estimate that could flatter a candidate.
    """
    declined = buffer.get("declined")
    if isinstance(declined, Mapping):
        return {
            "status": "unavailable",
            "reason": (
                "the compiler declined this whole-program lowering: "
                f"{str(declined.get('reason') or 'reason unavailable')[:400]}"
            ),
        }
    if not peak_macs_per_cycle:
        return {"status": "unavailable", "reason": "no derived structural peak for this target"}
    from merlin.perf.work_volume import work_from_command_buffer  # noqa: PLC0415

    work = work_from_command_buffer(buffer)
    macs = int(getattr(work, "known_macs", 0) or 0)
    tensors = buffer.get("tensors")
    if not macs or not isinstance(tensors, Mapping):
        return {"status": "unavailable", "reason": "the buffer declares no work or no tensors"}
    width = {"i8": 1, "u8": 1, "i16": 2, "bf16": 2, "f16": 2, "i32": 4, "f32": 4}
    operand_bytes = 0
    for spec in tensors.values():
        if not isinstance(spec, Mapping):
            continue
        shape, dtype = spec.get("shape"), str(spec.get("dtype") or "")
        if not isinstance(shape, Sequence) or dtype not in width:
            return {"status": "unavailable", "reason": f"an operand declares no shape or an unpriced dtype {dtype!r}"}
        count = 1
        for extent in shape:
            count *= int(extent)
        operand_bytes += count * width[dtype]
    return {
        "status": "derived",
        "compute_floor_cycles": macs / float(peak_macs_per_cycle),
        "declared_operand_bytes": operand_bytes,
        "exact": not bool(getattr(work, "is_lower_bound", False)),
        "licence": "a floor the arm cannot beat; never an estimate of what it will cost",
    }


def analyze_command_buffers(
    baseline_json: Path,
    candidate_json: Path,
    *,
    candidate_root: Path | None = None,
    peak_macs_per_cycle: int | None,
    achievable_macs_per_cycle: float | None,
    target: str = "",
) -> dict[str, Any]:
    """Compare declared work, movement and structure without claiming a timing ranking.

    Inputs are the candidate's emitted artifacts: no oracle, no golden, no holdout.
    The action derives demand bounds but constructs neither a composed timing envelope
    nor a qualified ordering estimator. Missing observations remain unknown.
    """
    from merlin.perf.command_buffer_diagnostics import representation_activity  # noqa: PLC0415
    from merlin.perf.movement_volume import movement_from_command_buffer  # noqa: PLC0415
    from merlin.perf.work_volume import work_from_command_buffer  # noqa: PLC0415

    def _load(path: Path) -> Mapping[str, Any]:
        # A RELATIVE PATH HERE HAD NO BASE, and this action runs in the HOST process rather than
        # under the sandbox's --chdir, so a relative argument resolved against a directory the agent
        # has never seen. Three bases were live in one tool: the agent's shell sees
        # `submission/performance/...`, a brokered subprocess is chdir'd into the submission so it
        # sees `performance/...`, and this host action saw neither. Measured: the agent spent two
        # calls discovering that, having been taught the second convention by the emit action one
        # call earlier, and the refusal it got back said "absent or linked" -- a claim about the
        # filesystem, when the actual fault was the base.
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (candidate_root / resolved) if candidate_root else resolved
        if resolved.is_symlink() or not resolved.is_file():
            hint = (
                ""
                if Path(path).is_absolute() or candidate_root is None
                else f" (a relative path is resolved against the candidate root {candidate_root})"
            )
            raise StageGateError(f"command buffer is absent or linked: {resolved}{hint}")
        return json.loads(resolved.read_text(encoding="utf-8"))

    buffers = {arm: _load(path) for arm, path in (("baseline", baseline_json), ("candidate", candidate_json))}
    out: dict[str, Any] = {
        "schema_version": 3,
        "kind": "host_owned_command_buffer_analysis",
        "basis": "emitted artifacts only; no oracle, no golden, no holdout",
    }
    arms: dict[str, Any] = {}
    for arm, buffer in buffers.items():
        work = work_from_command_buffer(buffer)
        movement = movement_from_command_buffer(buffer)
        macs = int(getattr(work, "known_macs", 0) or 0)
        declined = buffer.get("declined")
        declined = declined if isinstance(declined, Mapping) else None
        # A LOWER BOUND IS NOT A TOTAL. `work_volume` prices each command it can and records a
        # refusal for each it cannot, so `is_lower_bound` means "there is unpriced work here".
        # Reporting that as a total would understate the candidate's demand and silently flatter it.
        row: dict[str, Any] = {
            "status": "declined" if declined else "emitted",
            "declined": (
                {key: declined.get(key) for key in ("op", "reason", "shape") if key in declined} if declined else None
            ),
            "macs": None if declined else macs,
            "exact": False if declined else not bool(getattr(work, "is_lower_bound", False)),
            "unpriced_commands": (
                [f"whole-model lowering declined: {str(declined.get('reason') or 'reason unavailable')[:400]}"]
                if declined
                else [str(r) for r in (getattr(work, "refusals", ()) or ())][:8]
            ),
            "movement": {
                "known_bytes_in": None if declined else movement.known_bytes_in,
                "known_bytes_out": None if declined else movement.known_bytes_out,
                "known_bytes": None if declined else movement.known_bytes,
                "exact_bytes": False if declined else movement.exact_bytes,
                "is_lower_bound": True if declined else movement.is_lower_bound,
                "refusals": (
                    ["whole-model lowering declined before movement was emitted"]
                    if declined
                    else list(movement.refusals)[:8]
                ),
                "counts": "declared_by_command_buffer",
                "cannot_detect": (
                    "a lowering that re-loads a resident operand; compare issued "
                    "load count with declared resident-pack count"
                ),
            },
            "representation_activity": representation_activity(buffer),
        }
        if peak_macs_per_cycle and not declined:
            row["ideal_cycles_at_peak"] = macs / float(peak_macs_per_cycle)
        if achievable_macs_per_cycle and not declined:
            row["ideal_cycles_at_achievable"] = macs / float(achievable_macs_per_cycle)
        arms[arm] = row
    out["arms"] = arms
    out["peak_macs_per_cycle"] = peak_macs_per_cycle
    out["achievable_macs_per_cycle"] = achievable_macs_per_cycle

    # Structural diagnostics are not calibrated timing predictions. No empirical
    # accuracy claim can transfer from another target or workload by default.
    # Synchronization: how many completion points the candidate removed.
    try:
        from merlin.perf import barrier_arms as BARRIER  # noqa: PLC0415

        out["barriers"] = BARRIER.paired_removal(buffers["baseline"], buffers["candidate"])
    except Exception as exc:  # noqa: BLE001 - an uncountable stream is UNKNOWN, never zero
        out["barriers"] = {"status": BARRIER_UNKNOWN, "reason": f"barrier counting failed: {type(exc).__name__}"}

    # A lower bound on cycles from declared demand alone: what this arm cannot beat.
    out["lower_bound"] = {arm: _demand_lower_bound(buffer, peak_macs_per_cycle) for arm, buffer in buffers.items()}

    # Structural findings identify potential inefficiencies by level. They are not
    # cycle counts and may not be cited as measured savings.
    try:
        from merlin.perf import structural_levels as LEVELS  # noqa: PLC0415

        out["structural_levels"] = {arm: LEVELS.findings(buffer) for arm, buffer in buffers.items()}
    except Exception as exc:  # noqa: BLE001 - an unreadable buffer is UNKNOWN, never "clean"
        out["structural_levels"] = {
            "status": BARRIER_UNKNOWN,
            "reason": f"structural level analysis failed: {type(exc).__name__}",
        }

    b, c = arms["baseline"]["macs"], arms["candidate"]["macs"]
    if b and c and b != c:
        out["work_delta"] = {
            "candidate_over_baseline": c / b,
            "note": (
                "the candidate does a DIFFERENT amount of arithmetic; a cycle "
                "comparison between these two is not a schedule comparison"
            ),
        }
    # NO DIFFERENTIAL VERDICT IS ATTEMPTED HERE, and saying so is the point.
    #
    # This previously called `differential.compare(arms["baseline"], arms["candidate"])` on the two
    # plain dicts built just above. `compare` takes two `envelope.Composed` bounds and reads
    # `.operator` off them, so on a dict it raised AttributeError on EVERY call, the bare `except`
    # swallowed it, and the action reported a hardcoded `{"basis": "REFUSED"}` -- a refusal that
    # looked like the analyzer's considered verdict but was only a type error. A stale claim that
    # reads like evidence is worse than no claim, because it gets cited.
    #
    # The honest reason is structural, not incidental: this action compares DEMAND (the work each
    # command buffer declares) and never builds a composed envelope or per-resource demands, so it
    # has nothing a cycle-level differential could be computed from. The measurement path is what
    # carries a differential verdict.
    # No target/workload-bound estimator has been admitted by this action. A study's
    # historical agreement rates are not evidence about an arbitrary selected target.
    out["ordering_signals"] = {
        "status": ORDERING_REFUSED,
        "basis": "declared command-buffer structure only; no qualified ordering estimator",
        "measured": {},
        "reason": (
            "this action has no qualified ordering estimator for the selected target and workload. "
            "Work, movement, completion points and lower bounds are structural diagnostics, "
            "not proof that one legal schedule is faster. Use qualified measurement evidence "
            "to compare execution time."
        ),
        "artifact": None,
    }
    out["differential"] = {
        "basis": "not_attempted",
        "reason": (
            "this action prices declared WORK from the command buffers; a cycle-level "
            "differential needs a composed envelope and per-resource demands per arm, "
            "which it never builds. Read the measurement path for a differential verdict."
        ),
    }
    return out


def whole_program_schema_record(path: Path) -> dict[str, str]:
    """Current compiler API, separate from the immutable Phase-1 grading contract."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise StageGateError("current whole-program compiler schema is absent or linked")
    return {"path": str(path), "sha256": _sha256(path.read_bytes())}


def validate_whole_program_schema(buffer: Mapping[str, Any], record: Mapping[str, Any], *, arm: str) -> None:
    """Host-side validation is mandatory even when a compiler omits its own validator."""
    import jsonschema

    if (
        set(record) != {"path", "sha256"}
        or not isinstance(record.get("path"), str)
        or not _is_sha256(record.get("sha256"))
    ):
        raise StageGateError("whole-program compiler API schema binding changed")
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file():
        raise StageGateError("current whole-program compiler schema is absent or linked")
    payload = path.read_bytes()
    if _sha256(payload) != record["sha256"]:
        raise StageGateError("whole-program compiler API schema binding changed")
    schema = json.loads(payload)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(buffer))
    if errors:
        details = [f"/{'/'.join(map(str, error.path))}: {error.message}" for error in errors[:8]]
        raise StageGateError(
            f"whole-model {arm} command buffer violates current compiler API schema: " + "; ".join(details)
        )


_CAPSULE_DATAPATH_PATH = ("operation", "attributes", "dtype")


def declared_capsule_datapath(descriptor: Mapping[str, Any]) -> str | None:
    """The operand format the capsule declares its compile lowers to, or ``None``.

    ``None`` is returned rather than a guess: the census then judges at the capture's own element
    types and says so in its own ``datapath``/``dtype_authority`` fields, so a reader can tell which
    program the refusals are about.
    """
    node: Any = descriptor
    for key in _CAPSULE_DATAPATH_PATH:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return str(node) if isinstance(node, str) and node.strip() else None


def isa_capability_utilization(artifact_text: str, *, target: str) -> dict[str, Any]:
    """Declared-versus-emitted instruction use of the emitted artifact, from the target's own facts.

    THE KEY `agent_guidance.declared_capability_unused` READS. That finding has existed, complete
    with its magnitude and its licence, since it was written; nothing in production ever wrote the
    key, so it never fired once and the agent's only view of the machine was a count of the
    instructions it had already emitted. The producer is
    :func:`merlin.perf.isa_utilization.capability_utilization_for_target`, which is also what the
    capsule-bench ISA broker calls -- one producer, so the two phases cannot disagree about what a
    target declares.
    """
    try:
        from merlin.perf.isa_utilization import capability_utilization_for_target  # noqa: PLC0415

        return capability_utilization_for_target(artifact_text, target=target)
    except Exception as exc:  # noqa: BLE001 - an unmeasured ISA is stated, never a clean 100%
        return {
            "schema": "isa_capability_utilization_v1",
            "status": "UNKNOWN",
            "reason": (
                f"declared-versus-emitted instruction use was NOT measured: {type(exc).__name__}: {str(exc)[:200]}"
            ),
            "declared_count": None,
            "used_count": None,
            "unused": [],
            "undeclared_emitted": [],
        }


def capability_refusals(prepared_source: Any, *, target: str, datapath: str | None) -> list[dict[str, Any]]:
    """WHY the target's declared capabilities were refused, per site, aggregated by clause.

    THE KEY `agent_guidance.declared_capability_refused` READS, and the second half of the ISA
    story: :func:`isa_capability_utilization` can say a declared instruction is never emitted, and
    only this can say what the selector refused it on. The selector is the target's OWN capability
    map (``merlin.targetgen.eligibility`` through ``merlin.perf.placement_census``); the clause
    strings are its, not this module's, and nothing here names a target, a capability or a clause.

    Fail-closed: a census that could not be taken returns ONE entry that says so. Returning an empty
    list would read downstream as "nothing was refused", which is the flattering direction.
    """
    module = getattr(prepared_source, "parsed_module", None)
    if module is None:
        return [
            {
                "schema": "capability_refusal_census_v1",
                "capability": "accelerator_placement",
                "status": "UNKNOWN",
                "clauses": [],
                "caveat": (
                    "the captured source was not parsed in this analysis, so no site was asked "
                    "whether the target could run it; this is not a statement that none was refused"
                ),
            }
        ]
    try:
        from merlin.perf import capability_refusal as CR  # noqa: PLC0415
        from merlin.perf import lowering_coverage as LC  # noqa: PLC0415
        from merlin.perf import placement_census as PC  # noqa: PLC0415

        report = PC.census_of_module(module, target, datapath=datapath)
        sites = [
            CR.RefusalSite(
                site=f"{row['index']}:{row.get('op') or ''}",
                admitted=row.get("placement") != LC.HOST,
                clause=(
                    CR.SELECTED
                    if row.get("placement") != LC.HOST
                    else str(row.get("refusal") or "unjustified_host_placement")
                ),
                detail={
                    "family": row.get("family"),
                    "judged_dtype": row.get("dtype"),
                    "captured_dtype": row.get("captured_dtype"),
                    "dtype_authority": row.get("dtype_authority"),
                    "gap_class": row.get("gap_class"),
                    "reason": row.get("reason"),
                    "macs": row.get("macs"),
                },
            )
            for row in report.get("regions") or ()
            if isinstance(row, Mapping)
        ]
        census = dict(CR.census("accelerator_placement", sites))
        census["status"] = "measured"
        # WHICH PROGRAM these refusals are about. A census judged at a different operand format than
        # the compiler lowers to answers a different question with the same shape, so the format and
        # its authority ride with the verdict rather than being recoverable only from the rows.
        census["judged_datapath"] = report.get("datapath")
        census["datapath_authority"] = (
            "capsule declaration"
            if datapath is not None
            else "the capture's own element types; the capsule declared none"
        )
        census["silent_fallbacks"] = report.get("silent_fallbacks")
        census["unclassified_refusals"] = report.get("unclassified_refusals")
        census["host_by_gap_class"] = report.get("host_by_gap_class")
        return [census]
    except Exception as exc:  # noqa: BLE001 - an untaken census says so; it never reports zero refusals
        return [
            {
                "schema": "capability_refusal_census_v1",
                "capability": "accelerator_placement",
                "status": "UNKNOWN",
                "clauses": [],
                "caveat": (
                    f"the placement census could not be taken, so NO site was asked whether the "
                    f"target could run it: {type(exc).__name__}: {str(exc)[:200]}. This is not a "
                    f"statement that nothing was refused"
                ),
            }
        ]


def resolved_device(design_keys: Mapping[str, Any] | None, *, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """WHICH DEVICE a cycle count's design keys name, resolved through the pin registry.

    A cycle count is about a device, and a configuration name is not one. Two registered bitstreams
    in this repo elaborate the SAME configuration string onto the same board and are different
    machines, so :mod:`merlin.perf.design_identity` joins by the queue hw-config the run was
    submitted under CONFIRMED against the artifact digest the record carries -- two independent
    facts that have to agree before a device is named.

    Three states, never two: named, or ``UNKNOWN`` with the reason. "We could not tell which device
    this is" is not a softer "the designs differ"; it means the count must not be scored against
    anything measured elsewhere. An iteration with no measured record resolves to the first of
    those reasons, and says so rather than leaving the field out.
    """
    try:
        from merlin.perf.design_identity import design_string  # noqa: PLC0415

        if not isinstance(artifacts, Mapping):
            raise ValueError("device identity requires an explicit artifact registry mapping")
        return dict(design_string(design_keys or {}, artifacts=artifacts))
    except Exception as exc:  # noqa: BLE001 - an unresolved device is stated, never assumed
        return {
            "name": None,
            "config": None,
            "reason": (
                f"the device these design keys name could not be resolved: {type(exc).__name__}: {str(exc)[:200]}"
            ),
        }


def iteration_cost_plane(
    descriptor: Mapping[str, Any],
    *,
    target: str,
    arms: Mapping[str, Any],
    phase: str,
    artifacts: Mapping[str, Any],
    tiers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The cost plane for THIS iteration, on the same rule the capsule grade applies.

    WHY PHASE 2 NEEDS ITS OWN. The capsule-bench grade attaches a `cost_plane` to every row it
    scores, and it is pinned to `report` there deliberately. None of it crosses into the
    optimization loop: what the phase-2 agent gets instead is a structural dispatch delta whose own
    licence says "fewer dispatches do not prove fewer cycles", beside `cycle_selection: UNMEASURED`.
    So the agent optimizes against an accounting identity with no floor under it.

    This computes the SAME verdict, through :func:`merlin.perf.cost_plane.assess`, at the explicitly
    supplied declared phase. The caller also supplies the hardware artifact registry. The
    floor is DERIVED from the array this target's own facts declare and the work this program's own
    command buffer counted, so it needs no slack constant; the array geometry is never a literal and
    an underivable one leaves the floor an explicit UNKNOWN rather than dividing by a default.

    With no measured count in hand the verdict is ``incomplete`` -- which is a STATUS, orthogonal to
    phase, and never a pass at either. That is the point: the loop currently reports nothing at all
    here, and "not measured" and "measured fine" are indistinguishable in silence.
    """
    try:
        from merlin.liveness.facts import silicon_facts  # noqa: PLC0415
        from merlin.perf import cost_plane as CP  # noqa: PLC0415
        from merlin.perf.decompose import is_unknown  # noqa: PLC0415

        if not isinstance(artifacts, Mapping):
            raise ValueError("cost plane requires an explicit artifact registry mapping")
        facts = silicon_facts(target)
        rows = {}
        for arm in ("baseline", "candidate"):
            row = arms.get(arm)
            macs = row.get("macs") if isinstance(row, Mapping) else None
            floor = dict(
                CP.derived_floor(
                    macs=macs if isinstance(macs, int) and not isinstance(macs, bool) else None,
                    array_rows=facts.mesh_rows,
                    array_cols=facts.mesh_cols,
                    provenance=f"{arm} arm of this iteration",
                )
            )
            # An underivable floor comes back as the UNKNOWN sentinel, which is not JSON. Render it
            # as `None` BESIDE the reason the floor already carries -- a document that cannot be
            # written is a document nobody reads, and the reason is the part that matters.
            if is_unknown(floor.get("cycles")):
                floor["cycles"] = None
            rows[arm] = {
                "macs": macs,
                "work_is_exact": row.get("exact") if isinstance(row, Mapping) else None,
                "floor": floor,
            }
        verdict = dict(
            CP.assess(
                descriptor,
                tiers=tiers,
                macs=rows["candidate"]["macs"] if isinstance(rows["candidate"]["macs"], int) else None,
                array_rows=facts.mesh_rows,
                array_cols=facts.mesh_cols,
                phase=phase,
            )
        )
        verdict["arms"] = rows
        # THE DEVICE, not the configuration name. `cost_plane` refuses to compare two counts unless
        # every design key matches and is stated, and that identity was unreadable by the ledger
        # that holds the measured destinations: two registered bitstreams share one configuration
        # string. Resolving it here is what lets a measured count be scored against a measured
        # reference at all, and an unresolved one says so instead of being matched by name.
        verdict["device"] = resolved_device(verdict.get("design"), artifacts=artifacts)
        verdict["array"] = {
            "rows": facts.mesh_rows,
            "cols": facts.mesh_cols,
            "basis": "derived from this target's own RTL facts",
        }
        verdict["licence"] = (
            "the derived half of this plane is a FLOOR: it can refute a count below what the array "
            "must spend issuing this program's own tiles, and it cannot say whether a count is fast "
            "enough. A structural dispatch delta is not a cycle delta"
        )
        return verdict
    except Exception as exc:  # noqa: BLE001 - an undecided plane is stated; it never reads as within
        from merlin.perf.gate_phase import STATUS_INCOMPLETE  # noqa: PLC0415

        return {
            "schema": "cost_plane_verdict_v1",
            "status": STATUS_INCOMPLETE,
            "reason": (
                f"the cost plane could not be computed for this iteration: {type(exc).__name__}: {str(exc)[:200]}"
            ),
            "blocking": False,
            "admitted": False,
            "measured_cycles": None,
            "floor_cycles": None,
        }
