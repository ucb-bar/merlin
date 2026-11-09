"""Partial source-work floors joined only to original completed queue estimates.

These ratios order compiler exploration. They never supply measured priorities,
an incumbent, a performance target, plateau evidence, or convergence authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.perf.capsule_roofline import compute_peak_from_facts
from merlin.perf.model_macs import observe_model_macs

from . import _public_native_exploratory_estimate as ESTIMATE
from .contracts import StageGateError, document_sha256
from .exploratory_development_priority import _estimate, _signature
from .search_inputs import ORDINARY_FAST_KIND, ORDINARY_SELECTED_FAST_KIND

SCHEMA = "merlin.exploratory_capsule_priorities.v1"
MAX_SOURCE_BYTES = 16 << 20


def _read_pin(path, digest, limit=MAX_SOURCE_BYTES):
    path = Path(path)
    if not is_sha256(digest) or path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise StageGateError("exploratory floor requires a bounded exact source pin")
    data = path.read_bytes()
    if len(data) > limit or hashlib.sha256(data).hexdigest() != digest:
        raise StageGateError("exploratory floor source pin changed")
    return data


def derive_compute_floors(corpus, *, facts_pin):
    """Use the selected member's traced source, never its generated command plan.

    The caller validates the original frozen corpus. Source and facts bytes are
    independently reopened here; unsupported source work is UNKNOWN, not zero.
    """
    facts = json.loads(_read_pin(facts_pin["path"], facts_pin["sha256"]))
    peak, basis = compute_peak_from_facts(facts)
    floors = {}
    for member in corpus.capsules:
        key = member.family, member.capsule
        row = dict(member=list(key), capsule_sha256=member.source_sha256,
                   status="UNKNOWN", useful_macs=None, floor_cycles=None,
                   peak_macs_per_cycle=peak, peak_basis=basis, source=None,
                   facts_sha256=facts_pin["sha256"], work=[],
                   reason="frozen frontend source pin unavailable")
        try:
            trace = member.descriptor.get("frontend_trace")
            if not isinstance(trace, dict):
                raise StageGateError(row["reason"])
            relative = Path(trace.get("source_mlir", ""))
            if not relative.parts or relative.is_absolute() or any(p in {".", ".."} for p in relative.parts):
                raise StageGateError("frozen frontend source path escapes its member")
            source_root = Path(member.source_dir)
            path = source_root / relative
            if source_root.is_symlink() or any(p.is_symlink() for p in (path, *path.parents)
                                              if p == source_root or source_root in p.parents):
                raise StageGateError("frozen frontend source path contains a link")
            raw = _read_pin(path, trace.get("source_mlir_sha256"))
            row["source"] = dict(path=str(path), sha256=trace["source_mlir_sha256"])
            shapes = [shape for _op, shape in observe_model_macs(raw.decode("utf-8"))]
            row["work"] = [asdict(shape) for shape in shapes]
            if not shapes or any(shape.status != "derived" or shape.macs is None for shape in shapes):
                raise StageGateError("complete useful source MAC work is unproved; unsupported work remains UNKNOWN")
            if peak is None:
                raise StageGateError(basis)
            macs = sum(shape.macs for shape in shapes)
            if macs <= 0:
                raise StageGateError("source has no positive proved useful MAC work")
            row.update(status="partial", useful_macs=macs, floor_cycles=-(-macs // peak),
                       reason=("useful source MACs / simultaneous extracted arithmetic-site capacity; "
                               "other service terms unresolved"))
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, StageGateError) as error:
            row["reason"] = str(error)
        floors[key] = row
    return floors


def _usable_arm(arm):
    estimate, reason = _estimate(arm)
    if estimate is None:
        return None, reason
    native = arm["native_exploration"]
    counts = native.get("observed_counts")
    if not isinstance(counts, dict):
        return None, "original verified native counts unavailable"
    try:
        ESTIMATE.validate(estimate, counts=counts)
    except (KeyError, TypeError, StageGateError) as error:
        return None, str(error)
    if estimate["event_accounting_complete"] is not True or estimate["unresolved_events"]:
        return None, "fixed-model event accounting is incomplete"
    return estimate, None


def project(pool, *, observations=None):
    """Reopen original queue results before offering per-source capsule ordering.

    Only completed, host-issued compile/Spike jobs enter. The live original
    SearchInputs validator owns source/count/native custody; shape arithmetic
    and serialized estimate presence alone cannot enter this view.
    """
    observations = pool.observations() if observations is None else observations
    roster = tuple(sorted(pool.inputs.validate()))
    corpus = pool.inputs.corpus
    baseline = pool.baseline_revision.public_record()["source_sha256"]
    groups = {baseline: {}}
    current = baseline
    rejected = []
    owner = getattr(pool.policy, "_native_exploration", None)
    facts_pin = None
    floors = {}
    if owner is not None and owner.exploratory_estimate_protocol == ESTIMATE.PROTOCOL:
        facts_pin = pool.policy.binding().get("rtl_facts")
        if isinstance(facts_pin, dict):
            floors = derive_compute_floors(corpus, facts_pin=facts_pin)
    for observation in observations:
        if observation.get("state") != "completed" or observation.get("kind") not in (
            ORDINARY_FAST_KIND, ORDINARY_SELECTED_FAST_KIND
        ):
            continue
        source = observation.get("source_sha256")
        if not is_sha256(source):
            continue
        current = source
        groups.setdefault(source, {})
        try:
            issued = pool.status(observation["job_id"])
            if any(issued.get(key) != observation.get(key) for key in (
                "state", "kind", "revision_id", "source_sha256", "result", "selected_members"
            )) or issued.get("state") != "completed":
                raise StageGateError("exploratory observation differs from its original issued completed job")
            revision = pool.revisions.get(observation["revision_id"])
            if revision.source_sha256 != source:
                raise StageGateError("queued revision differs from its source group")
            data = pool.inputs.validate_result(observation["result"], revision, observation["kind"])
            if data["candidate_sha256"] != source or data["tuning_corpus_sha256"] != corpus.capsules_sha256:
                raise StageGateError("validated queue feedback changed source or corpus identity")
        except (KeyError, TypeError, ValueError, StageGateError) as error:
            rejected.append(dict(job_id=observation.get("job_id"), source_sha256=source, reason=str(error)))
            for key in roster:
                groups[source].setdefault(key, []).append(dict(
                    job_id=observation.get("job_id"), estimate=None,
                    reason="original completed-result validation refused this repeat",
                ))
            continue
        for cell in data["cells"]:
            key = cell["family"], cell["capsule"]
            if key not in roster or cell.get("baseline") is None or cell.get("candidate") is None:
                continue
            estimates = [_usable_arm(cell.get(arm)) for arm in ("baseline", "candidate")]
            pair_reason = None
            if key in floors and cell.get("source_sha256") != floors[key]["capsule_sha256"]:
                pair_reason = "estimate capsule differs from its frozen source-work member"
            if "diagnostic_pair_check" in cell and (cell["diagnostic_pair_check"] or {}).get("status") != "PASS":
                pair_reason = "original complete source/product pair check unavailable"
            if all(e is not None for e, _ in estimates) and _signature(estimates[0][0]) != _signature(estimates[1][0]):
                pair_reason = "paired estimates changed fixed model, comparison domain or scope"
            if all(e is not None for e, _ in estimates) and (
                cell["baseline"]["compiled"]["elf_sha256"] == cell["candidate"]["compiled"]["elf_sha256"]
                and estimates[0][0]["estimated_cycles"] != estimates[1][0]["estimated_cycles"]
            ):
                pair_reason = "identical compiled executables have inconsistent fixed-model estimates"
            for group, (estimate, reason), arm in zip(
                (baseline, source), estimates, ("baseline", "candidate"), strict=True
            ):
                groups[group].setdefault(key, []).append(dict(
                    job_id=observation.get("job_id"), revision_id=observation["revision_id"], arm=arm,
                    estimate=copy.deepcopy(estimate), reason=pair_reason or reason,
                    feedback_sha256=document_sha256(data),
                ))
    sources = []
    for source, members in sorted(groups.items()):
        ranked, unknown, under_floor = [], [], []
        for key in roster:
            floor = floors.get(key)
            repeats = members.get(key, [])
            reason = None
            if floor is None or floor["status"] != "partial":
                reason = (floor or {}).get("reason", "selected structural source/facts floor unavailable")
            elif not repeats:
                reason = "no completed validated estimate for this source and capsule"
            elif any(row["reason"] for row in repeats):
                reason = "; ".join(sorted({row["reason"] for row in repeats if row["reason"]}))
            elif len({_signature(row["estimate"]) for row in repeats}) != 1:
                reason = "repeats changed fixed model, comparison domain or scope"
            if reason is not None:
                unknown.append(dict(member=list(key), reason=reason))
                continue
            cycles = [row["estimate"]["estimated_cycles"] for row in repeats]
            bound = floor["floor_cycles"]
            if min(cycles) < bound:
                under_floor.append(dict(member=list(key), floor_cycles=bound, minimum_estimated_cycles=min(cycles),
                                        reason="estimate below structural floor; no attainment ordering is valid"))
                continue
            ranked.append(dict(member=list(key), floor_cycles=bound,
                               estimated_cycles=dict(lo=min(cycles), hi=max(cycles)),
                               attainment=dict(worst=bound / max(cycles), best=bound / min(cycles)),
                               repeat_count=len(repeats), provenance=repeats,
                               floor_status="partial", convergence_eligible=False))
        ranked.sort(key=lambda row: (row["attainment"]["worst"], row["member"]))
        sources.append(dict(source_sha256=source, ranked=ranked, unknown=unknown, under_floor=under_floor,
                            next=[row["member"] for row in ranked], convergence=False))
    selected = next(row for row in sources if row["source_sha256"] == current)
    return dict(schema=SCHEMA, status="EXPLORATORY", corpus_sha256=corpus.capsules_sha256,
                current_exploration_source_sha256=current, sources=sources,
                ranked=selected["ranked"], unknown=selected["unknown"], under_floor=selected["under_floor"],
                next=selected["next"], source_compute_floors=list(floors.values()),
                rejected_observations=rejected, facts=copy.deepcopy(facts_pin),
                order="ascending worst-repeat structural floor / fixed-model estimated cycles",
                convergence=False, completion_blocked=True, measured_cycles=None,
                qualified_for_promotion=False, promotion_authority="NONE", plateau_evidence=False,
                scope="partial ideal compute floors and fixed-model estimates; advisory compiler work ordering only")
