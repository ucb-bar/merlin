"""Optional known-answer recovery benchmark: how much of a DIRECTED reduction did a loop re-author?

An authoring loop that reaches a number nobody reached by hand is evidence. An authoring loop that
reaches a number somebody already reached by hand, unaided, is a MEASUREMENT -- and it is the only
one available before the hand result is beaten. This module is that measurement. It takes a
reduction a directed session already achieved, decomposed into the steps that achieved it, and
scores a candidate on how much of it the candidate re-authored by itself.

Three properties are the whole design.

**THE DENOMINATOR IS NOT THE TOTAL.** ``(base - candidate) / (base - destination)`` is the obvious
score and it is a bad one, for a reason that is measured rather than argued: in the reduction this
benchmark was built around, one step removes 54% of the operations and another removes EXACTLY ZERO
of them. A total-operations score pays a candidate 0.54 for re-authoring one idea out of seven, and
pays nothing at all -- can pay nothing at all -- for re-authoring the step that moves no operations
in this metric. So the score here is an attribution: the candidate's reduction is split by OPERATION
FAMILY, each family's recovery is the fraction of that family's reduction the candidate achieved,
and each step is credited with the families it actually moved. The per-step credits sum to the total
by construction (:func:`score_recovery` asserts it), and a step the metric cannot see is reported
with weight zero and a reason, never averaged away.

**A STEP THAT CHANGES THE CORRECTNESS CONTRACT IS NAMED, NOT AVERAGED.** The stack this was built
from asserts its correctness gate after EVERY step, so a step that breaks it is named rather than
folded into the headline -- and one of them does break it, in principle and not by accident. A run
declares the gate it is scored under and :func:`admitted_levers` excludes every step that only
clears a weaker one, so the headline number is never reachable by relaxing correctness. An
undeclared gate is UNKNOWN and excluded, never assumed to pass.

**THE FEEDBACK IS SEPARATE FROM THE SCORE, AND IT NEVER NAMES A STEP.** :func:`feedback` returns
what the loop is licensed to see; the key's step decomposition is scoring-side only. Handing a loop
"most of the remaining operations are integer arithmetic inside the addressing chain" is a
destination. Handing it the step list is the answer, because the step list IS the plan. The
instruments are individually switchable so the benchmark can say which one the loop needed --
that is what an ablation of this benchmark means -- and a test asserts that no subset of them ever
emits a step name.

**And a candidate that read the answer is REFUSED, not scored.** :func:`admit_candidate` consumes
the transcript audit's own verdict and fails closed: no audit is ``refused_unaudited``, not a pass.
A score is a claim about unaided work, so an unproven "unaided" is not a weaker score, it is no
score. See :mod:`merlin.targetgen.sandbox.answer_surfaces` for what "the answer" is derived to be.

Target-agnostic by construction: every fact about the reduction is in the key FILE, which is data,
and nothing here knows which target, model or compiler it is about.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from merlin.common.access import KEY_ENV, KEY_FILENAME, KEY_TOPIC, KEY_TOPIC_FOLDED
from merlin.common.paths import artifacts_dir

__all__ = [
    "GATE_UNKNOWN",
    "INSTRUMENTS",
    "KEY_ENV",
    "KEY_FILENAME",
    "KEY_SCHEMA",
    "SCHEMA",
    "Lever",
    "RecoveryKey",
    "RecoveryKeyError",
    "admit_candidate",
    "admitted_levers",
    "candidate_families",
    "feedback",
    "key_path",
    "load_key",
    "main",
    "score_recovery",
]

#: Schema of the record :func:`score_recovery` produces.
SCHEMA = "known_answer_recovery_v1"
#: Schema of the KEY FILE this module scores against.
KEY_SCHEMA = "known_answer_recovery_key_v1"
# Persistent key identity is re-exported from the core access registry. The sandbox must still find
# keys after this optional scorer is uninstalled, so it cannot import this module to classify them.

#: A step whose correctness gate was never established. Excluded from every scored denominator:
#: "we could not tell" is not "it passed". Declared here so the key file and the scorer agree.
GATE_UNKNOWN = "UNKNOWN"

#: The feedback channels a run may switch on, one at a time, so an ablation can say which one the
#: loop actually needed. Each is a DESTINATION or a VERDICT -- never a method, and never a step
#: name. Adding one is a deliberate decision about what the benchmark still measures.
INSTRUMENTS: tuple[str, ...] = (
    "destination_total",  # the operation count the directed session reached
    "destination_families",  # ...broken down by operation family (where the work is, not how)
    "structural_gap",  # signed per-field distance to the reference's emitted structure
    "refusal_brief",  # dead ends derived from the target's own RTL, as conclusions
    "correctness_gate",  # whether this candidate clears the declared gate
)


class RecoveryKeyError(ValueError):
    """The key could not be resolved, or does not describe a scoreable reduction."""


@dataclass(frozen=True)
class Lever:
    """One step of the directed reduction, with what it moved and what gate it cleared.

    ``family_delta`` is signed and POSITIVE MEANS REMOVED, so it sums to ``marginal``. A step may
    legitimately ADD operations in one family while removing more in another; the attribution below
    handles both directions, because a candidate that only did the cheap half of such a step has
    not re-authored it.
    """

    name: str
    order: int
    total_after: int
    marginal: int
    family_delta: dict[str, int]
    gate: str
    note: str = ""

    @property
    def movement(self) -> int:
        """Total absolute operation movement, the denominator of this step's identifiability."""
        return sum(abs(v) for v in self.family_delta.values())


@dataclass(frozen=True)
class RecoveryKey:
    """A directed reduction, decomposed. The ANSWER KEY -- never tracked, never agent-readable."""

    schema: str
    benchmark: str
    target: str
    metric: str
    base_total: int
    base_families: dict[str, int]
    destination_total: int
    destination_families: dict[str, int]
    levers: tuple[Lever, ...]
    program_identity: str | None = None
    reference: dict[str, Any] = field(default_factory=dict)
    refusals: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def lever(self, name: str) -> Lever:
        for lever in self.levers:
            if lever.name == name:
                return lever
        raise RecoveryKeyError(f"no step named {name!r} in this key")


# --------------------------------------------------------------------------------- resolving a key


def key_path(target: str | None = None, *, root: Path | None = None) -> Path:
    """Where this host's key lives, resolved rather than committed.

    ``MERLIN_RECOVERY_KEY`` wins. Otherwise the newest ``latest`` product under the topic, looked
    for BOTH at its minted location and at the location ``merlin-storage organize`` folds it to,
    so a reader finds it on either side of a fold. Returns the path whether or not it exists; the
    caller decides whether absence is an error (in a fresh worktree it legitimately is absent, and
    saying so beats inventing one).
    """
    override = os.environ.get(KEY_ENV)
    if override:
        return Path(override)
    base = artifacts_dir() if root is None else Path(root)
    roots = [base / KEY_TOPIC, base.joinpath(*KEY_TOPIC_FOLDED)]
    candidates: list[Path] = []
    for topic_root in roots:
        search = topic_root / target if target else topic_root
        if not search.is_dir():
            continue
        for version_dir in sorted(search.glob("v*")):
            key = version_dir / "latest" / KEY_FILENAME
            if key.is_file():
                candidates.append(key)
    if candidates:
        return max(candidates, key=lambda p: p.resolve().name)
    return roots[0] / (target or "") / "v1" / "latest" / KEY_FILENAME


def load_key(path: Path | str | None = None, *, target: str | None = None) -> RecoveryKey:
    """Load and validate a key file. Refuses a key that cannot support a score."""
    resolved = Path(path) if path is not None else key_path(target)
    if not resolved.is_file():
        raise RecoveryKeyError(
            f"no recovery key at {resolved}. The key is an ANSWER SURFACE: it is untracked by "
            f"design and absent from a fresh worktree. Mint it with `merlin-recovery mint`, or "
            f"point {KEY_ENV} at one -- do not reconstruct its numbers by hand"
        )
    document = yaml.safe_load(resolved.read_text())
    if not isinstance(document, Mapping):
        raise RecoveryKeyError(f"{resolved} is not a mapping")
    if document.get("schema") != KEY_SCHEMA:
        raise RecoveryKeyError(f"{resolved} declares schema {document.get('schema')!r}, expected {KEY_SCHEMA!r}")
    levers: list[Lever] = []
    for raw in document.get("levers") or ():
        if not isinstance(raw, Mapping):
            raise RecoveryKeyError(f"{resolved}: a step is not a mapping")
        delta = {str(k): int(v) for k, v in (raw.get("family_delta") or {}).items()}
        levers.append(
            Lever(
                name=str(raw["name"]),
                order=int(raw["order"]),
                total_after=int(raw["total_after"]),
                marginal=int(raw["marginal"]),
                family_delta=delta,
                gate=str(raw.get("gate") or GATE_UNKNOWN),
                note=str(raw.get("note") or ""),
            )
        )
    if not levers:
        raise RecoveryKeyError(f"{resolved} declares no steps; a key with no decomposition cannot attribute")
    base = document.get("base") or {}
    destination = document.get("destination") or {}
    key = RecoveryKey(
        schema=str(document["schema"]),
        benchmark=str(document.get("benchmark") or "unnamed"),
        target=str(document.get("target") or ""),
        metric=str(document.get("metric") or ""),
        base_total=int(base["total"]),
        base_families={str(k): int(v) for k, v in (base.get("families") or {}).items()},
        destination_total=int(destination["total"]),
        destination_families={str(k): int(v) for k, v in (destination.get("families") or {}).items()},
        levers=tuple(sorted(levers, key=lambda lever: lever.order)),
        program_identity=(str(document["program_identity"]) if document.get("program_identity") else None),
        reference=dict(document.get("reference") or {}),
        refusals=tuple(dict(r) for r in (document.get("refusals") or ()) if isinstance(r, Mapping)),
        provenance=dict(document.get("provenance") or {}),
        path=resolved,
    )
    _check_key_closes(key)
    return key


def _check_key_closes(key: RecoveryKey) -> None:
    """The decomposition must ACCOUNT for the reduction it claims, or it cannot attribute it.

    Two closures, both exact. A key whose steps do not sum to its own endpoints is describing a
    different reduction than the one it states, and every per-step weight computed from it would be
    wrong in a way no downstream check could see.
    """
    stepped = key.base_total - sum(lever.marginal for lever in key.levers)
    if stepped != key.destination_total:
        raise RecoveryKeyError(
            f"the steps do not close: base {key.base_total} minus the step marginals is {stepped}, "
            f"but the declared destination is {key.destination_total}"
        )
    for lever in key.levers:
        if sum(lever.family_delta.values()) != lever.marginal:
            raise RecoveryKeyError(
                f"step {lever.name!r}: its family deltas sum to "
                f"{sum(lever.family_delta.values())}, not its marginal {lever.marginal}"
            )


# ------------------------------------------------------------------------------------- admission


def admit_candidate(audit: Mapping[str, Any] | None) -> dict[str, Any]:
    """May this candidate be scored at all? FAIL CLOSED.

    The recovery score is a claim that a loop did the work UNAIDED. A candidate that read a
    withheld surface has not falsified that claim weakly, it has made it unmeasurable -- so it is
    refused rather than scored low, and a candidate with NO audit is refused for the same reason.
    An absent audit is the hollow positive this whole instrument exists to avoid: nothing was
    looked for, so nothing was found, and the result reads exactly like a clean run.

    The violation vocabulary is not this module's: it is
    :func:`merlin.common.access.audit_hit_is_violation`, which itself fails
    closed on an unrecognised hit kind. One vocabulary, so a new kind of leak disqualifies here the
    day it disqualifies there.
    """
    from merlin.common.access import audit_hit_is_violation

    if not isinstance(audit, Mapping):
        return {
            "admitted": False,
            "status": "refused_unaudited",
            "reason": (
                "no transcript audit was supplied. A recovery score claims the work was unaided; "
                "an unchecked run does not support that claim, and scoring it anyway would make a "
                "cheating run and a clean run produce the same record"
            ),
        }
    hits = audit.get("hits")
    if not isinstance(hits, (list, tuple)):
        return {
            "admitted": False,
            "status": "refused_unaudited",
            "reason": (
                f"the audit carries no `hits` sequence (got {type(hits).__name__}); it cannot be read as a verdict"
            ),
        }
    violations = [hit for hit in hits if audit_hit_is_violation(hit)]
    if violations:
        return {
            "admitted": False,
            "status": "refused_answer_access",
            "reason": (
                f"{len(violations)} audit hit(s) mean withheld content reached this candidate. A "
                "number reached by reading the answer is not a recovery of it"
            ),
            "violations": [dict(v) if isinstance(v, Mapping) else {"hit": repr(v)} for v in violations],
        }
    return {
        "admitted": True,
        "status": "admitted",
        "hits_reviewed": len(hits),
        "reason": "every audit hit is declared advisory; no withheld content reached this candidate",
    }


def admitted_levers(
    key: RecoveryKey, required_gate: str | None
) -> tuple[tuple[Lever, ...], tuple[dict[str, Any], ...]]:
    """Split the key's steps into those scored under ``required_gate`` and those excluded, with why.

    ``required_gate=None`` admits every step whose gate is KNOWN -- the permissive reading, for a
    run that has not declared one. It still excludes ``UNKNOWN``, because an unestablished gate is
    the one thing no reading may assume.
    """
    scored: list[Lever] = []
    excluded: list[dict[str, Any]] = []
    for lever in key.levers:
        if lever.gate == GATE_UNKNOWN:
            excluded.append(
                {
                    "lever": lever.name,
                    "gate": lever.gate,
                    "marginal": lever.marginal,
                    "reason": "this step's correctness gate was never established; UNKNOWN is not a pass",
                }
            )
        elif required_gate is not None and lever.gate != required_gate:
            excluded.append(
                {
                    "lever": lever.name,
                    "gate": lever.gate,
                    "marginal": lever.marginal,
                    "reason": (
                        f"this step clears {lever.gate!r}, not the run's declared {required_gate!r}; "
                        "a reduction bought by relaxing the correctness contract is not in the denominator"
                    ),
                }
            )
        else:
            scored.append(lever)
    return tuple(scored), tuple(excluded)


# --------------------------------------------------------------------------------------- scoring


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _family_keys(*maps: Mapping[str, int]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for mapping in maps:
        for name in mapping:
            seen.setdefault(str(name), None)
    return tuple(seen)


def score_recovery(
    candidate: Mapping[str, int],
    key: RecoveryKey,
    *,
    required_gate: str | None = None,
    audit: Mapping[str, Any] | None = None,
    candidate_label: str = "",
    candidate_total: int | None = None,
) -> dict[str, Any]:
    """How much of the directed reduction did ``candidate`` re-author?

    ``candidate`` is a per-family operation census of the candidate's emitted program -- the same
    census :func:`candidate_families` produces and the same one the key's endpoints are stated in.

    THE ATTRIBUTION, stated once so the number is readable. For each operation family ``f`` the key
    removed ``K_f`` operations across the scored steps and the candidate removed ``C_f``. The
    candidate's recovery IN THAT FAMILY is ``rho_f = clamp(C_f / K_f, 0, 1)`` -- clamped because
    removing more of a family than the reference did is a different program, not extra credit, and
    signed-consistently because a step that ADDS operations in a family is only re-authored by a
    candidate that also added them. Step ``l`` is then credited ``a_l = sum_f rho_f * delta_{l,f}``,
    and the headline is ``sum_l a_l / D`` where ``D = sum_l marginal_l``. The credits sum to the
    headline exactly, by construction, and this function asserts it rather than trusting it.

    WHAT THIS CANNOT DO, said in the record rather than in a comment: family attribution cannot
    tell two steps apart when they move the SAME families. Each step therefore carries
    ``identifiability`` -- the share of its movement that happens in families no other scored step
    touches. A step at 0.0 is credited entirely by inference from shared families, and its
    per-step number should be read as "consistent with", never "did".
    """
    verdict = admit_candidate(audit)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "benchmark": key.benchmark,
        "target": key.target,
        "metric": key.metric,
        "candidate": candidate_label or "UNLABELLED",
        "key": str(key.path) if key.path else "INLINE",
        "admission": verdict,
    }
    if not verdict["admitted"]:
        record["recovery"] = None
        record["consequence"] = (
            "no score is produced. A recovery fraction is a claim about unaided authoring, and "
            "this candidate does not support one"
        )
        return record

    scored, excluded = admitted_levers(key, required_gate)
    record["gate"] = {"required": required_gate or "ANY_KNOWN", "excluded_levers": list(excluded)}
    if not scored:
        record["recovery"] = None
        record["consequence"] = "every step was excluded by the declared gate; there is no denominator to score against"
        return record

    denominator = sum(lever.marginal for lever in scored)
    if denominator == 0:
        record["recovery"] = None
        record["consequence"] = "the admitted steps remove no operations in this metric; a ratio to zero is not a score"
        return record

    cand = {str(k): int(v) for k, v in candidate.items()}
    families = _family_keys(key.base_families, cand, *(lever.family_delta for lever in scored))
    key_removed = {f: sum(lever.family_delta.get(f, 0) for lever in scored) for f in families}
    cand_removed = {f: key.base_families.get(f, 0) - cand.get(f, 0) for f in families}
    rho: dict[str, float] = {}
    for f in families:
        k_f = key_removed[f]
        rho[f] = 0.0 if k_f == 0 else _clamp01(cand_removed[f] / k_f)

    # Which scored steps move a family at all -- the basis for identifiability.
    movers = {f: [lever.name for lever in scored if lever.family_delta.get(f, 0)] for f in families}

    per_lever: list[dict[str, Any]] = []
    attributed_total = 0.0
    for lever in scored:
        attributed = sum(rho[f] * delta for f, delta in lever.family_delta.items() if delta)
        attributed_total += attributed
        movement = lever.movement
        exclusive = sum(abs(d) for f, d in lever.family_delta.items() if d and len(movers[f]) == 1)
        row: dict[str, Any] = {
            "lever": lever.name,
            "order": lever.order,
            "gate": lever.gate,
            "marginal": lever.marginal,
            "weight": lever.marginal / denominator,
            "attributed": round(attributed, 3),
            "share": (attributed / lever.marginal) if lever.marginal else None,
            "identifiability": (exclusive / movement) if movement else None,
        }
        if not lever.marginal:
            row["note"] = (
                "this step moves ZERO operations in this metric, so its weight is zero and the "
                "benchmark cannot see whether the candidate re-authored it. That is a property of "
                "the metric, not evidence the step is worthless"
            )
        elif row["identifiability"] == 0.0:
            row["note"] = (
                "every family this step moves is also moved by another scored step, so its credit "
                "is inferred from shared families -- read it as 'consistent with', not 'did'"
            )
        if lever.note:
            row["key_note"] = lever.note
        per_lever.append(row)

    recovery = attributed_total / denominator
    # The closure the docstring promises, checked rather than trusted.
    assert abs(sum(row["attributed"] for row in per_lever) - recovery * denominator) < 1e-3

    total_now = int(candidate_total) if candidate_total is not None else sum(cand.values())
    proxy = (key.base_total - total_now) / denominator

    record.update(
        {
            "recovery": round(recovery, 6),
            "denominator": denominator,
            "denominator_basis": (
                "the sum of the marginal reductions of the steps admitted under the declared gate "
                "-- NOT the base-to-destination difference, which would include steps this run is "
                "not scored on"
            ),
            "levers": per_lever,
            "families": {
                f: {
                    "key_removed": key_removed[f],
                    "candidate_removed": cand_removed[f],
                    "recovered": round(rho[f], 6),
                }
                for f in families
                if key_removed[f] or cand_removed[f]
            },
            "proxy_total_recovery": round(proxy, 6),
            "proxy_divergence": round(proxy - recovery, 6),
            "proxy_licence": (
                "`proxy_total_recovery` is the naive (base - candidate) / denominator. It is "
                "reported only so the divergence is visible: a candidate that matches the total "
                "while missing the families removed DIFFERENT operations than the reference did, "
                "and the two numbers separating is the signal"
            ),
            "licence": (
                "operation counts are an emitted-program accounting, not cycles. A candidate that "
                "trades integer operations for floating-point ones can raise this score and lose "
                "time on hardware; a recovery fraction is evidence a loop found the same KIND of "
                "work to remove, never that it reached the same speed"
            ),
        }
    )
    return record


# -------------------------------------------------------------------------------------- feedback


def feedback(
    key: RecoveryKey,
    *,
    instruments: Sequence[str] = (),
    candidate: Mapping[str, int] | None = None,
    command_buffer: Mapping[str, Any] | None = None,
    gate_result: str | None = None,
) -> dict[str, Any]:
    """What the loop is LICENSED to see, under the instruments this run enabled.

    The ablation surface. Every channel here is a destination or a verdict; none is a method, and
    none names a step, because the step list is the plan and handing over the plan is handing over
    the answer. :func:`score_recovery` reads the step decomposition, this does not, and a test
    asserts that no subset of ``instruments`` ever emits a step name.
    """
    enabled = tuple(dict.fromkeys(str(i) for i in instruments))
    unknown = [i for i in enabled if i not in INSTRUMENTS]
    if unknown:
        raise RecoveryKeyError(f"unknown instrument(s) {unknown}; declared instruments are {list(INSTRUMENTS)}")
    out: dict[str, Any] = {
        "schema": "known_answer_recovery_feedback_v1",
        "benchmark": key.benchmark,
        "metric": key.metric,
        "instruments_enabled": list(enabled),
        "instruments_withheld": [i for i in INSTRUMENTS if i not in enabled],
    }
    if "destination_total" in enabled:
        out["destination_total"] = key.destination_total
        out["base_total"] = key.base_total
    if "destination_families" in enabled:
        out["destination_families"] = dict(key.destination_families)
        out["base_families"] = dict(key.base_families)
    if "structural_gap" in enabled:
        out["structural"] = _structural(key, command_buffer)
    if "refusal_brief" in enabled:
        out["refusals"] = [dict(r) for r in key.refusals]
    if "correctness_gate" in enabled:
        out["correctness_gate"] = gate_result or GATE_UNKNOWN
    if candidate is not None and "destination_total" in enabled:
        out["candidate_total"] = sum(int(v) for v in candidate.values())
    return out


def _structural(key: RecoveryKey, command_buffer: Mapping[str, Any] | None) -> dict[str, Any]:
    """The reference's emitted-structure distance -- a free, pre-hardware signal.

    This is :mod:`merlin.perf.target_reference`'s ``structural_gap``, which was complete, tested and
    uncalled. It is a legitimate instrument because the reference ledger is already the loop's
    declared destination: it says what an achieving program LOOKED like, not how to build one.
    """
    if command_buffer is None:
        return {"status": "no_command_buffer_supplied"}
    ref = key.reference
    if not ref.get("model") or not ref.get("design"):
        return {"status": "key_declares_no_reference"}
    from .target_reference import ReferenceError, find_reference, structural_gap

    try:
        name, reference = find_reference(
            str(ref["model"]),
            str(ref["design"]),
            program_identity=key.program_identity,
            device=(str(ref["device"]) if ref.get("device") else None),
        )
    except ReferenceError as exc:
        return {"status": "reference_unresolved", "reason": str(exc)}
    gap = dict(structural_gap(command_buffer, reference, program_identity=key.program_identity))
    gap["reference"] = name
    return gap


# ------------------------------------------------------------------------------ measuring a candidate


def candidate_families(artifact: Path | str) -> dict[str, Any]:
    """The per-family dynamic-operation census of one emitted host artifact.

    The same measurement the key's endpoints are stated in, so a candidate and the key are always
    counted by one instrument. ``status`` is propagated: an artifact whose control flow the
    analyzer cannot prove yields UNKNOWN rather than a total that silently counts a rolled loop
    once.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, func, llvm
    from xdsl.parser import Parser

    from .host_cfg_activity import analyze_host_cfg_activity

    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, llvm.LLVM, func.Func):
        ctx.load_dialect(dialect)
    module = Parser(ctx, Path(artifact).read_text()).parse_module()
    function = next(
        (op for op in module.walk() if op.name in ("llvm.func", "func.func") and op.regions and op.regions[0].blocks),
        None,
    )
    if function is None:
        return {"status": "UNKNOWN", "problems": ["no host function with a body in this artifact"], "families": {}}
    analysis = analyze_host_cfg_activity(function)
    families = analysis.get("dynamic_operations") or {}
    families = {str(k): int(v) for k, v in families.items() if isinstance(v, int)}
    return {
        "status": analysis.get("status", "UNKNOWN"),
        "problems": list(analysis.get("problems") or ()),
        "families": families,
        "total": sum(families.values()) if families else None,
    }


# ------------------------------------------------------------------------------------ minting a key


def mint_key(
    ledger: Iterable[Mapping[str, Any]],
    *,
    benchmark: str,
    target: str,
    metric: str,
    gates: Mapping[str, str] | None = None,
    program_identity: str | None = None,
    reference: Mapping[str, Any] | None = None,
    refusals: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a key document from a measured replay ledger.

    ``ledger`` is the ordered per-step measurement: the first entry is the base, each later entry
    is the cumulative state after one step. Every marginal and every family delta is DERIVED here
    from adjacent measurements, so no number in a key is typed by hand -- which is the difference
    between an answer key and a recollection of one. A step with no gate in ``gates`` is minted
    ``UNKNOWN`` and will be excluded from every scored denominator until someone establishes it.
    """
    rows = [dict(r) for r in ledger]
    if len(rows) < 2:
        raise RecoveryKeyError("a ledger needs a base measurement and at least one step")
    gate_map = {str(k): str(v) for k, v in (gates or {}).items()}

    def census(row: Mapping[str, Any]) -> dict[str, int]:
        fam = (row.get("ops") or {}).get("families") or row.get("families") or {}
        return {str(k): int(v) for k, v in fam.items() if isinstance(v, int)}

    def total(row: Mapping[str, Any]) -> int:
        value = (row.get("ops") or {}).get("total", row.get("total"))
        if not isinstance(value, int):
            raise RecoveryKeyError(f"ledger row {row.get('label')!r} carries no integer total")
        return value

    base_families, base_total = census(rows[0]), total(rows[0])
    levers: list[dict[str, Any]] = []
    previous_families, previous_total = base_families, base_total
    for order, row in enumerate(rows[1:], start=1):
        families, row_total = census(row), total(row)
        name = str(row.get("lever") or row.get("label") or f"step{order}")
        delta = {f: previous_families.get(f, 0) - families.get(f, 0) for f in _family_keys(previous_families, families)}
        levers.append(
            {
                "name": name,
                "order": order,
                "total_after": row_total,
                "marginal": previous_total - row_total,
                "family_delta": {f: v for f, v in delta.items() if v},
                "gate": gate_map.get(name, GATE_UNKNOWN),
                "note": str(row.get("note") or ""),
            }
        )
        previous_families, previous_total = families, row_total
    return {
        "schema": KEY_SCHEMA,
        "benchmark": benchmark,
        "target": target,
        "metric": metric,
        "program_identity": program_identity,
        "base": {"total": base_total, "families": base_families},
        "destination": {"total": previous_total, "families": previous_families},
        "levers": levers,
        "reference": dict(reference or {}),
        "refusals": [dict(r) for r in refusals],
        "provenance": dict(provenance or {}),
    }


# ------------------------------------------------------------------------------------------- CLI


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin-recovery", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser("measure", help="census one emitted host artifact")
    measure.add_argument("artifact")

    score = sub.add_parser("score", help="score a candidate census against the key")
    score.add_argument("--key", default=None)
    score.add_argument("--target", default=None)
    score.add_argument("--candidate", required=True, help="JSON from `measure`, or a raw family census")
    score.add_argument("--audit", default=None, help="JSON transcript-audit verdict; WITHOUT IT THE RUN IS REFUSED")
    score.add_argument("--gate", default=None)
    score.add_argument("--label", default="")

    mint = sub.add_parser("mint", help="derive a key from a measured replay ledger and file it as a product")
    mint.add_argument("--ledger", required=True, help="JSONL or JSON list: base first, then one row per step")
    mint.add_argument("--benchmark", required=True)
    mint.add_argument("--target", required=True)
    mint.add_argument("--metric", default="host_dynamic_operations")
    mint.add_argument("--gates", default=None, help="JSON map of step name -> the gate it CLEARED")
    mint.add_argument("--program-identity", default=None)
    mint.add_argument("--reference", default=None, help="JSON: model/design/device of the public ledger entry")
    mint.add_argument("--refusals", default=None, help="JSON list of dead-end conclusions (never their method)")
    mint.add_argument("--provenance", default=None, help="JSON: at minimum `answer_surfaces`, the material to mask")
    mint.add_argument("--version", type=int, default=1)

    show = sub.add_parser("feedback", help="what the loop may see under these instruments")
    show.add_argument("--key", default=None)
    show.add_argument("--target", default=None)
    show.add_argument("--instruments", default="")
    show.add_argument("--command-buffer", default=None)

    args = parser.parse_args(argv)

    if args.command == "measure":
        print(json.dumps(candidate_families(args.artifact), indent=1))
        return 0

    if args.command == "mint":
        from merlin.common.artifacts import new_product

        text = Path(args.ledger).read_text()
        rows = (
            json.loads(text)
            if text.lstrip().startswith("[")
            else [json.loads(line) for line in text.splitlines() if line.strip()]
        )
        document = mint_key(
            rows,
            benchmark=args.benchmark,
            target=args.target,
            metric=args.metric,
            gates=_read_json(args.gates) if args.gates else None,
            program_identity=args.program_identity,
            reference=_read_json(args.reference) if args.reference else None,
            refusals=_read_json(args.refusals) if args.refusals else (),
            provenance=_read_json(args.provenance) if args.provenance else None,
        )
        product = new_product(
            KEY_TOPIC,
            version=args.version,
            target=args.target,
            notes=(
                "ANSWER KEY for the known-answer recovery benchmark. Untracked by design and masked "
                "from every agent-under-test; see merlin.targetgen.sandbox.answer_surfaces."
            ),
        )
        out = product.add_artifact(KEY_FILENAME)
        out.write_text(yaml.safe_dump(document, sort_keys=False))
        product.write_manifest()
        # Prove the minted key is loadable and closes, here rather than at first use.
        load_key(out)
        print(str(out))
        return 0

    key = load_key(args.key, target=args.target)

    if args.command == "score":
        blob = _read_json(args.candidate)
        families = blob.get("families", blob) if isinstance(blob, Mapping) else {}
        record = score_recovery(
            families,
            key,
            required_gate=args.gate,
            audit=_read_json(args.audit) if args.audit else None,
            candidate_label=args.label,
            candidate_total=blob.get("total") if isinstance(blob, Mapping) else None,
        )
        print(json.dumps(record, indent=1))
        return 0 if record.get("recovery") is not None else 1

    instruments = [i.strip() for i in args.instruments.split(",") if i.strip()]
    command_buffer = _read_json(args.command_buffer) if args.command_buffer else None
    print(json.dumps(feedback(key, instruments=instruments, command_buffer=command_buffer), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
