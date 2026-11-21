"""Which PROGRAM a cycle count is about, resolved from stated facts rather than assumed.

WHY THIS EXISTS. :mod:`merlin.analysis.perf.search.references` already refuses to score a candidate against a
reference measured on another DESIGN -- "cycles are not comparable across designs" -- and that
refusal is right. It has an exact twin that the ledger did not make: cycles are not comparable
across PROGRAMS either, and three different programs in this project carry whole-model numbers for
the same network on the same device.

Measured, and this is the whole reason the module exists:

* one entry point takes the network's parameters as pointer arguments, its weights arrive already
  quantized, and each contraction's activation scale arrives as a calibrated constant argument;
* the same network's own capsule interface is a FLOAT program that a compiler must quantize
  itself -- not the first one repackaged, a different arithmetic with a different op census;
* a third program takes NO arguments at all -- parameters are linked in as constant blobs, the
  entry is a bare ``main``, its input quantizer runs at BUILD time and is in no measured number,
  and the figure it prints is the SUM OF PER-CALL CYCLE BRACKETS rather than one window.

A ratio between the first and the third was quoted widely as a compiler result. It is not one: the
numerator and the denominator are different programs, on different devices, measured over
different windows. And no compiler edit can move a candidate from one identity to another, because
the identity is a property of the lowering and of the CAPTURE it came from, not of the compiler's
quality. A gap that cannot be closed by the loop being scored is not a gap, it is a category error
with a number attached.

THREE STATES, NEVER TWO -- the rule :mod:`merlin.analysis.perf.measurement.authority` established for devices, for
the same reason. A program is NAMED, or it is ``UNKNOWN(reason)``. "We could not tell which program
this is" is not a softer "the programs differ": a caller that collapsed the two would discard
comparisons that are fine, and a caller that collapsed UNKNOWN into "same" would publish the ratio
this module exists to stop.

TWO AGREEING FACTS, NEVER ONE. An argument count alone does not name a program -- two lowerings can
land on the same arity by coincidence, and a reader who keys on it has keyed on a number, not on a
program. So an identity is resolved only when at least :data:`MIN_AGREEING_FACTS` of its declared
facts are STATED by the candidate and every stated fact agrees. Facts the candidate leaves out are
not held against it; they are recorded in ``confirmed_by`` so a reader can see how much evidence the
resolution actually had.

THE ROSTER IS DATA. The identities themselves are declared in the reference ledger
(``program_identities:``), not in this file: adding a fourth program identity must be a reviewed
edit to a contract file, and nothing here may know what any particular program is called. This
module knows only how to resolve facts against a declared roster and how to refuse.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from merlin.analysis.perf.contract.harvest import Observation
from merlin.analysis.programs.work_volume import work_from_command_buffer
from merlin.common import digest as _mdigest
from merlin.common.paths import contract_dir

# ---- program_identity --------------------------------------------------------------------------

#: How many of an identity's declared facts a candidate must STATE, and agree on, before the
#: identity is named. One fact is a coincidence waiting to happen: an argument count is a number two
#: unrelated lowerings can share, and the mistake this module exists to prevent was made by a reader
#: who had exactly one fact (a count) and treated it as a program.
MIN_AGREEING_FACTS = 2

#: The key a reference entry declares its program identity under.
REFERENCE_KEY = "program_identity"

#: The key the ledger declares the roster under.
ROSTER_KEY = "program_identities"

#: The name an entry states when the program behind its number is UNRECORDED. It is a DECLARATION,
#: not a resolution: a roster entry under this name carries no ``facts``, so :func:`resolve` can
#: never land on it, and an entry naming it is readable and unscoreable -- which is the third state.
UNKNOWN_IDENTITY = "UNKNOWN"


class ProgramIdentityError(ValueError):
    """A comparison was attempted between programs that are not the same program.

    Derives from ``ValueError`` and NOT from ``ReferenceError``-by-inheritance-only: callers that
    already catch ``ValueError`` around reference resolution keep working, and callers that want to
    distinguish this refusal from a missing reference can.
    """


class ProgramIdentity(dict):
    """The resolution, as data: ``name``, ``facts``, ``confirmed_by``, and ``reason`` when unresolved."""

    @property
    def resolved(self) -> bool:
        return bool(self.get("name"))


class Comparability(dict):
    """Whether two programs may be scored against each other: ``comparable`` plus a ``reason``."""

    @property
    def ok(self) -> bool:
        return bool(self.get("comparable"))


def _unknown(reason: str, **extra: Any) -> ProgramIdentity:
    return ProgramIdentity(name=None, facts={}, confirmed_by=(), reason=reason, **extra)


def declared_identities(ledger: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The roster the ledger declares, as ``name -> entry``.

    An empty roster is returned as empty rather than raised on: a ledger that declares no identities
    is a ledger whose entries cannot be checked, and saying so is the caller's job (the shipped
    ledger is gated by a test), not this function's.
    """
    roster = ledger.get(ROSTER_KEY) if isinstance(ledger, Mapping) else None
    if not isinstance(roster, Mapping):
        return {}
    return {str(name): dict(entry) for name, entry in roster.items() if isinstance(entry, Mapping)}


def _declared_facts(entry: Mapping[str, Any]) -> dict[str, Any]:
    facts = entry.get("facts")
    return dict(facts) if isinstance(facts, Mapping) else {}


def _stated(facts: Mapping[str, Any] | None) -> dict[str, Any]:
    """The facts a candidate actually states -- ``None`` values are NOT statements.

    A field present and null is the honest way to say "I do not know this about myself", and folding
    it into a statement would let a program resolve on evidence it never gave.
    """
    if not isinstance(facts, Mapping):
        return {}
    return {str(key): value for key, value in facts.items() if value is not None}


def resolve(facts: Mapping[str, Any] | None, roster: Mapping[str, Any]) -> ProgramIdentity:
    """Name the program whose declared facts these are, or ``UNKNOWN(reason)``.

    Resolution, and every step can refuse:

    1. the candidate states at least :data:`MIN_AGREEING_FACTS` facts;
    2. the roster declares at least one identity;
    3. for exactly one identity, every fact the candidate states that the identity also declares
       AGREES, and at least :data:`MIN_AGREEING_FACTS` of them do so;
    4. no second identity clears step 3 -- two matches is a ROSTER DEFECT and is reported rather
       than resolved by picking one, because the same facts would then name two programs.

    A fact the candidate states that NO identity declares is not an error: the roster describes what
    distinguishes the identities, not everything a program can say about itself.
    """
    stated = _stated(facts)
    if len(stated) < MIN_AGREEING_FACTS:
        return _unknown(
            f"the program states {len(stated)} fact(s) ({', '.join(sorted(stated)) or 'none'}) and "
            f"{MIN_AGREEING_FACTS} agreeing facts are required to name a program; one fact is a "
            "number two unrelated lowerings can share, which is how a program gets misidentified"
        )

    declared = {str(name): dict(entry) for name, entry in (roster or {}).items() if isinstance(entry, Mapping)}
    if not declared:
        return _unknown("the roster declares no program identities, so there is nothing to resolve against")

    matches: dict[str, tuple[str, ...]] = {}
    disagreements: dict[str, list[str]] = {}
    for name, entry in declared.items():
        expected = _declared_facts(entry)
        if not expected:
            continue
        agreeing: list[str] = []
        conflicts: list[str] = []
        for key, value in expected.items():
            if key not in stated:
                continue
            if stated[key] == value:
                agreeing.append(key)
            else:
                conflicts.append(f"{key}={stated[key]!r} (declared {value!r})")
        if conflicts:
            disagreements[name] = conflicts
        elif len(agreeing) >= MIN_AGREEING_FACTS:
            matches[name] = tuple(sorted(agreeing))

    if not matches:
        detail = "; ".join(f"{name}: {', '.join(rows)}" for name, rows in sorted(disagreements.items()))
        return _unknown(
            "no declared program identity matches these facts"
            + (f" -- {detail}" if detail else f"; the roster declares {sorted(declared)}"),
            stated_facts=dict(stated),
        )
    if len(matches) > 1:
        return _unknown(
            f"these facts match more than one declared program identity ({', '.join(sorted(matches))}); "
            "one set of facts cannot name two programs -- the roster does not distinguish them",
            stated_facts=dict(stated),
        )

    name, agreed = next(iter(matches.items()))
    return ProgramIdentity(
        name=name,
        facts=dict(stated),
        confirmed_by=agreed,
        unconfirmed=tuple(sorted(set(_declared_facts(declared[name])) - set(agreed))),
    )


def identity_of_reference(reference: Mapping[str, Any]) -> ProgramIdentity:
    """The identity a reference entry DECLARES, or ``UNKNOWN(reason)`` when it declares none.

    A reference states its identity by name -- it was measured, so the program that produced it is
    known and does not have to be inferred from a fact bundle. An entry that names none is unknown,
    never assumed to share the identity of whatever it is about to be compared with.
    """
    name = str((reference or {}).get(REFERENCE_KEY) or "").strip()
    if not name:
        return _unknown(
            "the reference states no `program_identity`, so which program produced its cycle count "
            "is unrecorded; a number whose program is unknown cannot be scored against, because the "
            "candidate may not be able to produce that program at all"
        )
    if name == UNKNOWN_IDENTITY:
        # NOT a name that can match another one. Two entries both declaring UNKNOWN are not two
        # entries of the same program -- they are two entries nobody identified, and treating the
        # label as an identity would make the honest declaration the most permissive thing in the
        # file. This is the difference between three states and two wearing three labels.
        return _unknown(
            "the reference declares its program identity as UNKNOWN: the program behind this number "
            "is unrecorded, which is a statement that it cannot be scored against, not a program"
        )
    return ProgramIdentity(name=name, facts={}, confirmed_by=("declared_by_the_reference",))


def comparability(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Comparability:
    """Whether a candidate program and a reference program may be scored against each other.

    Takes two :class:`ProgramIdentity` results (or anything with ``name``/``reason``). Comparable
    only when BOTH are named and the names are equal; unknown on either side is refused with the
    reason that side gave, so the refusal says what is missing rather than that something is.
    """
    candidate_name = str((candidate or {}).get("name") or "")
    reference_name = str((reference or {}).get("name") or "")
    if not reference_name:
        return Comparability(
            comparable=False,
            candidate=candidate_name or None,
            reference=None,
            reason="the reference's program identity is unknown: "
            + str((reference or {}).get("reason") or "no reason recorded"),
        )
    if not candidate_name:
        return Comparability(
            comparable=False,
            candidate=None,
            reference=reference_name,
            reason=f"the reference was measured on the {reference_name!r} program and the candidate's "
            "program identity is unknown: " + str((candidate or {}).get("reason") or "no reason recorded"),
        )
    if candidate_name != reference_name:
        return Comparability(
            comparable=False,
            candidate=candidate_name,
            reference=reference_name,
            reason=f"the candidate is a {candidate_name!r} program and the reference was measured on a "
            f"{reference_name!r} program; cycles are not comparable across program identities any more "
            "than across designs, and no compiler edit moves a candidate between them -- the identity "
            "is a property of the lowering, not of its quality",
        )
    return Comparability(comparable=True, candidate=candidate_name, reference=reference_name, reason="")


def require_comparable(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Comparability:
    """:func:`comparability`, raising :class:`ProgramIdentityError` instead of returning a refusal.

    The raising form is for call sites that RETURN A SCORE -- a per-field gap, a ratio -- where a
    refusal record could be read past. Call sites that return a record the loop reads should use
    :func:`comparability` and embed the refusal, because a raise into a path wrapped in
    ``try/except`` is indistinguishable from nothing happening.
    """
    verdict = comparability(candidate, reference)
    if not verdict.ok:
        raise ProgramIdentityError(verdict["reason"])
    return verdict


# ---- target_reference --------------------------------------------------------------------------
# Score a candidate against a MEASURED reference point, so the loop has a destination.
#
# An authoring loop that sees only its own last revision can tell whether it moved, never whether it
# moved far enough or in the right direction. Measured: a campaign optimizing one collapsed host
# operation count sealed a round at -2 operations -- an estimated +0.000% of the window -- while the
# program it was optimizing sat far from a number the same compiler family had already reached by
# hand. The number was known. It was simply not in the loop.
#
# This module exposes the reference as a GAP: per-field structural distance plus an estimated cycle
# distance. Two properties are deliberate.
#
# **The gap is signed and named.** `mesh_regions: -12` means twelve regions the reference puts on the
# matrix unit are somewhere else, and it says which field to look at. A scalar score would say only
# "worse".
#
# **A structural match is never reported as a cycle match.** Reaching the reference's command and
# region census means the candidate emits the same SHAPE of program, which is evidence and not a
# measurement -- the reference's own regression floor shares a byte-identical command buffer with it
# and still ran 29% slower, because the spread lived in host-lane codegen. So `cycles` carries
# `status: estimated` and the structural verdict is reported separately from it.
#
# Target-neutral by construction: every fact comes from the ledger, which is data, and the design is
# a parameter that must MATCH. Cycles are not comparable across designs (the same capsule reads 510
# on one and 317 on another), so a mismatched design is refused rather than approximated.
#
# **And cycles are not comparable across PROGRAMS either.** That refusal is the exact twin of the
# design one and it was missing until 2026-09-19. Three programs in this project carry whole-model
# numbers for one network on one device -- a static-calibrated recapture entry, the capsule's own
# lowering of the same network, and a standalone C program with its parameters linked in -- and a
# ratio between two of them was quoted for weeks as a compiler result. It is not one: no compiler
# edit moves a candidate between identities, because the identity is a property of the lowering and
# of the capture it came from. So :func:`structural_gap` and :func:`find_reference` REFUSE across
# program identities, and :func:`estimate_cycles` returns a refusal record rather than raising,
# because its result is embedded in a loop whose ``try/except`` would swallow a raise into silence.
# See :mod:`merlin.analysis.perf.search.references`.
#
# **`design` is no longer sufficient to name a device.** Two registered bitstreams elaborate the same
# configuration string onto the same board and are different devices. Every measured ledger entry now
# states ``device`` -- the registry artifact name, which is unique per device -- and ``find_reference``
# takes it as the stronger key. It is optional only so existing callers keep working, and a lookup
# that settles for ``design`` says so in the record rather than silently.

#: The tracked ledger of measured reference points, relative to the merlin package directory.
REFERENCE_LEDGER_NAME = "contract/perf_reference_targets.yaml"

#: Structural fields compared against a reference, in report order. Each is a count the emitted
#: command buffer declares, so the comparison needs no simulator.
_STRUCTURAL_FIELDS = ("commands", "mesh_regions", "host_lane_regions", "weight_prepack_recipes", "kernel_abi_args")


class ReferenceError(ValueError):
    """A reference could not be resolved, or was resolved against a design it does not describe."""


def load_ledger(path: Path | None = None) -> dict[str, Any]:
    """The whole ledger document -- ``references`` AND the ``program_identities`` roster.

    Separate from :func:`load_references` because the roster is not a reference and a caller that
    wants to resolve a candidate's identity needs it without pretending it is one.
    """
    ledger = Path(path) if path is not None else contract_dir() / Path(REFERENCE_LEDGER_NAME).relative_to("contract")
    document = yaml.safe_load(ledger.read_text())
    if not isinstance(document, Mapping) or not isinstance(document.get("references"), Mapping):
        raise ReferenceError(f"{ledger} declares no `references` mapping")
    return dict(document)


def load_references(path: Path | None = None) -> dict[str, Any]:
    """The reference ledger, as data."""
    return dict(load_ledger(path)["references"])


def load_program_identities(path: Path | None = None) -> dict[str, Any]:
    """The declared program-identity roster, as data. Empty when the ledger declares none."""

    return declared_identities(load_ledger(path))


def find_reference(
    model: str,
    design: str,
    *,
    references: Mapping[str, Any] | None = None,
    status: str = "achieved",
    program_identity: str | None = None,
    device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """The best measured reference for ``model`` on ``design`` -- the FEWEST cycles achieved.

    Refuses on a design mismatch instead of falling back to another design's number, because a
    cycle count from a different design is not a weaker answer, it is a wrong one. ``device`` and
    ``program_identity`` narrow the same way and refuse the same way:

    * ``device`` is the STRONGER form of ``design``. Two registered bitstreams can elaborate one
      configuration string onto one board and be different devices, so a ``design`` that survives
      to more than one ``device`` is refused rather than resolved by picking the fastest.
    * ``program_identity`` refuses a reference measured on a program the caller cannot produce.
      Passing it is how a caller says which program its candidate IS; omitting it leaves that
      unstated, which is weaker evidence and is reported by the caller's record, not here.
    """
    table = dict(references) if references is not None else load_references()
    candidates = {
        name: entry
        for name, entry in table.items()
        if isinstance(entry, Mapping)
        and entry.get("model") == model
        and entry.get("status") == status
        and isinstance((entry.get("measured") or {}).get("whole_model_cycles"), int)
    }
    if not candidates:
        known = sorted({str(entry.get("model")) for entry in table.values() if isinstance(entry, Mapping)})
        raise ReferenceError(
            f"no reference with status {status!r} carries a whole-model cycle count for model "
            f"{model!r}; the ledger describes {known}"
        )
    on_design = {name: entry for name, entry in candidates.items() if entry.get("design") == design}
    if not on_design:
        designs = sorted({str(entry.get("design")) for entry in candidates.values()})
        raise ReferenceError(
            f"model {model!r} has references on {designs} but none on design {design!r}; cycles "
            f"are not comparable across designs, so there is no reference to score against"
        )

    if device is not None:
        on_device = {name: entry for name, entry in on_design.items() if entry.get("device") == device}
        if not on_device:
            devices = sorted({str(entry.get("device")) for entry in on_design.values()})
            raise ReferenceError(
                f"design {design!r} has references on devices {devices} but none on {device!r}; a "
                "configuration name is not a device and cycles are not comparable across devices"
            )
        on_design = on_device
    else:
        devices = {str(entry.get("device") or "UNSTATED") for entry in on_design.values()}
        if len(devices) > 1:
            raise ReferenceError(
                f"design {design!r} does not pin a device -- its references span {sorted(devices)}; "
                "two bitstreams can elaborate one configuration string and be different devices, so "
                "pass `device=` rather than letting a design string select across them"
            )

    if program_identity is not None:
        on_identity = {
            name: entry for name, entry in on_design.items() if entry.get("program_identity") == program_identity
        }
        if not on_identity:
            identities = sorted({str(entry.get("program_identity") or "UNSTATED") for entry in on_design.values()})
            raise ReferenceError(
                f"model {model!r} has references of program identity {identities} but none of "
                f"{program_identity!r}; cycles are not comparable across program identities any more "
                "than across designs, and no compiler edit moves a candidate between them"
            )
        on_design = on_identity
    else:
        # The device rule, mirrored -- and this one has teeth. `min` over cycles is a race the most
        # different program wins: the surviving set here spans a callable compiler output and a
        # hand-written vendor benchmark that runs a BATCH OF FOUR and publishes it divided by four,
        # and the vendor entry is an order of magnitude smaller, so an unqualified lookup would
        # hand the authoring loop a destination whose batch size alone puts it out of reach. The
        # fastest number is exactly the one most likely to be a different program.
        identities = {str(entry.get("program_identity") or "UNSTATED") for entry in on_design.values()}
        if len(identities) > 1:
            raise ReferenceError(
                f"model {model!r} on device {device or '(any)'!r} has references of more than one "
                f"program identity ({sorted(identities)}); selecting the fewest cycles across them "
                "would pick whichever program is least like the candidate, so pass "
                "`program_identity=` rather than letting the ledger choose"
            )

    name = min(on_design, key=lambda key: on_design[key]["measured"]["whole_model_cycles"])
    return name, dict(on_design[name])


def find_reference_for_capsule(
    capsule: str, *, references: Mapping[str, Any] | None = None, status: str = "achieved"
) -> tuple[str, dict[str, Any]] | None:
    """The reference describing ``capsule``, or ``None`` when nothing measured describes it.

    ``None`` rather than a raise, and the caller is expected to RECORD it: an objective with no
    measured reference is the single most important thing a driver can say about itself, and the
    campaign this was written for was optimizing exactly such an objective for seventeen
    iterations without that fact appearing anywhere in the loop.
    """
    table = dict(references) if references is not None else load_references()
    matches = {
        name: dict(entry)
        for name, entry in table.items()
        if isinstance(entry, Mapping)
        and entry.get("status") == status
        and capsule in (entry.get("capsules") or ())
        and isinstance((entry.get("measured") or {}).get("whole_model_cycles"), int)
    }
    if not matches:
        return None
    devices = {str(entry.get("device") or "UNSTATED") for entry in matches.values()}
    if len(devices) > 1:
        raise ReferenceError(
            f"capsule {capsule!r} is named by references on devices {sorted(devices)}; picking the "
            "fastest would cross devices silently, which is the mistake `device` exists to stop"
        )
    name = min(matches, key=lambda key: matches[key]["measured"]["whole_model_cycles"])
    return name, matches[name]


def _candidate_identity(program_identity: str | None) -> ProgramIdentity:
    """A candidate's stated identity, or UNKNOWN with the reason it is unknown.

    A caller that states nothing gets UNKNOWN rather than a pass. That is the whole three-state
    rule: the candidate is the one thing the caller KNOWS about, so leaving it unstated is a
    decision not to say, not an absence of the fact.
    """
    name = str(program_identity or "").strip()
    if not name:
        return ProgramIdentity(
            name=None,
            facts={},
            confirmed_by=(),
            reason="the caller stated no `program_identity` for the candidate, so which program it "
            "is was never established; state it (the ledger's `program_identities` roster names "
            "the declared ones) rather than letting the comparison assume it",
        )
    if name == UNKNOWN_IDENTITY:
        return ProgramIdentity(
            name=None,
            facts={},
            confirmed_by=(),
            reason="the caller stated the candidate's program identity as UNKNOWN, which says the "
            "program is unrecorded -- it is not a program two entries can share",
        )
    return ProgramIdentity(name=name, facts={}, confirmed_by=("stated_by_the_caller",))


def _emitted_structure(command_buffer: Mapping[str, Any]) -> dict[str, int]:
    """The candidate's structural census, from its command buffer alone."""
    params = command_buffer.get("params")
    params = params if isinstance(params, Mapping) else {}
    abi = command_buffer.get("kernel_abi")
    abi = abi if isinstance(abi, Mapping) else {}

    def count(value: Any) -> int:
        return len(value) if isinstance(value, (list, tuple)) else 0

    return {
        "commands": count(command_buffer.get("commands")),
        "mesh_regions": count(params.get("mesh_regions")),
        "host_lane_regions": count(params.get("host_lane_regions")),
        "weight_prepack_recipes": count(params.get("weight_prepack_recipes")),
        "kernel_abi_args": count(abi.get("args")),
    }


def _opcode_census(command_buffer: Mapping[str, Any]) -> dict[str, int]:
    census: dict[str, int] = {}
    for command in command_buffer.get("commands") or ():
        if isinstance(command, Mapping):
            opcode = str(command.get("opcode"))
            census[opcode] = census.get(opcode, 0) + 1
    return census


def structural_gap(
    command_buffer: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    program_identity: str | None = None,
) -> dict[str, Any]:
    """Signed per-field distance from ``command_buffer`` to the reference's emitted structure.

    Negative means the candidate has FEWER than the reference. Only the reference's declared fields
    are compared -- a reference that records no structure yields no structural verdict rather than
    a vacuous pass.

    RAISES :class:`ProgramIdentityError` unless the candidate and the reference are the same
    program. This is the call site where a cross-identity comparison does its damage, and it did:
    scored against a reference whose lowering puts 54 contractions on the matrix unit, a candidate
    compiled from a capsule whose own interface is float reads ``mesh_regions: -53`` -- a number
    that looks exactly like a schedule the compiler failed to find, and is in fact the distance
    between two different programs, closable only by re-capturing the model. A signed per-field gap
    is the most quotable thing this module produces, so it is the one that must refuse rather than
    annotate: an UNKNOWN identity is refused too, because "we could not tell" is not "they match".
    """
    require_comparable(_candidate_identity(program_identity), identity_of_reference(reference))
    declared = reference.get("emitted_structure")
    if not isinstance(declared, Mapping):
        return {"status": "reference_declares_no_structure", "fields": {}, "matches": None}
    actual = _emitted_structure(command_buffer)
    fields = {
        name: {"candidate": actual[name], "reference": int(declared[name]), "gap": actual[name] - int(declared[name])}
        for name in _STRUCTURAL_FIELDS
        if isinstance(declared.get(name), int)
    }
    opcodes: dict[str, Any] = {}
    if isinstance(declared.get("opcodes"), Mapping):
        census = _opcode_census(command_buffer)
        for opcode, expected in declared["opcodes"].items():
            got = census.get(str(opcode), 0)
            opcodes[str(opcode)] = {"candidate": got, "reference": int(expected), "gap": got - int(expected)}
        for opcode, got in census.items():
            opcodes.setdefault(opcode, {"candidate": got, "reference": 0, "gap": got})
    unmatched = sorted(
        [name for name, row in fields.items() if row["gap"]]
        + [f"opcode:{name}" for name, row in opcodes.items() if row["gap"]]
    )
    return {
        "status": "derived",
        "fields": fields,
        "opcodes": opcodes,
        "matches": not unmatched,
        "unmatched": unmatched,
        "licence": (
            "a structural match means the candidate emits the same SHAPE of program, not "
            "that it runs as fast: the reference's own regression floor shares a "
            "byte-identical command buffer with it and ran 29% slower"
        ),
    }


def estimate_cycles(
    host_dynamic_operations: int,
    reference: Mapping[str, Any],
    *,
    baseline_host_dynamic_operations: int | None = None,
    program_identity: str | None = None,
) -> dict[str, Any]:
    """Estimated whole-model cycles for a candidate, anchored on the reference's measurement.

    The anchor is cycles-per-host-operation from a run that MEASURED both quantities, scaled by the
    host lane's share of the window. It prices every operation family alike, which is exactly why
    the returned record carries the caveat rather than only a number.

    RETURNS a refusal record -- it does not raise -- when the candidate and the reference are not
    the same program. Deliberately the opposite choice from :func:`structural_gap`, for a reason
    that is about where the result goes rather than about how bad the error is: this record is
    embedded in the feedback an authoring loop reads, and that loop wraps its analysis in
    ``try/except`` so nothing can stall a run. A raise there would become silence, and a broken
    comparison would be indistinguishable from a comparison nobody asked for. A loud record the
    agent READS is the stronger instrument; the raising form lives in :func:`structural_gap`.
    """
    verdict = comparability(_candidate_identity(program_identity), identity_of_reference(reference))
    if not verdict.ok:
        return {
            "status": "not_comparable_across_program_identities",
            "candidate_program_identity": verdict.get("candidate"),
            "reference_program_identity": verdict.get("reference"),
            "reason": verdict["reason"],
            "consequence": (
                "no cycle estimate is produced. Scaling this candidate's host operations onto that "
                "reference's measured window would price work the two programs do not share, and the "
                "resulting distance would be reported as an optimization gap the loop cannot close"
            ),
        }
    measured = reference.get("measured")
    measured = measured if isinstance(measured, Mapping) else {}
    anchor = measured.get("cycles_per_host_operation_anchor")
    share = measured.get("host_share_lower_bound")
    target = measured.get("whole_model_cycles")
    if not isinstance(anchor, (int, float)) or not isinstance(target, int):
        return {"status": "reference_carries_no_cycle_anchor"}
    if baseline_host_dynamic_operations:
        # A RELATIVE estimate against a measured point is the defensible form: the ratio of host
        # operations, applied to the host share of a measured window. It needs no claim that the
        # anchor prices this candidate's op mix correctly, only that the mix did not change much.
        share_value = float(share) if isinstance(share, (int, float)) else 0.93
        ratio = host_dynamic_operations / baseline_host_dynamic_operations
        estimated = target * (1.0 - share_value + share_value * ratio)
        basis = "relative_to_reference_measurement"
    else:
        estimated = float(anchor) * host_dynamic_operations
        basis = "absolute_from_anchor"
    return {
        "status": "estimated",
        "basis": basis,
        "estimated_whole_model_cycles": round(estimated),
        "reference_whole_model_cycles": target,
        "cycles_above_reference": round(estimated) - target,
        "fraction_of_reference": round(estimated / target, 6) if target else None,
        "licence": (
            "an estimate, never a measurement -- the anchor is a whole-window average and "
            "prices every host operation family identically, so a revision trading integer "
            "operations for floating-point ones can improve this number and lose cycles"
        ),
    }


def score_against_reference(
    command_buffer: Mapping[str, Any],
    *,
    model: str,
    design: str,
    host_dynamic_operations: int | None = None,
    baseline_host_dynamic_operations: int | None = None,
    references: Mapping[str, Any] | None = None,
    program_identity: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """The whole gap record for one candidate: which reference, structural distance, cycle distance.

    ``program_identity`` and ``device`` are the candidate's own; both are carried into the record so
    a reader can see what the score was licensed by, including when it settled for less.
    """
    name, reference = find_reference(
        model, design, references=references, program_identity=program_identity, device=device
    )
    record: dict[str, Any] = {
        "schema": "perf_reference_gap_v1",
        "reference": name,
        "model": model,
        "design": design,
        "device": device or reference.get("device") or "UNSTATED",
        "program_identity": {
            "candidate": program_identity or "UNSTATED",
            "reference": reference.get("program_identity") or "UNSTATED",
            "confirmed_by": "stated by the caller" if program_identity else "NOT stated by the caller",
        },
        "structural": structural_gap(command_buffer, reference, program_identity=program_identity),
    }
    if isinstance(host_dynamic_operations, int):
        record["cycles"] = estimate_cycles(
            host_dynamic_operations,
            reference,
            baseline_host_dynamic_operations=baseline_host_dynamic_operations,
            program_identity=program_identity,
        )
    for note in reference.get("caveats") or ():
        record.setdefault("caveats", []).append(str(note))
    return record


# ---- historical_reference ----------------------------------------------------------------------
# Bind existing harvester observations to public bytes without promoting their fidelity.
#
# The host supplies the explicitly public workload allowlist and pinned raw artifacts. This module
# does not discover files, read hidden grades, execute a target, fit rates, or authorize target timing.
# Unlike an isolated-latency bound, a historical program count needs no assumption that removing
# contention can only reduce latency. It remains a reference for that program and measurement window.

_digest = _mdigest.sha256_bytes


def _document_digest(doc: Mapping[str, Any]) -> str:
    return _digest(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


_sha = _mdigest.is_sha256


def _object(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON field in historical evidence")
            out[key] = value
        return out

    doc = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(doc, dict):
        raise ValueError("historical evidence must be a JSON object")
    _document_digest(doc)  # Reject non-finite values, even outside the selected tier.
    return doc


def bind_historical_reference(
    observation: Observation,
    *,
    target: str,
    receipt: bytes,
    receipt_sha256: str,
    receipt_path: str,
    public_workloads: set[str] | frozenset[str],
    artifacts: Mapping[str, bytes],
    artifact_sha256: Mapping[str, str],
    executable_names: Sequence[str],
    harness_name: str,
    command_buffer_name: str | None = None,
) -> dict[str, Any]:
    """Recheck an admitted harvester row against exact public receipt and generated bytes.

    ``public_workloads`` is host-derived from the frozen public grade, not candidate input.
    Artifact hashes establish retained bytes, not an independent proof that this ELF was executed.
    Historical execution lineage remains named explicitly; no old certificate is rewritten.
    """
    if not isinstance(target, str) or not target or not receipt_path:
        raise ValueError("target and public receipt identity are required")
    if not isinstance(receipt, bytes) or not _sha(receipt_sha256) or _digest(receipt) != receipt_sha256:
        raise ValueError("historical receipt bytes differ from their host pin")
    doc = _object(receipt)
    if (
        doc.get("label") not in {"public", "dev"}
        or doc.get("capsule") not in public_workloads
        or doc.get("capsule") != observation.workload
    ):
        raise ValueError("observation is not the explicitly allowed public workload")
    if doc.get("target") not in (None, target):
        raise ValueError("historical receipt target differs from the host-bound target")
    stages = doc.get("tiers")
    row = stages.get(observation.stage) if isinstance(stages, Mapping) else None
    cycles = row.get("cycles") if isinstance(row, Mapping) else None
    if (
        observation.quantity != "total_cycles"
        or observation.unit != "cycles"
        or observation.status != "pass"
        or not isinstance(row, Mapping)
        or row.get("status") != "pass"
        or isinstance(cycles, bool)
        or not isinstance(cycles, (int, float))
        or not math.isfinite(cycles)
        or cycles <= 0
        or isinstance(observation.value, bool)
        or observation.value != cycles
    ):
        raise ValueError("historical observation does not match a positive passing raw cycle row")
    if (
        set(artifacts) != set(artifact_sha256)
        or not executable_names
        or isinstance(executable_names, (str, bytes))
        or len(set(executable_names)) != len(executable_names)
    ):
        raise ValueError("exact artifact pins and unique executable names are required")
    required = {*executable_names, harness_name}
    if command_buffer_name is not None:
        required.add(command_buffer_name)
    if not required <= set(artifacts):
        raise ValueError("historical executable, harness or command buffer bytes are missing")
    for name, raw in artifacts.items():
        if not isinstance(raw, bytes) or not _sha(artifact_sha256[name]) or _digest(raw) != artifact_sha256[name]:
            raise ValueError("historical artifact bytes differ from their host pin")

    work: dict[str, Any] | None = None
    command_buffer = None
    work_location = None
    if command_buffer_name is not None:
        command_buffer = _object(artifacts[command_buffer_name])
        work_location = command_buffer_name
    elif isinstance(doc.get("command_buffer_artifact"), Mapping):
        from merlin.analysis.programs.command_buffers import REFERENCE_KEY, load_command_buffer

        embedded = doc["command_buffer_artifact"]
        # A stored buffer is digest-checked as it is read (CommandBufferIntegrityError is a ValueError).
        loaded = load_command_buffer(embedded)
        if isinstance(loaded, Mapping):
            command_buffer = loaded
            if embedded.get("artifact_sha256") != _document_digest(command_buffer):
                raise ValueError("embedded historical command buffer identity changed")
            work_location = (
                "receipt#/command_buffer_artifact/command_buffer"
                if isinstance(embedded.get("command_buffer"), Mapping)
                else f"receipt#/command_buffer_artifact/{REFERENCE_KEY}"
            )
    if command_buffer is not None:
        work = work_from_command_buffer(command_buffer).to_dict()
        historical_work = doc.get("work_volume") or {}
        if historical_work.get("artifact_sha256") != work["artifact_sha256"]:
            raise ValueError("retained command buffer does not match historical work identity")
    sim = row.get("sim_provenance") or {}
    engine_sha = sim.get("sha256")
    config_sha = row.get("engine_config_sha256") or sim.get("config_sha256")
    counter_sha = row.get("counter_semantics_sha256")
    missing = [
        name
        for name, value in (
            ("engine_binary", engine_sha),
            ("engine_configuration", config_sha),
            ("counter_semantics", counter_sha),
        )
        if not _sha(value)
    ]
    # Raw declared fields are evidence to inspect, not a host-verified ROI contract.
    missing.extend(("host_verified_warm_window", "physical_byte_counters", "independent_timing_validation"))
    if work is None or work["exact_macs"] is None:
        missing.append("exact_command_work")
    hardware = doc.get("toolchain_shas") or {}
    if not isinstance(hardware, Mapping):
        raise ValueError("historical toolchain identities must be a mapping")
    result = {
        "schema": "historical_program_reference_v1",
        "target": target,
        "submission": observation.submission,
        "workload": observation.workload,
        "stage": observation.stage,
        "declared_measurement_tier": observation.tier,
        "status": "pass",
        "cycles": cycles,
        "unit": "cycles",
        "engine": row.get("engine"),
        "engine_binary_sha256": engine_sha,
        "engine_configuration_sha256": config_sha,
        "counter_semantics_sha256": counter_sha,
        "toolchain_shas": dict(hardware),
        "receipt_path": receipt_path,
        "receipt_sha256": receipt_sha256,
        "artifact_sha256": dict(artifact_sha256),
        "executable_sha256": {name: artifact_sha256[name] for name in executable_names},
        "harness_sha256": artifact_sha256[harness_name],
        "retained_asset_binding": "host-pinned retained bytes; historical execution lineage is separate",
        "carried": dict(row.get("carried") or {}),
        "window": "UNKNOWN",
        "declared_measurement_contract": row.get("measurement_contract"),
        "work": work,
        "work_location": work_location,
        "work_scope": "receipt-bound compiler command buffer, not hardware work counters",
        "physical_moved_bytes": None,
        "physical_movement_basis": "UNKNOWN",
        "usable_as": "historical_engine_relative_proxy" if _sha(engine_sha) else "diagnostic_only",
        "target_cycle_authority": False,
        "warm_calibration": False,
        "physical_roofline_complete": False,
        "full_model_cycle_ordering": "UNKNOWN",
        "missing_contracts": missing,
    }
    result["reference_sha256"] = _document_digest(result)
    return result


def reference_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compact agent-facing inventory; no pooling into a latency, peak, rank or fitted constant."""
    unique: dict[str, Mapping[str, Any]] = {}
    for record in records:
        raw = dict(record)
        identity = raw.pop("reference_sha256", None)
        if (
            raw.get("schema") != "historical_program_reference_v1"
            or identity != _document_digest(raw)
            or raw.get("target_cycle_authority") is not False
            or raw.get("warm_calibration") is not False
        ):
            raise ValueError("historical reference identity/authority changed")
        unique[identity] = record
    groups: dict[str, dict[str, Any]] = {}
    for identity, record in sorted(unique.items()):
        group_identity = {
            key: record[key]
            for key in (
                "target",
                "engine",
                "engine_binary_sha256",
                "engine_configuration_sha256",
                "counter_semantics_sha256",
                "window",
                "toolchain_shas",
            )
        }
        key = _document_digest(group_identity)
        group = groups.setdefault(
            key, {"identity": group_identity, "references": [], "comparable_schedule_ranking": "UNKNOWN"}
        )
        work = record.get("work") or {}
        exact = work.get("exact_macs")
        group["references"].append(
            {
                "reference_sha256": identity,
                "workload": record["workload"],
                "cycles": record["cycles"],
                "executable_sha256": record["executable_sha256"],
                "harness_sha256": record["harness_sha256"],
                "command_macs": exact,
                "command_macs_per_observed_cycle": exact / record["cycles"]
                if type(exact) is int and exact > 0
                else None,
                "rate_interpretation": "historical command-work ratio; not a hardware peak or cycle bound",
                "missing_contracts": list(record["missing_contracts"]),
            }
        )
    return {
        "schema": "historical_reference_summary_v1",
        "reference_count": len(unique),
        "engine_groups": groups,
        "target_cycle_authority": False,
        "warm_calibration": False,
        "physical_roofline_complete": False,
        "full_model_cycle_ordering": "UNKNOWN",
        "next_action": "Match the changed source mechanism with a short warm paired probe; "
        "do not extrapolate these historical ratios into full-model latency.",
    }


__all__ = [
    "Comparability",
    "MIN_AGREEING_FACTS",
    "ProgramIdentity",
    "ProgramIdentityError",
    "REFERENCE_KEY",
    "REFERENCE_LEDGER_NAME",
    "ROSTER_KEY",
    "ReferenceError",
    "UNKNOWN_IDENTITY",
    "comparability",
    "declared_identities",
    "estimate_cycles",
    "find_reference",
    "find_reference_for_capsule",
    "identity_of_reference",
    "load_ledger",
    "load_program_identities",
    "load_references",
    "require_comparable",
    "resolve",
    "score_against_reference",
    "structural_gap",
]
