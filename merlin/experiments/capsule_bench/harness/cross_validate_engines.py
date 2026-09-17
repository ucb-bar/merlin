"""Prove a SECOND elaborated-RTL engine agrees with the established one, capsule by capsule.

``rtl_engine_policy`` picks the L3 engine by availability in cost order (vcs > gsim > verilator) on the
premise that every engine on that rung answers at the same fidelity. That premise is an ASSUMPTION until
it is measured. A second oracle that disagrees silently is worse than no second oracle: the selection is
by availability, so the day the faster engine appears it starts certifying capsules, and nothing in the
artifact says the verdict changed engine. This script is the evidence that has to exist first.

What it does: for each capsule it compiles the ELF **once** (through the same
``contract.compile.compile_lowered_to_elf`` seam a grade uses) and runs that **same ELF** on both
engines through the target backend's ``run_elf(elf, simulator=...)``. Nothing here builds a command
line, so what is compared is the path grading actually takes -- an engine cross-validated through a
bespoke driver proves the driver agrees, not the grader.

What is COMPARED is the output bytes. The strong evidence channel is the run's own result readback:
every backend prints the output tensors it read back out of DRAM as the shared ``OUT <name> <rows>
<cols> v...`` protocol lines, and that is the same data the grader gates on. Each engine's tensors are
digested PER OUTPUT TENSOR (see :func:`digest_outputs`), so a divergence names the tensor that diverged
rather than saying only "something differed". A console ``CHK`` value, where a capsule prints one, is
kept as a SECOND channel and still compared; it is the fallback when a run produced no output tensor.

This compares ENGINE vs ENGINE. The reference engine's own output IS the reference: no golden, expected
value, or answer key is read or reported anywhere below, and nothing here is a correctness claim about
the capsule -- only about whether two engines running one ELF produced the same bytes.

Three-state agreement, never two:

    AGREE           both engines ran, returned the same verdict, and produced byte-identical evidence
                    (the same output tensors; or, absent those, the same CHK bytes)
    DISAGREE        both engines ran and something differed -- verdict, an output tensor, or CHK
    UNDETERMINABLE  an engine could not run, or neither engine produced byte evidence to compare

The third state is the whole point. An engine that could not run has produced NO evidence, and folding
that into AGREE is exactly the "checks that skip and report success" failure this repo keeps paying for
(a codegen smoke that could not run reported ``codegen_ok: true`` for 101 minutes of a run that graded
nothing). Undeterminable capsules are reported SEPARATELY from disagreements, because they call for
different action: fix the engine vs investigate the divergence.

Exit codes: 0 every planned capsule agreed; 2 at least one DISAGREE (cross-validation refuted);
3 the cross-validation is INCOMPLETE (an undeterminable capsule, or nothing was proven at all) --
suppressible with ``--allow-incomplete`` when the incompleteness is understood and recorded.

Target is a PARAMETER (``--target``); this file names no engine as "the" engine either -- the reference
and candidate are both arguments, so the same script cross-validates any future pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# Only the stdlib at module scope: the planning half is a pure function and its unit tests must not need
# a built target, a backend, or a simulator to import this file. Everything from ``merlin`` is imported
# inside the function that needs it.

#: The agreement verdicts. Three, and the third is not a shade of the first.
AGREE = "AGREE"
DISAGREE = "DISAGREE"
UNDETERMINABLE = "UNDETERMINABLE"

#: A single engine's verdict on one capsule. ``did_not_run`` is NOT a fail: a fail is a measurement.
PASS = "pass"
FAIL = "fail"
DID_NOT_RUN = "did_not_run"

#: The digest algorithm for one engine's output tensors. Named in the recorded digest string itself
#: (``sha256:...``) so a report never has to be read against the version of the script that made it.
OUTPUT_DIGEST_ALGO = "sha256"

#: The console token carrying the run's self-computed check value. Matched as an exact TOKEN, never by
#: position: a simulator console has several writers (the harness, the model's own stderr, stray
#: ``%Warning:`` fragments) and they interleave, so "the value on line N" is not a stable address. Same
#: reason ``perf.hw_counters.parse_counter_output`` attributes by name rather than by line number.
CHK_MARKER = "CHK"

#: The elaborated-RTL rung. A tier index is a FIDELITY, not a simulator (see ``rtl_engine_policy``), so
#: the tier a capsule declares is what decides whether cross-validating an engine tells us anything
#: about how that capsule gets graded.
DEFAULT_TIER = "L3"

EXIT_OK = 0
EXIT_DISAGREE = 2
EXIT_INCOMPLETE = 3


# --------------------------------------------------------------------------------------------------
# Planning -- pure. No subprocess, no simulator, no filesystem.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CapsulePlan:
    """The work one capsule contributes to the cross-validation."""

    capsule: str
    capsule_dir: str | None
    artifact_dir: str | None          # holds command_buffer.json + lowered.llvm.mlir (the ELF's inputs)
    engines: tuple[str, ...]          # (reference, candidate) -- in the order they will be run
    tier: str
    included: bool
    reason: str                       # why it is included, or why it is not. Never empty.

    @property
    def engine_runs(self) -> int:
        return len(self.engines) if self.included else 0

    def to_dict(self) -> dict:
        return {"capsule": self.capsule, "capsule_dir": self.capsule_dir,
                "artifact_dir": self.artifact_dir, "engines": list(self.engines),
                "tier": self.tier, "included": self.included, "reason": self.reason}


@dataclass(frozen=True)
class CrossValidationPlan:
    """Every capsule considered, included or not, plus the pair being compared.

    Excluded capsules stay IN the plan with their reason. A plan that silently dropped them would make a
    narrow cross-validation indistinguishable from a broad one -- and the number that gets cited is
    "the engines agreed on N capsules", which is only interpretable next to what the other capsules did.
    """

    target: str
    reference_engine: str
    candidate_engine: str
    tier: str
    entries: tuple[CapsulePlan, ...]

    def included(self) -> tuple[CapsulePlan, ...]:
        return tuple(e for e in self.entries if e.included)

    def excluded(self) -> tuple[CapsulePlan, ...]:
        return tuple(e for e in self.entries if not e.included)

    @property
    def total_engine_runs(self) -> int:
        return sum(e.engine_runs for e in self.entries)

    def to_dict(self) -> dict:
        return {"target": self.target, "reference_engine": self.reference_engine,
                "candidate_engine": self.candidate_engine, "tier": self.tier,
                "planned": len(self.included()), "excluded": len(self.excluded()),
                "total_engine_runs": self.total_engine_runs,
                "entries": [e.to_dict() for e in self.entries]}


def _capsule_name(capsule: Any) -> str:
    """The capsule's name, whichever shape the caller had it in (a loaded dict, or a bare name)."""
    if isinstance(capsule, str):
        return capsule
    name = capsule.get("name") or capsule.get("id")
    if not name:
        # Fail closed: an unnamed capsule cannot be reported on, and quietly numbering it would put a
        # cross-engine verdict against a row nobody can trace back to a capsule.
        raise ValueError(f"capsule has neither 'name' nor 'id': {sorted(capsule)!r}")
    return str(name)


def _declared_tiers(capsule: Any) -> tuple[str, ...]:
    if isinstance(capsule, str):
        return ()
    return tuple(str(t) for t in (capsule.get("required_oracle_tiers") or ()))


def plan_cross_validation(capsules: Iterable[Any], *, target: str, reference_engine: str,
                          candidate_engine: str, tier: str = DEFAULT_TIER,
                          artifacts: dict[str, str] | None = None,
                          artifacts_root: str | Path | None = None,
                          require_declared_tier: bool = True) -> CrossValidationPlan:
    """The per-capsule work plan. PURE: no subprocess, no simulation, no filesystem access.

    ``artifacts`` maps capsule name -> the directory holding that capsule's ``command_buffer.json`` and
    ``lowered.llvm.mlir`` (the compiler outputs the graded ELF is built from). ``artifacts_root`` is the
    convenience form: the directory is taken to be ``<root>/<capsule name>``. The path is only COMPUTED
    here, never opened -- planning stays pure so the plan can be reviewed (and unit-tested) before any
    simulator time is bought.

    ``require_declared_tier`` keeps the plan honest about what it proves. A capsule that does not declare
    ``tier`` is never certified at that rung, so two engines agreeing on it says nothing about the grade
    the engines would issue; it is excluded WITH that reason rather than padding the agreement count.
    """
    if reference_engine == candidate_engine:
        # A self-comparison is trivially AGREE and proves nothing. Refuse it rather than emitting a
        # green report whose every row compared an engine with itself.
        raise ValueError(f"reference and candidate engine are both {reference_engine!r}: comparing an "
                         f"engine with itself cannot establish agreement")
    if artifacts and artifacts_root:
        raise ValueError("pass artifacts= or artifacts_root=, not both")

    entries: list[CapsulePlan] = []
    seen: set[str] = set()
    for capsule in capsules:
        name = _capsule_name(capsule)
        cap_dir = None if isinstance(capsule, str) else capsule.get("__dir__")
        art = None
        if artifacts is not None:
            art = artifacts.get(name)
        elif artifacts_root is not None:
            art = str(Path(artifacts_root) / name)

        if name in seen:
            # A duplicate would be counted twice in "the engines agreed on N capsules". Corpora are
            # discovered from several roots (a target's own suite plus the shared one), so this is a
            # real collision, not a defensive nicety.
            entries.append(CapsulePlan(name, cap_dir, art, (), tier, False,
                                       "duplicate capsule name already planned"))
            continue
        seen.add(name)

        declared = _declared_tiers(capsule)
        if require_declared_tier and declared and tier not in declared:
            entries.append(CapsulePlan(
                name, cap_dir, art, (), tier, False,
                f"declares tiers {list(declared)} and not {tier}: the engines never certify it at this "
                f"rung, so agreement here would not be about how it is graded"))
            continue
        if art is None:
            entries.append(CapsulePlan(
                name, cap_dir, art, (), tier, False,
                "no artifact directory located: the ELF's inputs (command_buffer.json + "
                "lowered.llvm.mlir) are unknown, so the two engines could not be given the same ELF"))
            continue

        entries.append(CapsulePlan(name, cap_dir, art, (reference_engine, candidate_engine), tier,
                                   True, f"runs one ELF on {reference_engine} then {candidate_engine}"))

    # Deterministic order: the plan is reviewed before it is paid for, and a plan whose rows move
    # between invocations cannot be diffed against the one that was reviewed.
    entries.sort(key=lambda e: e.capsule)
    return CrossValidationPlan(target, reference_engine, candidate_engine, tier, tuple(entries))


# --------------------------------------------------------------------------------------------------
# Console parsing
# --------------------------------------------------------------------------------------------------
def parse_chk(console: str) -> tuple[str, ...]:
    """Every ``CHK`` value the console carried, in order, as the RAW strings that were printed.

    Prefix/token matching, never positional: the marker is located by name anywhere on the line, exactly
    as ``perf.hw_counters.parse_counter_output`` locates its own, because a simulator console interleaves
    writers and "the token after the Nth" is not a stable address.

    The value is kept as a STRING and never coerced to a number. The comparison downstream is byte
    equality, and coercing (``0x10`` -> 16, ``1.0`` -> 1.0) would make two consoles that printed
    different bytes compare equal -- which is precisely the silent agreement this script exists to
    prevent. An empty tuple means the console carried no CHK at all, which is a MISSING reading and is
    handled as such by :func:`compare_runs`; it is never a zero and never an implicit match.
    """
    values: list[str] = []
    for raw in (console or "").splitlines():
        parts = raw.split()
        for i, tok in enumerate(parts):
            if tok != CHK_MARKER:            # exact token: "CHECK"/"CHKSUM" are different markers
                continue
            if i + 1 < len(parts):
                values.append(parts[i + 1])
            break                            # one reading per line; a second marker is a different line
    return tuple(values)


def parse_counters(console: str) -> dict:
    """The console's hardware-counter lines, through the shared reader (never a second parser here)."""
    from merlin.perf.hw_counters import parse_counter_output
    return parse_counter_output(console or "")


# --------------------------------------------------------------------------------------------------
# Output-byte evidence -- pure
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OutputDigest:
    """One output tensor's identity, as ONE engine produced it: shape + a digest of its bytes.

    Per TENSOR, not one digest over everything, so a divergence can name which output diverged. The
    values themselves are never carried: the digest is the comparable, and a report that reprinted
    tensor contents would be publishing result data this tool has no business publishing.
    """

    name: str
    rows: int
    cols: int
    elements: int
    digest: str                        # "<algo>:<hex>" -- self-describing, so an old report stays readable

    def to_dict(self) -> dict:
        return {"name": self.name, "rows": self.rows, "cols": self.cols,
                "elements": self.elements, "digest": self.digest}


def _canonical_element(value: Any) -> bytes:
    """One output element as the exact bytes that identify it. Fails CLOSED on anything else.

    ``float`` goes through its IEEE-754 bit pattern rather than a decimal rendering: ``repr`` collapses
    ``-0.0``/``0.0`` distinctions under some formats and cannot distinguish NaN payloads, and a
    canonicalization that loses a distinction is exactly how two engines that produced different bytes
    come to compare equal. ``int`` goes through its decimal digits, which are exact at any width. The
    two are tagged apart so ``1`` and ``1.0`` never collide.

    Anything else RAISES rather than being coerced or stringified: an element this function does not
    understand is an UNKNOWN, and the caller records it as missing evidence (UNDETERMINABLE) instead of
    hashing a repr that may or may not be stable.
    """
    if isinstance(value, float):
        return b"f" + struct.pack("<d", value)
    if isinstance(value, int):         # bool is an int; the console protocol never yields one
        return b"i" + str(int(value)).encode("ascii")
    raise ValueError(f"output element of type {type(value).__name__} cannot be canonicalized: "
                     f"refusing to digest a value whose byte identity is unknown")


def digest_outputs(outputs: Any) -> tuple[OutputDigest, ...]:
    """Digest ONE engine's output tensors, per tensor, sorted by name.

    ``outputs`` is what the backend's own ``parse_output`` recovered from the shared ``OUT`` protocol
    lines -- i.e. the result bytes the run read back out of DRAM, the same data the grader gates on.
    Taking the digest downstream of the backend's parser (rather than off the raw console text) is
    deliberate: the parser is where a backend's console quirks are already handled (Verilator
    ``%Warning:`` fragments are stripped, malformed ``METRIC`` lines tolerated), so two engines are
    compared on the DATA they produced and not on which of them interleaved a warning into the stream.

    The shape is folded into each digest, so a tensor with the same values in a different shape is a
    different digest rather than a silent match. Raises on a ragged or non-rectangular tensor, and on
    an element whose bytes cannot be canonicalized -- fail closed, never a partial digest.
    """
    if not isinstance(outputs, dict):
        raise ValueError(f"outputs is {type(outputs).__name__}, not a name -> tensor map: "
                         f"there is nothing to digest")
    digests: list[OutputDigest] = []
    for name in sorted(outputs):
        tensor = outputs[name]
        if not isinstance(tensor, (list, tuple)):
            raise ValueError(f"output {name!r} is {type(tensor).__name__}, not a list of rows")
        h = hashlib.new(OUTPUT_DIGEST_ALGO)
        h.update(str(name).encode("utf-8") + b"\0")
        cols: int | None = None
        count = 0
        for r, row in enumerate(tensor):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"output {name!r} row {r} is {type(row).__name__}, not a list of values")
            if cols is None:
                cols = len(row)
            elif len(row) != cols:
                raise ValueError(f"output {name!r} is ragged: row 0 has {cols} values, row {r} has "
                                 f"{len(row)} -- refusing to digest a tensor with no single shape")
            for value in row:
                h.update(_canonical_element(value))
                h.update(b" ")
                count += 1
            h.update(b"\n")
        cols = 0 if cols is None else cols
        h.update(f"{len(tensor)}x{cols}".encode("ascii"))
        digests.append(OutputDigest(str(name), len(tensor), cols, count,
                                    f"{OUTPUT_DIGEST_ALGO}:{h.hexdigest()}"))
    return tuple(digests)


def compare_output_digests(reference: Sequence[OutputDigest], candidate: Sequence[OutputDigest], *,
                           reference_engine: str = "reference",
                           candidate_engine: str = "candidate") -> tuple[bool, str]:
    """(match, why) for two engines' output tensors. PURE. Names the tensor(s) that diverged.

    A tensor present on one side only is a divergence, not a skipped comparison: an engine that produced
    fewer outputs than the other did not corroborate the outputs it omitted.
    """
    a = {d.name: d for d in reference}
    b = {d.name: d for d in candidate}
    only_ref, only_cand = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_ref or only_cand:
        return False, (f"output tensors differ: {reference_engine} produced {sorted(a)} and "
                       f"{candidate_engine} produced {sorted(b)} "
                       f"(only {reference_engine}: {only_ref}; only {candidate_engine}: {only_cand})")
    diverged = [n for n in sorted(a) if a[n] != b[n]]
    if diverged:
        detail = "; ".join(
            f"{n}: {reference_engine}={a[n].digest} [{a[n].rows}x{a[n].cols}] vs "
            f"{candidate_engine}={b[n].digest} [{b[n].rows}x{b[n].cols}]" for n in diverged)
        return False, f"output bytes differ on {len(diverged)} of {len(a)} tensor(s) -- {detail}"
    return True, (f"{len(a)} output tensor(s) byte-identical: "
                  + ", ".join(f"{n}={a[n].digest}" for n in sorted(a)))


# --------------------------------------------------------------------------------------------------
# One engine's answer
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineRun:
    """What one engine did with the ELF.

    ``ran`` means "produced a GRADEABLE console", not "the process started". An engine that exited
    non-zero, timed out, or printed a console the shared protocol parser could not complete has produced
    no verdict, and its ``verdict`` stays ``did_not_run`` -- never ``fail``, which would read as a
    measurement the engine never made.
    """

    engine: str
    ran: bool
    verdict: str                       # pass | fail | did_not_run
    chk: tuple[str, ...] = ()
    counters: dict = field(default_factory=dict)
    wall_s: float | None = None
    detail: str = ""
    console_bytes: int = 0
    # Appended, never inserted: the strong evidence channel. Empty means NO byte evidence from this
    # engine -- which is a missing reading, handled as such by :func:`compare_runs`, never an implicit
    # match. ``output_note`` says WHY it is empty when it is.
    outputs: tuple[OutputDigest, ...] = ()
    output_note: str = ""

    def to_dict(self) -> dict:
        return {"engine": self.engine, "ran": self.ran, "verdict": self.verdict,
                "chk": list(self.chk), "counters": dict(self.counters),
                "wall_s": self.wall_s, "detail": self.detail,
                "console_bytes": self.console_bytes,
                "outputs": [d.to_dict() for d in self.outputs],
                "output_note": self.output_note}


def unavailable_run(engine: str, detail: str) -> EngineRun:
    """The record for an engine that could not answer. Explicit constructor so no call site has to
    remember that ``did_not_run`` is the verdict and that the wall time is unknown rather than 0.0."""
    return EngineRun(engine, False, DID_NOT_RUN, (), {}, None, detail)


def run_on_engine(engine: str, elf: str | Path, *, backend, grade: Callable[[dict], bool],
                  timeout: int = 3600) -> EngineRun:
    """Run ``elf`` on ``engine`` through the backend's OWN seams and record what came back.

    Deliberately calls ``backend.available(engine)`` / ``backend.run_elf(elf, simulator=engine)`` /
    ``backend.parse_output(console)`` rather than assembling a command line: the comparison is only
    worth anything if it exercises the path a grade takes.

    ``available`` may RAISE for an engine the backend does not know (measured:
    ``available('gsim')`` -> ``GemminiError: unknown simulator 'gsim'`` before that branch existed). That
    is recorded as unavailability WITH the reason, exactly as ``rtl_engine_policy.select`` treats a
    raising probe -- an engine this target has no branch for is a different thing to fix than an engine
    whose binary is missing, and both are different from a crash.
    """
    try:
        ok = bool(backend.available(engine))
    except Exception as exc:               # noqa: BLE001 - an engine the backend does not know
        return unavailable_run(engine, f"availability probe raised {type(exc).__name__}: "
                                       f"{str(exc)[-200:]}")
    if not ok:
        return unavailable_run(engine, f"{engine} reports unavailable for this backend")

    t0 = time.perf_counter()
    try:
        console = backend.run_elf(elf, simulator=engine, timeout=timeout)
    except Exception as exc:               # noqa: BLE001 - non-zero exit, timeout, missing binary
        return unavailable_run(engine, f"run_elf raised {type(exc).__name__}: {str(exc)[-300:]}")
    wall = round(time.perf_counter() - t0, 3)

    # Counters and CHK come off the RAW console, before the protocol parser, so they survive a console
    # the parser refuses: a run that hung is still worth its counter lines when diagnosing why.
    chk = parse_chk(console)
    counters = parse_counters(console)
    try:
        outputs, _raw = backend.parse_output(console)
    except Exception as exc:               # noqa: BLE001 - console never reached DONE / malformed
        return EngineRun(engine, False, DID_NOT_RUN, chk, counters, wall,
                         f"console not gradeable: {type(exc).__name__}: {str(exc)[-200:]}",
                         len(console or ""))

    # The result readback is the strong evidence. Digested here, once, off the SAME parsed outputs the
    # grader is about to see -- so what the report calls "the bytes this engine produced" is the data
    # the verdict was formed on, not a second reading of the console taken somewhere else.
    try:
        digests = digest_outputs(outputs)
        note = "" if digests else "the run printed no OUT tensor: no output bytes to compare"
    except Exception as exc:               # noqa: BLE001 - ragged / uncanonicalizable element
        # Fail CLOSED: no digest rather than a partial one. The comparison then has no byte evidence
        # from this engine and says so, which is UNDETERMINABLE -- never a quiet pass on the verdict.
        digests, note = (), (f"outputs could not be digested: {type(exc).__name__}: {str(exc)[-200:]}")

    verdict = PASS if grade(outputs) else FAIL
    return EngineRun(engine, True, verdict, chk, counters, wall,
                     f"{engine} completed in {wall}s", len(console or ""), digests, note)


# --------------------------------------------------------------------------------------------------
# Agreement -- pure
# --------------------------------------------------------------------------------------------------
#: Which channel the verdict actually rested on. Recorded per capsule, because "the engines agreed on N
#: capsules" means something different for each of these, and a report that does not say which one it
#: used lets the weakest be read as the strongest.
EV_OUTPUT_BYTES = "output_bytes"       # the engines produced the same output tensors -- the strong one
EV_CONSOLE_CHK = "console_chk"         # no output tensors; the consoles' CHK values matched
EV_VERDICT_ONLY = "verdict_only"       # evidence explicitly WAIVED (--allow-missing-chk); weak
EV_NONE = "none"                       # no evidence at all -- always UNDETERMINABLE


@dataclass(frozen=True)
class CapsuleComparison:
    capsule: str
    reference: EngineRun
    candidate: EngineRun
    agreement: str                     # AGREE | DISAGREE | UNDETERMINABLE
    chk_match: bool | None             # None = could not be evaluated (no CHK to compare)
    speed_ratio: float | None          # reference wall / candidate wall: >1 = candidate is faster
    reason: str
    bytes_match: bool | None = None    # None = could not be evaluated (no output bytes to compare)
    evidence: str = EV_NONE            # which channel the verdict rests on

    def to_dict(self) -> dict:
        return {"capsule": self.capsule, "agreement": self.agreement,
                "bytes_match": self.bytes_match, "evidence": self.evidence,
                "chk_match": self.chk_match, "speed_ratio": self.speed_ratio,
                "reason": self.reason,
                "reference": self.reference.to_dict(), "candidate": self.candidate.to_dict()}


def _chk_match(reference: EngineRun, candidate: EngineRun) -> bool | None:
    """Whether the two consoles' CHK values matched. ``None`` when there is nothing to compare -- one or
    both consoles carried no value, which is a MISSING reading and never a match."""
    if reference.chk and candidate.chk:
        return reference.chk == candidate.chk
    if reference.chk or candidate.chk:
        return False                   # one-sided: present on one console, absent on the other
    return None


def _chk_disagreement(reference: EngineRun, candidate: EngineRun) -> str:
    """The CHK divergence, if any, as text -- empty when the two consoles do not contradict each other.

    Used as a SECOND channel once the output bytes have already matched: matching bytes with a differing
    CHK still means the two runs differ somewhere the console can see, and the whole point of this file
    is that such a difference is never quietly dropped."""
    if reference.chk and candidate.chk and reference.chk != candidate.chk:
        return (f"CHK bytes differ: {reference.engine}={list(reference.chk)} vs "
                f"{candidate.engine}={list(candidate.chk)}")
    if bool(reference.chk) != bool(candidate.chk):
        had = reference.engine if reference.chk else candidate.engine
        lacked = candidate.engine if reference.chk else reference.engine
        return f"only {had} printed a {CHK_MARKER} value; {lacked} printed none"
    return ""


def _speed_ratio(reference: EngineRun, candidate: EngineRun) -> float | None:
    """How many times faster the candidate was. ``None`` when either side has no wall time -- an engine
    that did not run took no measurable time, and printing 0.0 or ``inf`` would read as a speed."""
    a, b = reference.wall_s, candidate.wall_s
    if a is None or b is None or b <= 0:
        return None
    return round(a / b, 3)


def compare_runs(reference: EngineRun, candidate: EngineRun, *,
                 capsule: str = "", require_chk: bool = True) -> CapsuleComparison:
    """The three-state agreement verdict for one capsule. PURE.

    ``require_chk`` is the historical name of one knob: require BYTE EVIDENCE. With it (the default) a
    pair of runs that produced nothing comparable -- no output tensors on either side and no CHK on
    either console -- is UNDETERMINABLE. Clearing it (``--allow-missing-chk``) is the explicit statement
    that the pass/fail verdicts alone are the evidence being claimed, which is much weaker: two engines
    can both "pass" while computing different bytes.

    Order of the checks is deliberate:

    1. **Either engine did not run -> UNDETERMINABLE.** Always, and regardless of what the other engine
       found. This is the rule the whole script is built around: an absent result is not agreement.
       Reporting "AGREE" because the only engine that ran happened to pass is how a second oracle gets
       adopted on no evidence.
    2. **Verdicts differ -> DISAGREE.** Note that ``fail`` == ``fail`` is AGREEMENT: the claim under test
       is that the two oracles say the same thing, not that the capsule passes.
    3. **Exactly one engine produced output tensors -> DISAGREE.** An engine that read back no result
       cannot corroborate the results the other read back, and treating the silent side as "nothing to
       compare" would let a truncated readback pass as a match.
    4. **Output bytes -> the verdict.** Per output tensor, by digest, no tolerance: same tensor names,
       same shapes, same bytes, or DISAGREE naming the tensor that diverged. Two engines running one ELF
       over one elaborated design must read back the same bytes; "close" is not a thing an RTL engine is
       allowed to be. A CHK divergence on top of matching bytes is still a DISAGREE -- the consoles
       differ in a load-bearing field either way.
    5. **No output bytes anywhere -> fall back to the console CHK ladder**, unchanged: both values match
       -> AGREE; they differ -> DISAGREE; only one console carried a value -> DISAGREE (it cannot be
       corroborated); neither did -> UNDETERMINABLE unless the evidence was explicitly waived.
    """
    name = capsule or ""
    missing = [r.engine for r in (reference, candidate) if not r.ran]
    if missing:
        why = "; ".join(f"{r.engine}: {r.detail}" for r in (reference, candidate) if not r.ran)
        return CapsuleComparison(
            name, reference, candidate, UNDETERMINABLE, None, _speed_ratio(reference, candidate),
            f"no verdict from {', '.join(missing)} — undeterminable, never agreement ({why})",
            None, EV_NONE)

    if reference.verdict != candidate.verdict:
        return CapsuleComparison(
            name, reference, candidate, DISAGREE, None, _speed_ratio(reference, candidate),
            f"verdicts differ: {reference.engine}={reference.verdict} vs "
            f"{candidate.engine}={candidate.verdict}", None, EV_NONE)

    ratio = _speed_ratio(reference, candidate)
    chk_match = _chk_match(reference, candidate)

    # ---- the strong channel: the output bytes each engine read back --------------------------------
    if bool(reference.outputs) != bool(candidate.outputs):
        had = reference if reference.outputs else candidate
        lacked = candidate if reference.outputs else reference
        note = f" ({lacked.output_note})" if lacked.output_note else ""
        return CapsuleComparison(
            name, reference, candidate, DISAGREE, chk_match, ratio,
            f"only {had.engine} produced output tensors ({len(had.outputs)}); {lacked.engine} produced "
            f"none{note}, so it cannot corroborate them", False, EV_OUTPUT_BYTES)

    if reference.outputs and candidate.outputs:
        same, why = compare_output_digests(reference.outputs, candidate.outputs,
                                           reference_engine=reference.engine,
                                           candidate_engine=candidate.engine)
        if not same:
            return CapsuleComparison(name, reference, candidate, DISAGREE, chk_match, ratio, why,
                                     False, EV_OUTPUT_BYTES)
        chk_problem = _chk_disagreement(reference, candidate)
        if chk_problem:
            return CapsuleComparison(
                name, reference, candidate, DISAGREE, chk_match, ratio,
                f"output bytes matched but the consoles differ: {chk_problem}", True, EV_OUTPUT_BYTES)
        chk_note = ("; CHK byte-identical " + str(list(reference.chk))) if reference.chk else ""
        return CapsuleComparison(
            name, reference, candidate, AGREE, chk_match, ratio,
            f"both {reference.verdict}; {why}{chk_note}", True, EV_OUTPUT_BYTES)

    # ---- no byte evidence on either side: the console CHK fallback ---------------------------------
    notes = "; ".join(f"{r.engine}: {r.output_note}" for r in (reference, candidate) if r.output_note)
    if reference.chk and candidate.chk:
        if reference.chk != candidate.chk:
            return CapsuleComparison(
                name, reference, candidate, DISAGREE, False, ratio,
                f"CHK bytes differ: {reference.engine}={list(reference.chk)} vs "
                f"{candidate.engine}={list(candidate.chk)}", None, EV_CONSOLE_CHK)
        return CapsuleComparison(
            name, reference, candidate, AGREE, True, ratio,
            f"both {reference.verdict}; no output tensor on either console, CHK byte-identical "
            f"{list(reference.chk)}", None, EV_CONSOLE_CHK)

    if reference.chk or candidate.chk:
        had = reference.engine if reference.chk else candidate.engine
        lacked = candidate.engine if reference.chk else reference.engine
        return CapsuleComparison(
            name, reference, candidate, DISAGREE, False, ratio,
            f"only {had} printed a CHK value; {lacked} printed none, so it cannot corroborate one",
            None, EV_CONSOLE_CHK)

    if require_chk:
        return CapsuleComparison(
            name, reference, candidate, UNDETERMINABLE, None, ratio,
            f"both {reference.verdict}, but neither engine produced byte evidence: no output tensor was "
            f"read back and neither console carried a {CHK_MARKER} value, so agreement is unproven "
            f"({notes or 'no further detail'}). Two engines can both pass while computing different "
            f"bytes; pass --allow-missing-chk to rest the verdict on pass/fail alone",
            None, EV_NONE)
    return CapsuleComparison(
        name, reference, candidate, AGREE, None, ratio,
        f"both {reference.verdict}; no output tensor and no {CHK_MARKER} on either console "
        f"(byte evidence waived — this rests on the verdicts alone)", None, EV_VERDICT_ONLY)


# --------------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------------
def _load_capsule_artifacts(artifact_dir: str | Path) -> tuple[dict, str]:
    """(command buffer, lowered LLVM-dialect MLIR text) for one capsule, by the frozen ABI's own names."""
    d = Path(artifact_dir)
    cb_path, llvm_path = d / "command_buffer.json", d / "lowered.llvm.mlir"
    if not cb_path.is_file():
        raise FileNotFoundError(f"{cb_path} not found: no command buffer to build an ELF from")
    if not llvm_path.is_file():
        raise FileNotFoundError(f"{llvm_path} not found: no lowered MLIR to build an ELF from")
    return json.loads(cb_path.read_text(encoding="utf-8")), llvm_path.read_text(encoding="utf-8")


def build_shared_elf(entry: CapsulePlan, *, target: str, workdir: str | Path) -> tuple[Path, dict]:
    """Compile the ELF ONCE, through the same seam a grade compiles it with; return (elf, cb).

    Once, not once per engine. Two compilations of the same inputs are ALMOST certainly identical, and
    "almost" is the word that ruins a cross-validation: if the engines then disagreed, the first question
    would be whether they ran the same bytes, and there would be no way to answer it.
    """
    from merlin.targetgen.contract.compile import compile_lowered_to_elf
    cb, llvm_text = _load_capsule_artifacts(entry.artifact_dir)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    elf = compile_lowered_to_elf(cb, llvm_text, work, target=target)
    return Path(elf), cb


def _reference_grader(cb: dict) -> Callable[[dict], bool]:
    """pass/fail for a console's outputs, by the SAME reference equality the backend's own
    ``run_command_buffer`` gates on -- so a verdict here means what a verdict there means. Both engines
    are graded by this one callable, which is what makes their verdicts comparable at all."""
    from merlin.runtime.reference import outputs_match, reference_outputs
    ref = reference_outputs(cb)
    return lambda outputs: bool(outputs_match(outputs, ref))


def cross_validate(plan: CrossValidationPlan, *, workdir: str | Path, timeout: int = 3600,
                   require_chk: bool = True, backend=None,
                   build_elf: Callable[[CapsulePlan], tuple[Path, dict]] | None = None,
                   grader: Callable[[dict], Callable[[dict], bool]] | None = None,
                   log: Callable[[str], None] = print) -> list[CapsuleComparison]:
    """Run every included capsule on both engines and return the per-capsule comparisons.

    ``backend`` / ``build_elf`` / ``grader`` are injection points, present so the agreement logic can be
    exercised without a simulator; the defaults are the real seams (the target's registered backend, the
    contract compile path, the reference-equality grader).
    """
    if backend is None:
        from merlin.runtime.backends import base as _backends
        backend = _backends.get_backend(plan.target)
    if build_elf is None:
        def build_elf(entry: CapsulePlan):                       # noqa: D401 - default injection
            return build_shared_elf(entry, target=plan.target,
                                    workdir=Path(workdir) / entry.capsule)
    if grader is None:
        grader = _reference_grader

    out: list[CapsuleComparison] = []
    for entry in plan.included():
        try:
            elf, cb = build_elf(entry)
        except Exception as exc:                                 # noqa: BLE001
            # NEITHER engine ran, so this is undeterminable -- a build failure is not a verdict about
            # whether the engines agree, and recording it as one would blame the engines for the harness.
            why = f"could not build the shared ELF: {type(exc).__name__}: {str(exc)[-300:]}"
            out.append(compare_runs(unavailable_run(plan.reference_engine, why),
                                    unavailable_run(plan.candidate_engine, why),
                                    capsule=entry.capsule, require_chk=require_chk))
            log(f"[xval] {entry.capsule}: {why}")
            continue
        grade = grader(cb)
        runs = [run_on_engine(eng, elf, backend=backend, grade=grade, timeout=timeout)
                for eng in entry.engines]
        cmp_ = compare_runs(runs[0], runs[1], capsule=entry.capsule, require_chk=require_chk)
        out.append(cmp_)
        log(f"[xval] {entry.capsule}: {cmp_.agreement} — {cmp_.reason}")
    return out


# --------------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossValidationSummary:
    target: str
    reference_engine: str
    candidate_engine: str
    tier: str
    comparisons: tuple[CapsuleComparison, ...]
    excluded: tuple[CapsulePlan, ...]
    allow_incomplete: bool

    @property
    def agreed(self) -> tuple[CapsuleComparison, ...]:
        return tuple(c for c in self.comparisons if c.agreement == AGREE)

    @property
    def disagreed(self) -> tuple[CapsuleComparison, ...]:
        return tuple(c for c in self.comparisons if c.agreement == DISAGREE)

    @property
    def undeterminable(self) -> tuple[CapsuleComparison, ...]:
        return tuple(c for c in self.comparisons if c.agreement == UNDETERMINABLE)

    @property
    def evidence_census(self) -> dict:
        """How many AGREEs rest on each channel. Reported because "the engines agreed on N capsules"
        means something different for output-byte agreement than for a waived one, and a report that
        does not separate them lets the weakest be cited as the strongest."""
        census = {EV_OUTPUT_BYTES: 0, EV_CONSOLE_CHK: 0, EV_VERDICT_ONLY: 0, EV_NONE: 0}
        for c in self.agreed:
            census[c.evidence] = census.get(c.evidence, 0) + 1
        return census

    @property
    def median_speed_ratio(self) -> float | None:
        ratios = sorted(c.speed_ratio for c in self.comparisons if c.speed_ratio is not None)
        if not ratios:
            return None
        mid = len(ratios) // 2
        return ratios[mid] if len(ratios) % 2 else round((ratios[mid - 1] + ratios[mid]) / 2, 3)

    @property
    def exit_code(self) -> int:
        """0 only when the cross-validation actually established something.

        A DISAGREE is the loud failure (2). The quiet one is 3: undeterminable rows, or zero agreements.
        Zero agreements with zero disagreements is the shape a green report takes when NOTHING ran, and
        a cross-validation that proved nothing must not exit 0 -- that is the exact "a check that could
        not run reported success" failure this repo has paid for repeatedly.
        """
        if self.disagreed:
            return EXIT_DISAGREE
        if self.allow_incomplete:
            return EXIT_OK
        if self.undeterminable or not self.agreed:
            return EXIT_INCOMPLETE
        return EXIT_OK

    def to_dict(self) -> dict:
        return {
            "target": self.target, "reference_engine": self.reference_engine,
            "candidate_engine": self.candidate_engine, "tier": self.tier,
            "counts": {"agree": len(self.agreed), "disagree": len(self.disagreed),
                       "undeterminable": len(self.undeterminable),
                       "excluded_from_plan": len(self.excluded)},
            "agreement_evidence": self.evidence_census,
            "median_speed_ratio": self.median_speed_ratio,
            "exit_code": self.exit_code,
            "capsules": [c.to_dict() for c in self.comparisons],
            "excluded": [e.to_dict() for e in self.excluded],
        }


def summarize(plan: CrossValidationPlan, comparisons: Sequence[CapsuleComparison], *,
              allow_incomplete: bool = False) -> CrossValidationSummary:
    return CrossValidationSummary(plan.target, plan.reference_engine, plan.candidate_engine,
                                  plan.tier, tuple(comparisons), plan.excluded(), allow_incomplete)


def render(summary: CrossValidationSummary) -> str:
    """The per-capsule table plus the three counts. Undeterminable rows are listed in their OWN section:
    folded into the disagreements they would look like engine bugs, and folded into the agreements they
    would look like evidence."""
    ref, cand = summary.reference_engine, summary.candidate_engine
    lines = [f"cross-validation: {cand} vs {ref} on {summary.target} at {summary.tier}",
             "",
             f"{'capsule':<32} {ref:>12} {cand:>12} {'out-bytes':>10} {'chk':>9} "
             f"{'x' + ref[:6]:>10}  verdict",
             "-" * 108]
    for c in summary.comparisons:
        mark = {True: "exact", False: "DIFFER", None: "—"}
        ratio = "—" if c.speed_ratio is None else f"{c.speed_ratio:.2f}x"
        lines.append(f"{c.capsule:<32} {c.reference.verdict:>12} {c.candidate.verdict:>12} "
                     f"{mark[c.bytes_match]:>10} {mark[c.chk_match]:>9} {ratio:>10}  {c.agreement}")
    census = summary.evidence_census
    lines += ["", f"AGREE          {len(summary.agreed)}   "
                  f"(on output bytes {census[EV_OUTPUT_BYTES]}, "
                  f"on console {CHK_MARKER} {census[EV_CONSOLE_CHK]}, "
                  f"on the verdicts alone {census[EV_VERDICT_ONLY]})",
              f"DISAGREE       {len(summary.disagreed)}",
              f"UNDETERMINABLE {len(summary.undeterminable)}   "
              f"(no evidence — NOT counted as agreement)",
              f"excluded from the plan: {len(summary.excluded)}"]
    if summary.median_speed_ratio is not None:
        lines.append(f"median speed ratio ({cand} vs {ref}): {summary.median_speed_ratio}x")
    for c in summary.disagreed:
        lines.append(f"  DISAGREE {c.capsule}: {c.reason}")
    for c in summary.undeterminable:
        lines.append(f"  UNDETERMINABLE {c.capsule}: {c.reason}")
    lines.append(f"exit {summary.exit_code}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def _bootstrap_merlin_path() -> None:
    """Put ``merlin/python`` on ``sys.path`` when this file is run as a script.

    Walks UP looking for the directory rather than indexing ``parents[N]``: an index silently resolves to
    the wrong place the first time the harness moves, and this file has no other way to notice.
    """
    for parent in Path(__file__).resolve().parents:
        cand = parent / "merlin" / "python"
        if cand.is_dir():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", required=True,
                   help="the target whose backend runs both engines (a PARAMETER, never assumed)")
    p.add_argument("--capsules", nargs="+", required=True,
                   help="capsule root(s) to discover capsules under")
    p.add_argument("--artifacts-root", required=True,
                   help="directory holding <capsule>/command_buffer.json + lowered.llvm.mlir")
    p.add_argument("--reference-engine", default="verilator",
                   help="the ESTABLISHED engine the candidate must agree with")
    p.add_argument("--candidate-engine", required=True,
                   help="the engine under test; it certifies nothing until it agrees")
    p.add_argument("--tier", default=DEFAULT_TIER, help="the fidelity rung being cross-validated")
    p.add_argument("--workdir", required=True, help="scratch directory for the compiled ELFs")
    p.add_argument("--timeout", type=int, default=3600, help="per-engine wall budget, seconds")
    p.add_argument("--limit", type=int, default=None, help="cross-validate only the first N capsules")
    p.add_argument("--json", dest="json_out", default=None, help="write the summary record here")
    p.add_argument("--allow-missing-chk", "--allow-missing-evidence", action="store_true",
                   dest="allow_missing_chk",
                   help=("rest the verdict on pass/fail alone when NEITHER engine produced byte "
                         "evidence (no OUT tensor read back, no CHK on either console). Much weaker: "
                         "two engines can both pass while computing different bytes"))
    p.add_argument("--allow-incomplete", action="store_true",
                   help="exit 0 despite undeterminable capsules (state why in the run's record)")
    p.add_argument("--plan-only", action="store_true",
                   help="print the plan and exit without running a simulator")
    p.add_argument("--include-undeclared-tier", action="store_true",
                   help=("also cross-validate capsules that do not DECLARE --tier. The engines still "
                         "execute the same ELF, so agreement there is real evidence about the ENGINES; "
                         "it is simply not evidence about a rung those capsules are never certified at, "
                         "and the report must not be read as the latter"))
    return p


def main(argv: list[str] | None = None) -> int:
    _bootstrap_merlin_path()
    args = _parser().parse_args(argv)
    from merlin.targetgen.capsule_common import discover_capsules

    capsules = discover_capsules([Path(r) for r in args.capsules])
    if args.limit is not None:
        capsules = capsules[:args.limit]
    plan = plan_cross_validation(capsules, target=args.target,
                                 reference_engine=args.reference_engine,
                                 candidate_engine=args.candidate_engine, tier=args.tier,
                                 artifacts_root=args.artifacts_root,
                                 require_declared_tier=not args.include_undeclared_tier)
    print(f"[xval] planned {len(plan.included())} capsules "
          f"({plan.total_engine_runs} engine runs), excluded {len(plan.excluded())}")
    for e in plan.excluded():
        print(f"[xval]   excluded {e.capsule}: {e.reason}")
    if not plan.included():
        # An empty plan exits non-zero below (nothing was proven). Say WHY it is empty here, since the
        # commonest cause is a corpus whose capsules declare a shallower ladder than --tier, and the
        # remedy is a flag rather than a bug.
        print(f"[xval] nothing planned: no capsule declares {args.tier}. Pass "
              f"--include-undeclared-tier to compare the engines on them anyway (agreement about the "
              f"ENGINES, not about a rung these capsules are never certified at).")
    if args.plan_only:
        print(json.dumps(plan.to_dict(), indent=2))
        return EXIT_OK

    comparisons = cross_validate(plan, workdir=args.workdir, timeout=args.timeout,
                                 require_chk=not args.allow_missing_chk)
    summary = summarize(plan, comparisons, allow_incomplete=args.allow_incomplete)
    print(render(summary))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        print(f"[xval] wrote {out}")
    return summary.exit_code


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
