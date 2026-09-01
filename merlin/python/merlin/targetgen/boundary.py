"""Where a capsule's work crosses the host/accelerator seam — derived from the capsule, never typed.

The ``(semantic_family, dtype, tile_alignment)`` cells say WHAT a corpus computes. They are silent about
HOW the work is composed, and composition is where a whole-model compiler actually fails: a corpus of
single-dispatch capsules can prove every family and still never once hand a value from the accelerator to
the host and back. Measured on this repo's interlocked target: of 34 graded capsules, **every** small one
is a single accelerator program with no host-lane computation in it at all, and the only capsules that
compose anything are the three whole-model capstones — which run at the functional tier. So the corpus
proves families thoroughly and proves composition nowhere, and nothing reported that.

The vocabulary is the composition shape:

``A``        one accelerator region; nothing composed.
``A->A``     two or more accelerator regions in a row — the intermediate never leaves the accelerator.
``H->A->H``  host computation on both sides of one accelerator region — the ordinary seam.
``A->H->A``  an accelerator region, a host island, another accelerator region — the expensive seam,
             and the one a placement decision gets wrong.
``routing``  a genuinely mixed graph: several accelerator segments and several host segments.
``H``        no accelerator region at all — the negative case, and a real requirement: a target that
             accelerates something it should not is as wrong as one that misses work.
``UNKNOWN``  the capsule's interface could not be read. NEVER folded into ``H``: "no accelerator work"
             and "we could not tell" are different facts, and collapsing them reports an unread capsule
             as a proven host-only one.

Both capsule grammars are read through their own canonical parser — ``contract.interface_emit`` for the
frozen ``merlin_iface`` command grammar, ``model_coverage.regions_from_module`` plus
``eligibility.is_eligible`` for linalg-on-tensors — so this module never learns a second dialect and can
never disagree with the parser the grader uses.

**The ``merlin_iface`` grammar cannot express host computation.** Its whole op set (``resident_pack``,
``matmul``, ``commit``, ``evict``) is accelerator work; tensors enter and leave through host memory, but
host *memory* is not a host *computation* and counting it as one would label every capsule ``H->A->H``
and make the axis say nothing. A capsule in that grammar is therefore ``A`` or ``A->A``, decided by how
many dispatches it issues — which is exactly the residency property such a capsule exists to prove.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: An accelerator-eligible region, in the sequence vocabulary this module classifies.
ACCEL = "A"
#: Accelerator-ELIGIBLE, but no path in this repo can emit this target's host/device boundary, so
#: whether the work crosses the seam is UNDETERMINABLE. Never folded into ``A`` (which would claim a
#: crossing nothing can compile) nor into ``H`` (which would claim the target refused the work).
UNBUILDABLE = "?A"
#: A region the target's capability model does not admit — the host/scalar lane carries it.
HOST = "H"

A = "A"
A_A = "A->A"
H_A_H = "H->A->H"
A_H_A = "A->H->A"
ROUTING = "routing"
HOST_ONLY = "H"
UNKNOWN = "UNKNOWN"

#: Strongest-first. A capsule that exhibits several shapes is NAMED by the strongest one, because the
#: axis asks "what is the hardest composition this capsule proves", and the full segment counts stay on
#: the profile so a reader can see what was folded rather than having to trust the label.
PRECEDENCE = (ROUTING, A_H_A, H_A_H, A_A, A, HOST_ONLY)


@dataclass(frozen=True)
class BoundaryProfile:
    """One capsule's composition shape, with the evidence that produced it."""

    kind: str = UNKNOWN
    grammar: str = ""                    # "merlin_iface" | "linalg" | ""
    n_accel_regions: int = 0
    n_host_regions: int = 0
    n_unresolved: int = 0
    n_unbuildable: int = 0
    accel_segments: int = 0
    host_segments: int = 0
    detail: str = ""
    #: EVERY composition shape the capsule contains, not only ``kind`` (the strongest one it is named
    #: by). Coverage is measured on this; ``kind`` is what a reader sees. Keeping them apart is what
    #: lets the gate say a shape is covered only INCIDENTALLY -- by a capsule that is about something
    #: else -- rather than silently treating that as equivalent to a capsule built to prove it.
    contains: tuple = ()

    def to_dict(self) -> dict:
        return {"boundary": self.kind, "contains": sorted(self.contains), "grammar": self.grammar,
                "n_accel_regions": self.n_accel_regions, "n_host_regions": self.n_host_regions,
                "n_unresolved": self.n_unresolved, "n_unbuildable": self.n_unbuildable,
                "accel_segments": self.accel_segments,
                "host_segments": self.host_segments, "detail": self.detail}


def segments(seq) -> list[tuple[str, int]]:
    """Compress a region sequence into runs, dropping regions whose class could not be resolved.

    An unresolved region is dropped rather than treated as host work: calling it host manufactures a seam
    that no evidence supports, and a manufactured seam is exactly the kind of unearned coverage this whole
    derivation exists to prevent. How many were dropped is reported on the profile.
    """
    out: list[tuple[str, int]] = []
    for item in seq:
        if item not in (ACCEL, HOST):
            continue
        if out and out[-1][0] == item:
            out[-1] = (item, out[-1][1] + 1)
        else:
            out.append((item, 1))
    return out


def classify_sequence(seq) -> str:
    """The strongest composition shape a region sequence exhibits.

    ``A->H->A`` needs two accelerator segments, which by construction have a host segment between them;
    ``routing`` needs the mix to be genuine on both sides, so one stray host op between two accelerator
    runs is an island rather than a mixed graph.
    """
    segs = segments(seq)
    if not segs:
        return UNKNOWN
    n_a = sum(1 for k, _ in segs if k == ACCEL)
    n_h = sum(1 for k, _ in segs if k == HOST)
    if not n_a:
        return HOST_ONLY
    if n_a >= 2 and n_h >= 2:
        return ROUTING
    if n_a >= 2:
        return A_H_A
    # exactly one accelerator run
    idx = next(i for i, (k, _) in enumerate(segs) if k == ACCEL)
    if 0 < idx < len(segs) - 1:
        return H_A_H
    return A_A if segs[idx][1] >= 2 else A


def patterns_in_sequence(seq) -> set[str]:
    """EVERY composition shape a sequence contains, not just the strongest one.

    The requirement side needs this: a real model's region sequence classifies as ``routing`` as a whole,
    yet it contains isolated dispatches, adjacent accelerator pairs and host islands — and each of those
    is a composition the corpus must exercise somewhere. Taking only the whole-model label would demand
    ``routing`` and nothing else, which is the narrowest possible reading of the richest evidence.
    """
    segs = segments(seq)
    if not segs:
        return set()
    found: set[str] = set()
    n_a = sum(1 for k, _ in segs if k == ACCEL)
    n_h = sum(1 for k, _ in segs if k == HOST)
    if not n_a:
        return {HOST_ONLY}
    if n_h:
        found.add(HOST_ONLY)                       # the model does carry host-lane-only stretches
    if n_a >= 2 and n_h >= 2:
        found.add(ROUTING)
    if n_a >= 2:
        found.add(A_H_A)
    for i, (kind, run) in enumerate(segs):
        if kind != ACCEL:
            continue
        found.add(A_A if run >= 2 else A)
        if 0 < i < len(segs) - 1:
            found.add(H_A_H)
    return found


# ---------------------------------------------------------------------------------------------------
# reading a capsule
# ---------------------------------------------------------------------------------------------------

#: Ops the frozen ``merlin_iface`` grammar defines that are NOT accelerator dispatches.
_STRUCTURAL_OPS = frozenset({"tensor"})


def iface_mnemonics(text: str) -> list[str]:
    """Every ``merlin_iface`` OP mnemonic in a module, in order — a tokenizer, not a second parser.

    Needed because :func:`contract.interface_emit.parse_interface_mlir` returns only the commands it
    RECOGNISES and says nothing about the ones it did not, so a module using an op outside the frozen
    grammar parses "successfully" into a shorter command list. Measured across the shipped corpora: 15 of
    160 interface capsules contain an op the parser drops — every ``movement`` capsule parses to ZERO
    commands, and every flash-attention capsule loses its second matmul while still reporting success.
    Comparing this inventory against the parsed commands is how a dropped op becomes visible instead of
    becoming a wrong answer.

    Module attributes (``merlin_iface.version``) and types (``!merlin_iface.resident``) are not ops: the
    former only occur on the ``module`` line, the latter always carry the ``!`` sigil.
    """
    marker = "merlin_iface."
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("module") or line.startswith("}"):
            continue
        _, sep, rest = line.partition(marker)
        if not sep:
            continue
        # `!merlin_iface.resident` is a type, and a type never opens a statement; an op does.
        head = line.partition(marker)[0]
        if head.endswith("!"):
            continue
        mnem = rest
        for stop in (" ", "{", "(", ":", ","):
            mnem = mnem.partition(stop)[0]
        if mnem and mnem not in _STRUCTURAL_OPS:
            out.append(mnem)
    return out



def grammar_mnemonics() -> frozenset[str]:
    """The op mnemonics the frozen grammar defines, read from the PARSER'S OWN tables.

    Asking the parser what it knows, rather than restating the contract here, is the difference between
    one authority and two that can drift. If a mnemonic is added to the grammar, this follows for free.
    """
    from merlin.targetgen.contract import interface_emit as IE

    known = set(getattr(IE, "_OP_TO_OPCODE", {}) or {})
    known |= set(getattr(IE, "_NAMED_OP_OPERAND_KEYS", {}) or {})
    return frozenset(known | _STRUCTURAL_OPS)


def undefined_mnemonics(text: str) -> list[str]:
    """Mnemonics a module uses that the frozen grammar does not define, sorted and de-duplicated."""
    known = grammar_mnemonics()
    return sorted({m for m in iface_mnemonics(text) if m not in known})

def _whole_op_opcodes() -> frozenset[str]:
    """Opcodes that are a WHOLE op — they produce their output tensor themselves, with no separate
    commit. Read from the parser's own table so a newly-defined op class is counted for free."""
    from merlin.targetgen.contract import interface_emit as IE

    return frozenset(str(v) for v in (getattr(IE, "_NAMED_OP_TO_OPCODE", {}) or {}).values())


def _accel_dispatches(commands) -> int:
    """How many separate accelerator dispatches a ``merlin_iface`` command list issues.

    A dispatch is the point at which the accelerator produces a tensor the host can see. That is a
    COMMIT for the residency path -- and counting ``MATMUL_RESIDENT`` instead would miscount
    K-accumulation, where several matmuls feed ONE commit and the intermediate never becomes a value at
    all; that is one dispatch, and calling it two would report an accumulation capsule as proving
    composition it does not prove.

    But it is ALSO a whole-op command, which carries its own output and never commits. Counting only
    COMMIT made every such capsule read as zero dispatches and fall into the "commands but no commit"
    branch, which then described a real convolution as "configuration or movement only". Measured the
    moment the grammar gained ``conv2d``: all three conv capsules parsed as ``RES_PACK, CONV2D, EVICT``
    and reported no dispatch at all, so a capsule with two of them would have read ``A`` instead of
    ``A->A``. The whole-op set is read from the parser's own table rather than listed here, so this
    cannot drift the next time an op class is added.
    """
    whole = _whole_op_opcodes()
    n = 0
    for cmd in commands or ():
        opcode = str((cmd or {}).get("opcode") or "")
        if opcode == "COMMIT" or opcode in whole:
            n += 1
    return n


def _sequence_from_linalg(module, target: str) -> tuple[list[str], int]:
    """``[A|H|?]`` per linalg region, in program order, plus how many stayed unresolved."""
    from merlin.targetgen.eligibility import capability_map_for_target, is_eligible
    from merlin.targetgen.model_coverage import regions_from_module

    cap_map = capability_map_for_target(target)
    seq: list[str] = []
    unresolved = 0
    for region in regions_from_module(module):
        family = region.resolved_family()
        if family is None:
            unresolved += 1
            seq.append("?")
            continue
        seq.append(ACCEL if (family in cap_map and is_eligible(region, cap_map).eligible) else HOST)
    return seq, unresolved


def _unbuildable_seam(target: str) -> str | None:
    """Why ``target``'s host/device boundary cannot be emitted, or None when it can.

    Delegates to the ONE predicate that owns the answer (``llvmlower.device_build.boundary_buildable``)
    rather than re-deriving the transport here, so the composition axis and the builder can never
    disagree about which targets have a compilable seam. An unresolvable target answers None -- "we
    could not tell" is the caller's existing UNKNOWN path, not a claim of unbuildability.
    """
    try:
        from merlin.llvmlower.device_build import boundary_buildable
        return boundary_buildable(target)
    except Exception:                                          # noqa: BLE001
        return None


def profile_iface_text(text: str) -> BoundaryProfile:
    """Classify a ``merlin_iface`` capsule from its text.

    Separate from the linalg path because the two grammars answer the question differently, and because
    xDSL needs a FILE for the linalg one. Split so the merlin_iface classification -- the one that covers
    most of a corpus -- is testable from a string with no parser session at all.
    """
    from merlin.targetgen.contract import interface_emit as IE

    try:
        doc = IE.parse_interface_mlir(text)
    except Exception as e:                                     # noqa: BLE001
        return BoundaryProfile(grammar="merlin_iface",
                               detail=f"unparseable: {type(e).__name__}: {e}")
    commands = list(doc.get("commands") or ())
    undefined = undefined_mnemonics(text)
    if undefined:
        # THE PARSER DROPPED SOMETHING, so any shape derived from what it returned is a shape derived
        # from an incomplete program. UNKNOWN with the reason, never a confident label: a movement
        # capsule whose only op vanished would otherwise report as a clean single dispatch.
        return BoundaryProfile(grammar="merlin_iface",
                               n_accel_regions=len(commands),
                               detail=f"the frozen interface grammar does not define {undefined}; the "
                                      f"canonical parser returned {len(commands)} command(s) and "
                                      f"reported no error, so the program is only partly readable")
    n = _accel_dispatches(commands)
    if not n:
        # Commands but no commit: a configuration- or movement-only program. It is accelerator work
        # with no dispatch boundary, not host work -- and not UNKNOWN either, because we read it fine.
        if not commands:
            return BoundaryProfile(grammar="merlin_iface", detail="module declares no accelerator op")
        return BoundaryProfile(kind=A, grammar="merlin_iface", contains=(A,),
                               n_accel_regions=len(commands), accel_segments=1,
                               detail="accelerator commands that produce no host-visible tensor "
                                      "(configuration or residency only): one accelerator region, no "
                                      "dispatch seam")
    return BoundaryProfile(kind=A_A if n >= 2 else A, grammar="merlin_iface",
                           contains=((A, A_A) if n >= 2 else (A,)),
                           n_accel_regions=n, accel_segments=1,
                           detail=f"{n} accelerator dispatch(es); the merlin_iface grammar carries no "
                                  f"host computation, so no host seam can exist in it")


def profile_path(path: str | Path, target: str) -> BoundaryProfile:
    """Classify the capsule interface at ``path``."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        return BoundaryProfile(detail=f"unreadable: {type(e).__name__}: {e}")

    from merlin.targetgen.contract import linalg_iface as LI
    try:
        is_linalg = LI.is_linalg_on_tensors(text)
    except Exception as e:                                     # noqa: BLE001
        return BoundaryProfile(detail=f"grammar undecidable: {type(e).__name__}: {e}")
    if not is_linalg:
        return profile_iface_text(text)

    from merlin.targetgen.model_coverage import load_module
    try:
        module = load_module(p)
    except Exception as e:                                     # noqa: BLE001
        return BoundaryProfile(grammar="linalg", detail=f"unparseable: {type(e).__name__}: {e}")
    seq, unresolved = _sequence_from_linalg(module, target)
    # AN ELIGIBLE REGION IS NOT A CROSSING WE CAN EMIT. Eligibility says the capability manifest admits
    # the work; it says nothing about whether any path in this repo can compile the seam that carries it.
    # On a device_native target the boundary is a DRAM address contract honoured by the harness, not a
    # linkable call, so `device_build` refuses it outright -- and yet this function would happily label
    # every admitted region `A` and let a corpus report `H->A->H` covered on a target where that seam is
    # uncompilable. UNKNOWN with the reason instead, and NOT by dropping the regions: dropping them turns
    # [A,H,A] into [H] and reports a host-only capsule, which is a stronger false claim than the one it
    # replaces.
    n_eligible = sum(1 for x in seq if x == ACCEL)
    unbuildable = _unbuildable_seam(target) if n_eligible else None
    if unbuildable:
        return BoundaryProfile(
            kind=UNKNOWN, grammar="linalg", n_unresolved=unresolved, n_unbuildable=n_eligible,
            n_host_regions=sum(1 for x in seq if x == HOST),
            detail=f"{n_eligible} of {len(seq)} linalg region(s) are accelerator-eligible, but "
                   f"{unbuildable} -- so whether this program crosses the seam is undeterminable, "
                   f"not proven")
    segs = segments(seq)
    return BoundaryProfile(
        kind=classify_sequence(seq), contains=tuple(sorted(patterns_in_sequence(seq))),
        grammar="linalg",
        n_accel_regions=sum(1 for s in seq if s == ACCEL),
        n_host_regions=sum(1 for s in seq if s == HOST),
        n_unresolved=unresolved,
        accel_segments=sum(1 for k, _ in segs if k == ACCEL),
        host_segments=sum(1 for k, _ in segs if k == HOST),
        detail=f"{len(seq)} linalg region(s); eligibility from the target's capability map")


def capsule_interface(capsule_dir: str | Path) -> Path | None:
    """The interface MLIR a capsule declares, whichever field it declares it under."""
    import yaml

    d = Path(capsule_dir)
    cy = d / "capsule.yaml"
    if not cy.is_file():
        return None
    try:
        doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    for key in ("interface_mlir", "linalg_mlir"):
        name = doc.get(key)
        if name and (d / str(name)).is_file():
            return d / str(name)
    return None


def profile_capsule(capsule_dir: str | Path, target: str) -> BoundaryProfile:
    """Classify one capsule directory. A capsule declaring no interface is UNKNOWN, never ``H``."""
    iface = capsule_interface(capsule_dir)
    if iface is None:
        return BoundaryProfile(detail="capsule declares no readable interface MLIR")
    return profile_path(iface, target)


# ---------------------------------------------------------------------------------------------------
# the two sides of the gate
# ---------------------------------------------------------------------------------------------------

def corpus_boundaries(corpus_roots, target: str, *, labels=None, exclude=None) -> dict:
    """``kind -> [capsule names]`` for a corpus, plus the ones that could not be read.

    Mirrors ``cert_capsule_cover``'s selection rules (label filter, grading exclusions) so the boundary
    axis measures the same corpus the family/dtype axes do. A capsule the descriptor withholds from
    grading cannot evidence a boundary type any more than it can evidence a cell.
    """
    import yaml

    labels = set(labels or {"public"})
    exclude = set(exclude or ())
    roots = [corpus_roots] if isinstance(corpus_roots, (str, Path)) else list(corpus_roots)
    by_kind: dict[str, list[str]] = {}
    primary: dict[str, list[str]] = {}
    unread: dict[str, str] = {}
    for root in roots:
        for cy in sorted(Path(root).glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            name = cap.get("name") or cy.parent.name
            if name in exclude:
                continue
            prof = profile_capsule(cy.parent, target)
            if prof.kind == UNKNOWN:
                unread[name] = prof.detail
                continue
            primary.setdefault(prof.kind, []).append(name)
            # CREDIT EVERY SHAPE THE CAPSULE CONTAINS, not only the strongest one it is named by.
            # The requirement side already reads a capture's CONTAINED shapes, and crediting the corpus
            # side with one label each made the two sides ask different questions: a whole-model capsule
            # whose sequence genuinely opens an accelerator region, crosses to the host and comes back is
            # named `routing`, and `A->H->A` was then reported uncovered while a graded capsule was
            # exercising it. That is the same under-crediting as scoring a fused capsule for one family.
            for kind in sorted(prof.contains or {prof.kind}):
                by_kind.setdefault(kind, []).append(name)
    return {"by_kind": {k: sorted(v) for k, v in sorted(by_kind.items())},
            "primary": {k: sorted(v) for k, v in sorted(primary.items())},
            "unreadable": unread}


def required_boundaries(captures: dict, target: str) -> dict:
    """Which composition shapes the REAL captured models present — the requirement.

    Same evidential rule as the family axis: a shape no real model exhibits is not required of the
    corpus, however expressible it is. A shape several models exhibit and no capsule covers is a hole the
    pass-rate cannot express, because every capsule that exists still passes.
    """
    from merlin.targetgen.model_coverage import load_module

    by_kind: dict[str, list[str]] = {}
    unreadable: dict[str, str] = {}
    whole_model: dict[str, str] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            module = load_module(path)
        except Exception as e:                                 # noqa: BLE001
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"
            continue
        seq, _ = _sequence_from_linalg(module, target)
        whole_model[label] = classify_sequence(seq)
        for kind in sorted(patterns_in_sequence(seq)):
            by_kind.setdefault(kind, []).append(label)
    return {"by_kind": {k: sorted(v) for k, v in sorted(by_kind.items())},
            "whole_model_shape": whole_model, "captures_unreadable": unreadable}


def uncovered_boundaries(required: dict, corpus: dict) -> dict:
    """Required composition shapes no capsule in the corpus exercises."""
    want = list((required or {}).get("by_kind") or {})
    have = set((corpus or {}).get("by_kind") or {})
    missing = sorted(set(want) - have)
    return {
        "n_required": len(want),
        "n_covered": len(set(want) & have),
        "uncovered": missing,
        "corpus_kinds": sorted(have),
        "extra_kinds": sorted(have - set(want)),
        "covered_only_incidentally": sorted(
            k for k in set(want) & have
            if k not in ((corpus or {}).get("primary") or {})),
        "unreadable_capsules": dict((corpus or {}).get("unreadable") or {}),
        "note": ("a required composition shape with no capsule means the corpus proves families but not "
                 "composition; the pass-rate cannot express it because every capsule that exists passes"),
    }
