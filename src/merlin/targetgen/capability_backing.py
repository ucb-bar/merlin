"""A capability claim that NARROWS a target must cite what it was read off, not assert it.

THE DEFECT THIS EXISTS FOR. A hand-written contract line said an elementwise map on one systolic
target existed ONLY fused behind a contraction, with two lines of prose as its whole justification.
The RTL says the opposite in both directions: the device runs a STANDALONE two-operand accumulator
add (the loop unroller's two loads pass ``accumulate = false.B`` and ``accumulate = true.B`` under
``is_resadd``, K at 0 and D NULL), and the fused form the line licensed is the one it cannot express
(a contraction's only DRAM-to-accumulator port passes ``accumulate = false.B`` and the per-channel
bias already occupies the region). Consequence: no standalone elementwise capsule was ever derived,
``residual_add`` was demanded by 0 of 103 graded capsules, the agent was graded 101/103 "complete",
and the compiler it produced refused 16 of a 71-group ResNet-50.

WHY AN OP-COVERAGE CHECK COULD NOT CATCH IT. Such a check asks "is there an op the manifest
ADMITS that no capsule DEMANDS?" -- it reads the manifest, so a manifest that wrongly
says a capability does not exist produces SILENCE, not a finding. The removed demand is invisible
precisely because the declaration that removed it is the thing being trusted.

SO THE PROPERTY HERE IS ABOUT THE DECLARATION ITSELF, and it is deliberately not "is the claim true"
(nothing in this repo can decide that from the YAML). It is:

    **A claim that NARROWS what a target supports must carry a resolvable citation -- an RTL source in
    the read set of one of that target's own declared hardware pins, an in-repo test or module that
    pins it, or a derivation entry point -- and every citation it carries must still resolve.**

Narrowing is the dangerous direction and the only direction gated. Over-declaring a capability
produces capsules the hardware refuses, which is loud; under-declaring removes rows from a denominator,
which is silent and flatters every recall number computed afterwards.

A CITATION THAT NO LONGER RESOLVES IS A FAILURE, not a pass. Half the value is that the citation
cannot rot into decoration: a test that moved, a module that was renamed, or an RTL file that is in no
pin's ``requires_paths`` are each reported by name. The last is the one that bit hardest -- a claim
may cite exactly the right Chisel file while the pin that names the checkout never reads it, so an
edit to it changes the verdict and ``verify`` says the pin is clean.

WHAT IS IN SCOPE, and why it is not everything. The claim shapes below are the ones on the
ARR-DENOMINATOR axis: the semantic-capability declarations that :mod:`merlin.targetgen.eligibility`
turns into a per-region verdict, plus the families a contract exempts from ever being graded. A
contract narrows in other shapes too (a unit's ``ops`` enumeration, a ``backends`` list, a
``legality`` clause, a ``false`` feature flag), and :data:`DEFERRED_SHAPES` names them rather than
leaving their absence to be mistaken for coverage. They are not gated yet because they do not feed the
eligibility predicate, so a wrong one costs a capsule rather than a denominator.

Target-agnostic by construction: every fact about a target is read from that target's own contract and
its own declared pins, and the target is a parameter throughout.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Claim",
    "Citation",
    "DEFERRED_SHAPES",
    "NARROWING_SHAPES",
    "audit",
    "audit_document",
    "claims",
    "claims_in_document",
    "gated_targets",
    "debt_key",
    "problems",
]


# ---------------------------------------------------------------------------------------------------
# what counts as a narrowing claim
# ---------------------------------------------------------------------------------------------------

#: The gated shapes: ``key -> what the shape means``. DATA, not control flow -- adding a shape is an
#: edit here plus an extractor below, and the CLI prints this table so the scope is never implicit.
NARROWING_SHAPES: dict[str, str] = {
    "composed_with": (
        "the family is declared reachable ONLY attached to a named producer, so a region that asks "
        "for it standalone is refused"
    ),
    "ranks": (
        "the family is declared only at the listed operand ranks, so a region of any other rank is "
        "refused before it reaches the device rewrite"
    ),
    "family_dtypes": (
        "the family admits FEWER dtypes than the compute unit that carries it, so a region the unit's "
        "own datapath holds is refused on the family"
    ),
    "unmaterializable": (
        "the family is declared present in hardware but exempt from ever being graded, which the "
        "coverage gate reads as a reviewed exemption and stops asking about"
    ),
    "undetermined": (
        "the family is declared undecidable, so every region on it is scored unmeasured rather than either way"
    ),
    "no_semantic_capabilities": (
        "the compute unit declares NO semantic families at all, so every region routed at it is "
        "refused as `undeclared_family` -- the most total narrowing a unit can express, and the one "
        "that reads in every aggregate as work the hardware cannot do"
    ),
}

#: Shapes this gate does NOT decide, named so their absence is not read as coverage. Each narrows, and
#: each is a candidate for a later revision of this gate; none of them feeds the eligibility predicate,
#: which is why they are not failures today.
DEFERRED_SHAPES: dict[str, str] = {
    "unit_ops": "a compute unit's `ops` enumeration -- narrower than the unit's semantic families",
    "unit_dtypes": "a compute unit's own `dtypes` list, which has no wider declaration to compare to",
    "absent_family": "a family declared NOWHERE, which has no line to attach a citation to",
    "backends": "a `runtime.backends` enumeration -- a harness reach, not a hardware capability",
    "legality": "a `legality` / `compiler_obligations` clause -- an obligation, not a family verdict",
    "feature_flag": "a bare `false` feature flag (memory_model, readout, completion)",
    "capability_exclude": "a capsule exclusion list in an EXPERIMENT descriptor, not in the contract",
}

#: Suffixes that make a token a candidate RTL source -- something that lives in a pinned external
#: checkout rather than in this repository.
_RTL_SUFFIXES = (".scala", ".sv", ".v", ".fir", ".firrtl", ".vh", ".chisel")
#: Suffixes that make a token a candidate in-repo path.
_REPO_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".md", ".mlir", ".c", ".cc", ".cpp", ".h", ".txt")
#: Trailing/leading punctuation stripped off a token before it is classified. A citation is written
#: inside prose, so it arrives wrapped in backticks, parentheses and sentence punctuation.
_EDGE = "`'\"(),;:[]{}<>*|!?"


class BackingError(ValueError):
    """A contract this reader cannot parse structurally. Never softened into "no claims"."""


# ---------------------------------------------------------------------------------------------------
# the claim record
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """One thing a claim's comment block points at, and whether it still resolves."""

    #: The token exactly as the comment spells it.
    token: str
    #: ``repo_path`` | ``target_path`` | ``rtl_pin`` | ``module`` | ``unresolved_repo_path``
    #: | ``unpinned_rtl`` | ``missing_module``
    kind: str
    #: What it resolved to (a path, a pin name, a module name), or why it did not.
    detail: str

    @property
    def resolves(self) -> bool:
        return self.kind in ("repo_path", "target_path", "rtl_pin", "module")


@dataclass(frozen=True)
class Claim:
    """One narrowing declaration, where it is written, and what it cites."""

    target: str
    #: A key of :data:`NARROWING_SHAPES`.
    shape: str
    #: What is narrowed -- the family name, for every shape declared today.
    detail: str
    #: Repo-relative path of the contract document the claim is written in.
    source: str
    #: 1-based line of the declaration.
    line: int
    #: The declaration line, stripped.
    text: str
    citations: tuple[Citation, ...] = ()
    #: Comment lines attached to the claim, for a report that has to show the evidence.
    comment: tuple[str, ...] = field(default=(), repr=False)

    @property
    def key(self) -> str:
        return f"{self.shape}:{self.detail}"

    @property
    def backed(self) -> bool:
        return any(c.resolves for c in self.citations)

    @property
    def broken(self) -> tuple[Citation, ...]:
        """Citations that do NOT resolve. A claim with one of these fails even when it is backed --
        a citation that has rotted is how a reviewed claim becomes an unreviewed one without a diff."""
        return tuple(c for c in self.citations if not c.resolves)

    def why(self) -> str:
        if not self.citations:
            return "no citation at all: the comment block names no RTL source, no test, no module"
        if not self.backed:
            return "cites only things that do not resolve: " + "; ".join(f"{c.token} ({c.detail})" for c in self.broken)
        return "backed by " + ", ".join(f"{c.token} [{c.kind}]" for c in self.citations if c.resolves)


def debt_key(target: str, claim_key: str, *, axis: str) -> str:
    """The ratchet entry for one claim.

    Target-scoped, because a claim is debt on one target and evidence on another and a flat entry would
    forgive every target at once; and axis-scoped, because "cites nothing" and "cites something that no
    longer resolves" are different defects with different fixes, and one line may not forgive both.

    NOT document-scoped, deliberately. A target that ships both a curated contract and a residual
    declares the same claim twice, and the two are required to agree (a test compares them), so one
    ledger line covering both is the honest granularity -- a per-document key would let the pair
    disagree with one half forgiven.
    """
    return f"{target} {axis}:{claim_key}"


# ---------------------------------------------------------------------------------------------------
# reading the contract documents, with line numbers and the comments the loader throws away
# ---------------------------------------------------------------------------------------------------


def _repo_root() -> Path:
    from merlin.common.paths import repo_root

    return repo_root()


#: Where a target's contract documents live relative to its base, and which documents declare
#: capabilities. Both are the layout the target registry itself reads.
_CONTRACTS = "contracts"
_DOCUMENTS = ("target_contract.yaml", "residual.yaml")


def _declared_name(path: Path) -> str | None:
    """The ``name:`` a contract document declares for its target, read structurally."""
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    name = doc.get("name") if isinstance(doc, dict) else None
    return name if isinstance(name, str) and name else None


def _contract_bases() -> dict[str, Path]:
    """``target -> base`` for every in-tree target directory that ships a contract document, including
    one the registry does not resolve -- a target with only a ``residual.yaml``, which the capability
    deriver still reads.

    The name is the one the DOCUMENT declares, never the directory it sits in: the two differ, and a
    directory-keyed scan once attributed a pinned copy's capsules to a name no target owns. The reference
    examples are searched before the bundled contract mirrors, so a target present in both is audited
    where it is maintained.
    """
    repo = _repo_root()
    out: dict[str, Path] = {}
    for pattern in (f"examples/*/target/{_CONTRACTS}", f"merlin/targets/*/{_CONTRACTS}"):
        for contracts in sorted(repo.glob(pattern)):
            for document in _DOCUMENTS:
                path = contracts / document
                if not path.is_file():
                    continue
                name = _declared_name(path)
                if name is not None:
                    out.setdefault(name, contracts.parent)
                break
    return out


def _target_base(target: str) -> Path | None:
    try:
        from merlin.targetgen.rtl.facts import target_base

        base = target_base(target)
    except Exception:  # noqa: BLE001 -- not registry-resolvable; the residual-only scan below decides
        base = None
    if base and Path(base).is_dir():
        return Path(base)
    # A target that ships only a residual is not resolvable through the registry, and its declarations
    # are read all the same. Fail closed on the directory rather than on the resolver.
    return _contract_bases().get(target)


def gated_targets() -> tuple[str, ...]:
    """Every target this gate has a contract document for.

    WIDER than :func:`target_registry.all_targets`, and deliberately. That list needs a
    ``target_contract.yaml``; a target that ships only a ``residual.yaml`` is still a target whose
    declarations the capability deriver reads. Discovering by "has a contract document" rather than by a
    registry list is also what keeps a new target from arriving unsurveyed.
    """
    names: set[str] = set(_contract_bases())
    try:
        from merlin.targetgen import target_registry as tr

        names.update(tr.all_targets())
    except Exception as exc:  # noqa: BLE001 -- a broken registry must not read as "no targets"
        raise BackingError(f"the target registry could not be read: {type(exc).__name__}: {exc}") from exc
    return tuple(sorted(names))


def contract_documents(target: str) -> tuple[tuple[Path, str], ...]:
    """Every contract document that declares capabilities for ``target``, as ``(path, text)``.

    BOTH the curated reference contract AND the residual side-input, when a target ships both, because
    they are different files read by different consumers and they have drifted: a residual that denies
    a family the curated contract declares produces a derived manifest without it, which is the same
    invisible denominator loss by another route.
    """
    base = _target_base(target)
    if base is None:
        return ()
    out: list[tuple[Path, str]] = []
    for name in _DOCUMENTS:
        path = Path(base) / _CONTRACTS / name
        if path.is_file():
            out.append((path, path.read_text(encoding="utf-8")))
    return tuple(out)


def _comment_block(lines: list[str], line0: int) -> tuple[str, ...]:
    """The contiguous comment lines immediately above 0-based ``line0``, plus its own trailing one.

    Walks UPWARDS and stops at the first line that is neither a comment nor blank-inside-a-block. A
    blank line ends the block: a comment separated from a declaration by whitespace is prose about the
    section, not evidence for the line.
    """
    out: list[str] = []
    i = line0 - 1
    while i >= 0:
        stripped = lines[i].strip()
        if not stripped.startswith("#"):
            break
        out.append(stripped.lstrip("#").strip())
        i -= 1
    out.reverse()
    own = lines[line0] if 0 <= line0 < len(lines) else ""
    _before, sep, after = own.partition("#")
    if sep:
        out.append(after.strip())
    return tuple(out)


def _child(node: Any, key: str):
    """The value node under ``key`` of a YAML mapping NODE (not a loaded dict), or ``None``."""
    if not isinstance(getattr(node, "value", None), list):
        return None
    for k, v in node.value:
        if getattr(k, "value", None) == key:
            return v
    return None


def _scalar(node: Any):
    """A node's loaded Python value, via the same loader the rest of the repo uses."""
    if node is None:
        return None
    return yaml.safe_load(yaml.serialize(node))


def _seq(node: Any) -> list:
    return list(node.value) if node is not None and isinstance(getattr(node, "value", None), list) else []


# ---------------------------------------------------------------------------------------------------
# citation resolution
# ---------------------------------------------------------------------------------------------------


def _pin_read_paths(target: str) -> dict[str, tuple[str, ...]]:
    """``pin name -> its requires_paths``, for every pin THIS target's contract declares.

    The target's own pins and no others. A claim about one device may not be evidenced by a file in
    another device's checkout -- that is precisely the inheritance-by-assumption the defect came from.
    """
    try:
        from merlin.common import provenance
        from merlin.targetgen.provenance import declared_pins
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for name in declared_pins(target):
        try:
            out[name] = provenance.pin(name).requires_paths
        except Exception:  # noqa: BLE001 -- an undeclared pin name is the pin registry's problem
            continue
    return out


def _tokens(comment: tuple[str, ...]) -> list[str]:
    """Candidate citation tokens, split structurally. No regex.

    Every character in :data:`_EDGE` becomes a separator rather than being stripped only at the ends.
    That is not cosmetic: prose wraps a citation in backticks AND inflects it, so
    ``` `StoreController.scala`'s ``` ends in ``s`` and an end-strip leaves the backtick embedded --
    the citation is then invisible and the claim reads as bare prose. A too-narrow tokenizer that
    silently drops a valid spelling is the failure mode this repo bans regex for; the separator set is
    the conservative direction, because over-splitting can only produce non-citation fragments, which
    :func:`_classify` ignores.

    ``::`` and ``#`` are partitioned off afterwards so ``module.py::func`` and ``a.h#L3`` both reduce
    to the artifact being cited. ``.`` / ``/`` / ``_`` / ``-`` survive, because they are inside the
    names.
    """
    out: list[str] = []
    for line in comment:
        spaced = "".join(" " if ch in _EDGE else ch for ch in line)
        for raw in spaced.split():
            tok = raw.partition("::")[0].partition("#")[0]
            while tok and tok[-1] in ".,-":
                tok = tok[:-1]
            if tok:
                out.append(tok)
    return out


def _in_a_pin(token: str, pins: dict[str, tuple[str, ...]]) -> str | None:
    """``"<pin>:<path>"`` when ``token`` names a file some declared pin READS, else ``None``.

    Matched on the full relative path first and on the basename second: a comment cites the file, and
    whether it spells the whole source root is an editorial choice, not evidence about the device.
    """
    want = token.rsplit("/", 1)[-1]
    for pin_name in sorted(pins):
        for rel in pins[pin_name]:
            if rel == token or rel.rsplit("/", 1)[-1] == want:
                return f"{pin_name}:{rel}"
    return None


def _classify(token: str, *, repo: Path, base: Path | None, pins: dict[str, tuple[str, ...]]) -> Citation | None:
    """What ``token`` cites, or ``None`` when it is ordinary prose rather than a citation."""
    low = token.lower()
    path_shaped = low.endswith(_REPO_SUFFIXES) and "/" in token
    if path_shaped:
        if (repo / token).is_file():
            return Citation(token, "repo_path", str(repo / token))
        if base is not None and (Path(base) / token).is_file():
            return Citation(token, "target_path", str(Path(base) / token))
    if path_shaped or low.endswith(_RTL_SUFFIXES):
        # Not in this repository. It may still be a file a declared pin READS, which is the strongest
        # citation there is: the pin names the revision, so `verify` reports an edit to it.
        got = _in_a_pin(token, pins)
        if got is not None:
            return Citation(token, "rtl_pin", got)
        if not pins:
            return Citation(
                token,
                "unpinned_rtl" if not path_shaped else "unresolved_repo_path",
                "not in this repo and this target's contract declares no hardware_pins",
            )
        return Citation(
            token,
            "unpinned_rtl" if not path_shaped else "unresolved_repo_path",
            "not in this repo and in no requires_paths of "
            + ", ".join(sorted(pins))
            + "; add it to that pin's read set",
        )
    if token.startswith("merlin.") and "." in token[7:]:
        # The token itself, or the token minus ONE trailing segment (a function or attribute name).
        # Never an unbounded walk up the package tree: that accepted `merlin.compile.linalg_lower`,
        # a module that does not exist, because `merlin.compile` does -- so any dotted token beginning
        # with a real package read as a citation, and a wrong module name could not fail. The floor of
        # three segments is what stops the one permitted step from landing on a bare package.
        parent = token.rpartition(".")[0]
        for mod in (token, parent if parent.count(".") >= 2 else ""):
            if not mod:
                continue
            try:
                if importlib.util.find_spec(mod) is not None:
                    return Citation(token, "module", mod)
            except (ImportError, ValueError, ModuleNotFoundError):
                continue
        return Citation(token, "missing_module", "no importable module at this dotted path")
    return None


def _citations(comment: tuple[str, ...], *, repo: Path, base: Path | None, pins) -> tuple[Citation, ...]:
    seen: dict[str, Citation] = {}
    for token in _tokens(comment):
        got = _classify(token, repo=repo, base=base, pins=pins)
        if got is not None and got.token not in seen:
            seen[got.token] = got
    return tuple(seen.values())


# ---------------------------------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------------------------------


def claims_in_document(
    text: str,
    *,
    target: str,
    source: str,
    repo: Path | None = None,
    base: Path | None = None,
    pins: dict[str, tuple[str, ...]] | None = None,
) -> tuple[Claim, ...]:
    """Every gated narrowing claim in ONE contract document, with its evidence resolved.

    Public because a test of this gate must be able to hand it a contract it wrote itself. A gate
    whose only input is the repository it guards can only be tested by editing that repository, and a
    test that mutates a tracked contract is a test that can leave the tree wrong when it fails.
    """
    repo = repo if repo is not None else _repo_root()
    pins = {} if pins is None else pins
    out: list[Claim] = []
    if True:
        lines = text.splitlines()
        try:
            root = yaml.compose(text)
        except yaml.YAMLError as exc:
            raise BackingError(f"{source}: not composable as YAML: {exc}") from exc
        if root is None:
            return ()
        rel = source

        def _emit(shape: str, detail: str, line0: int, extra: tuple[str, ...] = ()) -> None:
            # THE CLAIM'S OWN PROSE COUNTS AS ITS COMMENT. `unmaterializable_families` states its
            # reason in the VALUE, not in a `#` line above the key, and a reader that only looked at
            # comments found "no citation at all" on entries that cite a source file by line range.
            comment = _comment_block(lines, line0) + tuple(extra)
            out.append(
                Claim(
                    target=target,
                    shape=shape,
                    detail=detail,
                    source=rel,  # noqa: B023 -- bound per document, consumed immediately
                    line=line0 + 1,
                    text=lines[line0].strip() if 0 <= line0 < len(lines) else "",  # noqa: B023
                    citations=_citations(comment, repo=repo, base=base, pins=pins),
                    comment=comment,
                )
            )

        for unit_node in _seq(_child(root, "compute_units")):
            unit_dtypes = _scalar(_child(unit_node, "dtypes")) or []
            caps_node = _child(unit_node, "semantic_capabilities")
            if not _seq(caps_node):
                unit_name = str(_scalar(_child(unit_node, "name")) or "?")
                _emit("no_semantic_capabilities", unit_name, unit_node.start_mark.line)
                continue
            for cap_node in _seq(caps_node):
                family = str(_scalar(_child(cap_node, "family")) or "?")
                entry_line = cap_node.start_mark.line
                for key, shape in (("composed_with", "composed_with"), ("ranks", "ranks")):
                    val_node = _child(cap_node, key)
                    if val_node is None:
                        continue
                    if not (_scalar(val_node) or []):
                        continue  # an EMPTY list is the refusal of a restriction, not a restriction
                    _emit(shape, family, min(entry_line, val_node.start_mark.line))
                dt_node = _child(cap_node, "dtypes")
                fam_dtypes = _scalar(dt_node) or []
                if unit_dtypes and fam_dtypes and set(fam_dtypes) < set(unit_dtypes):
                    _emit("family_dtypes", family, min(entry_line, dt_node.start_mark.line))

        unmat = _child(root, "unmaterializable_families")
        for k, v in _seq(unmat):
            reason = str(_scalar(v) or "").strip()
            if reason:
                _emit("unmaterializable", str(k.value), k.start_mark.line, tuple(reason.splitlines()))

        for entry in _seq(_child(root, "semantic_capabilities_unknown")):
            fam_node = _child(entry, "family")
            fam = _scalar(fam_node) if fam_node is not None else _scalar(entry)
            why_node = _child(entry, "reason")
            why = str(_scalar(why_node) or "").strip() if why_node is not None else ""
            _emit("undetermined", str(fam), entry.start_mark.line, tuple(why.splitlines()))
    return tuple(out)


def claims(target: str) -> tuple[Claim, ...]:
    """Every gated narrowing claim ``target`` declares, across every contract document it ships."""
    repo = _repo_root()
    base = _target_base(target)
    pins = _pin_read_paths(target)
    out: list[Claim] = []
    for path, text in contract_documents(target):
        rel = str(path.relative_to(repo)) if str(path).startswith(str(repo)) else str(path)
        out.extend(claims_in_document(text, target=target, source=rel, repo=repo, base=base, pins=pins))
    return tuple(out)


# ---------------------------------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------------------------------


def audit_document(
    text: str,
    *,
    target: str,
    source: str,
    repo: Path | None = None,
    base: Path | None = None,
    pins: dict[str, tuple[str, ...]] | None = None,
    ratchet: set[str] | None = None,
) -> dict:
    """:func:`audit` over ONE document handed in directly, for a test that writes its own contract."""
    found = claims_in_document(text, target=target, source=source, repo=repo, base=base, pins=pins)
    return _verdict(target, found, set(ratchet or ()), pins or {})


def audit(target: str, *, ratchet: set[str] | None = None) -> dict:
    """``target``'s narrowing claims, which are backed, and the ratchet reconciliation."""
    return _verdict(target, claims(target), set(ratchet or ()), _pin_read_paths(target))


def _verdict(target: str, found: tuple[Claim, ...], ratchet: set, pin_reads: dict) -> dict:
    unbacked = tuple(c for c in found if not c.backed)
    rotted = tuple(c for c in found if c.backed and c.broken)
    live = {debt_key(target, c.key, axis="unbacked") for c in unbacked}
    live |= {debt_key(target, c.key, axis="rotted") for c in rotted}
    mine = {e for e in ratchet if e.split(" ", 1)[0] == target}

    where = ", ".join(sorted(pin_reads)) if pin_reads else "a hardware pin this contract declares"
    probs: list[str] = []
    for c in unbacked:
        if debt_key(target, c.key, axis="unbacked") in mine:
            continue
        probs.append(
            f"{c.source}:{c.line} declares `{c.text}` -- {NARROWING_SHAPES[c.shape]} -- and {c.why()}. "
            f"Cite an RTL source in the read set of {where}, an in-repo test or module that pins it, "
            f"or a derivation entry point -- or drop the restriction."
        )
    for c in rotted:
        if debt_key(target, c.key, axis="rotted") in mine:
            continue
        probs.append(
            f"{c.source}:{c.line} declares `{c.text}` and its evidence has ROTTED: "
            + "; ".join(f"{b.token} -- {b.detail}" for b in c.broken)
            + ". Repoint the citation or remove it; a citation nobody can follow is prose."
        )
    stale = sorted(mine - live)

    return {
        "target": target,
        "n_claims": len(found),
        "claims": [
            {
                "shape": c.shape,
                "detail": c.detail,
                "source": c.source,
                "line": c.line,
                "text": c.text,
                "backed": c.backed,
                "why": c.why(),
                "citations": [{"token": x.token, "kind": x.kind, "detail": x.detail} for x in c.citations],
            }
            for c in found
        ],
        "unbacked": sorted(debt_key(target, c.key, axis="unbacked") for c in unbacked),
        "rotted": sorted(debt_key(target, c.key, axis="rotted") for c in rotted),
        "ratcheted": sorted(mine & live),
        "stale_ratchet_entries": stale,
        "problems": probs,
        "deferred_shapes": dict(DEFERRED_SHAPES),
        "pins": {k: list(v) for k, v in pin_reads.items()},
    }


def problems(target: str, *, ratchet: set[str] | None = None) -> tuple[list[str], dict]:
    doc = audit(target, ratchet=ratchet)
    return list(doc["problems"]), doc
