"""Materialize a sandbox view of the capsule corpus from the single source of truth.

The capsule-bench sandbox needs the *public* capsules (``label: public``) with their oracle-tier
requirement capped to what the sandbox can actually reach — the bwrap sandbox has no VCS/FireSim, so
tiers above L2 are unreachable. Everything else is byte-identical to ``merlin/contract/capsules/``.

Deriving this view (instead of hand-maintaining a committed copy) keeps ONE source of truth: the
committed sandbox copy at ``…/scripts/full_public_capsules/`` drifted from the contract (it had
``required_oracle_tiers`` reindented and L3 dropped by hand). ``materialize_public_capsules`` is that
transform, made explicit and testable.

Usage (as a library):
    from merlin.targetgen.contract.materialize import materialize_public_capsules
    materialize_public_capsules(dest, tier_ceiling="L2")

CLI:
    python -m merlin.targetgen.contract.materialize <dest_dir> [--tier-ceiling L2]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import yaml

from .schemas import contract_dir

# The 5 files that make up a capsule (see generate_corpus.py / AGENT.md).
_CAPSULE_FILES = ("capsule.yaml", "capsule.interface.mlir", "golden.yaml",
                  "expected_instruction_coverage.yaml", "README.md")
_TIER_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]
_DEFAULT_CEILING = "L2"  # bwrap sandbox: numerics + spike, no VCS/FireSim (L3+).


def _public_capsule_dirs(contract: str | Path | None = None) -> list[Path]:
    """Every capsule dir whose capsule.yaml is label: public, across isa/layers/model_slices.

    This materializes THIS (gemmini) contract's public capsules — its capsules live at
    ``capsules/<category>/<cap>/capsule.yaml`` (rel-depth 3). Another target's corpus that nests under the
    same root (e.g. ``capsules/atlas/<category>/<cap>/capsule.yaml``, rel-depth 4) must NOT be materialized
    into the gemmini sandbox, so restrict to the direct category/capsule depth."""
    root = contract_dir(contract) / "capsules"
    out = []
    for cap_yaml in sorted(root.rglob("capsule.yaml")):
        if len(cap_yaml.relative_to(root).parts) != 3:   # skip nested per-target corpora (atlas/, …)
            continue
        try:
            cap = yaml.safe_load(cap_yaml.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if cap.get("label") == "public":
            out.append(cap_yaml.parent)
    return out


def _public_capsule_dirs_in(corpus_roots: list[Path]) -> list[Path]:
    """Every ``label: public`` capsule dir found DIRECTLY under the given corpus roots (``<root>/<cap>/
    capsule.yaml``). Target-AGNOSTIC: the caller passes the descriptor's own ``capsule_corpus`` + sibling
    corpora, so any target's public set is materialized without a per-target root/depth assumption."""
    out: list[Path] = []
    for root in corpus_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for cap_yaml in sorted(root.glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cap_yaml.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") == "public":
                out.append(cap_yaml.parent)
    return out


def _cap_tiers(tier_ceiling: str) -> list[str]:
    if tier_ceiling not in _TIER_ORDER:
        raise ValueError(f"unknown tier ceiling {tier_ceiling!r} (expected one of {_TIER_ORDER})")
    keep = _TIER_ORDER[: _TIER_ORDER.index(tier_ceiling) + 1]
    return keep


# The RTL-derived / numeric oracle tiers. L0/L1 are the integer reference/simulate floor (marked
# not_applicable on a float datapath); a real numeric grade always rides one of these.
_RTL_TIERS = frozenset({"L2", "L3", "L4", "L5"})


def _cap_required(tiers: list[str], keep: set[str]) -> tuple[list[str], list[str]]:
    """Cap a capsule's ``required_oracle_tiers`` to what THIS phase can reach. Returns
    ``(kept, unreachable)`` — a PURE intersection plus the declared tiers this phase cannot reach.

    NEVER substitutes. An earlier revision, when capping removed every RTL/numeric tier a capsule
    required, appended the phase ceiling instead — so a capsule that declared the cycle-accurate cert
    tier was silently graded against a DIFFERENT, cheaper tier it had never asked for, and the result
    was reported as if it were the declared one. Measured cost: an entire agent run where the declared
    tier ran fine and returned real verdicts while the substituted tier's endpoint hung, and every
    capsule failed on a gate the capsule never named.

    The declared tier is the capsule's contract. When a phase cannot reach it the honest outcome is to
    say so (the caller fails closed on the empty/floor-only result and names the unreachable tier), not
    to quietly grade something else. Choosing a phase ceiling that the corpus actually declares is the
    caller's job — see :func:`declared_oracle_tiers` and ``qa_loop_adapters(declared_tiers=...)``."""
    kept = [t for t in tiers if t in keep]
    unreachable = [t for t in tiers if t not in keep]
    return kept, unreachable


def declared_oracle_tiers(*roots: str | Path) -> set[str]:
    """The union of every ``required_oracle_tiers`` entry declared by the capsules under ``roots``.

    The CORPUS is the authority on which oracle tiers a grade must ride. Deriving a phase's tier from
    this (rather than from "whichever tier the endpoint reaches fastest") is what keeps grading on the
    tier the capsule actually declared. Target-agnostic by construction: it reads capsule yaml, never a
    target name."""
    out: set[str] = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for cap_yaml in sorted(root.rglob("capsule.yaml")):
            try:
                cap = yaml.safe_load(cap_yaml.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            tiers = cap.get("required_oracle_tiers")
            if isinstance(tiers, list):
                out |= {str(t) for t in tiers}
    return out


def materialize_public_capsules(dest: str | Path, *, tier_ceiling: str = _DEFAULT_CEILING,
                                contract: str | Path | None = None,
                                corpus_roots: list[Path] | None = None,
                                exclude: tuple[str, ...] | set[str] | None = None) -> list[str]:
    """Derive the sandbox public-capsule view into ``dest``. Returns the capsule names written.

    Copies the 5 capsule files verbatim, then rewrites ``capsule.yaml``'s ``required_oracle_tiers``
    to the subset reachable at/below ``tier_ceiling`` (preserving every other field exactly).

    ``corpus_roots`` (target-AGNOSTIC): materialize the public capsules found directly under these roots
    (the descriptor's ``capsule_corpus`` + sibling corpora). When omitted, falls back to the legacy
    gemmini-contract discovery (``contract``) for backward compatibility.

    ``exclude`` is the descriptor's ``grading.exclude_capsules`` — capsule DIRECTORY NAMES this
    experiment withholds from the public graded set (see
    :attr:`~merlin.targetgen.target_experiment.TargetExperiment.graded_exclude` for why one exists). The
    names are matched against what the corpus actually holds and an exclusion matching NOTHING raises:
    a mistyped or stale name would otherwise silently re-admit an expensive capsule, and the failure
    mode of a *quietly wider* set is a run that blows its wall-clock budget with no signal saying why.
    """
    keep = set(_cap_tiers(tier_ceiling))
    dest = Path(dest)
    written: list[str] = []
    sources = (_public_capsule_dirs_in(corpus_roots) if corpus_roots is not None
               else _public_capsule_dirs(contract))
    drop = set(exclude or ())
    if drop:
        present = {s.name for s in sources}
        unknown = sorted(drop - present)
        if unknown:
            raise ValueError(
                f"grading.exclude_capsules names {unknown} but the public corpus holds no such "
                f"capsule(s); it has {sorted(present)}. Refusing to materialize, because an exclusion "
                f"that matches nothing silently GROWS the graded set back to full size.")
        sources = [s for s in sources if s.name not in drop]
    for src in sources:
        name = src.name
        d = dest / name
        d.mkdir(parents=True, exist_ok=True)
        for f in _CAPSULE_FILES:
            sp = src / f
            if not sp.is_file():
                continue
            if f == "capsule.yaml":
                cap = yaml.safe_load(sp.read_text(encoding="utf-8")) or {}
                tiers = cap.get("required_oracle_tiers")
                if isinstance(tiers, list):
                    kept, unreachable = _cap_required(tiers, keep)
                    cap["required_oracle_tiers"] = kept
                    if unreachable and not any(t in _RTL_TIERS for t in kept):
                        # Capping left NO numeric/RTL tier: this phase cannot certify the capsule at all.
                        # Carry the dropped DECLARED tiers forward so the grader can fail closed and NAME
                        # them, instead of reporting a bare "no oracle" — or, as it once did, silently
                        # substituting the ceiling tier and grading against something never declared.
                        # Only annotated in that case, so a capsule whose numeric floor survives capping
                        # (the ordinary case) stays byte-identical to the source.
                        cap["unreachable_required_oracle_tiers"] = unreachable
                        cap["oracle_tier_ceiling"] = tier_ceiling
                (d / f).write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
            else:
                shutil.copyfile(sp, d / f)
        written.append(name)
    return sorted(written)


def public_capsules_for(te, *, tier_ceiling: str | None = None) -> Path:
    """The public-capsule set to grade / self-check against for a target — DERIVED from its descriptor's
    ``capsule_corpus`` (+ sibling corpora), materialized into a per-target cache and returned. This is the
    target-agnostic replacement for the committed gemmini ``scripts/full_public_capsules`` set: gemmini
    reproduces its 20-capsule L2 set exactly, atlas gets its fp8/bf16 set at L3, any target its own — with
    NO per-target hardcode and no gemmini leak.

    ``tier_ceiling`` caps ``required_oracle_tiers`` to what the caller can reach. The default is the
    per-round loop tier the corpus itself DECLARES — ``qa_loop_adapters`` is asked for the fastest
    endpoint tier that is also a declared required tier, so the loop always grades against a tier the
    capsules asked for. When the endpoint reaches NONE of the declared tiers this FAILS CLOSED with the
    declared-vs-reachable sets named, rather than capping onto a substitute tier."""
    from merlin.common.artifacts import cache_dir
    from merlin.common.paths import repo_root
    from merlin.targetgen import capsule_runner as _CR
    root = repo_root()
    roots = ([te.capsule_corpus] if te.capsule_corpus else [])
    roots += [root / rel.rstrip("/") for rel in te.corpus_siblings()]
    if tier_ceiling is None:
        declared = declared_oracle_tiers(*roots)
        loop = _CR.qa_loop_adapters(te.target, te.sim_via, declared_tiers=declared)
        if not loop:
            reach = sorted(_CR.oracle_adapters(te.target, te.sim_via))
            if reach:
                # The endpoint DOES expose tiers, just none the corpus declared. Substituting one of them
                # is precisely the defect; refuse and name both sets.
                raise ValueError(
                    f"target {te.target!r}: its capsule corpus declares required oracle tiers "
                    f"{sorted(declared)} but the endpoint reaches {reach} — no declared tier is "
                    f"reachable, so this phase cannot grade these capsules. Refusing to substitute a "
                    f"tier the capsules never declared; make the declared tier reachable or fix the "
                    f"corpus.")
            # The endpoint reaches NOTHING (no mlc/sim/model venv in this environment). That is an
            # honestly ABSENT oracle, not a substitution risk: materialize at the legacy floor and let
            # the runner report each capsule's missing tier as unavailable, as it always has.
            tier_ceiling = _DEFAULT_CEILING
        else:
            tier_ceiling = max(loop, key=lambda t: _TIER_ORDER.index(t) if t in _TIER_ORDER else -1)
    # Concurrency-safe publish. Several A/B arms materialize the SAME target's public set at once; the old
    # ``rmtree(dest) + rebuild`` in place let one arm delete another's half-built cache mid-read (corrupt or
    # missing capsules -> wrong grades). Instead build into a UNIQUE versioned dir, then ATOMICALLY repoint a
    # per-target symlink at it (os.replace is atomic even over an existing symlink). A reader always follows
    # the symlink to a COMPLETE, immutable build; the last writer wins the symlink; nobody rmtrees a dir
    # another arm is reading. The cache namespace is purgeable, so stale builds are best-effort GC'd by age.
    base = cache_dir("capsule_bench_public")
    base.mkdir(parents=True, exist_ok=True)
    link = base / te.target
    ver = base / f".{te.target}.build.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    shutil.rmtree(ver, ignore_errors=True)
    materialize_public_capsules(ver, tier_ceiling=tier_ceiling, corpus_roots=roots,
                                exclude=getattr(te, "graded_exclude", ()))
    tmp_link = base / f".{te.target}.lnk.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(ver.name, tmp_link)            # relative target -> resolves beside the versioned dir
    if link.exists() and not link.is_symlink():
        shutil.rmtree(link, ignore_errors=True)   # one-time migration off the old in-place real dir
    os.replace(tmp_link, link)                # atomic publish (replaces the prior symlink in one step)
    # Best-effort GC: drop this target's OTHER finished builds that are old enough (>15 min) that no live
    # reader can still hold a path into them; never the just-published one (the symlink points at it).
    cutoff = time.time() - 900
    for old in base.glob(f".{te.target}.build.*"):
        try:
            if old.name != ver.name and old.stat().st_mtime < cutoff:
                shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass
    return link


def cert_capsule_cover(corpus_roots, *, labels: set[str] | None = None,
                       tile_dim: int | None = None) -> dict:
    """The REPRESENTATIVE subset a cycle-accurate cert tier should run, when the functional tier runs
    everything. Returns ``{"capsules": [...], "cells": [...], "uncovered": [...], "basis": {...}}``.

    Why a subset at all. The functional tier answers "does it compute the right value" and is cheap; the
    cert tier answers "does the hardware actually do it" -- encoding, protocol, resource limits -- and is
    minutes per capsule. Running cert on everything spends most of a run re-proving the same hardware
    facts, and on this repo's SIMT target that was 80% of an agent round for a verdict the score never
    read. Running it on nothing leaves the RTL claim unevidenced. A cover is the middle: full coverage at
    the functional tier, representative coverage at the cert tier.

    Why THESE axes. Representativeness must track where the HARDWARE differs, not where the numerics do:

      * ``semantic_family`` -- a contraction drives different RTL than a normalization or a movement.
      * operand ``dtype``   -- the proxy for WHICH compute unit runs it (block-scaled microscaling formats
                              go to the MX PE; ordinary floats to the SIMT lanes) and for datapath width.

      * tile ALIGNMENT (only when ``tile_dim`` is given) -- whether the capsule's extents divide evenly
        by the target's tile edge, or leave a partial tile. This is the axis a functional model is least
        able to stand in for: a partial tile changes addressing and the working-set boundary, and this
        repo has already been bitten by exactly that -- a taped-out unit computed partial N tiles
        (``n % 64 != 0``) wrongly while every functional check passed. A cover built on family and dtype
        alone can pick, for each cell, the one capsule whose extents happen to divide evenly, and then
        certify no partial tile anywhere. Passing ``tile_dim`` closes that blind spot; omitting it leaves
        it open, which is why ``basis`` reports which axes were actually used.

    All are declared per capsule and read as data, so a new target's cover falls out of its own corpus
    with no edit here. ``expected_instruction_coverage.instruction_classes`` would be the most faithful
    axis of all and is deliberately NOT used: every capsule in this repo declares it empty, so selecting on
    it would silently return a cover of one. That is recorded in ``basis`` so the caller can see which
    axes actually carried the choice rather than assuming all of them did.

    Greedy set cover, which is within a log factor of optimal and, more usefully, is explainable: each
    chosen capsule is the one adding the most uncovered cells. ``uncovered`` is returned rather than
    swallowed -- a cell no capsule can cover is a corpus gap the caller should surface, not hide.
    """
    labels = labels or {"public"}
    rows = []
    for root in ([corpus_roots] if isinstance(corpus_roots, (str, Path)) else corpus_roots):
        for cy in sorted(Path(root).glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if cap.get("label") not in labels:
                continue
            sem = cap.get("semantic") or {}
            dts = sorted({str(t.get("dtype")) for t in (cap.get("inputs") or []) if t.get("dtype")})
            if not dts:
                continue
            align = None
            if tile_dim and tile_dim > 0:
                extents = [int(x) for t in (cap.get("inputs") or [])
                           for x in (t.get("shape") or []) if str(x).lstrip("-").isdigit()]
                # "partial" if ANY extent leaves a remainder: one ragged axis is enough to exercise the
                # tile-edge path, and that is what we are trying to certify.
                align = "partial" if any(e % tile_dim for e in extents) else "aligned"
            rows.append({"name": cap.get("name") or cy.parent.name,
                         "family": sem.get("semantic_family"), "dtypes": dts, "align": align})

    def _cells(r):
        return {(r["family"], dt, r["align"]) for dt in r["dtypes"]}

    cells = {c for r in rows for c in _cells(r)}
    uncovered, chosen = set(cells), []
    while uncovered:
        best, gain = None, 0
        for r in rows:
            if r in chosen:
                continue
            g = len(_cells(r) & uncovered)
            if g > gain:
                best, gain = r, g
        if best is None:                      # nothing left can cover what remains -> report it
            break
        chosen.append(best)
        uncovered -= _cells(best)

    n_class = sum(1 for r in rows if r.get("instruction_classes"))
    return {
        "capsules": sorted(r["name"] for r in chosen),
        "cells": sorted("/".join(x for x in (f, dt, al) if x) for f, dt, al in cells),
        "uncovered": sorted("/".join(x for x in (f, dt, al) if x) for f, dt, al in uncovered),
        "basis": {"axes": ["semantic_family", "dtype"] + (["tile_alignment"] if tile_dim else []),
                  "tile_dim": tile_dim, "n_candidates": len(rows),
                  "n_cells": len(cells), "n_chosen": len(chosen),
                  "instruction_classes_available": n_class,
                  "note": ("instruction_classes is declared empty by every capsule in this corpus, so it "
                           "could not be used as an axis" if not n_class else "")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Materialize the sandbox public-capsule view.")
    ap.add_argument("dest", help="destination directory (gitignored sandbox / runs path)")
    ap.add_argument("--tier-ceiling", default=_DEFAULT_CEILING,
                    help=f"highest reachable oracle tier (default {_DEFAULT_CEILING})")
    ap.add_argument("--contract", default=None, help="contract dir override")
    a = ap.parse_args(argv)
    names = materialize_public_capsules(a.dest, tier_ceiling=a.tier_ceiling, contract=a.contract)
    print(f"materialized {len(names)} public capsules -> {a.dest} (tiers <= {a.tier_ceiling})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
