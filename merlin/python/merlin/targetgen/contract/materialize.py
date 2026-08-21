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
import shutil
import sys
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


def _cap_required(tiers: list[str], keep: set[str], tier_ceiling: str) -> list[str]:
    """Cap a capsule's ``required_oracle_tiers`` to what THIS phase can reach — WITHOUT ever stripping the
    numeric floor to nothing.

    Capping to the phase ceiling drops tiers above it. For a float capsule that requires the cycle-accurate
    cert (atlas: ``[L0, L1, L3]`` with the integer L0/L1 marked not_applicable), a naive intersection with
    a loop ceiling of L2 yields ``[L0, L1]`` — both N/A — so the grade enforces ZERO numeric tiers and any
    capsule that merely builds reads back as pass. That is the crash-pass regression: it appeared the moment
    the fast L2 npu-functional tier lowered the loop ceiling from L3 to L2.

    Rule: when capping removes every RTL/numeric tier the capsule required, the HIGHEST REACHABLE tier (the
    ceiling itself, when it is an RTL tier) BECOMES the required tier for this phase. The fast loop still
    enforces a real numeric oracle (atlas → L2 npu-functional is mandatory each round), while the
    cycle-accurate cert (L3) stays required at the checkpoint (ceiling L3, kept as-is). Target-general — it
    only fires when a required RTL tier was capped away; gemmini, whose loop tier L2 is already in its
    required set, is untouched."""
    kept = [t for t in tiers if t in keep]
    if (any(t in _RTL_TIERS for t in tiers)
            and not any(t in _RTL_TIERS for t in kept)
            and tier_ceiling in _RTL_TIERS):
        kept.append(tier_ceiling)
    return kept


def materialize_public_capsules(dest: str | Path, *, tier_ceiling: str = _DEFAULT_CEILING,
                                contract: str | Path | None = None,
                                corpus_roots: list[Path] | None = None) -> list[str]:
    """Derive the sandbox public-capsule view into ``dest``. Returns the capsule names written.

    Copies the 5 capsule files verbatim, then rewrites ``capsule.yaml``'s ``required_oracle_tiers``
    to the subset reachable at/below ``tier_ceiling`` (preserving every other field exactly).

    ``corpus_roots`` (target-AGNOSTIC): materialize the public capsules found directly under these roots
    (the descriptor's ``capsule_corpus`` + sibling corpora). When omitted, falls back to the legacy
    gemmini-contract discovery (``contract``) for backward compatibility.
    """
    keep = set(_cap_tiers(tier_ceiling))
    dest = Path(dest)
    written: list[str] = []
    sources = (_public_capsule_dirs_in(corpus_roots) if corpus_roots is not None
               else _public_capsule_dirs(contract))
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
                    cap["required_oracle_tiers"] = _cap_required(tiers, keep, tier_ceiling)
                (d / f).write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
            else:
                shutil.copyfile(sp, d / f)
        written.append(name)
    return sorted(written)


def _materialize_signature(roots, tier_ceiling) -> str:
    """A cheap fingerprint of the materialization inputs (source files + tier ceiling), so a fresh cache
    is REUSED instead of destructively rebuilt on every call — the rebuild raced a parallel grader
    reading the shared per-target cache."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(tier_ceiling).encode())
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            continue
        for f in sorted(rp.rglob("*")):
            if f.is_file():
                st = f.stat()
                h.update(f"{f}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()


def _materialize_public_locked(base: Path, target: str, tier_ceiling, roots) -> Path:
    """Concurrency-safe materialize of the per-target public cache. Arms grade IN PARALLEL against this
    shared dir, so the old ``rmtree(dest); copy`` raced — one arm wiped the tree mid-read of another
    (``FileNotFoundError`` on a capsule.yaml). Instead: hold a cross-process lock, REUSE the cache when
    the corpus is unchanged, and otherwise build into a temp dir and swap it in atomically (so a reader
    never observes a half-populated or vanished tree)."""
    import fcntl, os, tempfile
    base.mkdir(parents=True, exist_ok=True)
    # Namespace the cache by CEILING, not target alone: the QA loop grades the loop-ceiling view (e.g.
    # L2) while the cycle-accurate checkpoint grades the FULL-ladder view (e.g. L3) — in parallel. One
    # shared dir keyed only by target made each caller flip the signature and rebuild over the other's
    # live tree; worse, the checkpoint silently graded the LOOP-capped copy, so a required cycle-exact
    # tier was never mandatory at its own checkpoint (an L3 numeric mismatch read back status=pass).
    cache_key = f"{target}-{tier_ceiling}"
    dest = base / cache_key
    sig = _materialize_signature(roots, tier_ceiling)
    stamp = dest / ".materialize.sig"
    with open(base / (cache_key + ".lock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if stamp.is_file() and stamp.read_text(encoding="utf-8") == sig:
                return dest                                   # fresh — reuse, never rmtree under a reader
        except OSError:
            pass
        tmp = Path(tempfile.mkdtemp(prefix=cache_key + ".tmp-", dir=str(base)))
        try:
            materialize_public_capsules(tmp, tier_ceiling=tier_ceiling, corpus_roots=roots)
            (tmp / ".materialize.sig").write_text(sig, encoding="utf-8")
            old = None
            if dest.exists():
                old = base / f"{cache_key}.old-{os.getpid()}"
                os.replace(dest, old)                         # move current aside (same fs, atomic)
            os.replace(tmp, dest)                             # swap the fresh tree in
            tmp = None
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)        # swap failed: drop the temp build
    return dest


def public_capsules_for(te, *, tier_ceiling: str | None = None) -> Path:
    """The public-capsule set to grade / self-check against for a target — DERIVED from its descriptor's
    ``capsule_corpus`` (+ sibling corpora), materialized into a per-target cache and returned. This is the
    target-agnostic replacement for the committed gemmini ``scripts/full_public_capsules`` set: gemmini
    reproduces its 20-capsule L2 set exactly, atlas gets its fp8/bf16 set at L3, any target its own — with
    NO per-target hardcode and no gemmini leak.

    ``tier_ceiling`` caps ``required_oracle_tiers`` to what the caller can reach; default = the target's
    loop-reachable tier (``max(qa_loop_adapters)``), so gemmini→L2 (spike) and atlas→L3 (arc)."""
    from merlin.common.artifacts import cache_dir
    from merlin.common.paths import repo_root
    from merlin.targetgen import capsule_runner as _CR
    root = repo_root()
    roots = ([te.capsule_corpus] if te.capsule_corpus else [])
    roots += [root / rel.rstrip("/") for rel in te.corpus_siblings()]
    if tier_ceiling is None:
        loop = _CR.qa_loop_adapters(te.target, te.sim_via) or {"L2": None}
        tier_ceiling = max(loop, key=lambda t: _TIER_ORDER.index(t) if t in _TIER_ORDER else -1)
    # NOTE: arms grade IN PARALLEL against this shared per-target cache; a naive rmtree+recopy on every
    # call raced a concurrent grader reading it (capsule.yaml vanished mid-read -> FileNotFoundError).
    # Materialize under a cross-process lock, reuse when the corpus is unchanged, else swap a fresh tree
    # in atomically.
    return _materialize_public_locked(cache_dir("capsule_bench_public"), te.target, tier_ceiling, roots)


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
