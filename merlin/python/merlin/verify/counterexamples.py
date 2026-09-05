"""Turn refutations into corpus entries the normal generator builds.

When the lattice sweep refutes a point, the shape that exposed the defect should not stay in a log —
it should become a capsule the bench grades from then on. This module writes those shapes as PROFILE
ENTRIES rather than as capsule directories, which is the difference between joining the corpus and
sitting beside it.

**Why a profile entry and not a directory.** ``merlin/contract/capsules/generate_corpus.py`` is what
turns an entry into a capsule: it writes ``capsule.yaml``, ``capsule.interface.mlir`` and the README,
computes and writes the untracked ``golden.yaml``, scrubs the directory, and records the capsule in
``MANIFEST.yaml`` as *generated*. An earlier version of this work wrote the directory itself and got
none of that — no golden for the grader, and ``update_provenance_manifest`` would have classified a
solver-produced capsule as ``hand_authored``, which is exactly backwards.

``load_profile`` merges an explicit, hardcoded chain of sidecars -- ``<target>.yaml``,
``<target>.synth.yaml``, ``<target>.smt.yaml``, ``<target>.hidden.yaml``. It is NOT a glob, and this
docstring claimed it was: "``load_profile`` already merges ``profiles/<target>.*.yaml`` sidecars, so
emitting an entry gets all of it for free". It did not merge ``.smt.yaml`` at all, so every entry this
module wrote went to a filename nothing read — consistent with no ``*.smt.yaml`` ever having been
committed. The name is in the chain now; a NEW sidecar suffix still has to be added there by hand.

**What the entry carries, and what it does not.** It carries the SHAPE and configuration the solver
found. It does not carry the solver's input values, because a capsule has nowhere to put them:
``capsule.schema.json``'s ``inputs[]`` has no values field and sets ``additionalProperties: false``,
and ``capsule_golden.materialize_capsule_leaves`` fills every leaf unconditionally with
``Tensor.deterministic``. The values are written to an untracked artifact instead, where they are
evidence for a human rather than an input to the grader. Claiming otherwise would overstate what this
path does — see the 2026-09-05 correction in docs/design/compiler_verification.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

#: Prefix for solver-produced capsules. Distinct from ``corpus_synth.SYNTH_PREFIX`` ("SY") so a name
#: collision is impossible and a reader can tell at a glance which generator produced a capsule.
CX_PREFIX = "CX"

#: The schema's existing enum value for this provenance — no schema change is needed.
CX_ROLE = "smt_counterexample"


def profile_path(target: str) -> Path:
    from merlin.common.paths import merlin_dir

    return merlin_dir() / "contract" / "capsules" / "profiles" / f"{target}.smt.yaml"


def entry_name(m: int, k: int, n: int, *, dtype: str, family: str) -> str:
    """A stable, collision-free name. Schema requires ``^[A-Za-z0-9_]+$``."""
    return f"{CX_PREFIX}_{family}_{dtype}_{m}x{k}x{n}".replace("-", "_")


def counterexample_entry(*, target: str, m: int, k: int, n: int, dtype: str = "i8",
                         family: str = "contraction", obligation: str = "",
                         solver: str = "", bound_ms: int | None = None,
                         evidence_path: str | None = None) -> dict[str, Any]:
    """One profile entry for a refuted lattice point.

    Extents are CONCRETE integers rather than the tile-relative spellings the synthesized entries
    use ("tile", "2*tile-1"). The point of a counterexample is the specific shape that broke, so
    re-deriving it from a tile edge later would lose exactly the information worth keeping.
    """
    where = f"{m}x{k}x{n}"
    reference = (
        f"refuted by SMT translation validation at {where}"
        + (f" for obligation {obligation!r}" if obligation else "")
        + (f"; solver {solver}" if solver else "")
        + (f", bound {bound_ms} ms" if bound_ms is not None else "")
        + ". The SHAPE is the solver's; the stimulus is the corpus's own deterministic fill, because "
          "a capsule has no field for input values"
        + (f". Counterexample values: {evidence_path}" if evidence_path else "")
    )
    return {
        "cat": "isa",
        "kind": "isa",
        "name": entry_name(m, k, n, dtype=dtype, family=family),
        "op": "matmul",
        "operand_dtype": dtype,
        "out": "Y0",
        "lhs": "A0",
        "weight": "W",
        "source_role": CX_ROLE,
        "source_reference": reference,
        "label": "public",
        "modes": {},
        "M": int(m),
        "K": int(k),
        "N": int(n),
        "pass_requirements": ["target-isa-lowering", "tile-schedule"],
    }


def merge_entries(existing: list[dict], new: list[dict]) -> tuple[list[dict], int]:
    """Union by name, newest winning. Returns the merged list and how many were added.

    De-duplicating by name matters because the same shape can refute on successive runs; without it
    the profile would grow without bound and ``expand_sweeps`` would raise on the duplicate.
    """
    by_name = {e["name"]: e for e in existing}
    added = sum(1 for e in new if e["name"] not in by_name)
    by_name.update({e["name"]: e for e in new})
    return [by_name[k] for k in sorted(by_name)], added


def write_profile(target: str, entries: list[dict], *, provenance: dict | None = None) -> Path:
    """Write (or extend) ``profiles/<target>.smt.yaml``, the sidecar ``load_profile`` already reads."""
    import yaml

    path = profile_path(target)
    existing: list[dict] = []
    if path.is_file():
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        existing = list(doc.get("capsules") or [])
    merged, added = merge_entries(existing, entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "provenance": dict(provenance or {}, generator="merlin.verify.counterexamples",
                           note=("shapes refuted by SMT translation validation; the stimulus is the "
                                 "corpus's deterministic fill, not the solver's values")),
        "capsules": merged,
    }, sort_keys=True), encoding="utf-8")
    print(f"{path}: {len(merged)} entr{'y' if len(merged) == 1 else 'ies'} ({added} new)")
    return path


def write_evidence(target: str, records: list[dict]) -> Path | None:
    """Record the counterexample VALUES as an untracked artifact.

    Deliberately not in the corpus tree: a capsule directory's tracked files are its public contract,
    and input values that expose a defect are neither part of that contract nor something the grader
    can consume. Here they are auditable evidence for whoever reads the refutation.
    """
    if not records:
        return None
    import json

    from merlin.common.artifacts import new_product

    prod = new_product("verification", version=1, target=target, sources=[
        f"{len(records)} refuted lattice point(s) for {target}",
        "values are the solver's model; the corpus grades the deterministic fill instead",
    ], notes=("Counterexample values for refuted lattice points. Evidence for a human reading the "
              "refutation -- a refutation without its counterexample is an assertion -- and NOT an "
              "input to the grader, which has no field for input values."))
    out = prod.add_artifact("counterexamples.json")
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    prod.write_manifest()
    return out
