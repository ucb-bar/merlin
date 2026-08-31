#!/usr/bin/env python3
"""Gate: is a target's HELD-OUT capsule set actually disjoint from its PUBLIC one?

A holdout exists to measure transfer: the agent under test sees the OBLIGATION (the family, the dtype,
the contract) and never the concrete point. That only works if the concrete point is one it has not
already compiled. A holdout that coincides with a public capsule measures memorisation and reports it
as generalisation — and nothing checked, because both halves were hand-authored by an author who knew
the public set. A prior audit of this repo found exactly that: several "holdouts" were renames of
public capsules (same op, same dtype, same shape, different tensor labels), and the only reason anyone
knows is that a human read both files side by side.

WHAT A "POINT" IS. The parameterization, not the identity. Two entries are the same point when they
would build the same program: same op/kind/dtype/source, same extents, same epilogue and modes. The
capsule NAME and the operand TENSOR labels are deliberately NOT part of it — the corpus generator salts
operand DATA from the capsule name, so renaming ``lhs: A0`` to ``lhs: Ah5`` yields fresh numbers and a
byte-identical program. That is a data-independence check, which is worth having and is not transfer.
Tile-relative and absolute spellings are normalized against the target's DERIVED tile edge, so
``M_tiles: 2`` and ``M: 32`` are recognized as one point on a 16-wide array.

⚠ THIS SCRIPT NEVER EMITS A HIDDEN POINT. The whole hazard it guards against is that the specification
of a holdout is itself an answer; a checker that printed the colliding shape to fail a build would leak
precisely what the sidecar exists to hide (and this repo has had an answer-key incident). So the report
is COUNTS and a BOOLEAN. Values — shapes, dtypes, names, goldens — never leave the process. A
maintainer fixing a reported collision reads the untracked sidecar directly; the gate only says how
many there are.

Modes, mirroring the other gates in this directory:

  --target NAME        audit one target (repeatable); default: every target with a hidden sidecar
  --json               machine-readable (counts only)
  --ratchet PATH       pre-existing overlap that MAY ONLY SHRINK, scoped per target
  --fail-on-overlap    exit non-zero when a target's overlap exceeds its ratcheted allowance
                       (default: report only)
  --fail-on-unverifiable
                       exit non-zero when a target's comparison COULD NOT RUN (no descriptor, no
                       derivable tile edge, a malformed sweep). A check that could not run has
                       established nothing, and this repo has repeatedly shipped one reporting
                       success; "no sidecar here" is the one non-ok state that is a real answer.

Reporting-only by default for the same reason the sibling coverage gate is: the sidecars predate the
check, and turning an inherited duplicate into a hard failure on day one only teaches everyone to pass
``--no-verify``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
for _p in (_REPO / "merlin" / "python",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402

from merlin.common.paths import repo_root  # noqa: E402

PROFILES = _REPO / "merlin" / "contract" / "capsules" / "profiles"
HIDDEN_SUFFIX = ".hidden.yaml"

# --- what makes a POINT ---------------------------------------------------------------------------
# Identity / provenance / bookkeeping: present on every entry, load-bearing for none of them. Dropping
# these is what turns "a rename" into "the same point".
_IDENTITY_KEYS = frozenset({
    "name", "label", "cat", "source_role", "source_reference", "note", "notes", "description",
    "comment", "comparison_group", "gate",
})
# Keys whose VALUE is an operand tensor label. Dropped on purpose, and this is the crux of the check:
# the corpus generator salts operand DATA from the capsule name, so an entry that differs from a public
# one only in ``lhs: A0`` vs ``lhs: Ah5`` builds a byte-identical program over fresh numbers. Keeping
# these would let every rename read as a distinct holdout, which is precisely the failure the audit
# found. (The DROP itself comes from the general string rule below; naming them here is what keeps
# them out of ``unclassified_keys`` — i.e. it records that someone decided, rather than that nobody
# looked.)
_TENSOR_LABEL_KEYS = frozenset({"lhs", "weight", "out", "src", "q", "k", "v", "gamma"})
# ROLE labels: what the capsule is FOR (the semantic family the coverage requirement accounts it
# under), not what it computes. Two entries at one shape that declare different roles still build the
# same program, so a role is not a point. Kept apart from the identity keys because the distinction is
# worth reading: dropping ``semantic`` is a JUDGEMENT (it is real metadata) where dropping ``name`` is
# not. Measured: counting it as structural hides three of this repo's known rename-holdouts.
_ROLE_LABEL_KEYS = frozenset({"semantic"})
# String-valued keys that DO define the program: which op, which frontend, which dtype. Any other
# string-valued key is treated as a label and dropped — the conservative direction for a disjointness
# test, since dropping a key can only merge points and so can only make a collision MORE visible.
# Unrecognized string keys are reported (never silently accepted) as ``unclassified_keys``.
_STRUCTURAL_STRING_KEYS = frozenset({
    "op", "kind", "operand_dtype", "accum_dtype", "output_dtype", "source", "model", "spec_ref",
    "loader", "compare",
})
_TILES_SUFFIX = "_tiles"


def canonical_point(entry: dict, tile: int) -> tuple:
    """The entry's PARAMETERIZATION as a hashable, comparable value. Never printed.

    Normalizations, in order:
      * ``<X>_tiles`` folds into ``<X>`` at the derived tile edge, so the two spellings of one shape
        compare equal (a holdout authored in tiles cannot hide behind a public capsule authored in
        absolute extents, or the reverse);
      * identity/provenance keys are dropped;
      * a string value under a non-structural key is dropped as a label (operand tensor names are the
        motivating case: the corpus generator salts operand DATA from the capsule name, so a rename
        produces different numbers and the same program);
      * everything else — numbers, bools, lists, mappings — is kept and rendered order-independently.
    """
    flat: dict = {}
    for key, value in entry.items():
        if key in _IDENTITY_KEYS:
            continue
        if value is None:
            # A null carries no parameterization, so it cannot distinguish two points. It also shows up
            # for a reason worth naming: an unquoted prose value inside a YAML FLOW mapping splits on
            # its own commas, leaving the tail as a bare null-valued key. Keeping those would make every
            # differently-worded rationale look like a different program.
            continue
        if key.endswith(_TILES_SUFFIX) and len(key) > len(_TILES_SUFFIX):
            base = key[: -len(_TILES_SUFFIX)]
            if base in entry:            # both spellings present: the absolute one is authoritative
                continue
            flat[base] = int(value) * tile
            continue
        if isinstance(value, str) and key not in _STRUCTURAL_STRING_KEYS:
            continue
        flat[key] = value
    return tuple(sorted((k, _freeze(v)) for k, v in flat.items()))


def _freeze(value):
    """A hashable, order-stable rendering of a YAML value (mappings sort by key; lists keep order)."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    return ("str", str(value))


def unclassified_string_keys(entries: list[dict]) -> list[str]:
    """String-valued keys that are neither declared identity nor declared structural.

    Reported rather than defaulted. Such a key is currently DROPPED from the point, which is the
    conservative direction, but a reader must be able to see that the checker met a field it does not
    have an opinion about — a silently-classified new field is how a disjointness test stops testing.
    """
    seen = set()
    for e in entries:
        for k, v in e.items():
            if (isinstance(v, str) and k not in _IDENTITY_KEYS and k not in _TENSOR_LABEL_KEYS
                    and k not in _ROLE_LABEL_KEYS and k not in _STRUCTURAL_STRING_KEYS):
                seen.add(k)
    return sorted(seen)


# --- loading --------------------------------------------------------------------------------------
def _generator():
    """Import the corpus generator BY PATH and reuse its sweep machinery.

    Reused, never reimplemented: a second expander would drift from the one that actually mints the
    capsules, and a disjointness proof about points nobody generates is worth nothing.
    """
    path = _REPO / "merlin" / "contract" / "capsules" / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("merlin_corpus_generator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _descriptor_for(target: str) -> Path | None:
    cand = (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target
            / "target_experiment.yaml")
    return cand if cand.is_file() else None


def _needs_traits(*profiles: dict) -> bool:
    """Does any sweep in these profiles gate on a derived trait?

    Deriving the trait set costs a capability-manifest resolution (tens of seconds per target), and a
    gate is the only thing that reads it. Asked rather than always paid, so a gate that no sweep uses
    does not make this checker too slow to run in a hook. When no sweep gates, ``expand_sweeps`` never
    consults the traits at all, so passing None changes nothing.
    """
    for prof in profiles:
        for sweep in (prof.get("sweeps") or []):
            if isinstance(sweep, dict) and ((sweep.get("gate") or {}).get("requires")):
                return True
    return False


def _expand(gc, profile: dict, binding, skipped: list, traits) -> tuple[list[dict], int]:
    """``(entries, n_hand_authored)`` — the profile's hand-written entries plus its expanded sweeps."""
    hand = list(profile.get("capsules") or [])
    entries = gc.expand_sweeps(profile, binding, traits=traits, skipped=skipped)
    return entries, len(hand)


def compare(public_entries: list[dict], holdout_entries: list[dict], tile: int) -> dict:
    """The whole comparison, as a pure function of two entry lists — COUNTS ONLY.

    Kept separate from :func:`audit` (which does the file/descriptor plumbing) for two reasons. It is
    the part worth testing, and a test must be able to construct an OVERLAPPING pair and watch this
    catch it — a disjointness gate that has never been shown to fail is indistinguishable from one that
    always passes. And it is the part that must be provably leak-free: nothing it returns is derived
    from a holdout's values, so no caller can print one by accident.
    """
    pub = {canonical_point(e, tile) for e in public_entries}
    hid = [canonical_point(e, tile) for e in holdout_entries]
    overlap = sum(1 for p in hid if p in pub)
    return {
        "n_public_distinct_points": len(pub),
        "n_holdout_points": len(hid),
        "n_holdout_coinciding_with_public": overlap,
        "n_holdout_internal_duplicates": len(hid) - len(set(hid)),
        "disjoint": overlap == 0,
    }


def targets_with_holdouts() -> list[str]:
    return sorted(p.name[: -len(HIDDEN_SUFFIX)] for p in PROFILES.glob(f"*{HIDDEN_SUFFIX}"))


def audit(target: str) -> dict:
    """Measure one target's holdout set against its public set. Counts only — no point ever escapes."""
    public_path = PROFILES / f"{target}.yaml"
    hidden_path = PROFILES / f"{target}{HIDDEN_SUFFIX}"
    if not public_path.is_file():
        return {"target": target, "status": "no_public_profile",
                "detail": f"no {public_path.name}: nothing to be disjoint FROM"}
    if not hidden_path.is_file():
        # A public clone, or a sandbox where the sidecar is masked. Not an error and not a pass:
        # there is no holdout set here to check.
        return {"target": target, "status": "no_holdout_sidecar",
                "detail": "no hidden sidecar is readable here (public clone or masked sandbox)"}
    desc = _descriptor_for(target)
    if desc is None:
        return {"target": target, "status": "no_target_experiment",
                "detail": "no target_experiment.yaml: the tile edge cannot be derived, so tile-relative "
                          "and absolute spellings of one shape cannot be compared. Reported UNKNOWN "
                          "rather than compared at an assumed tile edge."}

    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.target_experiment import load_target_experiment

    public = yaml.safe_load(public_path.read_text(encoding="utf-8")) or {}
    hidden = yaml.safe_load(hidden_path.read_text(encoding="utf-8")) or {}
    try:
        binding = CS.derive_binding(load_target_experiment(desc), public.get("datapath", {}))
    except Exception as exc:  # noqa: BLE001 — an underivable binding is UNKNOWN, never a default tile
        return {"target": target, "status": "binding_underivable",
                "detail": f"{type(exc).__name__}: {exc}"}
    tile = int(getattr(binding, "tile_dim", 0) or 0)
    if tile < 1:
        return {"target": target, "status": "no_tile_edge",
                "detail": "the binding reports no tile edge; tile-relative extents cannot be resolved"}

    gc = _generator()
    skipped_pub: list = []
    skipped_hid: list = []
    traits = gc._corpus_traits(binding) if _needs_traits(public, hidden) else None
    try:
        pub_entries, pub_hand = _expand(gc, public, binding, skipped_pub, traits)
        hid_entries, hid_hand = _expand(gc, hidden, binding, skipped_hid, traits)
    except Exception as exc:  # noqa: BLE001 — a malformed sweep must fail loudly, not silently pass
        return {"target": target, "status": "sweep_error", "detail": f"{type(exc).__name__}: {exc}"}

    report = {
        "target": target,
        "status": "ok",
        "tile_edge": tile,
        "n_public_points": len(pub_entries),
        "n_public_hand_authored": pub_hand,
        "n_public_generated": len(pub_entries) - pub_hand,
        "n_holdout_hand_authored": hid_hand,
        "n_holdout_generated": len(hid_entries) - hid_hand,
    }
    report.update(compare(pub_entries, hid_entries, tile))
    report.update({
        "unclassified_keys": sorted(set(unclassified_string_keys(pub_entries))
                                    | set(unclassified_string_keys(hid_entries))),
        "traits_derived": traits is not None,
        "sweeps_skipped_public": skipped_pub,
        "sweeps_skipped_holdout": [{"reason": s.get("reason")} for s in skipped_hid],
    })
    return report


# --- ratchet --------------------------------------------------------------------------------------
def _debt_key(target: str) -> str:
    """The ratchet key, SCOPED TO ITS TARGET.

    A flat allowance would let one target's inherited duplicates excuse another's — and every target
    here inherits a different number of them, so a shared budget forgives all of them at once.
    """
    return f"{target} overlap"


def _load_ratchet(p: Path | None) -> dict[str, int]:
    """``{'<target> overlap': <allowed count>}``. An absent target allows zero."""
    out: dict[str, int] = {}
    if not p or not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, count = line.rpartition(":")
        if not sep:
            raise ValueError(f"{p}: ratchet line {line!r} is not '<target> overlap:<count>'")
        out[key.strip()] = int(count.strip())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path,
                    default=_HERE.parent / "holdout_disjointness_ratchet.txt")
    ap.add_argument("--fail-on-overlap", action="store_true")
    ap.add_argument("--fail-on-unverifiable", action="store_true")
    a = ap.parse_args(argv)

    ratchet = _load_ratchet(a.ratchet)
    names = a.target or targets_with_holdouts()
    reports = [audit(t) for t in names]

    if a.json:
        print(json.dumps(reports, indent=2))
    else:
        if not reports:
            print("no target declares a holdout sidecar here — nothing to check")
        for r in reports:
            if r["status"] != "ok":
                print(f"== {r['target']}: {r['status']} — {r.get('detail', '')}")
                continue
            allowed = ratchet.get(_debt_key(r["target"]), 0)
            print(f"== {r['target']}  (tile edge {r['tile_edge']})")
            print(f"   public        : {r['n_public_points']} point(s) "
                  f"[{r['n_public_hand_authored']} hand-authored, {r['n_public_generated']} generated]"
                  f" -> {r['n_public_distinct_points']} distinct")
            print(f"   holdout       : {r['n_holdout_points']} point(s) "
                  f"[{r['n_holdout_hand_authored']} hand-authored, {r['n_holdout_generated']} generated]")
            verdict = "DISJOINT" if r["disjoint"] else "NOT DISJOINT"
            print(f"   disjointness  : {verdict} — {r['n_holdout_coinciding_with_public']} holdout "
                  f"point(s) coincide with a public point (ratcheted allowance {allowed})")
            if r["n_holdout_internal_duplicates"]:
                print(f"   holdout self-overlap: {r['n_holdout_internal_duplicates']} holdout point(s) "
                      f"duplicate another holdout — the set is smaller than it counts")
            if not r["disjoint"]:
                print("     (the coinciding points are NOT printed: a holdout's specification is "
                      "itself an answer. Read the untracked sidecar to fix them.)")
            if r["unclassified_keys"]:
                print(f"   UNCLASSIFIED  : string field(s) {r['unclassified_keys']} are neither declared "
                      f"identity nor declared structural; they are DROPPED from the point (the "
                      f"conservative direction) — classify them rather than leaving this UNKNOWN")
            for s in r["sweeps_skipped_public"]:
                print(f"   note: public sweep {s.get('sweep')} generated nothing — {s.get('reason')}")
            for s in r["sweeps_skipped_holdout"]:
                print(f"   note: a holdout sweep generated nothing — {s.get('reason')}")

    over = [(r["target"], r["n_holdout_coinciding_with_public"],
             ratchet.get(_debt_key(r["target"]), 0))
            for r in reports if r["status"] == "ok"
            and r["n_holdout_coinciding_with_public"] > ratchet.get(_debt_key(r["target"]), 0)]
    # A target whose comparison could not run is neither a pass nor an overlap. It is UNKNOWN, and it
    # is tracked separately so it can never be mistaken for either. `no_holdout_sidecar` and
    # `no_public_profile` are excluded: those are real answers about a clone that has nothing to check.
    unverifiable = [(r["target"], r["status"]) for r in reports
                    if r["status"] not in ("ok", "no_holdout_sidecar", "no_public_profile")]
    rc = 0
    if over and a.fail_on_overlap:
        for target, seen, allowed in over:
            print(f"\nFAIL: {target} has {seen} holdout point(s) coinciding with a public point "
                  f"(allowance {allowed})", file=sys.stderr)
        rc = 1
    if unverifiable and a.fail_on_unverifiable:
        for target, status in unverifiable:
            print(f"\nFAIL: {target} could not be checked ({status}); a check that could not run has "
                  f"established nothing", file=sys.stderr)
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
