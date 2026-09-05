#!/usr/bin/env python3
"""Retire a hand-authored capsule entry ONLY when a derived one provably covers it.

The corpus is meant to be derived: a target's obligations come from its own facts and captures, and the
synthesizer turns them into capsules. What remains hand-written is a list nobody can reproduce for a new
target -- roughly 168 entries across six profiles -- and it is the last thing standing between this
corpus and "add a descriptor, get a corpus".

The obvious way to retire them is to delete the list and trust the synthesizer, and it is wrong. Measured
before this tool existed, the derivation could express **26%** of those entries; deleting the rest would
have removed coverage while every count went up, because the corpus would still be full of capsules and
the requirement would still report itself met. So this tool retires nothing it cannot show is covered,
and prints what it refused.

WHAT COUNTS AS COVERED. A hand entry is covered when some derived capsule exercises the same declared
axis VALUES -- the same signature `tier_policy.capsule_axes` already uses to choose certification
representatives, so this reuses the corpus's own notion of "interchangeable" rather than inventing a
second one. Three verdicts, and only the first retires:

``covered``            a derived capsule has the identical axis signature.
``covered_at_scale``   same signature, different extents. Reported, NOT retired by default: the
                       hand entry may exist precisely because of its extents (a ragged tail, a shape
                       that spills a store), and the signature cannot see that. ``--include-rescaled``
                       retires these too, for a caller who has checked.
``uncovered``          nothing derived matches, OR the obligation is one the vocabulary cannot express
                       at all. Stays, with its reason recorded.

MEASURED OUTCOME, and it is zero. Run across every target that classifies: **0 retirable**, 6 covered
only at a different scale, 159 that must stay. An earlier and weaker version of this test said 8 were
retirable, and inspecting the pairs is what refuted it -- one proposed retiring a block-scaled mxfp8
obligation against an f32 witness (the entry had been REBUILT in-process to a signature its own capsule
does not carry), and another proposed retiring a host/device interop member against a whole-model one
whose axis signature is byte-identical and which never crosses that seam. Both would have removed
coverage while every count rose. So the on-disk capsule is now the authority over any rebuild, and an
obligation the vocabulary cannot express never retires however well its signature matches.

NOTHING IS DELETED. A retired entry moves to ``profiles/retired/<target>.v0.yaml``, tracked, no longer
read by the generator, with ``RETIRED.md`` recording what covers each one. The repo's practice is to
untrack or supersede rather than delete, because a retired entry that turns out to have been load-bearing
must be readable, not recovered from a reflog.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common.paths import merlin_dir  # noqa: E402
from merlin.targetgen import corpus_spec as CS  # noqa: E402
from merlin.targetgen import tier_policy as TP  # noqa: E402

COVERED = "covered"
COVERED_AT_SCALE = "covered_at_scale"
UNCOVERED = "uncovered"


def _profiles_dir() -> Path:
    return merlin_dir() / "contract" / "capsules" / "profiles"


def _targets() -> list[str]:
    return sorted(p.stem for p in _profiles_dir().glob("*.yaml")
                  if not p.stem.startswith("_") and "." not in p.stem)


def _binding(target: str):
    import yaml

    from merlin.targetgen.target_experiment import load_target_experiment

    desc = (merlin_dir() / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml")
    prof = yaml.safe_load((_profiles_dir() / f"{target}.yaml").read_text())
    return CS.derive_binding(load_target_experiment(desc), prof["datapath"]), prof


def _shape_key(cap) -> tuple:
    return tuple(tuple(r.get("shape") or ()) for r in (cap.get("inputs") or ())
                 if isinstance(r, dict))


def _realize(entries, binding, on_disk: dict):
    """``name -> (axis signature, shape key)``, built the way the generator builds them.

    An entry whose builder is unavailable in this checkout (a whole-model capsule needing the capture
    venv, say) falls back to the capsule already written to disk. An entry that can be realized by
    NEITHER route is reported rather than treated as uncovered -- not being able to look at something is
    not evidence about it.
    """
    out, unrealized = {}, []
    for e in entries or []:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        name = str(e["name"])
        # THE CAPSULE ON DISK IS THE ONE THE GENERATOR ACTUALLY WROTE, so it is the authority. Rebuilding
        # the entry in-process looked equivalent and is not: measured, a block-scaled member rebuilt to a
        # different dtype signature than its own capsule carries, which would have retired an mxfp8
        # obligation against an f32 witness. The in-process build is the FALLBACK, for entries this
        # checkout cannot find on disk.
        cap = on_disk.get(name)
        if cap is None:
            try:
                built = CS.build(dict(e), binding)
                cap = built[0] if isinstance(built, tuple) else built
            except Exception:  # noqa: BLE001
                cap = None
        if cap is None:
            unrealized.append(name)
            continue
        out[name] = (frozenset(TP.capsule_axes(cap)), _shape_key(cap))
    return out, unrealized


def _on_disk(target: str, subtrees: set[str]) -> dict:
    import yaml

    root = merlin_dir() / "contract" / "capsules"
    paths = (sorted((root / target).rglob("capsule.yaml")) if target in subtrees
             else sorted(p for p in root.rglob("capsule.yaml")
                         if p.relative_to(root).parts[0] not in subtrees | {"profiles"}))
    out = {}
    for p in paths:
        try:
            d = yaml.safe_load(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and d.get("name"):
            out[str(d["name"])] = d
    return out


def classify(target: str) -> dict:
    import yaml

    binding, prof = _binding(target)
    synth_path = _profiles_dir() / f"{target}.synth.yaml"
    synth = yaml.safe_load(synth_path.read_text()) if synth_path.is_file() else {}
    subtrees = {t for t in _targets() if (merlin_dir() / "contract" / "capsules" / t).is_dir()}
    disk = _on_disk(target, subtrees)

    hand, hand_unrealized = _realize(prof.get("capsules"), binding, disk)
    derived, _ = _realize(synth.get("capsules"), binding, disk)

    by_sig: dict = {}
    for n, (sig, shp) in derived.items():
        by_sig.setdefault(sig, []).append((n, shp))

    # An entry whose obligation the vocabulary cannot express is never retirable, whatever its signature
    # says. Measured: a host/device interop member and a whole-model member have byte-identical axis
    # signatures -- same family, dtype, kind and instruction classes -- while the first exists to exercise
    # a seam the second never crosses. The signature is the corpus's own notion of interchangeable and it
    # is silent on that, so silence must not read as agreement.
    def _outside_the_vocabulary(name: str, cap) -> str | None:
        role = str((cap or {}).get("source_role") or "")
        if role in ("uplifted_from_bareMetalC", "handauthored_compiler_test", "spec_derived"):
            return f"source_role {role!r}: a person chose this case, and why is not in the axis vocabulary"
        lanes = (cap or {}).get("lanes")
        if lanes:
            return "declares a lane requirement, which the axis signature does not express"
        if str((cap or {}).get("kind")) == "model":
            return "a whole-model member's obligation is its network, which the signature cannot name"
        return None

    verdicts = {}
    for n, (sig, shp) in hand.items():
        blocked = _outside_the_vocabulary(n, disk.get(n))
        same = by_sig.get(sig)
        if blocked:
            verdicts[n] = (UNCOVERED, blocked)
        elif not same:
            verdicts[n] = (UNCOVERED, None)
        elif any(s == shp for _, s in same):
            verdicts[n] = (COVERED, next(m for m, s in same if s == shp))
        else:
            verdicts[n] = (COVERED_AT_SCALE, same[0][0])
    return {"target": target, "verdicts": verdicts, "unrealized": hand_unrealized,
            "n_hand": len(prof.get("capsules") or []), "n_derived": len(derived)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target")
    ap.add_argument("--include-rescaled", action="store_true",
                    help="also retire entries covered only at a different scale (check them first: a "
                         "hand entry often exists BECAUSE of its extents, which the signature cannot see)")
    ap.add_argument("--write", action="store_true", help="actually move the retired entries")
    args = ap.parse_args()

    targets = [args.target] if args.target else _targets()
    print(f"{'target':<16}{'hand':>6}{'derived':>9}{'covered':>9}{'at-scale':>10}{'uncovered':>11}{'unrealized':>12}")
    total = {COVERED: 0, COVERED_AT_SCALE: 0, UNCOVERED: 0}
    reports = {}
    for t in targets:
        try:
            rep = classify(t)
        except Exception as exc:  # noqa: BLE001
            print(f"{t:<16}  could not classify: {type(exc).__name__}: {exc}")
            continue
        reports[t] = rep
        c = sum(1 for v, _ in rep["verdicts"].values() if v == COVERED)
        s = sum(1 for v, _ in rep["verdicts"].values() if v == COVERED_AT_SCALE)
        u = sum(1 for v, _ in rep["verdicts"].values() if v == UNCOVERED)
        total[COVERED] += c; total[COVERED_AT_SCALE] += s; total[UNCOVERED] += u
        print(f"{t:<16}{rep['n_hand']:>6}{rep['n_derived']:>9}{c:>9}{s:>10}{u:>11}{len(rep['unrealized']):>12}")

    print(f"\n  retirable now (covered): {total[COVERED]}")
    print(f"  needs a look (covered only at a different scale): {total[COVERED_AT_SCALE]}")
    print(f"  must stay (uncovered): {total[UNCOVERED]}")
    if not args.write:
        print("\n  dry run — nothing moved. Re-run with --write to retire the covered entries.")
        return 0

    import yaml

    retire = {t: [n for n, (v, _) in r["verdicts"].items()
                  if v == COVERED or (args.include_rescaled and v == COVERED_AT_SCALE)]
              for t, r in reports.items()}
    if not any(retire.values()):
        print("\n[  ok] retirement: nothing is provably covered, so nothing moved. That is a RESULT, not "
              "an omission -- retiring on a weaker test is how a corpus loses coverage while every count "
              "goes up.")
        return 0

    out_dir = _profiles_dir() / "retired"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Retired hand-authored capsule entries",
             "",
             "Each entry below was removed from its target's live profile ONLY because a derived capsule",
             "exercises the same declared axis values. The covering capsule is named so the claim can be",
             "checked rather than trusted. Nothing here is deleted: the entries are kept verbatim beside",
             "this file so one that turns out to have been load-bearing is readable, not recovered from a",
             "reflog.", ""]
    for t, names in sorted(retire.items()):
        if not names:
            continue
        prof_path = _profiles_dir() / f"{t}.yaml"
        prof = yaml.safe_load(prof_path.read_text())
        keep = [e for e in (prof.get("capsules") or []) if str(e.get("name")) not in set(names)]
        moved = [e for e in (prof.get("capsules") or []) if str(e.get("name")) in set(names)]
        (out_dir / f"{t}.v0.yaml").write_text(
            "# DERIVED-OUT: entries retired from " + f"profiles/{t}.yaml" + " because a derived capsule\n"
            "# covers the same obligation. Tracked, and NOT read by the generator.\n"
            + yaml.safe_dump({"capsules": moved}, sort_keys=False))
        prof["capsules"] = keep
        prof_path.write_text(yaml.safe_dump(prof, sort_keys=False))
        lines.append(f"## {t}")
        lines.append("")
        for n in sorted(names):
            lines.append(f"- `{n}` — covered by `{reports[t]['verdicts'][n][1]}`")
        lines.append("")
    (out_dir / "RETIRED.md").write_text("\n".join(lines))
    print(f"\n  retired {sum(len(v) for v in retire.values())} entr(y/ies) to {out_dir}")
    print("  RE-GENERATE the affected targets and re-run check_conformance_coverage before committing: a "
          "profile and a generated tree that disagree read as a stale corpus, not as a retirement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
