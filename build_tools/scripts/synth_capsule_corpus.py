#!/usr/bin/env python3
"""Synthesize a target's capsule-profile entries from its DERIVED conformance requirement.

The corpus pipeline derives every per-capsule field already; what was hand-written was WHICH capsules
exist -- roughly 180 entries across six profiles, and the one input a new target's owner cannot
reasonably be asked to produce. This closes the loop: requirement in, entries out, in the shape
``generate_corpus.py`` already consumes.

The output is a TRACKED intermediate (``profiles/<target>.synth.yaml``) rather than an injection at
generation time, for three reasons: a reviewer diffs the requirement change the same way they diff
``conformance/<target>.yaml``; ``--check`` gives the byte-stability gate a cheap comparison that needs
neither torch nor an oracle; and every consumer that reads ``prof["capsules"]`` picks the entries up for
free.

Modes, mirroring the sibling gates in this directory:

  --target NAME   synthesize one target (repeatable); default: every target with a conformance spec
  --write         write profiles/<target>.synth.yaml
  --check         re-derive and diff against the tracked file; non-zero on drift
  --json          machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
# The generator lives beside the capsules, not in the package tree, and `_ungradeable` imports it to
# ask the golden engines what they can grade. Without this the gradeability check cannot run -- and a
# check that cannot run reports `ungradeable_unchecked`, which is honest but blocks every synthesis.
for _p in (_HERE.parents[2] / "merlin" / "python",
           _HERE.parents[2] / "merlin" / "contract" / "capsules"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin.common.paths import merlin_dir  # noqa: E402
from merlin.targetgen.corpus_synth import SynthesisError, synthesize  # noqa: E402

_CONFORMANCE = "contract/capsules/conformance"
_PROFILES = "contract/capsules/profiles"

_HEADER = (
    "# DERIVED — regenerate with:\n"
    "#   build_tools/scripts/synth_capsule_corpus.py --target {target} --write\n"
    "# Do not hand-edit. These entries exist because the target's own conformance requirement asks for\n"
    "# them: each carries the cell it was synthesized for in `source_reference`. Editing one here is a\n"
    "# claim the requirement does not make, and the next regeneration discards it.\n"
)


def _targets(explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    root = merlin_dir() / _CONFORMANCE
    return sorted(p.stem for p in root.glob("*.yaml")) if root.is_dir() else []


def _workload_spec(target: str) -> dict:
    """The declared workload spec for ``target``, or an empty one.

    Absent is not an error: a target that declares no preference simply gets no tie-break, and the
    requirement alone still determines the corpus.
    """
    from merlin.targetgen.target_experiment import load_target_experiment

    desc = merlin_dir() / "experiments/capsule_bench/targets" / target / "target_experiment.yaml"
    if not desc.is_file():
        return {}
    return dict(getattr(load_target_experiment(desc), "workload_spec", None) or {})


def _gradeable_candidates(entries: list[dict]) -> list[dict]:
    """Entries a GOLDEN ENGINE grades, i.e. the op-level ones.

    A model capsule carries no `op`: its program is the derived micro model and its verdict comes from
    the whole-model path, not from an engine selected by dtype. Asking the golden question of it raises
    on the missing key, which reads as "gradeability could not be decided" for the whole target and
    blocks a synthesis that is fine.
    """
    return [e for e in entries if e.get("op")]


def _ungradeable(entries: list[dict], target: str) -> list[dict]:
    """Entries whose (op, dtype) pair no golden engine can grade -- reported, never written.

    ⚠️ AN OP BEING MATERIALIZABLE IS NOT THE SAME AS BEING GRADEABLE. `corpus_synth` chooses the
    cheapest op that exercises a family and can be WRITTEN, which is the right question for a builder
    and the wrong one for a golden: the engine is picked by the entry's DTYPE, and each engine covers a
    different op set. Two measured cases, both of which crashed inside the writer rather than being
    reported here:

      * radiance's `attention` cells resolve to `attention_mx`, whose golden exists only in the
        block-scaled engine, while the cells are fp16/bf16/f32 -- so the SIMT engine raised
        "no SIMT golden for op 'attention_mx'".
      * a body-only op at a non-float dtype needs a `quant_scheme` (a weight-only capture emits a float
        matmul, which cannot grade an integer datapath), and without one the generator refuses it.

    The check lives here rather than in `corpus_synth` because which engine grades which op is the
    GENERATOR's knowledge; importing it into the synthesizer would make a pure module depend on the
    thing that consumes it. Reported as a cell that could not be expressed, with the reason, so the
    requirement shows an honest hole instead of a corpus that fails to build.
    """
    import generate_corpus as GC

    out = []
    for entry in entries:
        regime, _ = GC._entry_regime(entry, _binding(target))
        source = entry.get("source")
        why = None
        if source == "pytorch" and regime != "simt" and not entry.get("quant_scheme"):
            why = (f"a pytorch-sourced capsule needs a float dtype or a declared quant_scheme; this "
                   f"cell is {entry.get('operand_dtype')!r} (regime {regime!r})")
        elif source is None and regime == "simt" and entry["op"] in _MX_ONLY_GOLDEN:
            why = (f"{entry['op']!r} has a golden only in the block-scaled engine, and this cell is "
                   f"{entry.get('operand_dtype')!r} (regime {regime!r})")
        if why:
            out.append({"name": entry["name"], "op": entry["op"],
                        "dtype": entry.get("operand_dtype"), "regime": regime, "reason": why})
    return out


#: Ops whose golden exists ONLY in the block-scaled engine. Read from the engine's own dispatch rather
#: than guessed: `generate_corpus._simt_golden` and `_float_golden` raise by name for these, and the
#: block-scaled path is the only one that implements them.
_MX_ONLY_GOLDEN = frozenset({"attention_mx", "gemv_batched"})


def _binding(target: str):
    import generate_corpus as GC
    from merlin.targetgen import corpus_spec as CSPEC
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment
    prof = GC.load_profile(target)
    te = load_target_experiment(descriptor_path(target))
    return CSPEC.derive_binding(te, prof.get("datapath") or {})


def synth_for(target: str) -> dict:
    import yaml

    spec_path = merlin_dir() / _CONFORMANCE / f"{target}.yaml"
    if not spec_path.is_file():
        return {"target": target, "status": "no_conformance_spec",
                "detail": f"no derived requirement at {spec_path}; write one with "
                          f"check_conformance_coverage.py --target {target} --write"}
    doc = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    try:
        out = synthesize(doc, workload_spec=_workload_spec(target))
    except SynthesisError as exc:
        return {"target": target, "status": "unsynthesizable", "detail": str(exc)}

    try:
        bad = _ungradeable(_gradeable_candidates(list(out.get("capsules") or ())), target)
    except Exception as exc:                       # noqa: BLE001 -- cannot check is not "all fine"
        return {"target": target, "status": "ungradeable_unchecked",
                "detail": f"could not decide gradeability: {type(exc).__name__}: {exc}", **out}
    if bad:
        keep = {b["name"] for b in bad}
        out["capsules"] = [e for e in out["capsules"] if e["name"] not in keep]
        out["ungradeable"] = bad
    return {"target": target, "status": "ok", **out}


def _render(target: str, res: dict) -> str:
    import yaml

    return _HEADER.format(target=target) + yaml.safe_dump(
        {"provenance": res["provenance"], "capsules": res["capsules"]}, sort_keys=False, width=100)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    targets = _targets(a.target)
    if not targets:
        print("no --target given and no tracked conformance spec found; write one with "
              "check_conformance_coverage.py --target NAME --write", file=sys.stderr)
        return 2

    results = [synth_for(t) for t in targets]
    if a.json:
        print(json.dumps(results, indent=2))

    rc = 0
    for res in results:
        target = res["target"]
        out_path = merlin_dir() / _PROFILES / f"{target}.synth.yaml"
        if res["status"] != "ok":
            if not a.json:
                print(f"== {target}: {res['status']} — {res.get('detail', '')}")
            # An unsynthesizable requirement is a FAILURE under --check: it means the requirement asks
            # for something no capsule can express, which is the state this whole loop exists to make
            # impossible to ship silently.
            rc = rc or (1 if (a.check and res["status"] == "unsynthesizable") else 0)
            continue
        text = _render(target, res)
        if a.write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            if not a.json:
                print(f"wrote {out_path} — {len(res['capsules'])} entry/entries from "
                      f"{res['provenance']['n_required_cells']} required cell(s)")
        elif a.check:
            have = out_path.read_text(encoding="utf-8") if out_path.is_file() else ""
            if have != text:
                print(f"== {target}: DRIFT — {out_path} differs from a fresh derivation; "
                      f"re-run with --write", file=sys.stderr)
                rc = 1
            elif not a.json:
                print(f"== {target}: ok ({len(res['capsules'])} entry/entries)")
        elif not a.json:
            print(f"== {target}: {len(res['capsules'])} entry/entries from "
                  f"{res['provenance']['n_required_cells']} required cell(s)")
            for e in res["capsules"]:
                print(f"     {e['name']:38s} op={e['op']:12s} dtype={e['operand_dtype']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
