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
for _p in (_HERE.parents[2] / "merlin" / "python",):
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
