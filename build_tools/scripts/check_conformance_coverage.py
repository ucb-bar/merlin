#!/usr/bin/env python3
"""Gate: does a target's capsule corpus cover the coverage requirement DERIVED for it?

The requirement comes from :mod:`merlin.targetgen.conformance` — the intersection of what the target's
capability manifest admits with what the real target-models' captures contain, expressed in
``cert_capsule_cover``'s ``(semantic_family, dtype, tile_alignment)`` cells. This script reports which
required cells no capsule exercises.

An uncovered cell is NOT a failing capsule. It means the corpus contains no capsule that would exercise
that family/dtype/alignment at all, so a submission's silence there is unmeasured rather than correct —
which is the failure mode a pass-rate cannot express. Radiance measured 21 of 56 covered when this was
first run, while its headline scorecard read 36/39.

Modes, mirroring the other gates in this directory:

  --target NAME        audit one target (repeatable); default: every target with a conformance spec
  --spec PATH          compare against a tracked spec instead of re-deriving (drift check)
  --write PATH         regenerate the spec (this is how the tracked spec is produced)
  --json               machine-readable
  --ratchet PATH       pre-existing debt that MAY ONLY SHRINK; unlisted new gaps fail
  --fail-on-uncovered  exit non-zero when any non-ratcheted cell is uncovered (default: report only)

Reporting-only by default because the derivation is new and the corpus predates it: turning a 35-cell gap
into a hard failure on day one would only teach everyone to pass `--no-verify`.
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

from merlin.common.paths import artifacts_dir, repo_root  # noqa: E402
from merlin.targetgen import conformance as CF  # noqa: E402


def _captures(model_root: Path | None = None) -> dict[str, Path]:
    """Every captured model bundle available as derivation evidence.

    The bundle store is the one place a capture is guaranteed to be the SAME IR the grader compiles
    (``_ensure_bundle`` writes it), so deriving from it cannot drift from what is actually graded.
    """
    root = model_root or (artifacts_dir() / "recaptures")
    if not root.is_dir():
        return {}
    out = {}
    for d in sorted(root.iterdir()):
        m = d / "model.mlir"
        if m.is_file():
            out[d.name.replace("_fp32_consistent", "").replace("_consistent", "")] = m
    return out


def _target_experiment(target: str) -> Path | None:
    cand = (repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets" / target
            / "target_experiment.yaml")
    return cand if cand.is_file() else None


def audit(target: str, *, spec_path: Path | None = None) -> dict:
    """Derive (or load) the requirement and measure the corpus against it."""
    from merlin.targetgen.target_experiment import load_target_experiment

    desc = _target_experiment(target)
    if desc is None:
        return {"target": target, "status": "no_target_experiment",
                "detail": f"no target_experiment.yaml for {target!r}"}
    te = load_target_experiment(desc)
    roots = list(te.graded_roots())
    exclude = set(getattr(te, "graded_exclude", ()) or ())

    caps = _captures()
    if spec_path and spec_path.is_file():
        import yaml
        doc = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        origin = f"tracked spec {spec_path}"
    else:
        doc = CF.spec(target, caps)
        origin = "derived now"
    if not doc.get("cells"):
        return {"target": target, "status": "no_requirement", "spec_origin": origin,
                "detail": ("nothing was derived: no capability manifest resolved, or no captured model "
                           "was readable. This is 'we do not know', never 'nothing is required'."),
                "captures_available": sorted(caps)}

    tile = (doc.get("boundaries") or {}).get("tile_edge")
    gap = CF.uncovered(doc, roots, labels={"public", "dev"}, tile_dim=tile, exclude=exclude)
    by_cell = {c["cell"]: c for c in doc["cells"]}
    return {
        "target": target,
        "status": "ok",
        "spec_origin": origin,
        "graded_roots": [str(Path(r).name) for r in roots],
        "graded_exclude": sorted(exclude),
        "captures_used": (doc.get("diagnostics") or {}).get("captures_read", sorted(caps)),
        "tile_edge": tile,
        "n_required": gap["n_required"],
        "n_covered": gap["n_covered"],
        "uncovered": [{"cell": c, "basis": by_cell.get(c, {}).get("basis"),
                       "admitted_by": by_cell.get(c, {}).get("admitted_by", [])}
                      for c in gap["uncovered"]],
        "corpus_cells_not_required": gap["extra_cells"],
        "diagnostics": doc.get("diagnostics") or {},
    }


def _load_ratchet(p: Path | None) -> set[str]:
    if not p or not p.is_file():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ratchet", type=Path, default=None)
    ap.add_argument("--fail-on-uncovered", action="store_true")
    a = ap.parse_args(argv)

    # DEFAULT TARGET SET IS DISCOVERED, not named: every target that already has a tracked conformance
    # spec. A hardcoded default here would make this gate silently about one target forever, which is the
    # overfitting the whole module exists to prevent (and the no-target-name gate rightly rejects it).
    targets = a.target or sorted(
        p.stem for p in (repo_root() / "merlin" / "contract" / "capsules" / "conformance").glob("*.yaml"))
    if not targets:
        print("no --target given and no tracked conformance spec found under "
              "merlin/contract/capsules/conformance/; pass --target NAME (with --write to create one)",
              file=sys.stderr)
        return 2

    if a.write:
        import yaml
        if len(targets) != 1:
            print("--write takes exactly one --target", file=sys.stderr)
            return 2
        doc = CF.spec(targets[0], _captures())
        a.write.parent.mkdir(parents=True, exist_ok=True)
        a.write.write_text(
            "# DERIVED — regenerate with:\n"
            f"#   build_tools/scripts/check_conformance_coverage.py --target {targets[0]} "
            f"--write {a.write.relative_to(repo_root()) if a.write.is_absolute() else a.write}\n"
            "# Do not hand-edit: the point of this file is that it is evidence, not authorship.\n"
            + yaml.safe_dump(doc, sort_keys=False, width=100), encoding="utf-8")
        print(f"wrote {a.write} — {len(doc['cells'])} required cell(s)")
        return 0

    ratchet = _load_ratchet(a.ratchet)
    reports = [audit(t, spec_path=a.spec) for t in targets]
    if a.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            if r["status"] != "ok":
                print(f"== {r['target']}: {r['status']} — {r.get('detail', '')}")
                continue
            print(f"== {r['target']}  ({r['spec_origin']})")
            print(f"   captures used : {r['captures_used']}")
            print(f"   tile edge     : {r['tile_edge']}")
            print(f"   covered       : {r['n_covered']} / {r['n_required']} required cell(s)")
            new = [u for u in r["uncovered"] if u["cell"] not in ratchet]
            if r["uncovered"]:
                print(f"   UNCOVERED     : {len(r['uncovered'])}"
                      + (f" ({len(new)} not in the ratchet)" if ratchet else ""))
                for u in r["uncovered"]:
                    mark = " " if u["cell"] in ratchet else "*"
                    print(f"     {mark} {u['cell']:34s} basis={u['basis']} by={u['admitted_by']}")
            if r["corpus_cells_not_required"]:
                print(f"   corpus cells not in the requirement: {r['corpus_cells_not_required']}")
                print("     (a cell the hardware does not admit for that family — e.g. an int8 movement "
                      "capsule on a target whose movement datapath is float-only — is INTENTIONAL: it is "
                      "what forces the compiler off the accelerator path)")
            for n in (r["diagnostics"].get("notes") or []):
                print(f"   note: {n}")

    bad = [u["cell"] for r in reports if r["status"] == "ok"
           for u in r["uncovered"] if u["cell"] not in ratchet]
    if bad and a.fail_on_uncovered:
        print(f"\nFAIL: {len(bad)} required cell(s) uncovered and not ratcheted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
