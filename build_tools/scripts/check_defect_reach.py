#!/usr/bin/env python3
"""Could the corpus EXHIBIT the defects a hard-kernel campaign actually found?

A corpus is not evidence that a compiler is correct; it is evidence about the region it covers. When an
external campaign finds a defect the corpus never flagged, the useful question is not "why did the
capsules pass" but "could any capsule have failed" -- and that is answerable from the corpus alone,
without re-running anything.

This checks REACH, not outcome. For each defect class below it asks whether some capsule declares the
conditions under which the defect manifests. A class with no such capsule is UNREACHABLE: the corpus
cannot observe it, so a green run says nothing about it, and reporting that run as coverage is the
failure this repo keeps rediscovering.

THE CLASSES ARE DERIVED FROM MEASURED DEFECTS, not imagined. Each was found by a campaign over hard
kernels and is named with what it cost:

``epilogue_stage``      a fused per-column bias could not be emitted, and a standalone integer
                        requantization had no lowering. Both were omitted from the bundle rather than
                        counted as passing -- honest, but the corpus never demanded either.
``padding_identity``    a fused convolution/max-pool lost the declared padding identity. Wrong ONLY in
                        the border rows, so a corpus whose convolutions all declare zero padding cannot
                        see it whatever it runs.
``store_overflow``      a working set crossed the operand store and aliased, giving wrong outputs. The
                        corpus probes residency along a thin needle and never at both extents at once,
                        so no member's working set exceeds capacity.
``resident_reuse``      a weight that should stay resident was reconfigured per region. Numerically
                        correct, so only a capsule that DECLARES reuse across several activations can
                        expose it.

WHAT THIS IS NOT. Reach is necessary, not sufficient: a capsule can declare the conditions and still
fail to exhibit a defect the compiler only shows at another scale. An UNREACHABLE verdict is a proof
of absence; a REACHABLE one is only the absence of that proof.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common.paths import merlin_dir  # noqa: E402
from merlin.runtime.commandbuffer import EPILOGUE_STAGES  # noqa: E402


def _capsule_docs(root: Path, target: str, subtrees: set[str]):
    import yaml

    paths = (sorted((root / target).rglob("capsule.yaml")) if target in subtrees
             else sorted(p for p in root.rglob("capsule.yaml")
                         if p.relative_to(root).parts[0] not in subtrees | {"profiles"}))
    for p in paths:
        try:
            doc = yaml.safe_load(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(doc, dict):
            yield doc


def _attrs(cap):
    op = cap.get("operation") or {}
    a = op.get("attributes")
    return a if isinstance(a, dict) else {}


def _shapes(cap):
    out = []
    for row in cap.get("inputs") or ():
        if isinstance(row, dict) and isinstance(row.get("shape"), list):
            out.append(row)
    return out


def epilogue_reach(caps) -> dict:
    """Which declared epilogue stages does some capsule actually demand?"""
    seen: dict[str, list[str]] = {s: [] for s in EPILOGUE_STAGES}
    for c in caps:
        for stage in _attrs(c).get("epilogue") or ():
            seen.setdefault(str(stage), []).append(str(c.get("name") or ""))
        # a stage may also be the whole operation (a standalone bias or requant member)
        op = str((c.get("operation") or {}).get("op") or "")
        for stage in EPILOGUE_STAGES:
            if op == stage or op == f"{stage}_add" or op.endswith(f"_{stage}"):
                seen.setdefault(stage, []).append(str(c.get("name") or ""))
    return seen


def padding_reach(caps) -> dict:
    """Distinct convolution/pool geometries, and whether any declares a NON-ZERO padding."""
    geoms, padded, pad_values = set(), [], set()
    for c in caps:
        a = _attrs(c)
        if not any(k in a for k in ("kh", "kw", "pool_size")):
            continue
        pad = tuple(a.get("padding") or a.get("pool_padding") or ())
        geoms.add((pad, tuple(a.get("stride") or a.get("pool_stride") or ()),
                   tuple(a.get("dilation") or ())))
        if any(int(v) for v in pad if isinstance(v, int)):
            padded.append(str(c.get("name") or ""))
        if a.get("pad_value") is not None:
            pad_values.add(str(a.get("pad_value")))
    return {"distinct_geometries": len(geoms), "padded_members": padded,
            "declared_pad_values": sorted(pad_values)}


def store_overflow_reach(caps, *, target: str) -> dict:
    """Does any member's declared working set exceed the target's own operand store?"""
    try:
        from merlin.targetgen import memory_regime as MR

        store, capacity = MR.operand_store(target, dtype="i8")
    except Exception as exc:  # noqa: BLE001
        return {"capacity_elements": None, "over": [],
                "reason": f"the operand store is not derivable for this target ({type(exc).__name__}), "
                          "so overflow cannot be decided here -- unknown, not absent"}
    if not capacity:
        return {"capacity_elements": None, "over": [],
                "reason": "this target declares no operand-store capacity we can derive"}
    over = []
    for c in caps:
        total = 0
        for row in _shapes(c):
            n = 1
            for v in row["shape"]:
                n *= int(v)
            total += n
        if total > int(capacity):
            over.append((str(c.get("name") or ""), total))
    return {"capacity_elements": int(capacity), "over": over,
            "store": getattr(store, "name", None)}


def resident_reuse_reach(caps) -> dict:
    """Members declaring ONE weight reused across SEVERAL activations -- the only shape that can show a
    weight being reconfigured when it should have stayed resident."""
    chains = []
    for c in caps:
        rows = _shapes(c)
        weights = [r for r in rows if str(r.get("role")) == "weight"]
        acts = [r for r in rows if str(r.get("role")) == "input"]
        if len(weights) == 1 and len(acts) >= 2:
            chains.append((str(c.get("name") or ""), len(acts)))
    return {"chains": chains, "deepest": max([n for _, n in chains], default=0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="an unreachable defect class is a failure, not a note")
    args = ap.parse_args()

    root = merlin_dir() / "contract" / "capsules"
    targets = sorted(p.stem for p in (root / "profiles").glob("*.yaml")
                     if not p.stem.startswith("_") and "." not in p.stem)
    subtrees = {t for t in targets if (root / t).is_dir()}
    if args.target:
        targets = [args.target]

    report, unreachable = {}, []
    for t in targets:
        caps = list(_capsule_docs(root, t, subtrees))
        epi = epilogue_reach(caps)
        pad = padding_reach(caps)
        sto = store_overflow_reach(caps, target=t)
        res = resident_reuse_reach(caps)
        report[t] = {"n_capsules": len(caps), "epilogue": {k: len(v) for k, v in epi.items()},
                     "padding": pad, "store_overflow": sto,
                     "resident_reuse": {"n_chains": len(res["chains"]), "deepest": res["deepest"]}}

        for stage, members in epi.items():
            if not members:
                unreachable.append(f"{t}: epilogue stage {stage!r} is declared by the ABI and demanded "
                                   "by no capsule, so a backend that cannot emit it fails nothing here")
        if not pad["padded_members"]:
            unreachable.append(f"{t}: no capsule declares a non-zero padding, so a lowering that loses "
                               "the padding identity is wrong only in rows this corpus never computes")
        if sto["capacity_elements"] and not sto["over"]:
            unreachable.append(f"{t}: no capsule's working set exceeds the {sto['capacity_elements']}-element "
                               "operand store, so aliasing past capacity cannot be observed")
        if res["deepest"] < 2:
            unreachable.append(f"{t}: no capsule reuses one weight across several activations, so a weight "
                               "reconfigured when it should have stayed resident is invisible")

    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        print(f"{'target':<16}{'caps':>5}  {'epilogue reached':<34}{'padded':>7}{'over-store':>11}{'reuse':>7}")
        for t, r in report.items():
            reached = ",".join(k for k, n in r["epilogue"].items() if n) or "-"
            print(f"{t:<16}{r['n_capsules']:>5}  {reached[:33]:<34}"
                  f"{len(r['padding']['padded_members']):>7}{len(r['store_overflow']['over']):>11}"
                  f"{r['resident_reuse']['n_chains']:>7}")

    if unreachable:
        head = "[FAIL]" if args.strict else "[note]"
        print(f"\n{head} defect-reach: the corpus cannot EXHIBIT these classes, so a green run says "
              "nothing about them:")
        for line in unreachable:
            print(f"  - {line}")
        if args.strict:
            return 1
    else:
        print("\n[  ok] defect-reach: every declared class has at least one capsule that could exhibit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
