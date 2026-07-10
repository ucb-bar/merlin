#!/usr/bin/env python3
"""P26 final-presentation-pass checker.

Validates the curated clean figure set against the safe-wording / honesty rules. Reads
``figure_manifest.csv`` (source of truth: plot_id, class, evidence_tier, scale, title, caveat,
source_artifact, file) + ``final_slide_skeleton.md``. Pure stdlib; exits non-zero on any failure.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "figure_manifest.csv"
SLIDES = HERE / "final_slide_skeleton.md"

FORBIDDEN = ["speedup", "faster", "predicted performance", "achieved throughput",
             "hardware-independent roofline", "optimal", "best design"]
VALID_SCALE = {"captured-config", "deployment-composition", "structural", "proxy", "QA"}

fail = []
ok = []


def chk(cond, msg):
    (ok if cond else fail).append(msg)


def main() -> int:
    if not MANIFEST.is_file():
        print(f"[FAIL] missing {MANIFEST}")
        return 1
    rows = list(csv.DictReader(MANIFEST.read_text().splitlines()))
    chk(len(rows) > 0, f"manifest has {len(rows)} figures")

    for r in rows:
        pid, klass = r["plot_id"], r["class"]
        title = (r.get("title") or "").strip()
        sub = (r.get("caveat") or "")
        text = f"{title} {sub}".lower()
        # disclaimers like "not a speedup" / "not faster" are SAFE — strip negated forms before scanning
        scan = re.sub(r"\bnot\s+(a\s+|an\s+)?(speedup|speed-up|faster|optimal|hardware result|"
                      r"performance\s+\w+)", " ", text)
        fig = HERE / r["file"]
        # 1-5: existence + required fields
        chk(fig.is_file(), f"{pid}: figure file exists ({r['file']})")
        chk(bool(r.get("source_artifact")), f"{pid}: has a source artifact")
        chk(bool(r.get("evidence_tier")), f"{pid}: has an evidence tier")
        chk(r.get("scale") in VALID_SCALE, f"{pid}: scale '{r.get('scale')}' is valid")
        chk(bool(title), f"{pid}: has a title (DSE statement)")
        # 6: forbidden words (after stripping negated disclaimers)
        bad = [w for w in FORBIDDEN if w in scan]
        chk(not bad, f"{pid}: no forbidden wording {bad if bad else ''}")
        # 7: roofline / arithmetic-intensity qualifiers
        if "roofline" in text or "arithmetic intensity" in text:
            chk(("weight-stream" in text or "full-memory" in text), f"{pid}: AI/roofline has weight-stream/full-memory qualifier")
            chk("not measured performance" in text or "not a chip" in text or "modeling view" in text,
                f"{pid}: AI/roofline says not-measured-performance")
        # balance-band qualifier only where a machine balance is actually drawn
        if "roofline" in text or "machine-balance" in text or "machine balance" in text:
            chk(("hypothetical" in text or "parametric" in text), f"{pid}: roofline has hypothetical/parametric balance")
        # 8: real-time / requirement
        if "real-time" in text or "30hz" in text or "30 hz" in text or "requirement" in text:
            chk(("requirement" in text or "floor" in text), f"{pid}: real-time plot says requirement/floor")
            chk("not a chip" in text or "not measured" in text or "not a speedup" in text or "modeling" in text,
                f"{pid}: real-time plot disclaims measured HW")
        # 9: K provenance
        if re.search(r"\bK\b", title + " " + sub) and pid in ("decision_weight_residency",):
            chk(("ir" in text or "scf.for" in text or "source" in text or "config" in text or "scenario" in text),
                f"{pid}: K labeled IR-recovered/source-config/scenario")

    # 11: QA-only plots must not be classified main
    qa_only = {"evidence_type_by_workload", "evidence_type_by_phase", "required_command_rate_envelope",
               "primitive_regret_bar", "decision_primitive_choice", "avoidable_reload_by_region",
               "realtime_requirement_surface", "boundary_placement_heatmap"}
    main_ids = {r["plot_id"] for r in rows if r["class"] == "main"}
    chk(not (qa_only & main_ids), f"QA-only plots not in main set ({qa_only & main_ids or 'clean'})")
    chk(len(main_ids) >= 9, f"main set has {len(main_ids)} plots (>=9)")

    # 12: every slide in the skeleton has a caveat line
    if SLIDES.is_file():
        blocks = re.split(r"^##\s+Slide", SLIDES.read_text(), flags=re.M)[1:]
        for i, b in enumerate(blocks, 1):
            chk("caveat" in b.lower(), f"slide {i} has a caveat line")
    else:
        fail.append(f"missing {SLIDES}")

    for m in ok:
        print(f"[PASS] {m}")
    for m in fail:
        print(f"[FAIL] {m}")
    print(f"\n{len(ok)}/{len(ok) + len(fail)} checks passed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
