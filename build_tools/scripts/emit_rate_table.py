#!/usr/bin/env python3
"""Derive a target's per-class throughput table from its certification runs, and gate it.

Phase 1 certifies the functional corpus on the cycle-accurate tier and every one of those runs emits
a cycle count beside an emitted program. This turns that into the rate table two consumers refuse to
invent for themselves -- ``compose_estimate``'s empirical ceiling and ``routing.MeasuredCost`` -- and
writes it as a versioned product with its acceptance result attached.

THE GATE IS PART OF THE PRODUCT, not a separate report. ``compose_estimate``'s own contract is that
nothing it produces may be shown to an authoring agent until held-out containment has been run and
reported, so a table written without that number would be unusable by the rule that governs its only
consumer. Both travel in one file, and ``--check`` exits non-zero when the bound does not hold.

Usage::

    emit_rate_table.py --target gemmini                 # derive, report, write the product
    emit_rate_table.py --target gemmini --check         # non-zero exit if containment is not 1.0
    emit_rate_table.py --target gemmini --dry-run       # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common.artifacts import new_product  # noqa: E402
from merlin.perf import rate_table as RT  # noqa: E402

#: The product topic. Concern-first, target at folder level, per the generated-output convention.
TOPIC = "rate-table"
VERSION = 1


def derived_peak(target: str) -> tuple[float, dict]:
    """The target's structural peak, from its own RTL facts. Never assumed, never defaulted.

    A rate table over an assumed peak describes a machine nobody has, so a target whose facts do not
    carry a compute array is refused here rather than priced against a guess.
    """
    from merlin.targetgen.rtl import facts as F

    arrays = (F.load_facts(target).get("facts") or {}).get("arrays") or []
    usable = [a for a in arrays if isinstance(a, dict) and a.get("instances")]
    if not usable:
        raise SystemExit(f"{target}: RTL facts declare no compute array with an instance count, so "
                         f"this target has no derived structural peak and nothing here may price it")
    # The widest array is the compute peak; a target with several reports which one answered.
    chosen = max(usable, key=lambda a: int(a["instances"]))
    return float(chosen["instances"]), {
        "array": chosen.get("name"), "rows": chosen.get("rows"), "cols": chosen.get("cols"),
        "instances": chosen.get("instances"), "corroborated": chosen.get("corroborated"),
        "source": chosen.get("source")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless every held-out program is contained")
    ap.add_argument("--dry-run", action="store_true", help="report without writing a product")
    args = ap.parse_args(argv)

    peak, provenance = derived_peak(args.target)
    programs, _ = RT.observed_programs(args.target)
    table = RT.rates_for(args.target, peak_macs_per_cycle=peak, programs=programs)
    gate = RT.holdout_containment(args.target, peak_macs_per_cycle=peak, programs=programs)

    doc = {"table": table.to_dict(), "acceptance": gate, "peak_provenance": provenance}

    print(f"target {args.target}: peak {peak:g} MACs/cycle "
          f"({provenance['array']} {provenance['rows']}x{provenance['cols']}, "
          f"corroborated={provenance['corroborated']})")
    print(f"  programs seen : {table.n_programs_seen}")
    print(f"  classes rated : {len(table.rates)}  {sorted(table.rates)}")
    print(f"  UNPRICED      : {list(table.unpriced_classes)}")
    for name, rate in sorted(table.rates.items()):
        print(f"    {name:18s} slowest={rate.slowest_macs_per_cycle:9.4f} n={rate.n_programs:3d} "
              f"cycles {rate.cycles_min:.0f}..{rate.cycles_max:.0f}")
    rate_pct = gate["containment_rate"]
    print(f"  containment   : {gate['contained']}/{gate['n_decided']} "
          f"({'n/a' if rate_pct is None else f'{rate_pct:.3f}'}), "
          f"below={gate['below_floor']} above={gate['above_ceiling']}")
    width = gate["median_band_width"]
    print(f"  median width  : {'n/a' if width is None else f'{width:.1f}x'} "
          f"-- containment alone is cheap; this is the other half of the result")

    if not args.dry_run:
        product = new_product(TOPIC, version=VERSION, target=args.target,
                              notes="per-compute-class throughput harvested from certification runs")
        out = Path(product.path) / "rate_table.json"
        out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  wrote {out}")

    if args.check:
        if not gate["n_decided"]:
            print("REFUSED: no held-out program could be decided, so the bound is untested",
                  file=sys.stderr)
            return 2
        if rate_pct is not None and rate_pct < 1.0:
            print(f"FAILED: {gate['below_floor']} below the floor and {gate['above_ceiling']} above "
                  f"the ceiling; the bound does not hold and must not be used to price anything",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
