#!/usr/bin/env python3
"""Drive the mechanism calibration an analytical performance model has to be fitted against.

The performance phase deliberately keeps the cycle-accurate tier OUT of its inner loop: candidates go
``candidate -> L2 correctness -> analytical score -> roofline -> relative ranking``, and cycle-accurate
measurement appears twice -- **before** the run, on a small fixed set of mechanism-calibration capsules
whose measurements fit the model's coefficients, and **after**, as the functional regression basis.
Nothing drove the "before" half. The measurement libraries existed and were good; there was no driver.

This is the driver. It is deliberately thin: every reading is taken by
:mod:`merlin.perf.calibration`, which in turn delegates to the libraries that already know how each
number lies (:mod:`merlin.perf.occupancy`, :mod:`merlin.perf.falsifier`,
:mod:`merlin.perf.headroom`, :mod:`merlin.targetgen.memory_regime`,
:mod:`merlin.targetgen.rtl.fsm`). What lives here is only I/O: resolving a target's contract, its
graded corpus, its synthesis FSM inventory and its trace files, and writing the record through the
artifact helpers.

TWO MODES, AND THE FIRST ONE IS A REAL ANSWER
---------------------------------------------
With no ``--traces`` this reports a PLAN: the engine inventory and the memory-regime cover are derived
without running anything, so it can already say which mechanisms are uncalibratable on this target at
all and which capsules the cover would spend the expensive tier on. Every eta comes back UNKNOWN and
``ran_against_traces`` is False. It never reports a calibration that did not happen.

With ``--traces`` it consumes per-cycle occupancy traces through the ``MechanismTrace`` seam -- the
same shape :func:`merlin.perf.occupancy.calibrate_state_idle` and
:func:`merlin.perf.occupancy.joint_counts` already take, and the shape the co-simulation occupancy
drivers already produce -- and fills the cover.

TRACE FILE FORMAT (one JSON object per capsule, or a list of them)::

    {"capsule": "<name>",
     "columns": {"<signal>": ["<value per cycle>", ...], ...},
     "binding": {"<signal>": "<declared engine>", ...},
     "port_columns": ["<signal>", ...],      # top-level busy ports: their own reference
     "state_columns": ["<signal>", ...],     # internal state regs: encoding must be calibrated
     "unmeasured_units": [...],              # what the instrument states it did NOT read
     "work": "<work fingerprint>",           # null == not stated, recorded as such
     "completion_observable": true|false|null,
     "port_low": ["0", ""],                  # optional: what THIS instrument's ports read when low
     "provenance": "<how it was recorded>"}

Usage::

    perf_calibrate.py                                   # the harness's active target, plan mode
    perf_calibrate.py --target T                        # plan: cover + uncalibratable mechanisms
    perf_calibrate.py --target T --traces DIR_OR_FILE   # calibrate against supplied traces
    perf_calibrate.py --target T --dry-run              # print, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _common as C  # noqa: E402 -- bootstraps merlin/python and resolves the ACTIVE target

from merlin.common import artifacts as A                                        # noqa: E402
from merlin.perf import calibration as CAL                                      # noqa: E402

#: The product topic. ``perf-calibration`` is its own concern: the ledger records what a run cost,
#: this records what the MODEL was fitted against, and conflating them would let a stale calibration
#: be cited by a fresh ledger.
TOPIC = "perf-calibration"
PRODUCT_VERSION = 1
RECORD_NAME = "calibration.json"


def _contract(target: str) -> dict:
    """The target's capability contract, through the registry rather than by path."""
    from merlin.targetgen import target_registry as TR
    return TR.load_contract(target)


def _fsm_registers(target: str) -> list:
    """The synthesis FSM inventory, or an empty list meaning NO EXTRACTION WAS FOUND.

    Empty is a statement about the extraction and not about the design, and
    :func:`merlin.perf.calibration.engine_inventory` records it that way. This driver does not run
    synthesis: an inventory that is not on disk stays absent rather than being approximated.
    """
    from merlin.targetgen.rtl.fsm import fsm_inventory
    try:
        return list(fsm_inventory(target))
    except OSError:
        return []


def _capsule_dirs(target: str) -> dict[str, Path]:
    """``{capsule name: dir}`` over the target's own GRADED roots, public label only.

    The target's roots, not the corpus parent: grading one target's package against the parent pulled
    in 173 capsules from seven targets and reported ``1/84`` when that target's suite is 36.
    """
    import yaml
    from merlin.targetgen.corpora import graded_capsule_roots

    out: dict[str, Path] = {}
    for root in graded_capsule_roots(target):
        for cy in sorted(Path(root).rglob("capsule.yaml")):
            try:
                doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if doc.get("label") != "public":
                continue
            out.setdefault(str(doc.get("name") or cy.parent.name), cy.parent)
    return out


def _regimes(target: str, dirs: dict[str, Path]) -> tuple[dict, dict]:
    """``(corpus_regimes, regime_by_capsule)`` -- the regime cover and each capsule's own regime."""
    from merlin.targetgen import memory_regime as MR

    store, capacity = MR.operand_store(target)
    by_capsule = {name: MR.capsule_regime(d, target, store=store, capacity=capacity)
                  for name, d in sorted(dirs.items())}
    by_regime: dict[str, list[str]] = {}
    largest = {"name": None, "rows": 0, "fraction_of_capacity": 0.0}
    for name, got in by_capsule.items():
        by_regime.setdefault(got.get("regime") or MR.UNKNOWN, []).append(name)
        if int(got.get("rows") or 0) > int(largest["rows"] or 0):
            largest = {"name": name, "rows": got.get("rows"),
                       "fraction_of_capacity": got.get("fraction_of_capacity")}
    corpus = {"by_regime": {k: sorted(v) for k, v in sorted(by_regime.items())},
              "capacity_rows": int(capacity) if capacity else None,
              "largest_working_set": largest, "n_capsules": len(by_capsule)}
    return corpus, by_capsule


def _load_traces(where: Path) -> list[CAL.MechanismTrace]:
    """Read the trace seam. A file may hold one object or a list; a directory is read in name order.

    Anything that is not a mapping with per-cycle columns is refused loudly rather than skipped: a
    trace file this driver could not read is not a capsule with no overlap.
    """
    paths = sorted(where.glob("*.json")) if where.is_dir() else [where]
    if not paths:
        raise SystemExit(f"no trace files under {where}")
    out: list[CAL.MechanismTrace] = []
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        for raw in (doc if isinstance(doc, list) else [doc]):
            if not isinstance(raw, dict) or not isinstance(raw.get("columns"), dict):
                raise SystemExit(f"{p}: not a trace object (needs a 'columns' mapping)")
            name = str(raw.get("capsule") or p.stem)
            out.append(CAL.MechanismTrace(
                capsule=name,
                columns={str(k): [str(x) for x in v] for k, v in raw["columns"].items()},
                binding={str(k): str(v) for k, v in (raw.get("binding") or {}).items()},
                port_columns=tuple(str(x) for x in (raw.get("port_columns") or ())),
                state_columns=tuple(str(x) for x in (raw.get("state_columns") or ())),
                unmeasured_units=tuple(str(x) for x in (raw.get("unmeasured_units") or ())),
                work=(None if raw.get("work") is None else str(raw["work"])),
                completion_observable=raw.get("completion_observable"),
                port_low=(tuple(str(x) for x in raw["port_low"]) if raw.get("port_low") is not None
                          else CAL.DEFAULT_PORT_LOW),
                provenance=str(raw.get("provenance") or f"trace file {p.name}")))
    return out


def _provenance(target: str, dirs: dict[str, Path]) -> dict:
    """The provenance block. A calibration is cited as a hardware fact, so it records its revision.

    ``sources`` is the bytes actually READ -- the contract and the selected capsules' declarations --
    because a dirty tree changes what a derivation emitted while the commit still looks right.
    """
    from merlin.common import provenance as P
    from merlin.targetgen.rtl.facts import target_contract_path
    try:
        sources = [target_contract_path(target)]
    except Exception:                                        # noqa: BLE001 -- no contract path
        sources = []
    sources += [d / "capsule.yaml" for d in sorted(dirs.values())]
    present = [Path(s).resolve() for s in sources if Path(s).is_file()]
    try:
        from merlin.common.paths import repo_root
        got = P.record(sources=[str(s) for s in present],
                       extra={"trace_instrument": "supplied through the MechanismTrace seam"})
        # The digest is taken over the ABSOLUTE paths, so it is over bytes that were actually read;
        # the NAMES are then rewritten repo-relative, because an absolute path in a published artifact
        # says where one machine kept its checkout, which is not provenance.
        root = Path(repo_root()).resolve()
        got["sources"] = [str(s.relative_to(root)) if s.is_relative_to(root) else str(s)
                          for s in present]
        return got
    except Exception as exc:                                 # noqa: BLE001 -- registry unusable
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _print(rec: dict) -> None:
    inv = rec["engine_inventory"]
    print(f"target={rec['target']}  traces={rec['n_traces']}  "
          f"ran_against_traces={rec['ran_against_traces']}")
    print(f"  engines declared={inv['n_declared']} observable={inv['observable']}")
    for eng, why in inv["unobservable"].items():
        print(f"    UNOBSERVABLE {eng}: {why[:150]}")
    print(f"  fsm registers detected={inv['n_detected']} "
          f"undeclared={len(inv['detected_undeclared'])}")
    mr = rec["memory_regimes"]
    print(f"  operand store capacity_rows={mr['capacity_rows']} "
          f"regimes={ {k: len(v) for k, v in mr['by_regime'].items()} }")
    cs = rec["calibration_set"]
    print(f"  cover: calibrated={cs['n_calibrated']} uncovered={cs['n_uncovered']} "
          f"uncalibratable={cs['n_uncalibratable']}")
    for cell in cs["cells"]:
        print(f"    [{cell['state']:<15s}] {cell['axis']}:{cell['key']} "
              f"-> {list(cell['capsules'])}")
        print(f"        {cell['why'][:220]}")
    for cap in rec["capsules"]:
        eta = cap["eta"]
        shown = (f"{eta['value']:.4f}" if eta["state"] == CAL.MEASURED else "UNKNOWN")
        print(f"  {cap['capsule']:<28s} eta={shown} overlap_observable={cap['overlap_observable']} "
              f"live={list(cap['live_engines'])}")
        if eta["state"] != CAL.MEASURED:
            print(f"        why: {eta['why'][:220]}")
    comp = rec["composition"]
    for axis in ("engine_axis", "kind_axis"):
        entry = comp.get(axis) or {}
        op, eta = entry.get("operator") or {}, entry.get("eta") or {}
        shown = op.get("value") if op.get("state") == CAL.MEASURED else "UNKNOWN"
        eta_shown = (f"{eta['value']:.4f}" if eta.get("state") == CAL.MEASURED else "UNKNOWN")
        print(f"  composition[{axis}]: operator={shown} eta={eta_shown}")
        if op.get("state") != CAL.MEASURED:
            print(f"        why: {str(op.get('why'))[:220]}")
    print(f"  audit ok={rec['audit']['ok']} violations={len(rec['audit']['violations'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure the mechanisms an analytical performance model must be calibrated "
                    "against, and report which mechanisms nothing in this corpus can calibrate.")
    # OPTIONAL, defaulting to the harness's active target. The overnight runner invokes this script
    # with no arguments (`stage_calibration` in run_overnight.py), and a required flag would have made
    # that stage exit 2 and be journalled as a failed calibration -- a tooling limit wearing a
    # measurement's clothes, which is the failure mode this repo keeps re-finding. The active target
    # is the one _common already resolved from MERLIN_TARGET_EXPERIMENT, so both callers agree.
    ap.add_argument("--target", default=C.TARGET,
                    help="target name (its contract and corpus are looked up); defaults to the "
                         f"harness's active target ({C.TARGET!r}, from MERLIN_TARGET_EXPERIMENT)")
    ap.add_argument("--traces", type=Path, default=None,
                    help="per-cycle trace file or directory (see the module docstring for the shape); "
                         "omitted = plan mode, every eta UNKNOWN")
    ap.add_argument("--declared-idle-value", default=None,
                    help="the value the PRODUCER states its state registers hold when idle. Used ONLY "
                         "if the cycle-exact derivation refuses, never as a default, and stamped "
                         "declared_by_producer on every number downstream of it")
    ap.add_argument("--points-per-cell", type=int, default=CAL.POINTS_PER_CELL,
                    help="capsules per calibration cell (default 2: one point cannot separate a rate "
                         "from a fixed intercept)")
    ap.add_argument("--dry-run", action="store_true", help="print the record, write nothing")
    ap.add_argument("--notes", default="")
    args = ap.parse_args(argv)
    if not args.target:
        ap.error("no target: pass --target, or set MERLIN_TARGET_EXPERIMENT so the harness resolves one")

    contract = _contract(args.target)
    dirs = _capsule_dirs(args.target)
    corpus, by_capsule = _regimes(args.target, dirs)
    traces = _load_traces(args.traces) if args.traces else []

    rec = CAL.calibrate(
        target=args.target, contract=contract, traces=traces,
        corpus_regimes=corpus, regime_by_capsule=by_capsule,
        fsm_registers=_fsm_registers(args.target),
        declared_idle_value=args.declared_idle_value,
        points_per_cell=args.points_per_cell,
        provenance=_provenance(args.target, dirs), notes=args.notes)
    _print(rec)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    pd = A.new_product(TOPIC, version=PRODUCT_VERSION, target=args.target,
                       notes=args.notes or "mechanism calibration for the analytical performance model")
    out = pd.add_artifact(RECORD_NAME)
    out.write_text(json.dumps(rec, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    pd.write_manifest()
    print(f"\nwrote {out}")
    return 0 if rec["audit"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
