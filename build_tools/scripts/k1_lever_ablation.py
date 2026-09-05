#!/usr/bin/env python3
"""LEAVE-ONE-OUT lever ablation on the K1: what does each compiler lever contribute to the WALL?

The project claim is that the win comes from whole-compiler GLOBAL optimizations while the inner
loop stays competitive. Of the levers in ``merlin.mining.wholemodel_proposer.RANKED_LEVERS`` roughly
seven are whole-program (weight prepack, buffer promotion, memref-copy expansion, elementwise
fusion, self-copy erasure, whole-model accumulator residency, quantize-before-gather) and the rest
are inner-loop. There is NO per-lever attribution to the wall for any of them. Everything currently
known is STATIC: transposes removed, attributes stripped, bytes not moved.

Static evidence has been wrong twice on this tree. ``fold_weight_transpose`` had flawless static
evidence and cost 1.09x on the board. ``vectorize_non_contraction_generics`` produced 4.9x more
vector instructions, bit-identical output, and a 1.28x SLOWDOWN. This driver exists to replace that
class of evidence with a measured, anchored one.

WHAT IT DOES. Given ``--features A,B,C`` and ``--models m1,m2`` it runs, per model, one FULL cell
carrying the whole feature set and one cell per lever with that lever REMOVED, by invoking the
per-cell instrument ``k1_int8_fair_compare.py`` as a subprocess -- the same way
``k1_int8_et_campaign.py`` does. It owns nothing inside the instrument and re-implements none of its
guards.

THE COMPARISON RULE, WHICH IS THE WHOLE POINT.  Each cell measures ours AND ExecuTorch interleaved
in ONE session. Board conditions drift between cells, so the only sound comparison is RATIO VERSUS
RATIO: ``(ours/ET) with the lever`` against ``(ours/ET) without it``, with ExecuTorch as the common
anchor absorbing the drift. Comparing two ``ours_ns`` across sessions is exactly the unsound thing
this repo already got burned by -- a 1.61x weight-transpose result taken across two sessions whose
ExecuTorch arms were 13% apart. So:

  * the contribution is computed ONLY as a ratio of ratios, and
  * a cell that produced no ExecuTorch anchor yields a REFUSAL, never a fallback to a bare
    ``ours_ns`` delta. ``cell_ratio`` is the single door, and it has no such fallback behind it.

WHAT IT REFUSES. A cell with no verdict (a build that outran its ceiling, a gate that failed, an
ExecuTorch arm that did not export) records its refusal string and contributes NOTHING -- a missing
arm never becomes a zero contribution. A pair whose two cells were built from DIFFERENT compiler
sources (this tree is shared; other sessions commit mid-run) is refused on the instrument's own
``source_digest``, not attributed.

THE NOISE FLOOR IS PART OF THE VERDICT. The K1's measured noise band is 2.6%; a contribution inside
it is labelled ``within_noise`` and is NOT a result.

Usage::

  # what would run, and with which feature set -- spends no board time
  PYTHONPATH=merlin/python .venv/bin/python build_tools/scripts/k1_lever_ablation.py \
      --models small_llama,deepjscc \
      --features prepack_weight_layout,perop_register_block,promote_buffers_to_stack \
      --dry-run

  # run it (board serialized: one session at a time)
  MERLIN_K1_HOST=root@<board-ip> PYTHONPATH=merlin/python .venv/bin/python \
      build_tools/scripts/k1_lever_ablation.py --models ... --features ...

  # resume a session that died: every completed cell is kept
  ... k1_lever_ablation.py --models ... --features ... \
      --out-dir out/artifacts/lever-ablation/v1/lever-ablation_v1_<TS>_<sha7>
"""
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))

from merlin.common import provenance as _prov              # noqa: E402
from merlin.common.artifacts import new_product, utc_stamp  # noqa: E402
from merlin.common.paths import repo_root                   # noqa: E402
from merlin.compare import et_campaign as ec                # noqa: E402

#: The K1's measured noise band (noise floor >= 1.9%, band 2.6%). A contribution inside it is not a
#: result; see the `k1-measurement-noise` note and `k1_int8_fair_compare.session_drift`.
NOISE_BAND = 0.026

#: The per-cell instrument. READ-ONLY from here: this driver runs it and reads what it wrote.
INSTRUMENT = "build_tools/scripts/k1_int8_fair_compare.py"

#: Our int8 codegen package (repo-relative), overridable with ``--package``.
DEFAULT_PACKAGE = "out/artifacts/targets/rvv/hand_v0_int8"

#: Sources whose bytes decide what this ablation measured.
_DRIVER_SOURCES = ("build_tools/scripts/k1_lever_ablation.py",
                   INSTRUMENT,
                   "merlin/python/merlin/compare/et_campaign.py",
                   "merlin/python/merlin/llvmlower/impr_features.py",
                   "merlin/python/merlin/mining/wholemodel_proposer.py")

#: The cell id of the arm carrying every lever. The other cells are named for what they DROP.
FULL_CELL = "full"


# --------------------------------------------------------------------------------------------
# Feature names: validated BEFORE any board time, against the union of the real sources.
# --------------------------------------------------------------------------------------------

def feature_name_sources() -> tuple[dict, list[str]]:
    """``({source_label: {names}}, [failures])`` -- every place a lever name can legitimately live.

    THREE sources, because a valid name does not have to be in ``impr_features._REGISTRY``:

    1. the impr-feature registry, after importing the two modules that register the lazily-bound
       levers (``llvmlower.lower`` and ``mining.wholemodel_proposer``) -- without those imports
       ``prepack_weight_layout`` and ``cse_through_provenance`` are ABSENT from the registry, and a
       naive check against it would reject two of the levers this tool exists to ablate;
    2. the ``FEATURE`` constant of every ``merlin.llvmlower`` submodule that declares one, which is
       where a BUILD-PATH lever (handled in ``mining.k1`` / ``runtime.backends.zephyr_model``
       rather than by a pipeline edit) is named;
    3. ``RANKED_LEVERS`` itself, so the ranked list can never name something this driver rejects.

    A source that fails to import is REPORTED, never silently dropped: a narrowed universe would
    turn a valid lever into a startup error whose message blames the user. Validation fails closed
    on top of that -- an unknown name is an error, and the error says which sources were consulted
    and which of them could not be loaded.
    """
    sources: dict[str, set[str]] = {}
    failures: list[str] = []

    for mod in ("merlin.llvmlower.lower", "merlin.mining.wholemodel_proposer"):
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mod}: {type(e).__name__}: {e}")
    try:
        from merlin.llvmlower import impr_features as _impr
        sources["impr_features registry"] = set(_impr.known())
    except Exception as e:  # noqa: BLE001
        failures.append(f"merlin.llvmlower.impr_features: {type(e).__name__}: {e}")

    build_path: set[str] = set()
    try:
        import merlin.llvmlower as _ll
        for m in pkgutil.iter_modules(_ll.__path__):
            try:
                sub = importlib.import_module(f"merlin.llvmlower.{m.name}")
            except Exception as e:  # noqa: BLE001
                failures.append(f"merlin.llvmlower.{m.name}: {type(e).__name__}: {e}")
                continue
            name = getattr(sub, "FEATURE", None)
            if isinstance(name, str) and name:
                build_path.add(name)
        sources["llvmlower module FEATURE constants"] = build_path
    except Exception as e:  # noqa: BLE001
        failures.append(f"merlin.llvmlower package scan: {type(e).__name__}: {e}")

    try:
        from merlin.mining.wholemodel_proposer import RANKED_LEVERS
        sources["wholemodel_proposer.RANKED_LEVERS"] = {n for n, _ in RANKED_LEVERS}
    except Exception as e:  # noqa: BLE001
        failures.append(f"merlin.mining.wholemodel_proposer.RANKED_LEVERS: {type(e).__name__}: {e}")
    return sources, failures


def known_feature_names() -> set[str]:
    """Every lever name this driver will accept without further derivation."""
    sources, _ = feature_name_sources()
    out: set[str] = set()
    for names in sources.values():
        out |= names
    return out


def is_known_feature(name: str, known: set[str] | None = None) -> bool:
    """A name is known if it is in the union, or if the registry can DERIVE it from its own spelling
    (the ``accum_resident_v3*`` tuning points are registered on demand from the name).  Anything
    else is unknown -- fail closed."""
    if name in (known if known is not None else known_feature_names()):
        return True
    try:
        from merlin.llvmlower import impr_features as _impr
        _impr.get(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def unknown_features(names) -> list[str]:
    """The subset of ``names`` this driver cannot resolve. Empty means every name is real."""
    known = known_feature_names()
    return [n for n in names if not is_known_feature(n, known)]


# --------------------------------------------------------------------------------------------
# The cell list: the full set, plus one cell per lever with that lever removed.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AblationCell:
    """One measurement to make: a cell id, what it drops, and the feature set it carries."""

    cell_id: str
    dropped: str | None
    features: tuple

    @property
    def feature_arg(self) -> str:
        """What is handed to ``--features``. An EMPTY string is meaningful and is passed as such:
        the instrument reads ``--features ''`` as the frozen baseline (no features), which is
        exactly the leave-one-out arm of a single-lever set. Omitting the flag instead would make
        the instrument fall back to the PACKAGE's own certified feature list -- a different, larger
        set -- and the ablation would silently measure the wrong control."""
        return ",".join(self.features)

    def as_dict(self) -> dict:
        return {"cell_id": self.cell_id, "dropped": self.dropped,
                "features": list(self.features), "feature_arg": self.feature_arg}


def plan_cells(features) -> list[AblationCell]:
    """``[full] + [one cell per lever, that lever removed]`` -- in the given lever order.

    Order is preserved rather than sorted so the printed plan reads like the feature string the
    caller passed. Duplicate names collapse (a set is what the compiler sees anyway) but the first
    occurrence keeps its position."""
    seen: list[str] = []
    for f in features:
        f = f.strip()
        if f and f not in seen:
            seen.append(f)
    cells = [AblationCell(FULL_CELL, None, tuple(seen))]
    for lever in seen:
        cells.append(AblationCell(f"drop_{lever}", lever,
                                  tuple(f for f in seen if f != lever)))
    return cells


def cell_key(model: str, cell_id: str) -> str:
    """Ledger identity of one cell. Used for resume: a recorded cell is not re-run."""
    return f"{model}::{cell_id}"


def recorded_keys(rows, *, retry_refused: bool = False) -> set:
    """Cells with a recorded outcome. A refusal IS an outcome -- re-running a cell to re-derive a
    refusal it already recorded spends board time to learn nothing -- unless ``retry_refused``,
    which is for the case where the refusals were about the SESSION (a board that went away) rather
    than about the cell."""
    out = set()
    for r in rows:
        k = r.get("key")
        if not k:
            continue
        if retry_refused and r.get("status") != "measured":
            continue
        out.add(k)
    return out


# --------------------------------------------------------------------------------------------
# Reading one cell, and the ratio-of-ratios that is the only contribution this tool will report.
# --------------------------------------------------------------------------------------------

def cell_ratio(row: dict) -> tuple:
    """``(ours/executorch, None)`` or ``(None, refusal)`` -- the ONLY door to a cell's number.

    There is deliberately no path here from ``ours_ns`` alone to a number. A cell whose ExecuTorch
    arm did not produce a warm slope has NO anchor, and a bare ``ours_ns`` from it is not comparable
    with a bare ``ours_ns`` from another session: that is precisely the mistake that produced a
    1.61x weight-transpose "result" across two sessions whose ExecuTorch arms were 13% apart. Such a
    cell refuses."""
    if not row:
        return None, "no cell recorded"
    if row.get("status") != "measured":
        return None, (row.get("refusal") or "cell produced no verdict")
    ours = row.get("ours_ns")
    et = row.get("executorch_warm_ns")
    if not et:
        return None, ("the ExecuTorch arm produced no warm slope, so this cell has no anchor; "
                      "a bare ours_ns is not a comparand across sessions and is refused"
                      + (f" (ours_ns={ours} was measured)" if ours else ""))
    if not ours:
        return None, "our arm produced no gated wall"
    return ours / et, None


def _digest_of(row: dict) -> str:
    return str(row.get("source_digest") or "")


def source_mismatch_reason(full_row: dict, drop_row: dict) -> str:
    """Why the two cells of a pair may not be differenced, or ``""``.

    The tree is shared and other sessions commit mid-run, so two cells of one pair can be built from
    different compiler sources. The instrument already stamps ``source_digest`` over the bytes it
    actually READ; this compares those. UNKNOWN is refused as firmly as a mismatch -- a digest that
    could not be taken cannot show the two arms agree."""
    a, b = _digest_of(full_row), _digest_of(drop_row)
    for label, d in (("full", a), (f"{drop_row.get('cell_id', 'drop')}", b)):
        if not d or d.startswith("UNKNOWN"):
            return (f"the {label} cell carries no usable source_digest ({d or 'missing'}), so it "
                    "cannot be shown to have been built from the same compiler as its pair")
    if a != b:
        return (f"the two cells were built from DIFFERENT compiler sources "
                f"(full={a[:16]}… vs {drop_row.get('cell_id', 'drop')}={b[:16]}…); the difference "
                "between them is not attributable to the lever")
    return ""


def contribution(full_row: dict, drop_row: dict) -> dict:
    """RATIO OF RATIOS. ``(ours/ET) with the lever`` vs ``(ours/ET) without it``.

    ExecuTorch is the common anchor: it absorbs whatever the board did between the two cells, so the
    quantity that survives drift is the ratio of the two ratios, never a difference of two walls.

    ``contribution`` is ``ratio_without / ratio_with - 1``: POSITIVE means removing the lever made
    us relatively slower, i.e. the lever helps by that fraction. A refused or unpairable cell yields
    ``contribution: None`` -- NEVER 0.0, which would read as "measured, no effect"."""
    out: dict = {"dropped": (drop_row or {}).get("dropped"),
                 "ratio_with": None, "ratio_without": None,
                 "contribution": None, "speedup_attributable": None,
                 "within_noise": None, "noise_band": NOISE_BAND,
                 "status": "refused", "reason": "",
                 "method": "ratio_of_ratios(ours/executorch)"}
    if not full_row or not drop_row:
        missing = "full" if not full_row else "leave-one-out"
        out["reason"] = (f"the {missing} cell of this pair has not been recorded; a contribution "
                         "needs both arms and is not inferred from one")
        out["status"] = "incomplete"
        return out
    r_with, why_with = cell_ratio(full_row)
    r_without, why_without = cell_ratio(drop_row)
    out["ratio_with"], out["ratio_without"] = r_with, r_without
    if why_with or why_without:
        parts = []
        if why_with:
            parts.append(f"full cell: {why_with}")
        if why_without:
            parts.append(f"{drop_row.get('cell_id', 'leave-one-out')} cell: {why_without}")
        out["reason"] = "; ".join(parts)
        return out
    why_src = source_mismatch_reason(full_row, drop_row)
    if why_src:
        out["status"] = "source_mismatch"
        out["reason"] = why_src
        return out
    frac = r_without / r_with - 1.0
    out["contribution"] = frac
    out["speedup_attributable"] = r_without / r_with
    out["within_noise"] = abs(frac) <= NOISE_BAND
    out["status"] = "within_noise" if out["within_noise"] else ("helps" if frac > 0 else "hurts")
    if out["within_noise"]:
        out["reason"] = (f"|{frac:+.4f}| is inside the K1 noise band of {NOISE_BAND:.3f}; this is "
                         "not a result, it is an absence of one")
    dirty = sorted(set(full_row.get("source_dirty") or []) | set(drop_row.get("source_dirty") or []))
    if dirty:
        out["source_dirty"] = dirty
    return out


def attribute(rows, models, features) -> dict:
    """Roll a ledger up into one attribution per (model, lever), plus the counts that make the
    result falsifiable: how many pairs were attributable at all."""
    by_key = {r.get("key"): r for r in rows if r.get("key")}
    cells = plan_cells(features)
    levers = [c.dropped for c in cells if c.dropped]
    per_model: dict = {}
    for model in models:
        full = by_key.get(cell_key(model, FULL_CELL))
        entries = {}
        for lever in levers:
            drop = by_key.get(cell_key(model, f"drop_{lever}"))
            entries[lever] = contribution(full, drop)
        per_model[model] = {
            "full_cell": {"status": (full or {}).get("status", "not_run"),
                          "ratio": cell_ratio(full)[0],
                          "refusal": (full or {}).get("refusal", "")},
            "levers": entries,
        }
    flat = [(m, lever, c) for m, d in per_model.items() for lever, c in d["levers"].items()]
    attributed = [c for _, _, c in flat if c["contribution"] is not None]
    return {
        "noise_band": NOISE_BAND,
        "method": ("ratio of ratios: (ours/ExecuTorch) with the lever vs without it, ExecuTorch as "
                   "the common anchor. A bare ours_ns delta is never reported."),
        "models": list(models), "levers": levers,
        "per_model": per_model,
        "counts": {
            "pairs": len(flat),
            "attributed": len(attributed),
            "outside_noise": sum(1 for c in attributed if not c["within_noise"]),
            "within_noise": sum(1 for c in attributed if c["within_noise"]),
            "refused": sum(1 for _, _, c in flat if c["status"] == "refused"),
            "source_mismatch": sum(1 for _, _, c in flat if c["status"] == "source_mismatch"),
            "incomplete": sum(1 for _, _, c in flat if c["status"] == "incomplete"),
        },
    }


def format_report(summary: dict) -> str:
    """The attribution table, with every refusal visible. A reader must be able to see how much of
    the feature set produced no attribution at all."""
    L = []
    c = summary["counts"]
    L.append(f"LEVER ABLATION -- leave-one-out, {summary['method']}")
    L.append(f"noise band: {summary['noise_band']:.3f} (K1)")
    L.append(f"pairs={c['pairs']}  attributed={c['attributed']}  "
             f"outside_noise={c['outside_noise']}  within_noise={c['within_noise']}  "
             f"refused={c['refused']}  source_mismatch={c['source_mismatch']}  "
             f"incomplete={c['incomplete']}")
    for model, d in summary["per_model"].items():
        f = d["full_cell"]
        rat = f"{f['ratio']:.4f}" if f["ratio"] is not None else "n/a"
        L.append("")
        L.append(f"--- {model}   full-set ours/ET = {rat}  [{f['status']}]")
        if f["refusal"]:
            L.append(f"    full cell refused: {f['refusal'][:300]}")
        L.append(f"    {'lever':<44} {'with':>9} {'without':>9} {'contrib':>10}  verdict")
        for lever, e in d["levers"].items():
            if e["contribution"] is None:
                L.append(f"    {lever:<44} {'-':>9} {'-':>9} {'-':>10}  {e['status'].upper()}")
                L.append(f"        {e['reason'][:220]}")
                continue
            L.append(f"    {lever:<44} {e['ratio_with']:>9.4f} {e['ratio_without']:>9.4f} "
                     f"{e['contribution']:>+9.2%}  {e['status']}")
            if e.get("source_dirty"):
                L.append(f"        NOTE: uncommitted sources in this pair: {e['source_dirty']}")
    return "\n".join(L)


# --------------------------------------------------------------------------------------------
# Running the cells.
# --------------------------------------------------------------------------------------------

def instrument_command(plan, cell: AblationCell, a, out_json: Path) -> list:
    """The exact argv for one cell. Same shape as the campaign's -- this driver adds only
    ``--features`` (always present, possibly empty) and the per-cell ``--out``."""
    return [sys.executable, str(_ROOT / INSTRUMENT),
            "--model", plan.model,
            "--model-dir", str(plan.ours_bundle_root),
            "--baseline", str(_ROOT / a.package),
            "--features", cell.feature_arg,
            "--n", str(a.n), "--warmup", str(a.warmup), "--iters", str(a.iters),
            "--et-n-lo", str(a.et_n_lo), "--et-n-hi", str(a.et_n_hi),
            "--compile-timeout-s", str(a.compile_timeout_s),
            "--out", str(out_json)]


def ledger_row(plan, cell: AblationCell, record: dict | None, *, refusal: str = "",
               command=None, elapsed_s: float | None = None) -> dict:
    """One ledger row. A row is either ``measured`` with both walls and the source digest, or
    ``refused`` with the string that says why -- there is no third shape and no partial number."""
    row = {"key": cell_key(plan.model, cell.cell_id),
           "model": plan.model, "cell_id": cell.cell_id, "dropped": cell.dropped,
           "features": list(cell.features), "feature_arg": cell.feature_arg,
           "ours_bundle_id": plan.ours_bundle_id,
           "recorded": utc_stamp(), "elapsed_s": elapsed_s,
           "command": list(command or []),
           "status": "refused", "refusal": refusal,
           "ours_ns": None, "executorch_warm_ns": None,
           "source_digest": None, "source_dirty": []}
    if record is None:
        return row
    row["source_digest"] = record.get("source_digest")
    row["source_dirty"] = list(record.get("source_dirty") or [])
    row["host"] = record.get("host")
    row["session_drift"] = record.get("session_drift")
    v = record.get("verdict_qd8") or {}
    if v.get("status") == "measured":
        row["status"] = "measured"
        row["ours_ns"] = v.get("ours_ns")
        row["executorch_warm_ns"] = v.get("executorch_warm_ns")
        row["ours_over_executorch"] = v.get("ours_over_executorch")
        row["accuracy"] = v.get("accuracy")
        row["rvv"] = (record.get("ours") or {}).get("rvv")
    else:
        row["refusal"] = refusal or v.get("reason") or "the instrument recorded no verdict"
    return row


def _dirty(paths) -> list:
    out = []
    for rel in paths:
        got = subprocess.run(["git", "status", "--porcelain", "--", rel],
                             cwd=str(repo_root()), capture_output=True, text=True)
        if got.stdout.strip():
            out.append(rel)
    return sorted(out)


def _write_manifest(outdir: Path, product) -> None:
    """Keep manifest.yaml current in BOTH the fresh and the resumed case: a product dir without one
    fails the layout gate for everyone on this shared tree. On a resume the identity fields are
    PRESERVED -- re-stamping them would re-date somebody's cited result."""
    from merlin.common.yaml import dump_yaml, load_yaml

    files = sorted(p.name for p in outdir.iterdir() if p.is_file() and p.name != "manifest.yaml")
    cells = sorted(f"cells/{p.name}" for p in (outdir / "cells").iterdir()) \
        if (outdir / "cells").is_dir() else []
    mf = outdir / "manifest.yaml"
    if mf.is_file():
        existing = load_yaml(mf) or {}
        if isinstance(existing, dict):
            existing["artifacts"] = files + cells
            mf.write_text(dump_yaml(existing), encoding="utf-8")
            return
    if product is not None:
        product._artifacts = files + cells
        product.write_manifest()


def write_summary(outdir: Path, ledger: Path, product, models, features) -> dict:
    """Rewrite attribution.json / report.txt / manifest.yaml from the ledger. Called after EVERY
    cell, so a session killed mid-ablation still leaves a readable, layout-valid product."""
    summary = attribute(ec.read_ledger(ledger), models, features)
    summary["generated"] = utc_stamp()
    summary["ledger"] = ledger.name
    try:
        summary["provenance"] = _prov.record(
            sources=[str(repo_root() / s) for s in _DRIVER_SOURCES])
    except Exception as e:  # noqa: BLE001  -- a provenance stamp must not break a session
        summary["provenance"] = {"error": f"{type(e).__name__}: {e}"}
    summary["source_dirty"] = _dirty(_DRIVER_SOURCES)
    (outdir / "attribution.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "report.txt").write_text(format_report(summary) + "\n", encoding="utf-8")
    _write_manifest(outdir, product)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True,
                    help="comma-separated levers to ablate; one leave-one-out cell per lever")
    ap.add_argument("--models", required=True, help="comma-separated models, cheapest-first")
    ap.add_argument("--variant", default="int8")
    ap.add_argument("--package", default=DEFAULT_PACKAGE, help="our codegen package (repo-relative)")
    ap.add_argument("--out-dir", default=None,
                    help="resume into an existing ablation product dir; omitted = create one")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every cell, its feature set and its exact command; touch no board "
                         "and write no artifact")
    ap.add_argument("--force", action="store_true", help="re-run cells already recorded")
    ap.add_argument("--retry-refused", action="store_true",
                    help="on a resume, re-run the cells whose recorded outcome was a refusal (a "
                         "board that went away mid-session refuses every remaining cell, and those "
                         "refusals are about the session, not the lever)")
    ap.add_argument("--no-board-preflight", action="store_true",
                    help="skip the board reachability check. Do not use casually: without it an "
                         "unreachable board records a refusal on EVERY cell and a resume then "
                         "skips them forever as settled outcomes")
    ap.add_argument("--board-usable-bytes", type=int, default=ec.DEFAULT_BOARD_USABLE_BYTES)
    ap.add_argument("--prefer-rewritten", action="store_true",
                    help="measure ours on a declared LAYOUT-ONLY derivative of the resolved bundle")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--et-n-lo", type=int, default=1)
    ap.add_argument("--et-n-hi", type=int, default=6)
    ap.add_argument("--cell-timeout", type=int, default=14400, help="seconds per cell")
    ap.add_argument("--compile-timeout-s", type=int, default=7000,
                    help="ceiling on any single build command inside a cell. The module default of "
                         "900s is a KERNEL budget and a whole-model int8 clang invocation exceeds "
                         "it; sized here so the ceiling is a stated parameter of the run.")
    a = ap.parse_args(argv)
    if a.compile_timeout_s >= a.cell_timeout:
        ap.error(f"--compile-timeout-s={a.compile_timeout_s} is not under "
                 f"--cell-timeout={a.cell_timeout}: a build allowed to outlive its own cell can "
                 "never be the thing that stops the cell, and the cell would time out with no "
                 "record instead of reporting which command ran long.")

    features = [f.strip() for f in a.features.split(",") if f.strip()]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if not features:
        ap.error("--features named no lever; there is nothing to leave out")
    if not models:
        ap.error("--models named no model")

    # BEFORE any board time. An unknown lever name reaches the instrument as a feature the compiler
    # never applies, and the cell then measures the SAME code as the full arm -- a zero contribution
    # that looks like a measured absence of effect. Fail closed here instead.
    sources, load_failures = feature_name_sources()
    bad = unknown_features(features)
    if bad:
        print(f"ERROR: unknown lever name(s): {bad}")
        for label, names in sorted(sources.items()):
            print(f"  consulted {label}: {len(names)} names")
        for f in load_failures:
            print(f"  WARNING: a name source failed to load, so the universe checked against may "
                  f"be narrower than the real one: {f}")
        near = sorted(n for n in known_feature_names()
                      for b in bad if b[:8] and b[:8] in n)[:8]
        if near:
            print(f"  did you mean: {near}")
        return 2
    for f in load_failures:
        print(f"WARNING: feature-name source failed to load: {f}")

    plans = ec.plan_campaign(models, variant=a.variant, int8=True,
                             budget_bytes=a.board_usable_bytes,
                             budget_source=("declared via --board-usable-bytes"
                                            if a.board_usable_bytes != ec.DEFAULT_BOARD_USABLE_BYTES
                                            else "declared default (et_campaign)"),
                             prefer_rewritten=a.prefer_rewritten)
    cells = plan_cells(features)

    if a.dry_run:
        print(f"[dry-run] {len(plans)} model(s) x {len(cells)} cells = "
              f"{len(plans) * len(cells)} cells; package={a.package}; NO board time will be spent")
        print(f"[dry-run] levers ({len(features)}): {features}")
        print(f"[dry-run] contribution = ratio of ratios (ours/ET with) vs (ours/ET without); "
              f"noise band {NOISE_BAND:.3f}\n")
        for p in plans:
            print(f"===== {p.model} ({p.variant})")
            print(f"    resolved bundle : {p.reference_bundle_root}")
            print(f"    ours bundle     : {p.ours_bundle_id}")
            if p.refusals:
                print("    WOULD REFUSE ALL CELLS OFFLINE (no board time):")
                for r in p.refusals:
                    print(f"      - {r}")
                print()
                continue
            for cell in cells:
                out_json = Path("<out-dir>") / "cells" / f"{p.model}__{cell.cell_id}.json"
                label = "FULL SET" if cell.dropped is None else f"DROP {cell.dropped}"
                print(f"  --- cell {cell.cell_id}  [{label}]")
                print(f"      features ({len(cell.features)}): "
                      f"{cell.feature_arg or '<empty: frozen baseline>'}")
                print("      $ " + " ".join(instrument_command(p, cell, a, out_json)))
            print()
        runnable = [p.model for p in plans if p.runnable]
        print(f"[dry-run] would run {len(runnable)}/{len(plans)} models: {runnable}")
        print(f"[dry-run] would refuse offline: {[p.model for p in plans if not p.runnable]}")
        print("[dry-run] pairs that could be attributed if every cell lands: "
              f"{len(runnable) * len(features)}")
        return 0

    # The board, ONCE, before any row is written. Without it an unreachable board writes a refusal
    # on every cell and a resume then skips them all forever -- an outage recorded as a property of
    # the levers.
    if not a.no_board_preflight:
        from merlin.mining import k1 as k1mod
        if not k1mod.available():
            print("ERROR: the board is not reachable (host="
                  f"{k1mod.K1_HOST!r}, toolchain={k1mod.toolchain_cc()}). Nothing recorded: a run "
                  "now would write a refusal on every cell and a later resume would skip them as "
                  "settled outcomes. Set MERLIN_K1_HOST and retry.")
            return 2

    if a.out_dir:
        outdir = Path(a.out_dir)
        if not outdir.is_dir():
            print(f"ERROR: --out-dir {outdir} does not exist; omit it to create a new ablation")
            return 2
        if not (outdir / "manifest.yaml").is_file():
            print(f"ERROR: --out-dir {outdir} carries no manifest.yaml, so it is not an ablation "
                  "product dir. Resuming into it would leave a directory that fails the "
                  "artifact-layout gate for everyone on the tree; omit --out-dir to create one.")
            return 2
        product = None
        print(f"[resume] {outdir}")
    else:
        product = new_product("lever-ablation", version=1,
                              sources=[str(repo_root() / s) for s in _DRIVER_SOURCES],
                              notes=f"leave-one-out lever ablation over {features} on {models}")
        outdir = product.path
        print(f"[ablation] {outdir}")
    (outdir / "cells").mkdir(parents=True, exist_ok=True)
    (outdir / "plan.json").write_text(json.dumps(
        {"models": models, "features": features, "cells": [c.as_dict() for c in cells],
         "package": a.package, "noise_band": NOISE_BAND,
         "protocol": {"n": a.n, "warmup": a.warmup, "iters": a.iters,
                      "et_n_lo": a.et_n_lo, "et_n_hi": a.et_n_hi,
                      "compile_timeout_s": a.compile_timeout_s,
                      "cell_timeout": a.cell_timeout}}, indent=2), encoding="utf-8")
    ledger = outdir / "ledger.jsonl"
    done = set() if a.force else recorded_keys(ec.read_ledger(ledger),
                                               retry_refused=a.retry_refused)
    if done:
        print(f"[resume] already recorded, skipping {len(done)} cell(s)")

    for plan in plans:
        for cell in cells:
            key = cell_key(plan.model, cell.cell_id)
            if key in done:
                continue
            print(f"\n===== {key} =====", flush=True)
            if not plan.runnable:
                reason = " | ".join(plan.refusals)
                print(f"REFUSED offline: {reason}", flush=True)
                ec.append_row(ledger, ledger_row(plan, cell, None, refusal=reason))
                write_summary(outdir, ledger, product, models, features)
                continue
            cell_json = outdir / "cells" / f"{plan.model}__{cell.cell_id}.json"
            log = outdir / "cells" / f"{plan.model}__{cell.cell_id}.log"
            cmd = instrument_command(plan, cell, a, cell_json)
            print(f"features: {cell.feature_arg or '<empty: frozen baseline>'}", flush=True)
            print("$ " + " ".join(cmd), flush=True)
            t0 = time.time()
            record, refusal = None, ""
            try:
                with log.open("w", encoding="utf-8") as lf:
                    got = subprocess.run(cmd, cwd=str(repo_root()), stdout=lf,
                                         stderr=subprocess.STDOUT, timeout=a.cell_timeout)
                rc = got.returncode
            except subprocess.TimeoutExpired:
                rc = None
                refusal = (f"the instrument did not finish within --cell-timeout={a.cell_timeout}s; "
                           f"no record was written. See {log.name}.")
            if not refusal:
                if cell_json.is_file():
                    try:
                        record = json.loads(cell_json.read_text(encoding="utf-8"))
                    except ValueError as e:
                        refusal = f"the instrument's record at {cell_json.name} is unreadable: {e}"
                else:
                    tail = ""
                    if log.is_file():
                        tail = "\n".join(log.read_text(encoding="utf-8",
                                                       errors="replace").splitlines()[-12:])
                    refusal = (f"the instrument exited {rc} without writing a record; nothing to "
                               f"read a ratio from. Tail of {log.name}:\n{tail}")
            row = ledger_row(plan, cell, record, refusal=refusal, command=cmd,
                             elapsed_s=round(time.time() - t0, 1))
            ec.append_row(ledger, row)
            if row["status"] == "measured":
                print(f"MEASURED {key}: ours={row['ours_ns'] / 1e6:.3f} ms  et_warm="
                      f"{row['executorch_warm_ns'] / 1e6:.3f} ms  ours/ET="
                      f"{row['ours_ns'] / row['executorch_warm_ns']:.4f}", flush=True)
            else:
                print(f"REFUSED {key}: {row['refusal'][:400]}", flush=True)
            write_summary(outdir, ledger, product, models, features)

    summary = write_summary(outdir, ledger, product, models, features)
    print("\n" + format_report(summary))
    print(f"\n[out] {outdir}")
    print(f"[resume] re-run with --out-dir {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
