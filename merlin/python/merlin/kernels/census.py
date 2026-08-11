"""One row per contraction: its shape, its element types, its measured cost, and whether a target's
compute units can legally take it.

Three facts about a model's contractions live in three places today and have never been joined.
:mod:`kernels.shapes` reports extents per op. The element types are decided by a PIPELINE PASS, not by
the capture. Measured cost lives in a board profile keyed on provenance
(:mod:`llvmlower.op_profile`). Answering "which contractions would a new compute unit actually be worth
routing" needs all three at once, and needs them ranked by measured time rather than by arithmetic —
an op can be 40% of a model's multiply-accumulates and 3% of its wall.

**Which IR stage is observed is load-bearing, not a detail.** An ``int8`` bundle's ``model.mlir`` is
f32: the W8A8 rewrite is a pass (``llvmlower.quant_passes.apply_quant``, reached through
``prepare_for_lowering``), so the capture carries quantized WEIGHTS and an f32 graph. Measured on
deepjscc, all 20 contractions read ``(f32, f32, f32)`` from the capture and ``(i8, i8, i32)`` from the
prepared module. A census run on the capture therefore concludes that an int8-only unit is legal for
nothing at all — a wrong answer produced by observing the wrong stage rather than by any bad
reasoning. :func:`census_bundle` prepares first, records which stage it read, and keeps the prepared
module beside its own output so the conclusion can be re-checked against the exact IR it came from.

**Legality is delegated, never invented.** The verdict comes from :mod:`targetgen.routing` against the
target's own contract. That is deliberately narrower than "this contraction runs on that unit": the
contract expresses op name and element types, and nothing in it expresses tile geometry, accumulator
depth, layout or capacity — ``capabilities.tile`` / ``mrf_depth`` are rendered for a human and read by
no legality check. So every verdict carries a :attr:`Legality.scope` naming exactly which axes it
covers, and a "legal" row may never be read as "fits". A row whose element types cannot be resolved,
or whose target declares no compute units, is ``unknown`` with the reason — not ``illegal``, and not
dropped.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..common import mlir_query as mq
from .microkernel import ContractionShape

__all__ = ["Census", "CensusRow", "Legality", "census", "census_bundle", "legality_of",
           "to_markdown"]

#: Verdicts. ``unknown`` is a first-class outcome: it means the contract could not decide, which is a
#: different statement from "the hardware refuses" and must not collapse into it.
LEGAL, ILLEGAL, UNKNOWN = "legal", "illegal", "unknown"

#: What a routing verdict actually covers, spelled out so it cannot be over-read. Kept as a constant
#: because it is a claim about the capability contract's expressiveness, and it changes only when the
#: contract gains an axis.
ROUTING_SCOPE = "op_name+element_types"


@dataclass(frozen=True)
class Legality:
    """Whether one contraction is routable, and how much that verdict is worth.

    ``scope`` names the axes the verdict covers; anything outside it is undecided regardless of
    ``verdict``. ``reason`` is mandatory unless the verdict is :data:`LEGAL`.
    """

    verdict: str
    scope: str
    unit: str | None = None
    acc: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in (LEGAL, ILLEGAL, UNKNOWN):
            raise ValueError(f"unknown verdict {self.verdict!r}")
        if self.verdict != LEGAL and not self.reason:
            raise ValueError(f"a {self.verdict!r} verdict must carry a reason")


@dataclass(frozen=True)
class CensusRow:
    """One contraction, joined across every axis that was observable."""

    index: int
    key: str                      # provenance join key (see llvmlower.op_profile.join_key)
    mlir_op: str                  # the op as written (linalg.generic / linalg.matmul)
    op_class: str                  # the class it specializes to, i.e. what a schedule matches
    family: str                    # semantic family from prov.op/prov.family, for grouping
    role: str = ""                 # prov.role when a rewrite split one captured op into pieces
    parallel: tuple[int, ...] = ()
    reduction: tuple[int, ...] = ()
    dtypes: tuple[str, ...] = ()   # (lhs, rhs, out) MLIR tokens, () when unobserved
    work: int = 0
    work_complete: bool = True     # False => `work` is a lower bound (see kernels.work)
    bytes: int = 0
    ticks: int | None = None
    ticks_ops: int = 0             # profiled ops sharing this key; >1 => `ticks` is an upper bound
    pct_model: float | None = None
    legality: Legality | None = None


@dataclass(frozen=True)
class Census:
    """Every contraction in one model, plus what the census could and could not decide."""

    model: str
    stage: str                     # which IR the rows were read from
    source: str                    # path of that IR
    target: str | None = None
    rows: tuple[CensusRow, ...] = ()
    total_work: int = 0
    model_ticks: int | None = None   # whole-model ticks, i.e. the pct_model denominator
    ranked_by: str = "work"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def by_verdict(self, verdict: str) -> tuple[CensusRow, ...]:
        return tuple(r for r in self.rows if r.legality is not None
                     and r.legality.verdict == verdict)

    def measured_share(self, rows: "Sequence[CensusRow] | None" = None) -> float | None:
        """Share of measured whole-model time covered by ``rows``, counting each tick bucket ONCE.

        Per-row ``pct_model`` values must not be summed. Several contractions can join the same
        provenance bucket — an attention layer's two contractions carry one fqn — so adding their
        percentages counts that bucket twice: measured on whisper, summing the 91 rows gives 106.23% of a
        model that is by definition 100%. Deduplicating by ``(key, role)`` gives 96.01%.

        The result is an UPPER BOUND on the contractions themselves, because a bucket that covers a
        contraction also covers whatever else shares its key — the quantize prologue and requant
        epilogue the int8 rewrite attributes to the same layer.
        """
        if self.model_ticks in (None, 0):
            return None
        pool = self.rows if rows is None else rows
        buckets = {(r.key, r.role): (r.ticks or 0) for r in pool}
        return sum(buckets.values()) / float(self.model_ticks)

    @property
    def distinct_tick_buckets(self) -> int:
        return len({(r.key, r.role) for r in self.rows if r.ticks is not None})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# legality
# ---------------------------------------------------------------------------------------------


def _demand_op(op_class: str) -> str:
    """The op name a capability contract's ``ops`` list is written in.

    Derived from the STRUCTURAL class (``linalg.batch_matmul`` -> ``batch_matmul``) rather than from
    the capture's semantic ``prov.op`` label, because a compute unit's declared vocabulary is about
    what it computes, not about which model layer asked. A unit that declares only ``matmul`` does not
    thereby declare ``batch_matmul``: that gaps, and the reason says which name was demanded, which is
    the fail-closed reading."""
    return op_class.split(".", 1)[1] if "." in op_class else op_class


def _as_format(token: str) -> tuple[str | None, str]:
    """``(quant-format name, reason)`` for an MLIR element token, resolved through the format registry.

    An accumulator token (``i32``) is deliberately NOT resolvable here — the registry describes storage
    formats, and a contract's ``acc`` field is a raw MLIR token. So this is only ever asked of the two
    input operands."""
    from ..common import quant_formats as qf
    try:
        return qf.get(token).name, ""
    except KeyError:
        return None, f"element type {token!r} is not a registered quantization format"


def legality_of(shape: ContractionShape, units: Sequence[Any], *, site: str = "",
                undecidable: str = "") -> Legality:
    """Route one contraction against ``units`` and report the verdict with its scope.

    ``units`` are :class:`targetgen.compute_units.ComputeUnit` values from the target's contract. An
    empty ``units`` is :data:`UNKNOWN`, not :data:`ILLEGAL` — a contract that declares no compute units
    has said nothing about the hardware, and reporting that as a refusal would credit the census with a
    measurement it never made. ``undecidable`` carries WHY there are no units (contract missing, no
    target given, unparseable), so the row states the real cause rather than a generic one.
    """
    from ..targetgen import routing

    if not units:
        return Legality(UNKNOWN, ROUTING_SCOPE,
                        reason=undecidable or "no compute units to route against")
    if len(shape.dtypes) < 3:
        return Legality(UNKNOWN, ROUTING_SCOPE,
                        reason="operand element types not observed on this op")
    lhs, rhs = shape.dtypes[0], shape.dtypes[1]
    in_fmt, why_in = _as_format(lhs)
    weight_fmt, why_w = _as_format(rhs)
    if in_fmt is None or weight_fmt is None:
        return Legality(UNKNOWN, ROUTING_SCOPE, reason=why_in or why_w)

    demand = routing.OpDemand(op=_demand_op(shape.op), in_fmt=in_fmt, weight_fmt=weight_fmt,
                              site=site)
    result = routing.route([demand], list(units))[0]
    if result.unit is None:
        return Legality(ILLEGAL, ROUTING_SCOPE, reason=result.gap or "unroutable")
    return Legality(LEGAL, ROUTING_SCOPE, unit=result.unit, acc=result.acc)


def _units_for(target: str | None) -> tuple[list[Any], str]:
    """``(compute units, why-undecidable)`` for a target, degrading honestly when unresolvable.

    The second element is empty exactly when the units can decide something. It is carried into every
    row's reason so an unknown verdict names its own cause — "contract not found at <path>" and
    "contract declares no compute_units" are different failures with different fixes, and collapsing
    them into one message loses the only actionable part.
    """
    if not target:
        return [], "no target given: legality was not evaluated"
    from ..targetgen import compute_units as cu
    from ..targetgen import target_registry as tr
    try:
        contract = tr.load_contract(target)
    except Exception as exc:                                          # noqa: BLE001
        return [], f"target {target!r} contract unavailable ({type(exc).__name__}: {exc})"
    try:
        units = cu.compute_units(contract)
    except Exception as exc:                                          # noqa: BLE001
        return [], f"target {target!r} compute_units unparseable ({type(exc).__name__}: {exc})"
    if not units:
        return [], f"target {target!r} contract declares no compute_units"
    return units, ""


def _unit_vocabulary(units: Sequence[Any]) -> str:
    """What the units actually declare — so a row gapped on a name mismatch is diagnosable."""
    vocab = [f"{u.name}(kind={u.kind}, ops=[{','.join(u.ops) or '<any>'}], "
             f"dtypes=[{','.join(u.dtypes) or '<none declared>'}])" for u in units]
    return f"routed against {len(units)} compute unit(s): " + "; ".join(vocab)


# ---------------------------------------------------------------------------------------------
# the tick join
# ---------------------------------------------------------------------------------------------


def _ticks_by_key(prof_table: Sequence[Mapping[str, Any]] | None,
                  prof_ticks: Mapping[int, tuple[int, int]] | None
                  ) -> tuple[dict[str, tuple[int, int]], int | None]:
    """``({join key: (ticks, n_ops)}, whole-model ticks)`` from an op table + a board profile.

    Several profiled ops can share one join key (a whole layer stamped with a single ``prov.fqn``), so
    the ticks are SUMMED per key and the contributing op count is carried alongside. A row whose key
    covers more than one op therefore states an upper bound on that contraction rather than silently
    attributing a shared bucket to it — splitting the bucket would be an invention.

    The denominator is every profiled op, not only the contractions, so ``pct_model`` is a share of the
    whole model and the shares of all contractions can legitimately sum to well under 1.
    """
    if not prof_table or not prof_ticks:
        return {}, None
    from ..llvmlower.op_profile import join_key
    out: dict[str, tuple[int, int]] = {}
    total = 0
    for rec in prof_table:
        rec = dict(rec)
        got = prof_ticks.get(int(rec.get("id", -1)))
        if got is None:
            continue
        ticks = int(got[0])
        total += ticks
        # A rewrite that split one captured op into several stamps each piece with `prov.role`
        # (llvmlower.passes_quant_int._carry_prov). Keying on fqn+role keeps a contraction's own cost
        # separate from its requant epilogue's, which share an fqn by construction. Keying on fqn alone
        # would sum them and make either one only an upper bound.
        key = join_key(rec)
        role = rec.get("role")
        prev = out.get(key, (0, 0))
        out[key] = (prev[0] + ticks, prev[1] + 1)
        if role:
            keyed = f"{key}#{role}"
            prev_r = out.get(keyed, (0, 0))
            out[keyed] = (prev_r[0] + ticks, prev_r[1] + 1)
    return out, total


# ---------------------------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------------------------


def census(src: "str | Path | Any", *, model: str, stage: str = "unspecified",
           target: str | None = None,
           prof_table: Sequence[Mapping[str, Any]] | None = None,
           prof_ticks: Mapping[int, tuple[int, int]] | None = None) -> Census:
    """Census the contractions of one module.

    ``src`` is whatever :func:`kernels.shapes.observe_contractions` accepts. ``stage`` is a label for
    WHICH IR this is (``capture`` / ``prepared``); it is recorded rather than inferred, because the
    element types — and therefore every legality verdict — depend on it.
    """
    from . import work as wk
    from .shapes import observe_contractions

    observed = observe_contractions(src)
    units, undecidable = _units_for(target)
    notes: tuple[str, ...] = (undecidable + "; legality is unknown for every row",) if undecidable \
        else (_unit_vocabulary(units),)
    by_key, model_ticks = _ticks_by_key(prof_table, prof_ticks)
    if prof_table and not prof_ticks:
        notes += ("an op table was supplied without board ticks: cost columns are unmeasured",)

    rows: list[CensusRow] = []
    for i, (op, shape) in enumerate(observed):
        prov = mq.provenance(op)
        mlir_op = mq.op_name(op)
        key = (prov.get("prov.fqn") or prov.get("prov.region_id") or mlir_op)
        family = prov.get("prov.op") or prov.get("prov.family") or mlir_op.split(".", 1)[-1]
        w, complete = wk.work_of(op)
        # Prefer the role-qualified bucket (this contraction alone) over the fqn bucket (the whole
        # layer the rewrite split it out of), so a joined tick count is the op's own where it can be.
        role = prov.get("prov.role")
        ticks, n_ops = (by_key.get(f"{key}#{role}") if role else None) or by_key.get(key, (None, 0))
        rows.append(CensusRow(
            index=i, key=key, mlir_op=mlir_op, op_class=shape.op, family=family, role=role or "",
            parallel=tuple(shape.parallel), reduction=tuple(shape.reduction),
            dtypes=tuple(shape.dtypes), work=int(w), work_complete=bool(complete),
            bytes=wk.footprint_bytes(op), ticks=ticks, ticks_ops=n_ops,
            pct_model=(ticks / model_ticks) if (ticks is not None and model_ticks) else None,
            legality=legality_of(shape, units, site=key, undecidable=undecidable),
        ))

    # Rank by MEASURED cost when there is any, by arithmetic otherwise — and say which, because the
    # two orders disagree (a memory-bound contraction is heavy on wall and light on work).
    ranked_by = "ticks" if any(r.ticks for r in rows) else "work"
    rows.sort(key=lambda r: (-(r.ticks or 0), -r.work) if ranked_by == "ticks" else -r.work)
    if ranked_by == "work":
        notes += ("no board profile joined: ranked by arithmetic, which is NOT a cost measurement",)
    shared = sum(1 for r in rows if r.ticks_ops > 1)
    if shared:
        notes += (f"{shared} row(s) share a provenance key with another profiled op: their ticks are "
                  "an upper bound on the contraction alone",)
    partial = sum(1 for r in rows if not r.work_complete)
    if partial:
        notes += (f"{partial} row(s) have a partially recovered iteration space: work is a lower bound",)

    return Census(model=model, stage=stage, source=str(src) if isinstance(src, (str, Path)) else "",
                  target=target, rows=tuple(rows),
                  total_work=sum(r.work for r in rows), model_ticks=model_ticks,
                  ranked_by=ranked_by, notes=notes)


def census_bundle(bundle: "str | Path", *, model: str = "", target: str | None = None,
                  int8_compute: bool = False, work_dir: "str | Path | None" = None,
                  prof_table: Sequence[Mapping[str, Any]] | None = None,
                  prof_ticks: Mapping[int, tuple[int, int]] | None = None) -> Census:
    """Census a recapture bundle through the SAME preparation the compiler applies.

    Reuses ``runtime.backends.zephyr_model.prepare_for_lowering`` (with ``blocking=False``, so no
    per-op tagging is done and nothing is decided here) rather than reimplementing the quant rewrite:
    a census that prepared the module its own way would be reporting element types no build produces.
    Falls back to the raw capture, with a note, when preparation fails — an unpreparable bundle is
    still worth counting, as long as the stage it was counted at is stated.
    """
    bundle = Path(bundle)
    src = bundle / "model.mlir" if bundle.is_dir() else bundle
    model = model or (bundle.name if bundle.is_dir() else bundle.stem)
    if not src.is_file():
        return Census(model=model, stage="missing", source=str(src), target=target,
                      notes=(f"no model.mlir at {src}",))

    stage, observed, extra = "capture", src, ()
    if work_dir is not None:
        from ..runtime.backends.zephyr_model import prepare_for_lowering
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        try:
            prepared, _ = prepare_for_lowering(src, work, int8_compute=int8_compute, blocking=False)
            stage, observed = "prepared", prepared
        except Exception as exc:                                      # noqa: BLE001
            extra = (f"preparation failed ({type(exc).__name__}: {exc}); read the raw capture, so "
                     "element types are the CAPTURE's and not the ones a build would compile",)
    else:
        extra = ("no work dir given: read the raw capture, whose element types precede the quant "
                 "rewrite",)

    got = census(observed, model=model, stage=stage, target=target,
                 prof_table=prof_table, prof_ticks=prof_ticks)
    return Census(model=got.model, stage=got.stage, source=str(observed), target=got.target,
                  rows=got.rows, total_work=got.total_work, model_ticks=got.model_ticks,
                  ranked_by=got.ranked_by, notes=got.notes + extra)


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------


def _extents(row: CensusRow) -> str:
    par = "x".join(str(d) for d in row.parallel) or "-"
    red = "x".join(str(d) for d in row.reduction) or "-"
    return f"{par} / {red}"


def to_markdown(censuses: Sequence[Census], *, top: int = 20) -> str:
    """A summary a reader can audit: per model, the ranked rows and every note."""
    out: list[str] = ["# Contraction census", ""]
    for c in censuses:
        verdicts = {v: len(c.by_verdict(v)) for v in (LEGAL, ILLEGAL, UNKNOWN)}
        out += [f"## {c.model}", "",
                f"- IR stage read: `{c.stage}` (`{c.source}`)",
                f"- target: `{c.target or 'none'}`",
                f"- contractions: {len(c.rows)}; ranked by **{c.ranked_by}**",
                f"- legality ({ROUTING_SCOPE}): "
                f"{verdicts[LEGAL]} legal, {verdicts[ILLEGAL]} illegal, {verdicts[UNKNOWN]} unknown",
                f"- whole-model ticks: {c.model_ticks if c.model_ticks is not None else 'not measured'}",
                ""]
        overall = c.measured_share()
        if overall is not None:
            out += [f"- measured share of whole-model time, each tick bucket counted once "
                    f"({c.distinct_tick_buckets} buckets for {len(c.rows)} rows) — an **upper bound**, "
                    f"since a bucket also covers whatever shares its provenance key:", ""]
            for label, verdict in (("all contractions", None), ("legal", LEGAL), ("illegal", ILLEGAL),
                                   ("unknown", UNKNOWN)):
                rows = c.rows if verdict is None else c.by_verdict(verdict)
                if rows:
                    out.append(f"  - {label}: {c.measured_share(rows) * 100:.2f}%")
            out += ["", "The per-row `% model` column below must NOT be summed: rows sharing a bucket "
                    "would be counted more than once.", ""]
        if c.notes:
            out += ["Notes:", ""] + [f"- {n}" for n in c.notes] + [""]
        if not c.rows:
            out += ["_no contractions observed_", ""]
            continue
        out += ["| # | family | op class | parallel / reduction | dtypes | work | ticks | % model | verdict | reason |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in c.rows[:top]:
            lg = r.legality
            pct = f"{r.pct_model * 100:.2f}%" if r.pct_model is not None else "-"
            out.append(
                f"| {r.index} | {r.family}{'/' + r.role if r.role else ''} | `{r.op_class}` | {_extents(r)} | "
                f"{','.join(r.dtypes) or '-'} | {r.work:,} | {r.ticks if r.ticks is not None else '-'} | "
                f"{pct} | {lg.verdict if lg else '-'} | {(lg.reason if lg else '')[:80]} |")
        if len(c.rows) > top:
            out.append(f"| … | _{len(c.rows) - top} more rows omitted from this table_ | | | | | | | | |")
        out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def load_profile(path: "str | Path") -> tuple[list[dict[str, Any]], dict[int, tuple[int, int]]]:
    """``(op_table, {id: (ticks, hits)})`` from a board per-op profile summary.

    The summary's ``op_table`` already carries the measured ticks merged onto each op record, so the two
    halves the census wants are one file. A record missing ``ticks`` is skipped rather than counted as
    zero — an op the board never reported is unmeasured, and calling that "free" would understate the
    model and inflate every other row's share of it.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    table = list(doc.get("op_table") or [])
    ticks: dict[int, tuple[int, int]] = {}
    for rec in table:
        if rec.get("ticks") is None:
            continue
        ticks[int(rec["id"])] = (int(rec["ticks"]), int(rec.get("hits") or 1))
    return table, ticks


def _profile_for(spec: str, bundle_name: str) -> "Path | None":
    """The profile file for one model: ``spec`` itself when it is a file, else ``<spec>/<name>*.json``."""
    if not spec:
        return None
    p = Path(spec)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted((p / bundle_name).glob("*.json")) or sorted(p.glob(f"{bundle_name}*.json"))
        return cands[0] if cands else None
    return None


def _resolve_bundle(name: str, dtype: str) -> tuple[Path | None, str]:
    """The recapture dir for a model name, trying the name verbatim before the dtype conventions."""
    from ..common.artifacts import recaptures_dir
    root = Path(recaptures_dir())
    for cand in (name, f"{name}_{dtype}_full", f"{name}_{dtype}_consistent"):
        p = root / cand
        if (p / "model.mlir").is_file():
            return p, cand
    return None, ""


def main(argv: "Sequence[str] | None" = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", required=True, help="comma-separated model or bundle names")
    ap.add_argument("--dtype", default="int8", help="bundle dtype convention and compute datapath")
    ap.add_argument("--target", default="", help="target whose contract decides legality")
    ap.add_argument("--top", type=int, default=20, help="rows per model in the markdown table")
    ap.add_argument("--no-prepare", action="store_true",
                    help="read the raw capture (element types will PRECEDE the quant rewrite)")
    ap.add_argument("--profile", default="",
                    help="a k1_op_profile.py summary JSON (or a dir of them named <model>.json) whose "
                         "op_table supplies measured ticks. Without it the ranking is by arithmetic, "
                         "which the output labels as NOT a cost measurement.")
    a = ap.parse_args(argv)

    from ..common.artifacts import new_product

    prod = new_product("target-evolution", version=1, target=a.target or None,
                       notes=f"contraction census, dtype={a.dtype}")
    censuses: list[Census] = []
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        bundle, resolved = _resolve_bundle(name, a.dtype)
        if bundle is None:
            print(f"skip {name}: no bundle with a model.mlir under recaptures/")
            censuses.append(Census(model=name, stage="missing", source="",
                                   target=a.target or None,
                                   notes=("bundle not found",)))
            continue
        work = None if a.no_prepare else Path(prod.path) / "prepared" / resolved
        prof_path = _profile_for(a.profile, resolved)
        table, ticks = load_profile(prof_path) if prof_path else (None, None)
        got = census_bundle(bundle, model=resolved, target=a.target or None,
                            int8_compute=(a.dtype == "int8"), work_dir=work,
                            prof_table=table, prof_ticks=ticks)
        if a.profile and prof_path is None:
            print(f"  no profile found for {resolved} under {a.profile}")
        censuses.append(got)
        n_legal = len(got.by_verdict(LEGAL))
        print(f"{resolved}: {len(got.rows)} contractions at stage={got.stage}, "
              f"{n_legal} legal ({ROUTING_SCOPE}), ranked by {got.ranked_by}")

    payload = {"dtype": a.dtype, "target": a.target or None, "legality_scope": ROUTING_SCOPE,
               "models": [c.to_dict() for c in censuses]}
    prod.add_artifact("workload_census.json").write_text(json.dumps(payload, indent=1) + "\n",
                                                        encoding="utf-8")
    prod.add_artifact("workload_census.md").write_text(to_markdown(censuses, top=a.top),
                                                       encoding="utf-8")
    prod.write_manifest()
    print(f"\nwrote {prod.path}")
    return 0


if __name__ == "__main__":                                            # pragma: no cover
    raise SystemExit(main())
