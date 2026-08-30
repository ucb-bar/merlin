#!/usr/bin/env python3
"""R7 -- the headline experiment, reported as TWO claims that are never combined.

Off the measured capsule corpus **no reference implementation exists**. Nobody ships a hand-written
kernel for a shape merlin invented, so "fraction of reference" has no denominator there. That single
fact splits the headline in half, and the split is the point of this driver:

* **RECOVERS (7.1)** -- on the eligible capsules, where a shipped reference program was measured:
  what fraction of that reference's measured runtime does the derived envelope account for? The
  denominator is a *reference implementation's* cycles, so the number says something about the
  hardware model standing next to code somebody else wrote.
* **PREDICTS (7.2)** -- at shapes merlin emits itself: predicted cycles against measured cycles for
  **merlin's own** kernel. There is no reference, so this says nothing about how good the emitted
  code is. A predictor can be perfect on a kernel that is ten times slower than it should be.

:func:`report` therefore returns them under separate keys with separate ``n``, and
:func:`_written_result` states in words what the evidence does not support. Conflating the two would
report a prediction result as a recovery result, which is the specific error this task exists to
avoid.

**No "% of peak" anywhere.** The target's ``speed_of_light`` is null and the one candidate model is
GEMM-only, so attainment is UNKNOWN and every claim here is kernel-relative. This driver never
divides by a nameplate rate.

Two envelope variants are reported for the generated set, both derived and neither fitted:

``corpus_only``
    Exactly the terms the corpus calibrated -- a measured movement-rate ceiling, the systolic law
    ``fill + the delays the program schedules``, and the measurement source's reset intercept --
    transferred to the generated shape unchanged. Nothing new is learned from the generated program.
``plus_program_schedule``
    The same, plus the stall cycles the emitted program *itself* schedules and which no unit law
    already claims. Every input is read off the plan (its tile counts and its measured settle
    contract); there are no free parameters. This is the same move the systolic law already makes --
    "the delays the program schedules" -- applied to the delays that belong to no unit.

Usage::

    headline.py measure --target T --set size --tier vsim      # run the generated shapes, serially
    headline.py report  --target T                             # both claims + the written result
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.common import artifacts as A                                          # noqa: E402
from merlin.common.paths import env                                               # noqa: E402
from merlin.perf.attribution import buckets_from_kinds                            # noqa: E402
from merlin.perf.composer import (                                                # noqa: E402
    compose_corpus,
    peaks_from_observations,
    structural_unit_time,
)
from merlin.perf.decompose import UNKNOWN, ResourceKind, activity_from_busy, is_unknown  # noqa: E402
from merlin.perf.envelope import (                                                # noqa: E402
    Basis,
    ResourceDemand,
    ResourceTime,
    compose,
    resource_time,
)
from merlin.perf.headroom import composition_operator                             # noqa: E402
from merlin.perf.record import (                                                  # noqa: E402
    compose_unit_busy,
    derive_delay_mnemonic,
    derive_unit_roles,
    fill_cycles,
    read_digest_triple,
)

# The activity buckets the measurement source decomposes a run into, and what kind of resource each
# one is. A target fact supplied at the experiment edge, exactly as the composer requires.
BUCKET_KINDS = {
    "dma": ResourceKind.MOVEMENT,
    "mxu": ResourceKind.COMPUTE,
    "vpu": ResourceKind.COMPUTE,
    "none": ResourceKind.FIXED,
}
#: Which RTL module implements which activity bucket, for the timing walk.
RESOURCE_MODULES = {"dma": "DmaEngine", "mxu": "SystolicArray", "vpu": "VectorEngine"}
#: The measurement source's own field name for each bucket. Its vocabulary, not ours.
BUCKET_FIELDS = {"dma": "dma_busy", "mxu": "mxu", "vpu": "vpu", "none": "none"}
#: The op-stream family whose unit carries a structural law, and the metadata key holding its
#: dimension. The law itself lives in :mod:`merlin.perf.record`.
COMPUTE_FAMILY = "MatrixSystolic"
COMPUTE_BUCKET = "mxu"
MOVEMENT_BUCKET = "dma"
VECTOR_BUCKET = "vpu"
VECTOR_FAMILY = "Vector"
FIXED_BUCKET = "none"

#: The declared provenance a result about this substrate must carry.
PIN_NAMES = ("atlas_npu", "atlas_fp_units", "atlas_npu_model")
ARTIFACT_NAMES = ("atlas_arc_cycle_suite",)

#: Generated shapes. Two families, chosen for what each can support:
#:
#: ``size``  -- one contraction per size, spanning three orders of magnitude, every one a single
#:              K-tile so the systolic law stays inside its validated regime.
#: ``iso``   -- a genuine CHOICE SET: seven shapes doing IDENTICAL arithmetic (4,194,304 MACs) in an
#:              identical number of tile passes (128), differing only in how the contraction is
#:              decomposed. Ranking these is a real question with a real answer, which is what
#:              top-K recall and regret need and what a set of differently-sized shapes cannot give.
SHAPE_SETS = {
    "size": ("32x32x32", "64x32x64", "128x32x128", "256x32x256", "384x32x384", "512x32x512",
             "640x32x640", "704x32x704"),
    "iso": ("256x32x512", "256x64x256", "128x128x256", "128x256x128", "128x512x64", "64x1024x64",
            "32x4096x32"),
}
ISO_MACS = 4194304

MEASURE_CACHE = "perf_headline"


# ==================================================================================================
# the measured corpus -- one shipped reference implementation per kernel
# ==================================================================================================
def load_suite() -> dict:
    root = env("MERLIN_MLC_DIR")
    if not root:
        raise SystemExit("MERLIN_MLC_DIR unset: the measured cycle suite lives in the model checkout")
    path = suite_path()
    if not path.is_file():
        raise SystemExit(f"measured cycle suite not present at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def suite_path() -> Path:
    return Path(str(env("MERLIN_MLC_DIR"))) / "mlc" / "validate" / "npu_model_suite.json"


def corpus_sources(suite: dict) -> list:
    """One activity source per measured kernel, from the shipped reference program's own run."""
    out = []
    for name, body in suite["kernels"].items():
        arc = body["arc"]
        out.append(activity_from_busy(
            name, arc["truth"],
            {b: arc[BUCKET_FIELDS[b]] for b in BUCKET_KINDS},
            BUCKET_KINDS, partitioned=True, completion_observable=True,
            provenance="per-cycle activity decomposition from the cycle-accurate model"))
    return out


def corpus_law(suite: dict):
    """``(delay_mnemonic, roles, fill)`` -- the systolic law, derived from the corpus itself."""
    streams = [k["op_stream"] for k in suite["kernels"].values() if k.get("op_stream")]
    delay = derive_delay_mnemonic(streams)
    roles = derive_unit_roles(streams, COMPUTE_FAMILY, delay)
    return delay, roles, fill_cycles("systolic_2d", suite["_meta"]["mxu_dim"])


def corpus_demands(suite: dict) -> dict:
    """The two resources with a demand/peak form, per kernel."""
    out: dict[str, dict[str, ResourceDemand]] = {}
    for name, body in suite["kernels"].items():
        arc = body["arc"]
        out[name] = {
            MOVEMENT_BUCKET: ResourceDemand(
                MOVEMENT_BUCKET, ResourceKind.MOVEMENT, arc["reads"] + arc["writes"], "beats",
                basis=Basis.MOVED, provenance="measured read/write beats x the port width"),
            VECTOR_BUCKET: ResourceDemand(
                VECTOR_BUCKET, ResourceKind.COMPUTE,
                sum(1 for fam, _m, _i in body["op_stream"] if fam == VECTOR_FAMILY), "ops",
                basis=Basis.MOVED, provenance="program op stream"),
        }
    return out


def corpus_peaks(suite: dict, sources) -> dict:
    return peaks_from_observations(
        corpus_demands(suite), sources, units={MOVEMENT_BUCKET: "beats", VECTOR_BUCKET: "ops"},
        provenance="per-cycle activity decomposition")


def corpus_times(suite: dict, sources) -> dict:
    """Per-resource times for every measured kernel: the landed R4 configuration, unchanged."""
    delay, roles, fill = corpus_law(suite)
    demands = corpus_demands(suite)
    peaks = corpus_peaks(suite, sources)
    out: dict[str, list[ResourceTime]] = {}
    for name, body in suite["kernels"].items():
        out[name] = [
            resource_time(demands[name][MOVEMENT_BUCKET], peaks[MOVEMENT_BUCKET]),
            structural_unit_time(COMPUTE_BUCKET, ResourceKind.COMPUTE,
                                 compose_unit_busy(body["op_stream"], roles, fill, delay),
                                 provenance=f"fill={fill} plus the program's scheduled delays"),
            resource_time(demands[name][VECTOR_BUCKET], peaks[VECTOR_BUCKET]),
            ResourceTime(resource=FIXED_BUCKET, kind=ResourceKind.FIXED,
                         cycles=float(suite["_meta"]["reset_cycles"]), unit="cycles",
                         basis=Basis.MOVED, evidence_kind="measured",
                         provenance="reset_cycles declared by the measurement source"),
        ]
    return out


def derived_operator(sources):
    """The composition operator, DERIVED. Never defaulted, and it refuses from the buckets alone."""
    got = composition_operator(list(sources),
                               observed_overlap_cycles={s.workload: 0 for s in sources})
    if not isinstance(got, tuple):
        raise SystemExit(f"the composition operator could not be derived: {got}")
    return got


def timing_records(target: str):
    """``facts['timing']`` for the target, or None -- UNCACHED is not a design with no logic."""
    try:
        from merlin.targetgen.rtl import facts as F
        return (F.load_facts(target).get("facts") or {}).get("timing")
    except Exception:                                       # noqa: BLE001 - absent is recorded, not raised
        return None


def corpus_prediction(target: str, suite: dict):
    sources = corpus_sources(suite)
    op, eta = derived_operator(sources)
    records = timing_records(target)
    from merlin.perf.composer import fixed_terms_from_timing
    structural, _refused = fixed_terms_from_timing(records, RESOURCE_MODULES)
    return compose_corpus(
        sources, times=corpus_times(suite, sources), operator=op, eta=eta,
        buckets=buckets_from_kinds(BUCKET_KINDS, fixed_bucket="control"),
        timing_records=records, structural_resources=list(structural)), (op, eta)


# ==================================================================================================
# CLAIM 7.1 -- RECOVERS, where a shipped reference exists
# ==================================================================================================
def claim_recovers(target: str, suite: dict) -> dict:
    """Fraction of a shipped reference implementation's measured runtime the envelope accounts for.

    Reported over what actually RESOLVES. The partial bounds are reported beside it under their own
    name and never summed into the headline: a bound over a subset of the resources answers a
    different question from a bound over all of them, and averaging the two would hide which is
    which.
    """
    pred, (op, eta) = corpus_prediction(target, suite)
    resolved = pred.resolved
    rows = []
    for name, p in sorted(pred.predictions.items()):
        rows.append({
            "kernel": name,
            "measured_cycles": p.measured_cycles,
            "resolved": not is_unknown(p.predicted_cycles),
            "predicted_cycles": (None if is_unknown(p.predicted_cycles)
                                 else round(float(p.predicted_cycles), 2)),
            "fraction_of_reference": (None if is_unknown(p.recovered_share)
                                      else round(float(p.recovered_share), 4)),
            "partial_fraction_of_reference": round(p.partial_recovered_share, 4),
            "unresolved_resources": list(p.envelope.unresolved),
            "limiter": (None if is_unknown(p.envelope.limiter) else p.envelope.limiter),
        })
    full = [r for r in rows if r["resolved"]]
    return {
        "claim": "RECOVERS: fraction of a shipped reference implementation's measured cycles that "
                 "the derived structural envelope accounts for",
        "denominator": "the measured cycles of the reference program shipped for that kernel",
        "kernel_relative_only": True,
        "percent_of_peak": "UNCLAIMABLE: speed_of_light is null for this target and the one "
                           "candidate attainment model is GEMM-only, so no attainment denominator "
                           "is derived and none is invented",
        "operator": {"composition": op.value, "eta": eta,
                     "derivation": "from an overlap observation independent of the activity "
                                   "buckets; the buckets partition the timeline and cannot settle "
                                   "it"},
        "n_measured": len(rows),
        "n_resolved_end_to_end": len(full),
        "n_partial_only": len(rows) - len(full),
        "corpus_fraction_of_reference": (
            None if is_unknown(pred.corpus_recovered_share())
            else round(float(pred.corpus_recovered_share()), 4)),
        "corpus_fraction_of_reference_note": (
            f"summed predicted over summed measured across the {len(full)} kernel(s) whose bound "
            "resolves end to end. NOT a corpus average: the other kernels are absent from both "
            "sides of the ratio"),
        "floor_violations": list(pred.floor_violations),
        "bound_violations": list(pred.bound_violations),
        "unresolved_reason_by_resource": dict(pred.coverage.unresolved_reasons),
        "time_weighted_resolved_share": (
            None if is_unknown(pred.coverage.time_weighted_resolved_share)
            else round(float(pred.coverage.time_weighted_resolved_share), 4)),
        "kernels": rows,
    }


# ==================================================================================================
# the generated side -- merlin's own emitted contractions
# ==================================================================================================
def _layer_workload():
    """The N1 generator driver, loaded as a sibling module."""
    path = Path(__file__).resolve().parent / "layer_workload.py"
    spec = importlib.util.spec_from_file_location("_r7_layer_workload", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def emitted_op_stream(plan, *, delay_mnemonic: str, family_of) -> list:
    """The DYNAMIC op stream the emitted kernel executes, in the measurement source's vocabulary.

    ``(family, mnemonic, immediate)`` triples, the same contract
    :func:`merlin.perf.record.compose_unit_busy` consumes for a shipped program -- so the generated
    kernel is priced by the *same* law, through the same code, with no second path.

    The stream is reconstructed from the plan's own tile counts and the settle contract the plan
    carries, because the emitter loops and therefore its static text is not its trace. Three of the
    plan's independently-computed accessors cross-check the reconstruction
    (:func:`_check_stream`); a reconstruction that disagreed with them would be caught rather than
    priced.
    """
    mt, kt, nt = plan.tiles
    ops = plan.ops
    s = plan.settle

    def op(mnemonic: str, delay: int) -> list:
        out = [[family_of(mnemonic), _vocab(mnemonic), 0]]
        if delay:
            out.append(["Scalar", delay_mnemonic, int(delay)])
        return out

    body: list = []
    for _tile in range(mt * nt):
        for k in range(kt):
            body += op(ops.tile_load, s.tensor)          # weight tile into a matrix register
            body += op(ops.transpose, s.tensor)
            body += op(ops.weight_push, s.tensor)
            body += op(ops.tile_load, s.tensor)          # activation tile
            body += op(ops.contract if k == 0 else ops.contract_accumulate, s.mxu)
        body += op(ops.acc_read, s.vpu)                  # drain this output tile
        for _b in range(plan.facts.banks_per_tile):
            body += op(ops.tile_store, s.tensor)
    return body


def _vocab(mnemonic: str) -> str:
    """A mnemonic in the measurement source's spelling.

    The two vocabularies differ by one mechanical convention -- the ISA definition separates
    qualifiers with ``_`` and upper-cases, the measurement source separates with ``.`` and lower-
    cases -- so the transform is a case fold and a separator swap, applied to the whole token. It is
    never a pattern match, and a mnemonic that does not land in the source's vocabulary is reported
    by name rather than assumed into a bucket (see :func:`vector_demand`).
    """
    return str(mnemonic).lower().replace("_", ".")


def family_map(suite: dict) -> dict:
    """``mnemonic -> family``, read out of the measurement source's own op streams."""
    out: dict[str, str] = {}
    for body in suite["kernels"].values():
        for fam, mnemonic, _imm in body.get("op_stream") or ():
            out[str(mnemonic)] = str(fam)
    return out


def vector_demand(plan, suite: dict) -> tuple[float, str, list]:
    """``(demand, reason, unmapped)`` for the vector engine on a generated kernel.

    The corpus prices this bucket by counting ops of the vector family. That proxy can only be
    applied to a mnemonic the measurement source has actually seen: an instruction absent from every
    measured program has no established bucket, and guessing one from its spelling is how a unit
    gets credited with work it never did. So an unmapped mnemonic makes the demand a LOWER BOUND and
    the resource's time UNKNOWN, with the instruction named.
    """
    fam = family_map(suite)
    ops = plan.ops
    emitted = {ops.tile_load, ops.tile_store, ops.transpose, ops.weight_push, ops.contract,
               ops.contract_accumulate, ops.acc_read, ops.dma_load, ops.dma_store, ops.dma_wait}
    unmapped = sorted(m for m in emitted if _vocab(m) not in fam)
    known_vector = sorted(m for m in emitted if fam.get(_vocab(m)) == VECTOR_FAMILY)
    if unmapped:
        return (float(len(known_vector)),
                f"{len(unmapped)} emitted instruction(s) do not appear in any measured program, so "
                f"this bucket's op-count proxy cannot be evaluated on them: {unmapped}. The demand "
                "is a lower bound, not a total",
                unmapped)
    return float(len(known_vector)), "", []


def scheduled_stall_cycles(stream, *, delay_mnemonic: str, roles, unit_busy) -> dict:
    """Stall cycles the program schedules, split by whether a unit law already counted them.

    The systolic law is literally ``fill + the delays the program schedules for its compute op``, so
    a delay following a compute op is already inside that unit's composed busy time and counting it
    again as control time would charge it twice. Every other scheduled stall belongs to no unit and
    is serial control time on the scalar core -- the instruction occupies the core for its immediate
    whatever the engines are doing, which is what makes it a lower bound rather than an estimate.

    When the unit's composed busy is UNKNOWN the law counts nothing at all, so nothing is withheld
    from control: subtracting delays that no term is claiming would silently drop real cycles.
    """
    counted_by_unit = unit_busy.cycles is not None
    claimed = 0
    unclaimed = 0
    prev_base = None
    for _fam, mnemonic, imm in stream:
        if str(mnemonic) == delay_mnemonic and imm:
            if counted_by_unit and prev_base == roles.compute:
                claimed += int(imm)
            else:
                unclaimed += int(imm)
        prev_base = str(mnemonic).split(".", 1)[0]
    return {"counted_inside_a_unit_law": claimed, "unclaimed": unclaimed,
            "total": claimed + unclaimed,
            "unit_law_resolved": counted_by_unit}


def _check_stream(plan, stream, roles) -> dict:
    """Cross-check the reconstructed trace against the plan's own independently-computed accessors."""
    mt, kt, nt = plan.tiles
    computes = sum(1 for fam, m, _i in stream
                   if fam == roles.family and str(m).split(".", 1)[0] == roles.compute)
    drains = sum(1 for fam, m, _i in stream
                 if fam == roles.family and str(m).split(".", 1)[0] == roles.drain)
    return {"compute_ops": computes, "expected_compute_ops": mt * kt * nt,
            "drains": drains, "expected_drains": mt * nt,
            "agrees": computes == mt * kt * nt and drains == mt * nt}


def generated_times(plan, suite: dict, sources, *, variant: str) -> tuple:
    """Per-resource times for one generated contraction, under one declared envelope variant."""
    delay_mnemonic, roles, fill = corpus_law(suite)
    peaks = corpus_peaks(suite, sources)
    meta = suite["_meta"]
    fam = family_map(suite)
    stream = emitted_op_stream(plan, delay_mnemonic=delay_mnemonic,
                              family_of=lambda m: fam.get(_vocab(m), "Unmapped"))
    busy = compose_unit_busy(stream, roles, fill, delay_mnemonic)
    stalls = scheduled_stall_cycles(stream, delay_mnemonic=delay_mnemonic, roles=roles,
                                    unit_busy=busy)

    beats = plan.moved_bytes() / int(meta["beat_bytes"])
    move = resource_time(
        ResourceDemand(MOVEMENT_BUCKET, ResourceKind.MOVEMENT, beats, "beats", basis=Basis.MOVED,
                       provenance="bytes the emitted schedule moves, over the port width"),
        peaks[MOVEMENT_BUCKET])

    comp = structural_unit_time(COMPUTE_BUCKET, ResourceKind.COMPUTE, busy,
                                provenance=f"fill={fill} plus the delays the emitted program "
                                           "schedules")

    vdem, vreason, unmapped = vector_demand(plan, suite)
    if vreason:
        vec = ResourceTime(resource=VECTOR_BUCKET, kind=ResourceKind.COMPUTE, cycles=UNKNOWN,
                           unit="ops", basis=Basis.MOVED, evidence_kind="trace_derived",
                           provenance="emitted instruction inventory", reason=vreason)
    else:
        vec = resource_time(
            ResourceDemand(VECTOR_BUCKET, ResourceKind.COMPUTE, vdem, "ops", basis=Basis.MOVED,
                           provenance="emitted instruction inventory"),
            peaks[VECTOR_BUCKET])

    fixed_cycles = float(meta["reset_cycles"])
    fixed_prov = "reset_cycles declared by the measurement source"
    if variant == "plus_program_schedule":
        fixed_cycles += float(stalls["unclaimed"])
        fixed_prov += ("; plus the stall cycles the emitted program schedules that no unit law "
                       "claims, read from the plan's own settle contract and tile counts")
    elif variant != "corpus_only":
        raise ValueError(f"unknown envelope variant {variant!r}")
    fixed = ResourceTime(resource=FIXED_BUCKET, kind=ResourceKind.FIXED, cycles=fixed_cycles,
                         unit="cycles", basis=Basis.MOVED, evidence_kind="measured",
                         provenance=fixed_prov)
    detail = {"moved_bytes": plan.moved_bytes(), "beats": beats,
              "scheduled_stalls": stalls, "unmapped_mnemonics": unmapped,
              "stream_check": _check_stream(plan, stream, roles),
              "compute": {"cycles": busy.cycles, "groups": busy.groups, "computes": busy.computes,
                          "lower_bound": busy.lower_bound, "reason": busy.reason}}
    return [move, comp, vec, fixed], detail


def predict_generated(plan, suite: dict, sources, *, variant: str, operator, eta: float) -> dict:
    times, detail = generated_times(plan, suite, sources, variant=variant)
    c = compose(times, operator=operator, eta=eta)
    return {
        "variant": variant,
        "predicted_cycles": (None if is_unknown(c.cycles) else round(float(c.cycles), 2)),
        "partial_predicted_cycles": round(c.partial_cycles, 2),
        "unresolved": list(c.unresolved),
        "floor_cycles": round(c.floor_cycles, 2),
        "terms": {t.resource: (None if not t.known else round(float(t.cycles), 2)) for t in times},
        "term_reasons": {t.resource: t.reason for t in times if t.reason},
        "detail": detail,
    }


# ==================================================================================================
# measuring the generated shapes -- serially, one oracle at a time
# ==================================================================================================
def measure_shape(lw, target: str, tier: str, shape: str, workdir: Path, *, timeout: int) -> dict:
    """Build the plan for ``shape``, refuse it if its footprint aliases, otherwise run and grade it.

    The alias verdict is recorded for EVERY shape, run or refused. A wrapped footprint does not make
    a run slow, it makes the number wrong -- the window masks the address and drops the bytes with
    no error anywhere in the returned dict -- so the check is reported rather than assumed.
    """
    plan = lw.build_plan(target, tier, shape, workdir)
    rep = lw.WG.alias_report(plan.placements, plan.facts.dram_window)
    mt, kt, nt = plan.tiles
    row = {
        "shape": shape, "m": plan.m, "k": plan.k, "n": plan.n,
        "tiles": {"m": mt, "k": kt, "n": nt, "passes": mt * kt * nt},
        "macs": plan.total_macs(),
        "footprint_bytes": rep.footprint_bytes,
        "moved_bytes": plan.moved_bytes(),
        "kernel_words": len(plan.words),
        "unrolled_words": plan.unrolled_word_estimate(),
        "alias": {"ok": rep.ok, "window": rep.window, "reason": rep.reason,
                  "wrapped": list(rep.wrapped), "collisions": [list(c) for c in rep.collisions]},
        "tier": tier,
        "settle": {"tensor": plan.settle.tensor, "mxu": plan.settle.mxu, "vpu": plan.settle.vpu},
    }
    if not rep.ok:
        row["result"] = {"ran": False, "refused_because": rep.reason}
        return row
    runner = lw.make_runner(target, tier, workdir)
    budget = plan.suggested_max_cycles()
    t0 = time.time()
    try:
        cb = plan.command_buffer()
        for p in plan.placements:
            raw = plan.operands.get(p.name + "_bytes")
            if raw is not None:
                cb["tensors"][p.name]["preload_b64"] = base64.b64encode(raw).decode()
        res = runner(cb, plan.kernel_s, budget, timeout)
        out = next(iter(res["outputs"].values()))
    except lw.PO.ProgramDidNotHalt as e:
        row["result"] = {"ran": True, "halted": False, "budget": budget, "detail": str(e)}
        return row
    wall = time.time() - t0
    ok = plan.matches(out)
    row["result"] = {"ran": True, "halted": True, "bit_exact": ok,
                     "cycles": int(res["cycles"]), "wall_seconds": round(wall, 2),
                     "cycles_per_second": round(int(res["cycles"]) / max(wall, 1e-9), 1),
                     "oracle": res.get("oracle")}
    if not ok:
        row["result"]["divergence"] = plan.divergence(out)
    return row


def measure_path(target: str) -> Path:
    return A.cache_dir(f"{MEASURE_CACHE}/{target}") / "generated_runs.json"


def load_measurements(target: str) -> dict:
    p = measure_path(target)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"runs": {}}


def save_measurements(target: str, body: dict) -> None:
    measure_path(target).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")


def prior_runs(target: str) -> list:
    """Generated contractions measured by the N1 driver and left on disk as products.

    They are the size axis's large end and re-running them would cost 8 minutes of oracle for a
    number that is already recorded, so they are read back -- with their own product path, so a
    reader can see they came from a different invocation.
    """
    base = A.artifacts_dir() / "perf-workload" / target
    out = []
    for rec_path in sorted(base.glob("v*/*/workload_record.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        res = rec.get("result") or {}
        sh = rec.get("shape") or {}
        if not res.get("ran") or not res.get("halted"):
            continue
        out.append({"shape": f"{sh.get('m')}x{sh.get('k')}x{sh.get('n')}",
                    "cycles": int(res["cycles"]), "bit_exact": bool(res.get("bit_exact")),
                    "alias_ok": bool((rec.get("alias") or {}).get("ok")),
                    "tier": rec.get("tier"), "product": str(rec_path.parent)})
    return out


def cmd_measure(args) -> int:
    lw = _layer_workload()
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    shapes: list[str] = []
    for name in (SHAPE_SETS if args.set == "all" else [args.set]):
        shapes += list(SHAPE_SETS[name])
    body = load_measurements(args.target)
    runs = body.setdefault("runs", {})
    for shape in shapes:
        key = f"{args.tier}:{shape}"
        if key in runs and not args.force:
            print(f"# have {key}", file=sys.stderr)
            continue
        print(f"# running {key} ...", file=sys.stderr, flush=True)
        row = measure_shape(lw, args.target, args.tier, shape, wd, timeout=args.timeout)
        runs[key] = row
        save_measurements(args.target, body)
        r = row.get("result") or {}
        print(f"#   alias_ok={row['alias']['ok']} ran={r.get('ran')} "
              f"bit_exact={r.get('bit_exact')} cycles={r.get('cycles')} "
              f"wall={r.get('wall_seconds')}s", file=sys.stderr, flush=True)
    save_measurements(args.target, body)
    print(json.dumps({"measured": len(runs), "path": str(measure_path(args.target))}, indent=2))
    return 0


# ==================================================================================================
# CLAIM 7.2 -- PREDICTS, where no reference exists
# ==================================================================================================
def claim_predicts(target: str, suite: dict, tier: str) -> dict:
    lw = _layer_workload()
    sources = corpus_sources(suite)
    op, eta = derived_operator(sources)
    body = load_measurements(target)
    wd = Path("/scratch/agustin/tmp/perf_headline")
    wd.mkdir(parents=True, exist_ok=True)

    rows = []
    for key in sorted(body.get("runs", {})):
        run = body["runs"][key]
        if run.get("tier") != tier:
            continue
        res = run.get("result") or {}
        row = {"shape": run["shape"], "macs": run["macs"], "tiles": run["tiles"],
               "alias": run["alias"], "measured": res}
        if not (res.get("ran") and res.get("halted")):
            row["priced"] = False
            row["not_priced_because"] = ("the shape was refused for a wrapping footprint"
                                         if not run["alias"]["ok"] else "the program did not halt")
            rows.append(row)
            continue
        if not res.get("bit_exact"):
            row["priced"] = False
            row["not_priced_because"] = ("the run is not bit-exact, so its cycle count belongs to a "
                                         "different computation than the one predicted")
            rows.append(row)
            continue
        plan = lw.build_plan(target, tier, run["shape"], wd, with_operands=False)
        row["priced"] = True
        row["variants"] = {}
        for variant in ("corpus_only", "plus_program_schedule"):
            p = predict_generated(plan, suite, sources, variant=variant, operator=op, eta=eta)
            measured = int(res["cycles"])
            full = p["predicted_cycles"]
            partial = p["partial_predicted_cycles"]
            p["measured_cycles"] = measured
            p["prediction_accuracy"] = (None if full is None else round(full / measured, 4))
            p["partial_prediction_accuracy"] = round(partial / measured, 4)
            p["exceeds_measurement"] = bool(full is not None and full > measured)
            p["partial_exceeds_measurement"] = bool(partial > measured)
            row["variants"][variant] = p
        rows.append(row)

    priced = [r for r in rows if r.get("priced")]
    out = {
        "claim": "PREDICTS: predicted cycles against measured cycles for merlin's OWN emitted "
                 "kernel, at shapes for which no reference implementation exists",
        "denominator": "the measured cycles of merlin's own emitted kernel",
        "does_not_measure": "how good the emitted code is. With no shipped reference at these "
                            "shapes there is no fraction-of-reference to compute, and a predictor "
                            "can be exact on a kernel that is far slower than it should be",
        "percent_of_peak": "UNCLAIMABLE for the same reason as 7.1; nothing here divides by a "
                           "nameplate rate",
        "tier": tier,
        "operator": {"composition": op.value, "eta": eta},
        "n_shapes_attempted": len(rows),
        "n_priced": len(priced),
        "shapes": rows,
    }
    for variant in ("corpus_only", "plus_program_schedule"):
        vals = [r["variants"][variant] for r in priced]
        full = [v for v in vals if v["predicted_cycles"] is not None]
        out[variant] = {
            "n_resolved_end_to_end": len(full),
            "n_partial_only": len(vals) - len(full),
            "median_partial_prediction_accuracy": _median(
                [v["partial_prediction_accuracy"] for v in vals]),
            "range_partial_prediction_accuracy": (
                (min(v["partial_prediction_accuracy"] for v in vals),
                 max(v["partial_prediction_accuracy"] for v in vals)) if vals else None),
            "median_prediction_accuracy": _median([v["prediction_accuracy"] for v in full]),
            "shapes_exceeding_measurement": [r["shape"] for r in priced
                                             if r["variants"][variant]["exceeds_measurement"]
                                             or r["variants"][variant]["partial_exceeds_measurement"]],
        }
    out["prior_generated_runs_on_disk"] = prior_runs(target)
    return out


# ==================================================================================================
# 7.4 -- error, rank correlation, top-K recall and regret, as a function of size
# ==================================================================================================
def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return round(xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2, 4)


def _ranks(xs):
    """Average ranks, ties shared -- the ranking a rank correlation is defined over."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = shared
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return None if da == 0 or db == 0 else num / (da * db)


def kendall_tau_b(a, b):
    n = len(a)
    con = dis = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ta += 1
                tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif (da > 0) == (db > 0):
                con += 1
            else:
                dis += 1
    d1 = math.sqrt((con + dis + ta) * (con + dis + tb))
    return None if d1 == 0 else (con - dis) / d1


#: Below this many points a rank statistic is a number, not a result. Four points can only take a
#: handful of distinct correlation values and every one of them is within sampling noise of the
#: others, so the statistic is refused with its n rather than printed with a caveat.
MIN_RANK_N = 6


def rank_agreement(pairs, *, what: str) -> dict:
    """Spearman and Kendall over ``(predicted, measured)``, or the reason they were not computed."""
    n = len(pairs)
    if n < MIN_RANK_N:
        return {"n": n, "computed": False,
                "not_computed_because": f"{n} point(s); a rank correlation needs more than "
                                        f"{MIN_RANK_N - 1} to be distinguishable from chance, and "
                                        "reporting one over this sample would be a number rather "
                                        "than a result",
                "over": what}
    pred = [p for p, _m in pairs]
    meas = [m for _p, m in pairs]
    return {"n": n, "computed": True, "over": what,
            "spearman": _r(spearman(pred, meas)), "kendall_tau_b": _r(kendall_tau_b(pred, meas))}


def _r(v):
    return None if v is None else round(v, 4)


def top_k(pairs, k: int) -> dict:
    """Top-K recall and regret over a CHOICE SET -- candidates that do the same work.

    ``pairs`` are ``(name, predicted, measured)`` for candidates that are alternatives for one
    another. Recall is how many of the truly cheapest K the prediction's cheapest K contains; regret
    is how much worse the prediction's single best candidate actually is than the true best, as a
    fraction. Over candidates that do DIFFERENT amounts of work neither means anything, so this is
    only ever called on the iso-work set.
    """
    if len(pairs) < k + 1:
        return {"k": k, "n": len(pairs), "computed": False,
                "not_computed_because": f"{len(pairs)} candidate(s) cannot support a top-{k} claim"}
    by_pred = sorted(pairs, key=lambda p: p[1])
    by_meas = sorted(pairs, key=lambda p: p[2])
    pred_k = {p[0] for p in by_pred[:k]}
    meas_k = {p[0] for p in by_meas[:k]}
    best_pred, best_meas = by_pred[0], by_meas[0]
    regret = (best_pred[2] - best_meas[2]) / best_meas[2]
    return {"k": k, "n": len(pairs), "computed": True,
            "recall": round(len(pred_k & meas_k) / k, 4),
            "picked": best_pred[0], "true_best": best_meas[0],
            "regret_fraction": round(regret, 6),
            "regret_cycles": best_pred[2] - best_meas[2],
            "measured_spread_fraction": round(by_meas[-1][2] / by_meas[0][2] - 1.0, 6)}


def size_analysis(recovers: dict, predicts: dict, *, variant: str) -> dict:
    """Error and ranking as a function of size, over the two claims' points kept apart."""
    corpus_pts = [(r["kernel"], r["measured_cycles"], r["partial_fraction_of_reference"],
                   r["fraction_of_reference"]) for r in recovers["kernels"]]
    gen_pts = []
    for r in predicts["shapes"]:
        if not r.get("priced"):
            continue
        v = r["variants"][variant]
        gen_pts.append((r["shape"], v["measured_cycles"], v["partial_prediction_accuracy"],
                        v["prediction_accuracy"], v["partial_predicted_cycles"], r["macs"]))

    buckets: dict[str, list] = {}
    for name, measured, partial, _full in corpus_pts:
        buckets.setdefault(_decade(measured), []).append(("reference", name, measured, partial))
    for name, measured, partial, _full, _pp, _macs in gen_pts:
        buckets.setdefault(_decade(measured), []).append(("generated", name, measured, partial))

    by_decade = []
    for dec in sorted(buckets, key=lambda d: int(d.split("e")[1])):
        rows = buckets[dec]
        by_decade.append({
            "decade": dec, "n": len(rows),
            "sides": sorted({r[0] for r in rows}),
            "median_partial_accuracy": _median([r[3] for r in rows]),
            "members": [r[1] for r in rows],
        })

    iso = [(n, pp, m) for (n, m, _pa, _fa, pp, macs) in gen_pts if macs == ISO_MACS]
    size_span = None
    all_measured = [p[1] for p in corpus_pts] + [p[1] for p in gen_pts]
    if all_measured:
        size_span = {"smallest_measured_cycles": min(all_measured),
                     "largest_measured_cycles": max(all_measured),
                     "span": round(max(all_measured) / max(1, min(all_measured)), 1)}

    return {
        "variant": variant,
        "size_span": size_span,
        "by_decade": by_decade,
        "rank_agreement_generated": dict(
            rank_agreement([(pp, m) for (_n, m, _pa, _fa, pp, _mc) in gen_pts],
                           what="generated shapes, predicted vs measured cycles"),
            caveat="these shapes do DIFFERENT amounts of work, so a high rank correlation here is "
                   "mostly the ordering of the work itself. It is reported because it was asked "
                   "for, and it is not evidence that the model discriminates between candidates"),
        "rank_agreement_reference": dict(
            rank_agreement([(r["partial_fraction_of_reference"] * r["measured_cycles"],
                             r["measured_cycles"]) for r in recovers["kernels"]],
                           what="reference kernels, partial predicted vs measured cycles"),
            caveat="same caveat: the corpus kernels are different workloads, not alternatives for "
                   "one another"),
        "choice_set": {
            "definition": f"{len(iso)} shapes doing IDENTICAL arithmetic ({ISO_MACS:,} MACs) in an "
                          "identical number of tile passes, differing only in how the contraction "
                          "is decomposed. This is the only set here where ranking is a real "
                          "question: the candidates are alternatives for one another",
            "candidates": [{"shape": n, "predicted_cycles": pp, "measured_cycles": m}
                           for n, pp, m in sorted(iso, key=lambda p: p[2])],
            "rank_agreement": rank_agreement([(pp, m) for _n, pp, m in iso],
                                             what="iso-work choice set"),
            "top_1": top_k(iso, 1),
            "top_3": top_k(iso, 3),
        },
    }


def _decade(cycles: int) -> str:
    return f"1e{int(math.floor(math.log10(max(1, cycles))))}"


# ==================================================================================================
# provenance + emission
# ==================================================================================================
def digest_triple():
    return read_digest_triple(pin_names=list(PIN_NAMES), artifact_names=list(ARTIFACT_NAMES),
                              sources=[suite_path(), Path(__file__),
                                       Path(__file__).resolve().parent / "layer_workload.py"])


def _written_result(recovers: dict, predicts: dict, sizes: dict) -> list:
    """The result in words -- each claim with its n, and what the evidence does NOT support."""
    v = predicts["plus_program_schedule"]
    c = predicts["corpus_only"]
    lines = [
        f"RECOVERS (7.1), n={recovers['n_resolved_end_to_end']} of {recovers['n_measured']}: over "
        f"the kernels whose bound resolves end to end, the derived envelope accounts for "
        f"{recovers['corpus_fraction_of_reference']} of the reference implementations' measured "
        f"cycles. The other {recovers['n_partial_only']} carry a PARTIAL bound only and are not in "
        "that ratio.",
        f"PREDICTS (7.2), n={predicts['n_priced']} of {predicts['n_shapes_attempted']} generated "
        f"shapes: with the corpus-calibrated terms alone the partial bound reaches a median "
        f"{c['median_partial_prediction_accuracy']} of measured; adding the stall cycles the "
        f"emitted program itself schedules -- derived from the plan, no free parameters -- takes it "
        f"to {v['median_partial_prediction_accuracy']}.",
        "The two are NOT the same measurement. 7.1's denominator is somebody else's kernel; 7.2's "
        "denominator is merlin's own. A perfect 7.2 would still say nothing about whether the "
        "emitted code is fast.",
        "NOT SUPPORTED: any '% of peak' or attainment claim -- speed_of_light is null for this "
        "target and no attainment denominator is derived, so every number here is kernel-relative.",
        "NOT SUPPORTED: a fraction-of-reference at generated shapes. No reference implementation "
        "exists off the corpus, so that ratio has no denominator and is not reported.",
        f"NOT SUPPORTED: a claim that the model discriminates between ALTERNATIVES. The only real "
        f"choice set here is {sizes['choice_set']['top_1'].get('n')} iso-work shapes; the rank "
        "agreement over differently-sized shapes is mostly the ordering of the work itself.",
        "NOT SUPPORTED: an end-to-end bound at generated shapes. Every generated shape carries at "
        "least one unresolved resource, so the reported accuracies are PARTIAL bounds and are "
        "named as such.",
    ]
    if recovers["bound_violations"] or recovers["floor_violations"]:
        lines.append(f"WARNING: the reference-side bound is violated on "
                     f"{recovers['bound_violations']} / floors {recovers['floor_violations']}.")
    if v["shapes_exceeding_measurement"]:
        lines.append("WARNING: the program-schedule variant predicts MORE cycles than measured on "
                     f"{v['shapes_exceeding_measurement']}; a lower bound that exceeds its "
                     "measurement falsifies an input rather than being a good fit.")
    return lines


def report(target: str, tier: str) -> dict:
    suite = load_suite()
    recovers = claim_recovers(target, suite)
    predicts = claim_predicts(target, suite, tier)
    sizes = size_analysis(recovers, predicts, variant="plus_program_schedule")
    sizes_corpus_only = size_analysis(recovers, predicts, variant="corpus_only")
    return {
        "experiment": "R7 headline",
        "target": target,
        "digest": digest_triple().to_dict(),
        "claim_7_1_recovers": recovers,
        "claim_7_2_predicts": predicts,
        "claim_7_3_reported_separately": (
            "7.1 and 7.2 appear under separate keys, with separate n and separate denominators. "
            "They are never averaged, summed, or presented as one headline number: 7.1's "
            "denominator is a shipped reference implementation's measured cycles and 7.2's is "
            "merlin's own emitted kernel's, so a single combined figure would report a prediction "
            "result as a recovery result"),
        "analysis_7_4_vs_size": sizes,
        "analysis_7_4_vs_size_corpus_only": sizes_corpus_only,
        "written_result": _written_result(recovers, predicts, sizes),
    }


def cmd_report(args) -> int:
    body = report(args.target, args.tier)
    if not args.no_product:
        pd = A.new_product("perf-headline", version=1, target=args.target,
                           sources=[str(suite_path()), str(measure_path(args.target))],
                           notes="R7: the recovery claim and the prediction claim, reported "
                                 "separately with their own n and their own denominator")
        pd.add_artifact("headline.json").write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pd.add_artifact("result.md").write_text(
            "# R7 headline\n\n" + "\n\n".join(f"- {line}" for line in body["written_result"])
            + "\n", encoding="utf-8")
        pd.write_manifest()
        print(f"# product: {pd.path}", file=sys.stderr)
    print(json.dumps(body["written_result"], indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="run the generated shapes on an oracle, serially")
    m.add_argument("--target", required=True)
    m.add_argument("--tier", default="vsim")
    m.add_argument("--set", default="all", choices=[*SHAPE_SETS, "all"])
    m.add_argument("--workdir", default="/scratch/agustin/tmp/perf_headline")
    m.add_argument("--timeout", type=int, default=7200)
    m.add_argument("--force", action="store_true")
    m.set_defaults(func=cmd_measure)

    r = sub.add_parser("report", help="both claims, separately, plus the written result")
    r.add_argument("--target", required=True)
    r.add_argument("--tier", default="vsim")
    r.add_argument("--no-product", action="store_true")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
