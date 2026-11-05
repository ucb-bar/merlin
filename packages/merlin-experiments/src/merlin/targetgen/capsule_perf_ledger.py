"""Publish measured capsule timing and decomposition without changing grading."""
from __future__ import annotations

import json
from pathlib import Path


def _activity_source(workload: str, block, *, total_cycles: int, provenance: str):
    """An :class:`~merlin.perf.decompose.ActivitySource` from one validated timing block, or the
    reason there is none. Returns ``(source, refusal)`` with exactly one of them set.

    The KINDS come from the producer. They are not inferred from a unit's spelling and they are not
    defaulted: a unit whose kind the producer did not state is a hole in the input, and reading a
    role out of a name is how a local register load once became "DMA".
    """
    from merlin.perf.decompose import ResourceKind, activity_from_busy
    from merlin.perf.observations import IDLE_QUANTITY

    busy = block.busy_by_unit()
    if not busy:
        return None, f"{workload}: the block carries no per-unit busy count"
    declared = block.kinds()
    missing = sorted(set(busy) - set(declared))
    if missing:
        return None, (
            f"{workload}: the producer stated no kind for unit(s) {missing}. A role read "
            f"out of a unit's NAME is not a derivation, so the source is refused"
        )
    kinds: dict = {}
    for unit, kind in declared.items():
        try:
            kinds[unit] = ResourceKind(kind)
        except ValueError:
            return None, (
                f"{workload}: unit {unit!r} declares kind {kind!r}, which is not a "
                f"resource kind this layer knows; refusing to map it onto one"
            )
    idle = block.quantity(IDLE_QUANTITY)
    if idle is not None:
        # Cycles charged to NO engine are an intercept, not a rate, and they are first-class here:
        # on the measured corpus this is the single largest bucket, and an occupancy-only schema
        # cannot see it at all.
        busy = dict(busy)
        busy[IDLE_QUANTITY] = int(idle)
        kinds[IDLE_QUANTITY] = ResourceKind.FIXED
    try:
        src = activity_from_busy(
            workload,
            int(total_cycles),
            busy,
            kinds,
            # NOT a partition, asserted by the producer and re-asserted here.
            # If this ever becomes True the overlap reading is refused rather
            # than silently reported as zero.
            partitioned=bool(block.partitioned),
            completion_observable=None,
            provenance=provenance,
        )
    except ValueError as exc:
        return None, f"{workload}: {exc}"
    return src, None


def emit_perf_ledger(
    results: list, *, target: str, runs_root: str | Path, version: int = 1, activity_source, suite_for
) -> dict:
    """Write this grade's ``perf-ledger`` product and return a pointer + summary for the score.

    Products: ``ledger.json`` (per-tier rows plus the corpus-level composition question),
    ``observations.jsonl`` (one recovered observation per line) and ``refusals.json`` (everything
    that could have contributed and did not, with the reason). Per-cycle occupancy CSVs are NOT
    products -- they are ~34 B/cycle of replay-regenerable detail and live under
    ``cache_dir("occupancy")``, which is purgeable.

    Never fatal: a grade that ran must still report its numbers, with the ledger's gap visible.
    """
    from merlin.common.artifacts import cache_dir, new_product
    from merlin.kernels.measurement import authority_for
    from merlin.perf import harvest as HV
    from merlin.perf import observations as OBS
    from merlin.perf.decompose import Unavailable, decompose_corpus, is_unknown
    from merlin.perf.headroom import composition_operator

    auth = authority_for(target)
    rr = Path(runs_root) / "runs" / suite_for(target)
    rows: list[dict] = []
    obs_rows: list[dict] = []
    refusals: list[dict] = []
    sources: list = []
    overlaps: dict[str, int] = {}
    no_capability: list[str] = []

    for r in results:
        capsule = str(r.get("capsule") or "")
        path = rr / capsule / "capsule_result.json"
        tiers = r.get("tiers") or {}
        saw_block = False
        for tier, rec in sorted(tiers.items()):
            if not isinstance(rec, dict):
                continue
            # Recomposed from its two halves: the observation list and the instrument's own
            # capability record, which the tier keeps beside it rather than inside it.
            block = OBS.block_from_tier_record(rec)
            row = {
                "capsule": capsule,
                "tier": tier,
                "status": rec.get("status"),
                "cycles": rec.get("cycles"),
                "fidelity": rec.get("fidelity"),
                "evidence": rec.get("evidence"),
                # WHICH ENGINE produced these seconds and cycles, which BUILD of it, and where its
                # console went. A perf ledger without the engine is a set of numbers from an unknown
                # instrument: GSIM and Verilator run the same design at the same fidelity and cost
                # ~20-30x apart, so a fit over rows that mix them silently averages two instruments.
                # Until now the only trace of the engine was inside the `evidence` FILENAME, which
                # every consumer here would have had to string-parse to recover.
                "engine": rec.get("engine"),
                "engine_selection": rec.get("engine_selection"),
                "sim_provenance": rec.get("sim_provenance"),
                "console_log": rec.get("console_log"),
                "console_bytes": rec.get("console_bytes"),
                # STATED, not inferred from the file's path. See TierResult.submission.
                "submission": rec.get("submission"),
                # A run taken before the stamp existed is MARKED, never retro-labelled: the
                # concurrency it ran at is not recoverable and pretending otherwise is worse.
                "concurrency": OBS.concurrency_of(rec),
                "timing": rec.get("timing"),
                "timing_capability": rec.get("timing_capability"),
                "has_timing_block": block is not None,
            }
            rows.append(row)
            if block is None:
                continue
            saw_block = True
            for why in block.refusals:
                refusals.append({"what": f"{capsule}@{tier}", "reason": why, "where": str(path)})
            if block.alias_collisions is None:
                refusals.append(
                    {
                        "what": f"{capsule}@{tier} alias accounting",
                        "reason": (
                            "the producer carried no alias-collision count, so whether this run's "
                            "addresses stayed inside one window page is UNKNOWN, not zero"
                        ),
                        "where": str(path),
                    }
                )
            elif block.alias_collisions > 0:
                refusals.append(
                    {
                        "what": f"{capsule}@{tier}",
                        "reason": (
                            f"{block.alias_collisions} access(es) collided inside the wrapping "
                            "memory window: two addresses a window apart became one byte, so "
                            "these numbers are not about the program that was submitted"
                        ),
                        "where": str(path),
                    }
                )
            total = block.quantity(OBS.SAMPLED_QUANTITY)
            if total is None:
                total = rec.get("cycles")
            overlap = block.overlap_cycles()
            key = f"{capsule}@{tier}"
            if total is None:
                refusals.append(
                    {"what": key, "reason": "no cycle total to place the buckets against", "where": str(path)}
                )
                continue
            src, why = activity_source(
                key, block, total_cycles=int(total), provenance=str(rec.get("evidence") or tier)
            )
            if src is None:
                refusals.append({"what": key, "reason": why, "where": str(path)})
                continue
            sources.append(src)
            if overlap is None:
                refusals.append(
                    {
                        "what": key,
                        "reason": (
                            "the block licenses no overlap reading (it asserts a partitioned "
                            "bucket set, or carries no joint-occupancy entry), so this workload "
                            "cannot contribute to the composition operator"
                        ),
                        "where": str(path),
                    }
                )
            else:
                overlaps[key] = int(overlap)
        if not saw_block and tiers:
            no_capability.append(capsule)
        if path.is_file():
            # Reuse the harvest path rather than a second reader: same citability gate, same
            # contended-provenance rules. The submission is passed EXPLICITLY so the path heuristic
            # in ``_submission_of`` is only a fallback, never the key.
            _sub = next(
                (
                    str((t.get("submission") or {}).get("package"))
                    for t in tiers.values()
                    if isinstance(t, dict) and (t.get("submission") or {}).get("package")
                ),
                None,
            )
            o, rf = HV.harvest_capsule_result(path, authority=auth, submission=_sub)
            obs_rows.extend(x.to_dict() for x in o)
            refusals.extend(x.to_dict() for x in rf)

    # WHERE THE CYCLES WENT, per workload and for the corpus. `decompose` was reachable the moment an
    # activity source existed and nothing called it, so a run reported a total and never said which
    # resource bound it. The FIXED bucket is the one to watch: cycles charged to no engine at all --
    # layout conversion, dispatch, scalar epilogues, the allocate/copy/pack/evict traffic a tight
    # hand-written program does not pay. A compute-vs-movement split alone cannot see it, and it is
    # where this compiler loses to a directly-compiled baseline.
    #
    # THE LIMITER CANNOT REPORT IT. `binding` names the busiest ENGINE, and the fixed bucket is not an
    # engine -- so a workload spending 70% of its cycles on no engine at all still reports "bound by
    # the load unit". A reader who optimises the limiter alone would tune DMA and never touch the
    # larger cost, which is why `fixed_share` is carried BESIDE the binding rather than left to be
    # inferred from it.
    corpus = decompose_corpus(sources)
    modal = corpus.modal_binding_kind()
    decomposition = {
        "workloads": {
            name: {
                "binding": d.binding,
                "binding_kind": getattr(d.binding_kind, "value", str(d.binding_kind)),
                "binding_share": round(float(d.binding_share), 4),
                "fixed_share": round(float(d.fixed_share), 4),
                "margin_to_second": (None if is_unknown(d.margin_to_second) else round(float(d.margin_to_second), 4)),
                "total_cycles": d.total_cycles,
            }
            for name, d in sorted(corpus.workloads.items())
        },
        # A workload whose decomposition REFUSED is named, not dropped: a corpus regime read off the
        # survivors is a claim about a different corpus.
        "unavailable": {
            name: {"what": u.what, "missing": list(u.missing)} for name, u in sorted(corpus.unavailable.items())
        },
        "modal_binding_kind": (None if is_unknown(modal) else getattr(modal, "value", str(modal))),
        "binding_kind_counts": {getattr(k, "value", str(k)): n for k, n in corpus.binding_kind_counts.most_common()},
    }

    comp = composition_operator(sources, observed_overlap_cycles=overlaps or None)
    if isinstance(comp, Unavailable):
        # NAMED, not a bare "unknown": a reader has to be able to go and buy the missing evidence.
        composition = {"resolved": False, "what": comp.what, "missing": list(comp.missing), "detail": comp.detail}
    else:
        op, eta = comp
        composition = {
            "resolved": True,
            "operator": op.value,
            "eta": round(float(eta), 4),
            "note": (
                "derived from a JOINT occupancy observation independent of the "
                "activity buckets -- never from a partitioned bucket set, which "
                "reports zero overlap by construction"
            ),
        }

    occ = cache_dir("occupancy") / target
    pd = new_product(
        "perf-ledger",
        version=version,
        target=target,
        sources=[str(rr)],
        notes="what ONE graded run measured: per-tier cycle counts with the submission "
        "and the concurrency they were taken at, plus whatever per-unit "
        "decomposition the oracle could see. Every value is trace_derived and "
        "contended; none of it may become a calibrated constant. Distinct from "
        "perf-records (a designed suite) and perf-harvest (retro-mined).",
    )
    (pd.path / "observations.jsonl").write_text(
        "".join(json.dumps(o, sort_keys=True) + "\n" for o in obs_rows), encoding="utf-8"
    )
    (pd.path / "refusals.json").write_text(
        json.dumps(
            {
                "n": len(refusals),
                "refusals": refusals,
                "capsules_with_no_timing_capability": sorted(no_capability),
                "note": (
                    "a capsule listed here ran on an oracle that reports no per-unit "
                    "timing. That is UNMEASURED, not zero -- the tier emitted no block at "
                    "all, which is the correct output for an instrument without the "
                    "capability."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (pd.path / "ledger.json").write_text(
        json.dumps(
            {
                "target": target,
                "authority": {"citable_tier": auth.citable_tier, "declared": auth.declared, "gaps": list(auth.gaps())},
                "n_rows": len(rows),
                "n_observations": len(obs_rows),
                "n_refusals": len(refusals),
                "rows": rows,
                "composition_operator": composition,
                # WHICH RESOURCE BOUND EACH WORKLOAD, and the share charged to no engine at all.
                "decomposition": decomposition,
                "per_cycle_occupancy": {
                    "dir": str(occ),
                    "purgeable": True,
                    "note": (
                        "per-cycle occupancy CSVs are ~34 B/cycle and are regenerable by replaying the "
                        "same program, so they are a cache, never a product. Written only when "
                        "MERLIN_PERF_PER_CYCLE=1 was set for the run."
                    ),
                    "present": sorted(x.name for x in occ.glob("*.csv")) if occ.is_dir() else [],
                },
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.write_manifest()
    return {
        "product": str(pd.path),
        "n_rows": len(rows),
        "n_observations": len(obs_rows),
        "n_refusals": len(refusals),
        "capsules_with_no_timing_capability": len(no_capability),
        "composition_operator": composition,
        # The SUMMARY carries the corpus regime and the overhead share, not the per-workload
        # table -- a reader of the score should see which resource this corpus is bound by, and
        # how much of it went to no engine at all, without opening the product.
        "modal_binding_kind": decomposition["modal_binding_kind"],
        "binding_kind_counts": decomposition["binding_kind_counts"],
        "n_workloads_decomposed": len(decomposition["workloads"]),
        "n_workloads_undecomposable": len(decomposition["unavailable"]),
    }
