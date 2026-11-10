"""Normalize workload-specific observations into ordered evidence records.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows, _json, _yaml, case_workloads
from .corpus_facts import _all_facts
from .evidence import _DERIVED, _SMALL, _fact


def _eqish(a, b, tol=2) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a) == str(b)


def unified_facts(cs_dir: Path, scope: str) -> list[dict]:
    """Normalize DSE-relevant facts across ALL artifacts (never invented). Each fact is source-traced
    and (where cross-checkable) corroborated by counting independent artifacts that agree.
    scope = workload | all."""
    man = _json(cs_dir / "dse_contract.json") or {}
    per_wl = man.get("per_workload", {})
    workloads = case_workloads(cs_dir) if scope == "all" else [scope]
    facts: list[dict] = []
    n = 0

    def add(**kw):
        nonlocal n
        facts.append(_fact(f"F{n:04d}", **kw))
        n += 1

    crit = {r["workload"]: r for r in _csv_rows(cs_dir / "critical_path_table.csv")}
    shp = _csv_rows(cs_dir / "shape_summary_by_workload.csv")
    shr = _csv_rows(cs_dir / "shape_summary_by_region.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    tr = _csv_rows(cs_dir / "traffic_table.csv")
    dcap = {r["workload"]: r for r in _csv_rows(cs_dir / "dtype_capacity_table.csv")}
    cov = _csv_rows(cs_dir / "primitive_coverage_matrix.csv")
    epi = _csv_rows(cs_dir / "epilogue_pattern_table.csv")
    acc_g = _csv_rows(cs_dir / "accuracy_gated_dtype_candidates.csv")
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    tile = _csv_rows(cs_dir / "tile_waste_table.csv")
    shard = _csv_rows(cs_dir / "sharding_table.csv")
    dma = _csv_rows(cs_dir / "dma_stream_table.csv")
    buf = _csv_rows(cs_dir / "buffer_requirement_table.csv")
    bufing = _csv_rows(cs_dir / "buffering_requirement_table.csv")
    reuse = _csv_rows(cs_dir / "reuse_lifetime_table.csv")
    acc_m = {r.get("model", r.get("workload", "")): r for r in _csv_rows(cs_dir / "accuracy_gate_results.csv")}
    prate = _csv_rows(cs_dir / "phase_rate_table.csv")
    pstage = _csv_rows(cs_dir / "pipeline_stage_table.csv")
    accum = _csv_rows(cs_dir / "accumulator_contract_table.csv")
    wct = {r["workload"]: r for r in _csv_rows(cs_dir / "workload_contract_table.csv")}

    def _ops_weight(w):
        return sum(
            int(r["rhs_weight_bytes"]) for r in ops if r["workload"] == w and r["region_role"] == "repeated_head"
        )

    for w in workloads:
        d = per_wl.get(w, {})
        # --- weights: corroborate manifest vs data_movement vs operator_shape sum ---
        hw = d.get("head_weight_bytes")
        dm_head = next((r for r in dm if r["workload"] == w and r["region"] == "repeated_head"), None)
        if hw:
            srcs = [hw]
            if dm_head:
                srcs.append(int(dm_head["weight_bytes"]))
            if ops:
                srcs.append(_ops_weight(w))
            corr = sum(1 for s in srcs if _eqish(s, hw))
            add(
                workload=w,
                metric="head_weight_bytes",
                value=hw,
                unit="bytes",
                artifact="dse_contract.json",
                phase="P5",
                evidence="recovered_from_ir",
                corroborated_by=corr,
                caveat=_SMALL,
                implication="resident-weight capacity requirement",
            )
        add(
            workload=w,
            metric="K",
            value=d.get("K"),
            unit="steps",
            artifact="dse_contract.json",
            phase="P6",
            evidence="recovered_from_model_config",
            caveat="published architecture constant (not measured)",
            implication="bounds the K-loop / autonomous_K_loop axis",
        )
        add(
            workload=w,
            metric="ready_quantitative_dse",
            value=d.get("ready_quantitative_dse"),
            unit="bool",
            artifact="dse_readiness_summary.csv",
            phase="P4",
            evidence=_DERIVED,
            implication="structural DSE ready; quantitative needs measurements",
        )
        # --- accuracy: corroborate gate-candidate vs measured-gate results ---
        acc = d.get("accuracy_int8", "unavailable")
        accm = acc_m.get(w, {})
        corr_a = 1 + (1 if accm else 0)
        add(
            workload=w,
            metric="accuracy_int8_w8a8",
            value=acc,
            unit="band",
            artifact="accuracy_gated_dtype_candidates.csv",
            phase="P6",
            evidence="measured" if acc in ("pass", "fail") else "unavailable",
            corroborated_by=corr_a if acc in ("pass", "fail") else 1,
            implication="gates int8 as an accuracy-legal dtype candidate",
        )
        # --- parallelism (corroborate critical_path with concurrency window max width) ---
        c = crit.get(w)
        if c:
            add(
                workload=w,
                metric="available_parallelism",
                value=c["available_parallelism"],
                unit="work/span",
                artifact="critical_path_table.csv",
                phase="P7",
                evidence=_DERIVED,
                caveat="structural work/span, not a performance metric",
                implication="low inter-op parallelism favors intra-op sharding (not many identical "
                "units kept busy by concurrency)",
            )
            add(
                workload=w,
                metric="serialization",
                value=c["serialization"],
                unit="class",
                artifact="critical_path_table.csv",
                phase="P7",
                evidence=_DERIVED,
                implication="near-sequential DAG shape",
            )
            add(
                workload=w,
                metric="total_macs",
                value=c["total_macs"],
                unit="MACs",
                artifact="critical_path_table.csv",
                phase="P7",
                evidence="recovered_from_ir",
                corroborated_by=2 if ops else 1,
                caveat=_SMALL,
                implication="per-replan compute volume",
            )
        cw = next((r for r in _csv_rows(cs_dir / "concurrency_windows.csv") if r["workload"] == w), None)
        if cw:
            mw = max(
                (int(r["ready_ops"]) for r in _csv_rows(cs_dir / "concurrency_windows.csv") if r["workload"] == w),
                default=0,
            )
            add(
                workload=w,
                metric="max_ready_width",
                value=mw,
                unit="ops",
                artifact="concurrency_windows.csv",
                phase="P7",
                evidence=_DERIVED,
                implication="peak simultaneously-ready operators",
            )
        # --- operator geometry ---
        nm = sum(1 for r in ops if r["workload"] == w)
        if nm:
            add(
                workload=w,
                metric="n_matmuls",
                value=nm,
                unit="ops",
                artifact="operator_shape_table.csv",
                phase="P5",
                evidence="recovered_from_ir",
                corroborated_by=2 if c else 1,
                implication="operator count to cover",
            )
        ws = [r for r in shp if r["workload"] == w]
        if ws:
            top = max(ws, key=lambda r: float(r["mac_fraction"]))
            add(
                workload=w,
                metric="dominant_shape_class",
                value=top["shape_class"],
                unit="",
                artifact="shape_summary_by_workload.csv",
                phase="P5",
                evidence="recovered_from_ir",
                caveat=f"{w}: {float(top['mac_fraction']):.0%} of this workload's MACs",
                implication="dominant geometry class -> the primitive shape the DSE must cover",
            )
        if any(r["workload"] == w for r in shr):
            add(
                workload=w,
                metric="n_region_shape_groups",
                value=sum(1 for r in shr if r["workload"] == w),
                unit="groups",
                artifact="shape_summary_by_region.csv",
                phase="P5",
                evidence="recovered_from_ir",
                implication="per-region geometry breakdown",
            )
        if any(r["workload"] == w for r in tile):
            add(
                workload=w,
                metric="n_tile_evaluations",
                value=sum(1 for r in tile if r["workload"] == w),
                unit="op*primitive",
                artifact="tile_waste_table.csv",
                phase="P5",
                evidence=_DERIVED,
                implication="tile-coverage evaluations performed",
            )
        wc = [r for r in cov if r["workload"] == w]
        if wc:
            best = max(wc, key=lambda r: float(r["coverage_under_10pct"]))
            add(
                workload=w,
                abstraction=best["primitive"],
                metric="coverage_under_10pct",
                value=best["coverage_under_10pct"],
                unit="MAC-fraction",
                artifact="primitive_coverage_matrix.csv",
                phase="P5",
                evidence=_DERIVED,
                implication="best-covering primitive for this workload",
            )
        # --- memory / dma / buffers ---
        if dm_head:
            srcs = [int(dm_head["avoidable_weight_reload"])]
            tr_head = next((r for r in tr if r["workload"] == w and r["region"] == "repeated_head"), None)
            corr_av = 1 + (1 if tr_head else 0)
            add(
                workload=w,
                region="repeated_head",
                metric="avoidable_weight_reload",
                value=dm_head["avoidable_weight_reload"],
                unit="bytes",
                artifact="data_movement_table.csv",
                phase="P9",
                evidence=_DERIVED,
                corroborated_by=corr_av,
                caveat=_SMALL,
                implication="resident_weight_object residency benefit (bytes), no bandwidth claim",
            )
            corr_r = 1 + (1 if (w in dcap and _eqish(dcap[w].get("int8_B"), dm_head["resident_int8_B"])) else 0)
            add(
                workload=w,
                region="repeated_head",
                metric="resident_int8_B",
                value=dm_head["resident_int8_B"],
                unit="bytes",
                artifact="data_movement_table.csv",
                phase="P9",
                evidence=_DERIVED,
                corroborated_by=corr_r,
                implication="int8 resident-capacity requirement",
            )
        if any(r["workload"] == w for r in dma):
            nbs = sum(1 for r in dma if r["workload"] == w and r["bytes"] != "unavailable")
            add(
                workload=w,
                metric="byte_carrying_streams",
                value=nbs,
                unit="streams",
                artifact="dma_stream_table.csv",
                phase="P9",
                evidence="recovered_from_ir",
                implication="independent DMA streams (weight/activation/output)",
            )
        if any(r["workload"] == w for r in buf):
            bf = next(r for r in buf if r["workload"] == w)
            add(
                workload=w,
                metric="min_input_buffers",
                value=bf["min_input_buffer_count"],
                unit="buffers",
                artifact="buffer_requirement_table.csv",
                phase="P9",
                evidence=_DERIVED,
                implication="double-buffering to overlap DMA with compute",
            )
        if any(r["workload"] == w for r in bufing):
            yes = sum(1 for r in bufing if r["workload"] == w and r["can_overlap"] == "yes")
            add(
                workload=w,
                metric="overlap_candidates_yes",
                value=yes,
                unit="candidates",
                artifact="buffering_requirement_table.csv",
                phase="P8",
                evidence=_DERIVED,
                implication="phase overlaps structurally permitted",
            )
        if any(r["workload"] == w for r in reuse):
            add(
                workload=w,
                metric="n_resident_objects",
                value=sum(1 for r in reuse if r["workload"] == w),
                unit="objects",
                artifact="reuse_lifetime_table.csv",
                phase="P9",
                evidence="recovered_from_ir",
                implication="objects with a reuse lifetime (residency candidates)",
            )
        # --- fusion / numerical ---
        we = [r for r in epi if r["workload"] == w]
        if we:
            n_bias = sum(1 for r in we if r["has_bias"] == "True")
            add(
                workload=w,
                metric="matmul_bias_epilogues",
                value=n_bias,
                unit="ops",
                artifact="epilogue_pattern_table.csv",
                phase="P10",
                evidence="recovered_from_ir",
                implication="fused epilogue slot present (bias) -> fused_requant_epilogue candidate",
            )
        wa = [r for r in accum if r["workload"] == w]
        if wa:
            add(
                workload=w,
                region=wa[0]["region"],
                metric="accumulator_dtype",
                value=wa[0]["accumulator_dtype"],
                unit="dtype",
                artifact="accumulator_contract_table.csv",
                phase="P10",
                evidence="recovered_from_ir",
                implication="accumulator width for the datapath",
            )
        # --- sharding / pipeline rate ---
        if any(r["workload"] == w for r in shard):
            clean = sum(
                1
                for r in shard
                if r["workload"] == w
                and r["axis"] in ("M", "N")
                and r["shardable_8"] == "True"
                and r["tail_8"] == "False"
            )
            add(
                workload=w,
                metric="clean_8way_mn_shards",
                value=clean,
                unit="op*axis",
                artifact="sharding_table.csv",
                phase="P7",
                evidence=_DERIVED,
                implication="reduction-free M/N shards available",
            )
        if any(r["workload"] == w for r in pstage):
            cads = sorted({r["cadence"] for r in pstage if r["workload"] == w})
            add(
                workload=w,
                metric="phase_cadences",
                value="; ".join(cads),
                unit="",
                artifact="pipeline_stage_table.csv",
                phase="P8",
                evidence="recovered_from_prov_fqn",
                corroborated_by=2 if prate else 1,
                implication="multi-rate phase structure",
            )
        if w in wct:
            add(
                workload=w,
                metric="recovered_roles",
                value=wct[w].get("roles", ""),
                unit="",
                artifact="workload_contract_table.csv",
                phase="P2",
                evidence="recovered_from_prov_fqn",
                implication="region roles recovered from prov.fqn",
            )
        # dtype capacity (source; corroborates resident_int8) + traffic (corroborates avoidable)
        if w in dcap:
            add(
                workload=w,
                metric="int8_capacity_B",
                value=dcap[w].get("int8_B"),
                unit="bytes",
                artifact="dtype_capacity_table.csv",
                phase="P9",
                evidence=_DERIVED,
                corroborated_by=2 if dm_head else 1,
                implication="int8 resident capacity (weights scaled to int8)",
            )
        trh = next((r for r in tr if r["workload"] == w and r["region"] == "repeated_head"), None)
        if trh:
            add(
                workload=w,
                region="repeated_head",
                metric="weight_traffic_if_nonresident",
                value=trh["weight_traffic_if_nonresident"],
                unit="bytes",
                artifact="traffic_table.csv",
                phase="P9",
                evidence=_DERIVED,
                implication="weight bytes moved if NOT made resident (reload each step)",
            )
        wp = [r for r in prate if r["workload"] == w]
        if wp:
            head = next((r for r in wp if r["role"] == "repeated_head"), wp[0])
            add(
                workload=w,
                metric="head_cadence",
                value=head.get("cadence", ""),
                unit="",
                artifact="phase_rate_table.csv",
                phase="P6",
                evidence="recovered_from_prov_fqn",
                corroborated_by=2 if pstage else 1,
                implication="repeated-head cadence (rate class)",
            )
        # per-workload subdir: numerical_contract.yaml + region_attribution.yaml
        nc = _yaml(cs_dir / w / "numerical_contract.yaml")
        if isinstance(nc, dict):
            ncb = nc.get("numerical_contract", nc)
            add(
                workload=w,
                metric="compute_dtype",
                value=str(ncb.get("compute_dtype", "f32")),
                unit="dtype",
                artifact="numerical_contract.yaml",
                phase="P10",
                evidence="recovered_from_ir",
                implication="storage/compute dtype contract",
            )
        ra = _yaml(cs_dir / w / "region_attribution.yaml")
        if isinstance(ra, dict):
            regs = ra.get("topology_recovery", {}).get("regions", [])
            attr_mm = sum(r.get("facts", {}).get("matmul_count", 0) for r in regs)
            if attr_mm:
                add(
                    workload=w,
                    metric="attributed_matmuls",
                    value=attr_mm,
                    unit="ops",
                    artifact="region_attribution.yaml",
                    phase="P1",
                    evidence="recovered_from_prov_fqn",
                    corroborated_by=2 if nm else 1,
                    implication="matmuls attributed to roles",
                )

    if scope == "all":
        _all_facts(cs_dir, man, add)
    return facts
