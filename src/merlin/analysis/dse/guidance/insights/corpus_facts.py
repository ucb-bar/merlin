"""Collect cross-workload evidence through the original ordered fact writer.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows, _json, _yaml
from .evidence import _DERIVED


def _all_facts(cs_dir: Path, man: dict, add) -> None:
    """Cross-workload facts spanning every remaining artifact (so coverage == all)."""
    bm = _csv_rows(cs_dir / "hw_sw_boundary_matrix.csv")
    bc = _yaml(cs_dir / "boundary_candidate_contracts.yaml")
    bc_n = len((bc or {}).get("boundary_candidate_contracts", {}).get("certificates", []))
    for b in sorted(bm, key=lambda r: -int(r["boundary_pressure_score"]))[:8]:
        strong = [
            lv
            for lv in (
                "compiler_transform",
                "runtime_hal_object",
                "command_buffer_or_command_isa",
                "accelerator_isa",
                "device_microcode_or_controller",
                "fixed_hardware_datapath",
            )
            if b.get(lv) == "strong_candidate"
        ]
        add(
            workload="ALL",
            abstraction=b["abstraction"],
            level=";".join(strong),
            metric="boundary_pressure_score",
            value=b["boundary_pressure_score"],
            unit="evidence",
            artifact="hw_sw_boundary_matrix.csv",
            phase="P12",
            evidence=_DERIVED,
            corroborated_by=2 if bc_n else 1,
            caveat="evidence breadth, not performance/priority",
            implication="strong candidate boundary placement(s)",
        )

    def count(fname, key, phase, metric, impl, evidence=_DERIVED, sub=None):
        if fname.endswith((".yaml", ".json")):
            obj = (_json if fname.endswith(".json") else _yaml)(cs_dir / fname)
            if obj is None:
                return
            node = obj.get(key, obj) if isinstance(obj, dict) else obj
            if sub:
                node = node.get(sub, []) if isinstance(node, dict) else node
            val = len(node) if isinstance(node, (list, dict)) else node
        else:
            rows = _csv_rows(cs_dir / fname)
            if not rows:
                return
            val = len(rows)
        add(
            workload="ALL",
            metric=metric,
            value=val,
            unit="count",
            artifact=fname,
            phase=phase,
            evidence=evidence,
            implication=impl,
        )

    # P12 candidate lists + responsibility + knobs
    count(
        "runtime_object_candidates.yaml",
        "runtime_object_candidates",
        "P12",
        "n_runtime_objects",
        "objects that could cross the HAL boundary",
        sub="candidates",
    )
    count(
        "command_isa_candidates.yaml",
        "command_isa_candidates",
        "P12",
        "n_command_candidates",
        "command-ISA abstractions structurally suggested",
        sub="candidates",
    )
    count(
        "isa_candidate_primitives.yaml",
        "isa_candidate_primitives",
        "P12",
        "n_isa_primitives",
        "accelerator-ISA primitives structurally suggested",
        sub="primitives",
    )
    count(
        "boundary_dse_knobs.yaml",
        "boundary_dse_knobs",
        "P12",
        "n_boundary_knobs",
        "DSE knobs created by boundary placement",
        sub="knobs",
    )
    resp = _csv_rows(cs_dir / "responsibility_split_matrix.csv")
    if resp:
        add(
            workload="ALL",
            metric="compiler_owned_functions",
            value=sum(1 for r in resp if r["compiler"] == "owns"),
            unit="functions",
            artifact="responsibility_split_matrix.csv",
            phase="P12",
            evidence=_DERIVED,
            implication="functions the compiler owns in the HW/SW split",
        )
    # P11 knobs
    knobs = _yaml(cs_dir / "dse_search_space_knobs.yaml")
    if isinstance(knobs, dict):
        grp = knobs.get("dse_search_space_knobs", {}).get("knob_groups", [])
        add(
            workload="ALL",
            metric="enabled_knob_groups",
            value=sum(1 for g in grp if g.get("enabled")),
            unit="groups",
            artifact="dse_search_space_knobs.yaml",
            phase="P11",
            evidence=_DERIVED,
            implication="enabled DSE search-space knob groups (P5-P12)",
        )
    # P4 compiler proofs (KEY: proof status distribution)
    cpm = _csv_rows(cs_dir / "compiler_proof_matrix.csv")
    if cpm:
        for st in ("proven_for_workload", "assumed", "unknown"):
            add(
                workload="ALL",
                metric=f"compiler_proofs_{st}",
                value=sum(1 for r in cpm if r["status"] == st),
                unit="axes",
                artifact="compiler_proof_matrix.csv",
                phase="P4",
                evidence=_DERIVED,
                implication=f"abstraction axes with compiler proof status = {st}",
            )
    count(
        "abstraction_pressure_ranking.csv",
        None,
        "P4",
        "n_abstraction_axes_ranked",
        "abstraction axes by cross-workload pressure",
    )
    count(
        "abstraction_pressure_table.csv",
        None,
        "P4",
        "n_abstraction_pressure_rows",
        "per-workload abstraction pressure rows",
    )
    count("workload_family_table.csv", None, "P4", "n_workload_families", "workloads clustered into families")
    count("resident_state_table.csv", None, "P4", "n_resident_states", "states with a residency lifetime")
    count("requirements_table.csv", None, "P3", "n_requirements", "hardware-independent requirements derived")
    count("cross_workload_provenance.csv", None, "P1", "n_provenance_rows", "flat-vs-recovered provenance items")
    count(
        "operator_cluster_table.csv",
        None,
        "P5",
        "n_shape_clusters",
        "cross-workload geometry clusters",
        evidence="recovered_from_ir",
    )
    count(
        "operator_cluster_to_hierarchy.csv",
        None,
        "P7",
        "n_cluster_hierarchy_maps",
        "shape-cluster -> hierarchy-option mappings",
    )
    count(
        "multi_rate_contract.yaml",
        "multi_rate_contract",
        "P6",
        "n_multi_rate_workloads",
        "workloads with a recovered multi-rate contract",
        sub="workloads",
    )
    count(
        "numerical_epilogue_candidates.yaml",
        "numerical_epilogue_candidates",
        "P10",
        "n_fused_candidates",
        "fused-epilogue abstraction candidates",
        sub="candidates",
    )
    count(
        "lost_numerical_contracts.csv",
        None,
        "P10",
        "n_erased_numerical_contracts",
        "numerical contracts erased by the capture (scoped limitation)",
        evidence="unavailable",
    )
    count(
        "processing_unit_candidates.yaml",
        "processing_unit_candidates",
        "P7",
        "n_unit_candidates",
        "processing-unit candidates",
        sub="units",
    )
    count(
        "boundary_candidate_contracts.yaml",
        "boundary_candidate_contracts",
        "P12",
        "n_boundary_certificates",
        "HW/SW boundary-placement certificates",
        sub="certificates",
    )
    count(
        "pipeline_candidates.yaml",
        "pipeline_candidates",
        "P8",
        "n_pipeline_candidate_workloads",
        "workloads with pipeline overlap candidates",
        sub="workloads",
    )
    count(
        "pipeline_envelope.yaml",
        "pipeline_envelope",
        "P8",
        "n_pipeline_envelope_workloads",
        "workloads with a multi-rate phase model",
        sub="workloads",
        evidence="recovered_from_prov_fqn",
    )
    # memory_hierarchy_envelope regions
    mh = _yaml(cs_dir / "memory_hierarchy_envelope.yaml")
    if isinstance(mh, dict):
        regs = sum(len(x.get("regions", [])) for x in mh.get("memory_hierarchy_envelope", {}).get("workloads", []))
        add(
            workload="ALL",
            metric="n_memory_regions",
            value=regs,
            unit="regions",
            artifact="memory_hierarchy_envelope.yaml",
            phase="P9",
            evidence="recovered_from_ir",
            implication="per-region memory envelopes",
        )
    # sharding_opportunities required abstractions
    so = _yaml(cs_dir / "sharding_opportunities.yaml")
    if isinstance(so, dict):
        ra = so.get("sharding_opportunities", {}).get("required_abstractions", [])
        add(
            workload="ALL",
            metric="n_sharding_abstractions",
            value=len(ra),
            unit="abstractions",
            artifact="sharding_opportunities.yaml",
            phase="P7",
            evidence=_DERIVED,
            implication="reduction/broadcast abstractions sharding requires",
        )
    # accuracy_gate_report summary (corroborates the measured results)
    if (cs_dir / "accuracy_gate_report.md").is_file():
        npass = (cs_dir / "accuracy_gate_report.md").read_text().lower().count("pass")
        add(
            workload="ALL",
            metric="accuracy_gate_report_present",
            value="yes",
            unit="",
            artifact="accuracy_gate_report.md",
            phase="P4",
            evidence="measured",
            corroborated_by=2,
            implication="measured int8 accuracy summary (real measurement)",
        )
    # operator_geometry total operators (corroborates operator_shape_table)
    og = _yaml(cs_dir / "operator_geometry.yaml")
    if isinstance(og, dict):
        tot = sum(x.get("n_operators", 0) for x in og.get("operator_geometry", {}).get("workloads", []))
        add(
            workload="ALL",
            metric="total_operators",
            value=tot,
            unit="ops",
            artifact="operator_geometry.yaml",
            phase="P5",
            evidence="recovered_from_ir",
            corroborated_by=2 if _csv_rows(cs_dir / "operator_shape_table.csv") else 1,
            implication="total matmul operators across workloads",
        )
    # workload_contract_graph node count
    wcg = _yaml(cs_dir / "workload_contract_graph.yaml")
    if isinstance(wcg, dict):
        nodes = sum(len(g.get("nodes", [])) for g in wcg.get("workload_contract_graph", {}).get("graphs", []))
        add(
            workload="ALL",
            metric="contract_graph_nodes",
            value=nodes,
            unit="nodes",
            artifact="workload_contract_graph.yaml",
            phase="P6",
            evidence="recovered_from_ir",
            implication="multi-rate contract graph size",
        )
    # resource pressure dense vs skinny
    rp = {r["resource_class"]: r for r in _csv_rows(cs_dir / "resource_pressure_table.csv")}
    if rp:
        for rc in ("dense_gemm", "skinny_gemm_or_gemv"):
            if rc in rp:
                add(
                    workload="ALL",
                    metric=f"mac_fraction_{rc}",
                    value=rp[rc]["mac_fraction"],
                    unit="MAC-fraction",
                    artifact="resource_pressure_table.csv",
                    phase="P7",
                    evidence="recovered_from_ir",
                    implication="distinct compute family -> specialized vs monolithic units",
                )
    # processing unit multiplicity verdict
    pug = _yaml(cs_dir / "processing_unit_guidance.yaml")
    if isinstance(pug, dict):
        impl = pug.get("processing_unit_guidance", {}).get("search_space_implication", "")
        add(
            workload="ALL",
            metric="unit_multiplicity_implication",
            value="heterogeneous" if "heterogeneous" in impl.lower() else "see report",
            unit="",
            artifact="processing_unit_guidance.yaml",
            phase="P8",
            evidence=_DERIVED,
            implication="monolithic vs replicated vs specialized search-space implication",
        )
    # primitive regret (broad coverage + spread)
    reg = _csv_rows(cs_dir / "primitive_regret_table.csv")
    for r in sorted(reg, key=lambda x: -float(x["coverage_under_10pct"]))[:2]:
        add(
            workload="ALL",
            abstraction=r["primitive"],
            metric="coverage_under_10pct",
            value=r["coverage_under_10pct"],
            unit="MAC-fraction",
            artifact="primitive_regret_table.csv",
            phase="P5",
            evidence=_DERIVED,
            corroborated_by=2 if _csv_rows(cs_dir / "primitive_coverage_matrix.csv") else 1,
            caveat="structural coverage, not a performance metric",
            implication="broadly-covering primitive for the DSE search space",
        )
        add(
            workload="ALL",
            abstraction=r["primitive"],
            metric="max_regret",
            value=r["max_regret"],
            unit="MAC-fraction",
            artifact="primitive_regret_table.csv",
            phase="P5",
            evidence=_DERIVED,
            implication="cross-workload coverage spread (overfit risk if high)",
        )
    # cap: emit the top-5 measurements by candidates-unblocked (not all 35 rows -> de-noise)
    mp = sorted(_csv_rows(cs_dir / "measurement_priority_table.csv"), key=lambda r: -int(r["n_candidates_unblocked"]))[
        :5
    ]
    for r in mp:
        add(
            workload="ALL",
            metric="measurement_unblocks",
            value=r["n_candidates_unblocked"],
            unit="candidates",
            artifact="measurement_priority_table.csv",
            phase="P4",
            evidence=_DERIVED,
            caveat=r["measurement"],
            implication="measurement that unblocks candidates",
        )
    # REAL measured data: accuracy gate + dispatch coupling
    for r in _csv_rows(cs_dir / "accuracy_gate_results.csv"):
        mdl = r.get("model", r.get("workload", "?"))
        st = r.get("status", r.get("int8_status", ""))
        add(
            workload=mdl,
            metric="accuracy_int8_w8a8",
            value=st,
            unit="band",
            artifact="accuracy_gate_results.csv",
            phase="P4",
            evidence="measured" if st else "unavailable",
            corroborated_by=2 if st else 1,
            implication="MEASURED int8 W8A8 accuracy (real measurement)",
        )
    disp = _csv_rows(cs_dir / "dispatch_coupling.csv")
    if disp:
        add(
            workload="ALL",
            metric="measured_dispatch_ratio",
            value=disp[0].get(disp[0] and list(disp[0])[-1], "see file"),
            unit="ratio",
            artifact="dispatch_coupling.csv",
            phase="P4",
            evidence="measured",
            implication="MEASURED host dispatch coupling (real runtime measurement)",
        )
    # zoo low-bit storage finding (real low-bit evidence already in the package)
    fid = cs_dir / "numerical_contract_fidelity_report.md"
    if fid.is_file():
        add(
            workload="ZOO",
            metric="lowbit_storage_dequantized_finding",
            value="present",
            unit="",
            artifact="numerical_contract_fidelity_report.md",
            phase="P4",
            evidence="recovered_from_ir",
            implication="quantized zoo stores weights low-bit but runs f32 matmuls (native low-bit "
            "compute + packed layout absent) -- real low-bit storage evidence",
        )
