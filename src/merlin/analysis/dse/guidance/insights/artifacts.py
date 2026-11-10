"""Read case-study artifacts and inventory their explicit evidence fields.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from merlin.common.yaml import load_yaml

# artifacts the verify_implementation.py harness independently re-derives (drives verification_status
# + the tier-A/B gate). This is the P13-a "checked by verify_implementation.py" column.
_VERIFIED_ARTIFACTS = {
    "operator_shape_table.csv",
    "tile_waste_table.csv",
    "primitive_coverage_matrix.csv",
    "primitive_regret_table.csv",
    "shape_summary_by_workload.csv",
    "operator_cluster_table.csv",
    "workload_contract_graph.yaml",
    "critical_path_table.csv",
    "traffic_table.csv",
    "data_movement_table.csv",
    "dtype_capacity_table.csv",
    "dma_stream_table.csv",
    "buffer_requirement_table.csv",
    "sharding_table.csv",
    "region_attribution.yaml",
    "accuracy_gated_dtype_candidates.csv",
    "dispatch_granularity_table.csv",
    "epilogue_pattern_table.csv",
    "accumulator_contract_table.csv",
    "hw_sw_boundary_matrix.csv",
    "boundary_candidate_contracts.yaml",
    "responsibility_split_matrix.csv",
    "dse_search_space_knobs.yaml",
    "dse_contract.json",
    "resident_state_table.csv",
}


# expected root artifacts -> source phase (P13-a inventory). Per-workload subdir artifacts handled
# separately.
_ARTIFACT_PHASE = {
    "cross_workload_provenance.csv": "P1",
    "workload_contract_table.csv": "P2",
    "requirements_table.csv": "P3",
    "dse_readiness_summary.csv": "P4",
    "abstraction_pressure_table.csv": "P4",
    "abstraction_pressure_ranking.csv": "P4",
    "compiler_proof_matrix.csv": "P4",
    "resident_state_table.csv": "P4",
    "workload_family_table.csv": "P4",
    "measurement_priority_table.csv": "P4",
    "numerical_contract_fidelity_report.md": "P4",
    "accuracy_gate_results.csv": "P4",
    "operator_shape_table.csv": "P5",
    "operator_cluster_table.csv": "P5",
    "shape_summary_by_workload.csv": "P5",
    "shape_summary_by_region.csv": "P5",
    "operator_geometry.yaml": "P5",
    "tile_waste_table.csv": "P5",
    "primitive_coverage_matrix.csv": "P5",
    "primitive_regret_table.csv": "P5",
    "workload_contract_graph.yaml": "P6",
    "phase_rate_table.csv": "P6",
    "multi_rate_contract.yaml": "P6",
    "critical_path_table.csv": "P7",
    "concurrency_windows.csv": "P7",
    "sharding_table.csv": "P7",
    "sharding_opportunities.yaml": "P7",
    "operator_cluster_to_hierarchy.csv": "P7",
    "resource_pressure_table.csv": "P7",
    "processing_unit_candidates.yaml": "P7",
    "pipeline_envelope.yaml": "P8",
    "pipeline_stage_table.csv": "P8",
    "pipeline_candidates.yaml": "P8",
    "buffering_requirement_table.csv": "P8",
    "processing_unit_guidance.yaml": "P8",
    "data_movement_table.csv": "P9",
    "memory_hierarchy_envelope.yaml": "P9",
    "reuse_lifetime_table.csv": "P9",
    "dma_stream_table.csv": "P9",
    "buffer_requirement_table.csv": "P9",
    "dtype_capacity_table.csv": "P9",
    "traffic_table.csv": "P9",
    "epilogue_pattern_table.csv": "P10",
    "accumulator_contract_table.csv": "P10",
    "numerical_epilogue_candidates.yaml": "P10",
    "lost_numerical_contracts.csv": "P10",
    "dse_search_space_knobs.yaml": "P11",
    "accuracy_gated_dtype_candidates.csv": "P6",
    "hw_sw_boundary_matrix.csv": "P12",
    "boundary_candidate_contracts.yaml": "P12",
    "responsibility_split_matrix.csv": "P12",
    "boundary_dse_knobs.yaml": "P12",
    "isa_candidate_primitives.yaml": "P12",
    "runtime_object_candidates.yaml": "P12",
    "command_isa_candidates.yaml": "P12",
    "dse_contract.json": "P12",
    "accuracy_gate_report.md": "P4",
}


# --------------------------------------------------------------------------- io helpers


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(csv.DictReader(io.StringIO(path.read_text())))


def _yaml(path: Path):
    if not path.is_file():
        return None
    try:
        return load_yaml(path)
    except Exception:
        return None


def _json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _count(text: str, label: str) -> int:
    return text.count(label)


# --------------------------------------------------------------------------- P13-a inventory


def _file_meta(path: Path) -> dict:
    suffix = path.suffix.lstrip(".")
    rowcount = None
    keys = None
    schema_valid = True
    has_evidence = False
    counts = {lbl: 0 for lbl in ("unavailable", "unknown", "assumed_reference", "derived_requirement", "measured")}
    if path.is_file():
        text = path.read_text(errors="ignore")
        for lbl in counts:
            counts[lbl] = _count(text, lbl)
        has_evidence = ("evidence" in text) or ("recovered_from" in text) or ("measured" in text)
        if suffix == "csv":
            rows = _csv_rows(path)
            rowcount = len(rows)
            schema_valid = rowcount >= 0 and ("," in text.splitlines()[0] if text else True)
        elif suffix in ("yaml", "yml"):
            obj = _yaml(path)
            keys = sorted(obj)[:8] if isinstance(obj, dict) else None
            schema_valid = obj is not None
        elif suffix == "json":
            obj = _json(path)
            keys = sorted(obj)[:8] if isinstance(obj, dict) else None
            schema_valid = obj is not None
    return {
        "file_type": suffix,
        "row_count": rowcount,
        "top_level_keys": keys,
        "schema_valid": schema_valid,
        "has_evidence_fields": has_evidence,
        **counts,
    }


def artifact_inventory(cs_dir: Path) -> list[dict]:
    """One row per expected root artifact (+ any extra files found): exists, phase, type, counts."""
    rows = []
    seen = set()
    for name, phase in sorted(_ARTIFACT_PHASE.items()):
        p = cs_dir / name
        seen.add(name)
        m = _file_meta(p)
        rows.append(
            {
                "artifact": name,
                "path": str(p),
                "exists": p.is_file(),
                "source_phase": phase,
                "checked_by_verify": name in _VERIFIED_ARTIFACTS,
                **m,
            }
        )
    # any other generated root file we did not expect (so the inventory is complete)
    if cs_dir.is_dir():
        for p in sorted(cs_dir.glob("*")):
            if p.is_file() and p.name not in seen:
                m = _file_meta(p)
                rows.append(
                    {
                        "artifact": p.name,
                        "path": str(p),
                        "exists": True,
                        "source_phase": "other",
                        "checked_by_verify": False,
                        **m,
                    }
                )
    return rows


# --------------------------------------------------------------------------- P13-b unified facts


def case_workloads(cs_dir: Path) -> list[str]:
    man = _json(cs_dir / "dse_contract.json")
    if man and man.get("workloads"):
        return list(man["workloads"])
    return sorted({r["workload"] for r in _csv_rows(cs_dir / "critical_path_table.csv") if r.get("workload")})


def _inventory_md(inv) -> str:
    present = sum(1 for r in inv if r["exists"])
    L = [
        "# Artifact inventory\n",
        f"{present}/{len(inv)} expected artifacts present. Missing artifacts are recorded explicitly (partial mode).\n",
        "| artifact | exists | phase | type | rows | checked_by_verify | unavailable |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(inv, key=lambda x: (not x["exists"], x["source_phase"], x["artifact"])):
        L.append(
            f"| {r['artifact']} | {r['exists']} | {r['source_phase']} | {r['file_type']} | "
            f"{r['row_count'] if r['row_count'] is not None else '—'} | "
            f"{r['checked_by_verify']} | {r['unavailable']} |"
        )
    return "\n".join(L) + "\n"
