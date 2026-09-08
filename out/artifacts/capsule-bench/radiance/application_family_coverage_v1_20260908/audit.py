#!/usr/bin/env python3
"""Fail-closed census of Radiance's declared application captures and search cohort.

The evidence side is restricted to ``workload_spec.applications``.  In particular, the
kernel-library comparison cohort is never admitted as application evidence.  Parsing uses Merlin's
normalized structural MLIR reader so multi-result model2MLIR generic operations are handled by the
same xDSL path as other coverage consumers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml


NEW_CAPSULE = "SY_app_elementwise_add_f32_rank3_1x113x1_l2"
EXPECTED_APPLICATIONS = (
    "smolvla_fp32_consistent",
    "smolvla_int8_consistent",
    "smolvla_denoise_step_fp32_app",
    "lstmnetvit_fp8_consistent",
    "lstmnetvit_int8_consistent",
    "lstmnetvit_int8_w8a8_consistent",
)


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "merlin/python/merlin").is_dir() and (parent / ".git").exists():
            return parent
    raise RuntimeError("could not find repository root from audit location")


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "merlin/python"))

from merlin.common import mlir_query as mq  # noqa: E402
from merlin.targetgen import model_coverage as mc  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capsule_doc(name: str) -> tuple[Path, dict]:
    hits = list((ROOT / "merlin/contract/capsules/radiance").glob(f"*/{name}/capsule.yaml"))
    if len(hits) != 1:
        raise RuntimeError(f"capsule {name!r}: expected one source document, found {len(hits)}")
    return hits[0], yaml.safe_load(hits[0].read_text(encoding="utf-8")) or {}


def census_capture(path: Path) -> dict:
    module = mq.parse(path)
    descriptors = mc.regions_from_module(module)
    operations = mc.region_ops(module)
    if len(descriptors) != len(operations):
        raise RuntimeError(f"region/operation structural join mismatch for {path}")
    families: Counter[str] = Counter()
    ops_by_family: dict[str, Counter[str]] = {}
    exact_add = 0
    for operation, region in zip(operations, descriptors):
        family = region.resolved_family() or "unclassified"
        op = mq.attr_str(operation, "prov.op") or region.op or "?"
        families[family] += 1
        ops_by_family.setdefault(family, Counter())[op] += 1
        ins = [mq.type_shape_dtype(value.type) for value in operation.operands]
        outs = [mq.type_shape_dtype(value.type) for value in operation.results]
        if (op == "add" and len(ins) >= 2 and outs
                and ins[0] == ins[1] == outs[0] == ([1, 113, 1], "f32")):
            exact_add += 1
    return {
        "sha256": _sha(path),
        "regions": len(descriptors),
        "families": dict(sorted(families.items())),
        "operations_by_family": {
            family: dict(sorted(counts.items())) for family, counts in sorted(ops_by_family.items())
        },
        "exact_add_f32_1x113x1": exact_add,
    }


def make_report() -> dict:
    started = time.monotonic()
    descriptor = ROOT / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"
    doc = yaml.safe_load(descriptor.read_text(encoding="utf-8")) or {}
    declared = tuple((doc.get("workload_spec") or {}).get("applications") or ())
    if declared != EXPECTED_APPLICATIONS:
        raise RuntimeError(f"declared application roster drift: {declared!r}")

    per_capture = {}
    total_families: Counter[str] = Counter()
    total_ops: dict[str, Counter[str]] = {}
    exact_add = 0
    for name in declared:
        path = ROOT / "out/artifacts/recaptures" / name / "model.mlir"
        if not path.is_file():
            raise RuntimeError(f"declared capture is missing: {path}")
        row = census_capture(path)
        per_capture[name] = row
        total_families.update(row["families"])
        for family, counts in row["operations_by_family"].items():
            total_ops.setdefault(family, Counter()).update(counts)
        exact_add += row["exact_add_f32_1x113x1"]

    search = list(((doc.get("grading") or {}).get("search_cohort") or {}).get("include_capsules") or ())
    baseline = [name for name in search if name != NEW_CAPSULE]
    if len(baseline) != 14 or len(search) != 15 or NEW_CAPSULE not in search:
        raise RuntimeError(
            f"expected the audited 14-member baseline plus {NEW_CAPSULE}, got {len(baseline)} + "
            f"{len(search) - len(baseline)}")
    comparison = set((((doc.get("grading") or {}).get("evaluation_cohorts") or {})
                      .get("kernel_library_comparison") or {}).get("include_capsules") or ())
    if comparison & set(search):
        raise RuntimeError("PR-kernel comparison capsule leaked into the application search cohort")

    def summarize_capsules(names: list[str]) -> tuple[dict, dict]:
        counts: Counter[str] = Counter()
        docs = {}
        for name in names:
            path, cap = _capsule_doc(name)
            family = str((cap.get("semantic") or {}).get("semantic_family") or "unclassified")
            counts[family] += 1
            docs[name] = {
                "sha256": _sha(path), "family": family,
                "source_role": cap.get("source_role"), "operation": (cap.get("operation") or {}).get("op"),
                "required_oracle_tiers": cap.get("required_oracle_tiers") or [],
            }
        return dict(sorted(counts.items())), docs

    baseline_families, baseline_docs = summarize_capsules(baseline)
    current_families, current_docs = summarize_capsules(search)
    if baseline_families != {"contraction": 14}:
        raise RuntimeError(f"14-member baseline is not contraction-only: {baseline_families}")
    if any(row["source_role"] != "model_derived" for row in current_docs.values()):
        raise RuntimeError("application search cohort contains a non-model-derived capsule")
    probe = current_docs[NEW_CAPSULE]
    if (probe["family"], probe["operation"], probe["required_oracle_tiers"]) != (
            "elementwise_map", "add", ["L0", "L1", "L2"]):
        raise RuntimeError(f"wired probe does not express the audited L2 gap: {probe}")
    if exact_add != 99:
        raise RuntimeError(f"exact add shape count drifted from 99 to {exact_add}")

    classified = sum(n for family, n in total_families.items() if family != "unclassified")
    missing_before = {family: count for family, count in total_families.items()
                      if family not in baseline_families and family != "unclassified"}
    missing_after = {family: count for family, count in total_families.items()
                     if family not in current_families and family != "unclassified"}
    return {
        "schema": "radiance_application_family_coverage_v1",
        "scope": {
            "authority": "only workload_spec.applications in the Radiance target descriptor",
            "applications": list(declared),
            "excluded_as_application_evidence": "all kernel_library_comparison / PR-kernel capsules",
            "descriptor": str(descriptor.relative_to(ROOT)),
            "descriptor_sha256": _sha(descriptor),
        },
        "capture_census": {
            "total_regions": sum(total_families.values()),
            "classified_regions": classified,
            "unclassified_regions": total_families.get("unclassified", 0),
            "families": dict(sorted(total_families.items())),
            "operations_by_family": {
                family: dict(sorted(counts.items())) for family, counts in sorted(total_ops.items())
            },
            "per_capture": per_capture,
        },
        "search_comparison": {
            "pre_wire_14_capsules": baseline,
            "pre_wire_family_counts": baseline_families,
            "pre_wire_missing_family_occurrences": dict(sorted(missing_before.items())),
            "pre_wire_classified_occurrences_in_a_represented_family": sum(
                total_families.get(family, 0) for family in baseline_families),
            "pre_wire_fraction_of_classified_in_a_represented_family": round(
                sum(total_families.get(family, 0) for family in baseline_families) / classified, 6),
            "post_wire_15_capsules": search,
            "post_wire_family_counts": current_families,
            "post_wire_missing_family_occurrences": dict(sorted(missing_after.items())),
            "capsules": current_docs,
        },
        "wired_probe": {
            "capsule": NEW_CAPSULE,
            "family": "elementwise_map",
            "operation": "add",
            "captured_shape": [1, 113, 1],
            "dtype": "f32",
            "exact_occurrences_in_declared_captures": exact_add,
            "capture_anchor": {
                "application": "smolvla_fp32_consistent", "region_index": 888,
                "region_id": "add_100",
                "fqn": "model.vlm_with_expert.vlm.model.text_model.layers.0.input_layernorm",
                "model_mlir_sha256": per_capture["smolvla_fp32_consistent"]["sha256"],
            },
            "qualification_scope": (
                "one exact f32 rank-3 add program at L2; this does not qualify other elementwise "
                "operations, shapes, dtypes, families, or physical RTL"),
            "l2_receipt": "l2_receipt.json",
        },
        "other_non_contraction_disposition": {
            "movement": {
                "occurrences": total_families.get("movement", 0), "wired": False,
                "reason": "not admitted by Radiance's effective capability map; a must_accelerate capsule would be false",
            },
            "normalization": {
                "occurrences": total_families.get("normalization", 0), "wired": False,
                "reason": "not admitted by Radiance's effective capability map; existing neutral writers describe host work here",
            },
            "reduction": {
                "occurrences": total_families.get("reduction", 0), "wired": False,
                "reason": "not admitted by Radiance's effective capability map; existing reduce_sum synthesis cannot establish accelerator eligibility",
            },
            "attention": {
                "occurrences": total_families.get("attention", 0), "wired": False,
                "reason": "not admitted by Radiance's effective capability map; no honest standalone must_accelerate claim",
            },
            "unclassified": {
                "occurrences": total_families.get("unclassified", 0), "wired": False,
                "reason": "capture semantics do not resolve a family; retained as unclassified",
            },
        },
        "fail_closed": {
            "unclassified_regions": total_families.get("unclassified", 0),
            "families_with_no_search_capsule_after_wire": dict(sorted(missing_after.items())),
            "application_regions_physically_qualified_by_this_audit": 0,
            "note": (
                "family representation is a census, not a qualification claim. Only the wired probe's "
                "exact program may inherit its L2 receipt; no GSIM result is physical RTL evidence."),
        },
        "audit_wall_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write coverage.json beside this script")
    args = parser.parse_args()
    report = make_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        Path(__file__).with_name("coverage.json").write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
