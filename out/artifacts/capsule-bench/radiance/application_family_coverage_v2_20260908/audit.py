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


ADD_CAPSULE = "SY_app_elementwise_add_f32_rank3_1x113x1_l2"
MUL_CAPSULE = "SY_app_elementwise_mul_f32_rank1_32_l2"
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
    exact_mul = 0
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
        if (op == "mul" and len(ins) >= 2 and outs
                and ins[0] == ins[1] == outs[0] == ([32], "f32")):
            exact_mul += 1
    return {
        "sha256": _sha(path),
        "regions": len(descriptors),
        "families": dict(sorted(families.items())),
        "operations_by_family": {
            family: dict(sorted(counts.items())) for family, counts in sorted(ops_by_family.items())
        },
        "exact_add_f32_1x113x1": exact_add,
        "exact_mul_f32_32": exact_mul,
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
    exact_mul = 0
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
        exact_mul += row["exact_mul_f32_32"]

    search = list(((doc.get("grading") or {}).get("search_cohort") or {}).get("include_capsules") or ())
    baseline = [name for name in search if name not in (ADD_CAPSULE, MUL_CAPSULE)]
    after_add = [name for name in search if name != MUL_CAPSULE]
    if (len(baseline) != 14 or len(after_add) != 15 or len(search) != 16
            or ADD_CAPSULE not in search or MUL_CAPSULE not in search):
        raise RuntimeError(
            "expected the audited 14-member baseline plus exact add and mul capsules, got "
            f"{len(baseline)} + {len(search) - len(baseline)}")
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
    after_add_families, after_add_docs = summarize_capsules(after_add)
    current_families, current_docs = summarize_capsules(search)
    if baseline_families != {"contraction": 14}:
        raise RuntimeError(f"14-member baseline is not contraction-only: {baseline_families}")
    if any(row["source_role"] != "model_derived" for row in current_docs.values()):
        raise RuntimeError("application search cohort contains a non-model-derived capsule")
    add_probe = current_docs[ADD_CAPSULE]
    if (add_probe["family"], add_probe["operation"], add_probe["required_oracle_tiers"]) != (
            "elementwise_map", "add", ["L0", "L1", "L2"]):
        raise RuntimeError(f"wired add probe does not express the audited L2 gap: {add_probe}")
    mul_probe = current_docs[MUL_CAPSULE]
    if (mul_probe["family"], mul_probe["operation"], mul_probe["required_oracle_tiers"]) != (
            "elementwise_map", "mul", ["L0", "L1", "L2"]):
        raise RuntimeError(f"wired mul probe does not express the audited L2 gap: {mul_probe}")
    if exact_add != 99:
        raise RuntimeError(f"exact add shape count drifted from 99 to {exact_add}")
    if exact_mul != 168:
        raise RuntimeError(f"exact mul shape count drifted from 168 to {exact_mul}")

    from merlin.targetgen import compute_units as CU
    contract_path = ROOT / "merlin/targets/muon/contracts/target_contract.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    units = CU.compute_units(contract)
    effective = CU.semantic_capability_map(units)
    elementwise = effective.get("elementwise_map")
    if elementwise is None or elementwise.composed_with or "float32" not in elementwise.dtypes:
        raise RuntimeError(
            "effective Muon capability map does not admit standalone float32 elementwise_map")

    classified = sum(n for family, n in total_families.items() if family != "unclassified")
    missing_before = {family: count for family, count in total_families.items()
                      if family not in baseline_families and family != "unclassified"}
    missing_after = {family: count for family, count in total_families.items()
                     if family not in current_families and family != "unclassified"}
    return {
        "schema": "radiance_application_family_coverage_v2",
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
            "after_add_15_capsules": after_add,
            "after_add_family_counts": after_add_families,
            "post_wire_16_capsules": search,
            "post_wire_family_counts": current_families,
            "post_wire_missing_family_occurrences": dict(sorted(missing_after.items())),
            "capsules": current_docs,
        },
        "selection": {
            "highest_frequency_missing_family": {"family": "movement", "occurrences": 3469,
                "admitted": False, "reason": "absent from effective capability map"},
            "highest_frequency_safe_unrepresented_operation": {
                "family": "elementwise_map", "operation": "mul", "occurrences": 1533,
                "admitted": True, "reason": "standalone float32 elementwise_map is effective"},
        },
        "effective_capability": {
            "contract": str(contract_path.relative_to(ROOT)),
            "contract_sha256": _sha(contract_path),
            "family": elementwise.family,
            "dtypes": list(elementwise.dtypes),
            "ranks": list(elementwise.ranks),
            "rank_interpretation": "empty means any rank",
            "engines": list(elementwise.engines),
            "composed_with": list(elementwise.composed_with),
            "providers": [list(row) for row in CU.providers_of("elementwise_map", units)],
        },
        "wired_probes": {
            "add": {
                "capsule": ADD_CAPSULE, "family": "elementwise_map", "operation": "add",
                "captured_shape": [1, 113, 1], "dtype": "f32",
                "exact_occurrences_in_declared_captures": exact_add,
                "capture_anchor": {
                    "application": "smolvla_fp32_consistent", "region_index": 888,
                    "region_id": "add_100",
                    "fqn": "model.vlm_with_expert.vlm.model.text_model.layers.0.input_layernorm",
                    "model_mlir_sha256": per_capture["smolvla_fp32_consistent"]["sha256"],
                },
            },
            "mul": {
                "capsule": MUL_CAPSULE, "family": "elementwise_map", "operation": "mul",
                "captured_shape": [32], "dtype": "f32",
                "exact_occurrences_in_declared_captures": exact_mul,
                "capture_anchor": {
                    "application": "smolvla_fp32_consistent", "region_index": 904,
                    "region_id": "mul_22", "prov_aten": "aten.mul.Tensor",
                    "model_mlir_sha256": per_capture["smolvla_fp32_consistent"]["sha256"],
                },
                "l2_receipt": "l2_receipt.json",
            },
            "qualification_scope": (
                "two exact f32 add/mul programs at L2; this does not qualify other elementwise "
                "operations, shapes, dtypes, families, or physical RTL"),
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
                "family representation is a census, not a qualification claim. Only each wired probe's "
                "exact program may inherit its own L2 receipt; no GSIM result is physical RTL evidence."),
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
