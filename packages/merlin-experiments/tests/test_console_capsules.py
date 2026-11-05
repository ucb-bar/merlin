"""Console capsule catalog, performance, golden and pass sources read real record shapes, read-only."""

import json
import time

import pytest
import yaml
from merlin_experiments.tracking.console.io import Reader
from merlin_experiments.tracking.console.model import Context
from merlin_experiments.tracking.console.sources import capsules, golden, passes, perf
from merlin_experiments.tracking.terminal_capsules import full_source

SOURCE = "a" * 64
OTHER = "b" * 64


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value)
    elif path.suffix == ".yaml":
        path.write_text(yaml.safe_dump(value, sort_keys=False))
    else:
        path.write_text(json.dumps(value))


def _performance(cycles, floor=64, bound=True):
    digests = {
        k: k[0] * 64
        for k in (
            "capsule_sha256",
            "facts_sha256",
            "selection_sha256",
            "model_sha256",
            "comparison_domain_sha256",
            "scope_sha256",
        )
    }
    binding = dict(digests) if bound else {**digests, "model_sha256": "f" * 64}
    return {
        "schema": "merlin.phase1.joint_performance.v1",
        "status": "PROVISIONAL",
        "estimated_whole_soc_cycles": cycles,
        "complete_event_accounting": True,
        "structural_floor_to_estimate_ratio": floor / cycles,
        "structural_comparison_binding": binding,
        "structural_roofline": {
            "useful_macs": 4096,
            "structural_peak_macs_per_cycle": 256,
            "structural_compute_floor_cycles": floor,
            "logical_request_ideal_service": {"read": {"requested_bytes": 768}, "write": {"requested_bytes": 256}},
        },
        "modeled_resource_activity": {"resources": [{"resource_id": 3, "busy_unit_cycles": cycles // 2}]},
        "timing_mode": "untuned",
        **digests,
    }


@pytest.fixture()
def experiment(tmp_path):
    run, workspace, launch = tmp_path / "run", tmp_path / "ws" / "workspace", tmp_path / "launch"
    corpus = run / "private_corpus_input"
    _write(
        corpus / "public/capA/capsule.yaml",
        {
            "name": "capA",
            "kind": "layer",
            "source_role": "model_derived",
            "source_reference": "layer 3 of a cnn",
            "inputs": [
                {"name": "W", "shape": [64, 32], "dtype": "i8"},
                {"name": "IFM", "shape": [1, 8, 8, 64], "dtype": "i8"},
            ],
            "operation": {
                "op": "conv2d",
                "attributes": {"ci": 64, "kh": 3, "kw": 3, "stride": [2, 2], "ifm": "IFM", "output_dtype": "i32"},
            },
            "semantic": {"semantic_family": "contraction", "generalization_axis": "composition"},
            "pass_requirements": ["tile-schedule", "target-isa-lowering"],
            "required_oracle_tiers": ["L0", "L2", "L3"],
        },
    )
    _write(
        corpus / "scalability/cases/capB/capsule.yaml",
        {
            "name": "capB",
            "kind": "model",
            "inputs": [{"name": "x", "shape": [1, 2048], "dtype": "i8"}],
            "operation": {"op": "model", "attributes": {}},
            "pass_requirements": ["tile-schedule"],
        },
    )
    _write(
        corpus / "scalability/cases/capB/source-shape-binding.json",
        {
            "shape_mkn": [1, 2048, 32000],
            "model": "resnet50",
            "program": "decode",
            "operation": "aten._int_mm.default",
            "occurrences": [{}, {}],
            "full_model": True,
        },
    )
    selection = tmp_path / "scale"
    _write(
        selection / "selection.json",
        {"schema": "merlin.corpus.public_scalability.staged.v1", "cases": {"capB": "c" * 64}},
    )
    _write(
        run / "environment.yaml",
        {"public_corpus_input": {"public_scalability": {"path": str(selection)}}, "workspace_path": str(workspace)},
    )
    _write(run / "joint_campaign.json", {"policy": {"correctness_curriculum": {"original_capsules": ["capA"]}}})
    _write(
        workspace / "qa/progress.json",
        {
            "updated_at": "2026-10-23T10:00:00+00:00",
            "per_capsule": [
                {
                    "capsule": "capA",
                    "status": "incomplete",
                    "numeric_status": "pass",
                    "tiers": {"L0": "pass", "L2": "pass", "L3": "unavailable"},
                    "tier_cycles": {"L2": 900},
                    "failure_plane": "oracle_unavailable",
                    "failure_category": "NOT_RUN_IS_NOT_PASS",
                    "failure_tier": "L3",
                    "tier_reasons": {"L3": "no adapter"},
                }
            ],
        },
    )
    _write(
        workspace / "qa/tier_state.json",
        {
            "capA": {"L2": {"status": "pass", "components": {"parse": "x", "lower": "y", "<unattributed>": "z"}}},
            "capB": {"L2": {"status": "fail", "components": {"parse": "x"}}},
        },
    )
    channel = workspace / ".qa_channel"
    _write(
        channel / "resp_2_1792000000000000000_aa.json",
        {
            "submission_sha256": SOURCE,
            "graded_at_unix_ns": 1792000000000000000,
            "cases": [
                {"capsule": "capB", "status": "pass", "performance": _performance(4096)},
                {"capsule": "capC", "status": "pass", "performance": _performance(5000, bound=False)},
                {"capsule": "capA", "status": "pass", "performance": _performance(700, floor=7)},
            ],
        },
    )
    _write(
        channel / "simresp_promo0_1_capB.json",
        {
            "per_capsule": [
                {
                    "capsule": "capB",
                    "barrier_cycles": 5120,
                    "barrier_tier": "L3",
                    "barrier_status": "pass",
                    "barrier_timing_identity": {"engine": "gsim"},
                    "tier_boundary": {"common": {"compiler_sha256": SOURCE}},
                },
                {
                    "capsule": "capA",
                    "barrier_cycles": 777,
                    "barrier_tier": "L3",
                    "barrier_status": "pass",
                    "barrier_timing_identity": {"engine": "gsim"},
                    "tier_boundary": {"common": {"compiler_sha256": OTHER}},
                },
            ]
        },
    )
    _write(
        workspace / "submission/manifest.yaml",
        {
            "commands": {"parse": {"argv": ["python3", "{tool}"]}, "lower": {"argv": ["x"]}, "emit": {"argv": ["y"]}},
            "shared_pass_contract": {
                "families": [
                    {
                        "name": "fuse",
                        "boundary": "computation",
                        "status": "unknown",
                        "requirements": ["ownership"],
                        "reason": "not supplied",
                    }
                ],
                "controls": {"choices": {"dma_coalescing": {"enabled": [False, True]}}},
                "boundaries": {"computation": {"command": "lower"}},
            },
        },
    )
    _write(
        run / "OPTIMIZATION_CATALOG.json",
        {
            "mechanisms": [
                {"id": "capacity_tiles", "facet": "storage", "compiler": {"module": "m.capacity", "symbol": "fit"}}
            ]
        },
    )
    _write(run / "COMPILER_LIBRARY_API.json", {"public_modules": [{}, {}, {}]})
    _write(
        launch / "process/native.log",
        "\n".join(
            [
                "[promote] capA L2 invalidated by parse (changed)",
                "[promote] capA L3 invalidated by parse (changed)",
                "[promote] capB L2 invalidated by lower (changed)",
                "[promote] capA L2 pass -> L3: [capA]",
                "",
            ]
        ),
    )
    return {
        "id": str(run),
        "run_id": "run",
        "run_dir": str(run),
        "workspace": str(workspace),
        "launch_dir": str(launch),
        "state": "FINISHED",
    }


def _ctx(experiment, private=True, shared=None):
    return Context(
        experiment=experiment,
        reader=Reader(),
        state={},
        operator_private=private,
        now=time.time(),
        shared=shared if shared is not None else {},
    )


def test_catalog_carries_shapes_origin_cohort_and_newest_result(experiment):
    section = capsules.read(_ctx(experiment))
    rows = {r["name"]: r for r in section["capsules"]}
    a, b = rows["capA"], rows["capB"]
    assert a["cohort"] == "functional" and b["cohort"] == "scalability"
    assert a["shapes"] == ["W:64x32 i8", "IFM:1x8x8x64 i8"] and "ci=64" in a["attrs"] and "stride=2x2" in a["attrs"]
    assert "ifm" not in a["attrs"] and a["dtype"] == "i32" and a["family"] == "contraction"
    assert a["tiers"]["L2"] == "pass" and a["cycles"]["L2"] == 900 and a["cycles"]["L3"] is None
    assert a["failure"] == {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS", "tier": "L3"}
    assert b["origin"]["model"] == "resnet50" and b["origin"]["aten_op"] == "aten._int_mm.default"
    assert b["origin"]["occurrences"] == 2 and b["origin"]["full_model"] is True
    assert b["status"] is None and b["tiers"] == {"L2": "fail"}  # ledger only; no result row is not a pass
    assert section["counts"]["pass_requirements"] == {"tile-schedule": 2, "target-isa-lowering": 1}
    detail = capsules.inspect(_ctx(experiment), "capA")
    text = json.dumps(detail)
    assert "no adapter" in text and "input IFM:1x8x8x64 i8" in text


def test_public_mode_hides_capsule_names_but_keeps_counts(experiment):
    section = capsules.read(_ctx(experiment, private=False))
    assert section["capsules"] == [] and section["counts"]["cohort"]["functional"] == 1
    assert "capA" not in json.dumps(section)
    assert capsules.inspect(_ctx(experiment, private=False), "capA")["sections"] == []
    assert perf.read(_ctx(experiment, private=False))["rows"] == []
    assert golden.read(_ctx(experiment, private=False))["references"] == []


def test_perf_binding_refusal_and_divergence_only_on_one_source(experiment):
    shared = {"capsules": capsules.read(_ctx(experiment))}
    rows = {r["capsule"]: r for r in perf.read(_ctx(experiment, shared=shared))["rows"]}
    b, c, a = rows["capB"], rows["capC"], rows["capA"]
    assert b["spike_model_cycles"] == 4096 and b["gsim_cycles"] == 5120
    assert b["comparable"] and b["divergence_pct"] == 25.0
    assert b["floor_cycles"] == 64 and b["intensity"] == 4.0 and b["achieved_macs_per_cycle"] == 1.0
    assert b["top_resource"]["activity_pct"] == 50.0 and "capacity not recorded" in b["top_resource"]["basis"]
    assert c["spike_model_cycles"] is None and c["floor_cycles"] is None  # mismatched binding is refused
    assert a["gsim_cycles"] == 777 and a["spike_model_cycles"] == 700 and not a["comparable"]
    assert a["divergence_pct"] is None and "different compiler sources" in a["basis"]
    assert a["l2_cycles"] == 900


def test_golden_refuses_across_identities_and_lists_the_ledger(experiment, tmp_path, monkeypatch):
    from merlin.perf import target_reference

    ledger_path = tmp_path / "synthetic-reference-ledger.yaml"
    _write(
        ledger_path,
        {
            "references": {
                "synthetic_measured": {
                    "model": "resnet50",
                    "program_identity": "UNKNOWN",
                    "device": "UNKNOWN",
                    "status": "external_measured",
                    "measured": {"whole_model_cycles": 8192},
                },
                "synthetic_unmeasured": {
                    "model": "resnet50",
                    "program_identity": "UNKNOWN",
                    "device": "UNKNOWN",
                    "status": "unmeasured",
                },
            }
        },
    )
    ledger = target_reference.load_ledger(ledger_path)
    monkeypatch.setattr(target_reference, "load_ledger", lambda: ledger)
    shared = {"capsules": capsules.read(_ctx(experiment))}
    shared["perf"] = perf.read(_ctx(experiment, shared=shared))
    section = golden.read(_ctx(experiment, shared=shared))
    names = {r["name"]: r for r in section["references"]}
    assert names["synthetic_measured"]["cycles"] == 8192
    assert names["synthetic_unmeasured"]["cycles"] is None
    rows = [c for c in section["comparisons"] if c["subject"] == "capB"]
    assert rows and all(not c["comparable"] and c["ratio"] is None for c in rows)
    assert all(c["candidate_cycles"] == 5120 for c in rows)
    assert any("program identity" in c["reason"] for c in rows)


def test_pass_matrix_counts_exercise_churn_and_requirements(experiment):
    shared = {"capsules": capsules.read(_ctx(experiment))}
    section = passes.read(_ctx(experiment, shared=shared))
    rows = {(r["kind"], r["name"]): r for r in section["passes"]}
    assert section["pipeline"] == ["parse", "lower", "emit"]
    assert rows[("command", "parse")]["exercised_by"] == 2 and rows[("command", "parse")]["churn"] == 2
    assert rows[("command", "lower")]["churn"] == 1 and rows[("command", "emit")]["churn"] == 0
    assert rows[("family", "fuse")]["status"] == "unknown" and rows[("control", "dma_coalescing")]["used"] is True
    assert rows[("requirement", "tile-schedule")]["required_by"] == 2
    assert rows[("mechanism", "capacity_tiles")]["source"] == "m.capacity"
    assert section["counts"]["library_modules"] == 3 and section["counts"]["profiled"] == 0
    assert passes.churn(["[promote] x L3 invalidated by emit (changed)", "noise"]) == {"emit": {"L3": 1}}
    detail = json.dumps(passes.inspect(_ctx(experiment, shared=shared), "parse"))
    assert "invalidated L2 results 1 times" in detail


def test_full_source_is_structural():
    assert full_source(SOURCE) == SOURCE
    assert full_source("A" * 64) is None and full_source("a" * 63) is None and full_source(None) is None
