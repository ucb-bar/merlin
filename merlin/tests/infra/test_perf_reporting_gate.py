"""Arm4 reports consume only sealed exact-cell measurements and state their non-claims."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import copy
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root


SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


import perf_reporting as PR  # noqa: E402  (experiment-local module under test)
import perf_agent_stage as PAS  # noqa: E402
import perf_campaign as PC  # noqa: E402
import perf_pk_claim as PK  # noqa: E402

AR = _load("assemble_results")
GR = _load("gen_perf_report")
RR = _load("run_perf_bench")

_HANDOFFS: dict[str, SimpleNamespace] = {}


@pytest.fixture(autouse=True)
def _verified_candidate_handoffs(monkeypatch: pytest.MonkeyPatch):
    def verify(path: Path, **_kwargs):
        return _HANDOFFS[str(Path(path).resolve(strict=True))]

    monkeypatch.setattr(PAS, "verify_candidate_handoff", verify)
    monkeypatch.setattr(
        PC, "verify_functional_host_lane_snapshot",
        lambda host: (dict(host["run_snapshot"]), dict(host), Path(host["resolved_package"])))


def _canonical(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rows() -> list[dict]:
    rows: list[dict] = []
    for index, capsule in enumerate(("PK00_k8", "PK01_k16")):
        common = {
            "approach": "arm4", "correct": True, "tier_status": "pass",
            "grade_status": "pass", "numeric_status": "pass",
        }
        rows += [
            {
                **common,
                "identity": {"family": "PK", "capsule": capsule,
                             "simulator": "spike", "replicate": "r000"},
                "tier": "L2", "purpose": "correctness_screen", "citable": False,
                "cycles": None,
            },
            {
                **common,
                "identity": {"family": "PK", "capsule": capsule,
                             "simulator": "verilator", "replicate": "r000"},
                "tier": "L3", "purpose": "performance_certification", "citable": True,
                "cycles": 101 + index,
            },
        ]
    return rows


def _completion() -> dict:
    return {
        "expected": 4, "reported": 4, "correct": 4, "failed": 0, "missing": 0,
        "screen_expected": 2, "screen_passed": 2, "citable_expected": 2,
        "citable_measured": 2, "citable_passed": 2, "complete": True,
    }


def _campaign() -> dict:
    digest = "d" * 64
    probes = [{"label": "python3", "returncode": 0},
              {"label": "verilator", "returncode": 0}]
    rows = _rows()
    return {
        "status": "GO", "measurement_status": "GO", "claim_status": "NOT_ESTABLISHED",
        "experiment_mode": "measurement_smoke_only", "claim_launch_status": "NOT_REQUESTED",
        "claim_launch_blocker": PC.SMOKE_CLAIM_NONCLAIM,
        "claim_preflight": None, "claim_decision": None,
        "approach": "arm4", "refusal": None,
        "functional_run_id": "arm4_functional_exact",
        "functional_submission_sha256": digest, "snapshot_sha256": digest,
        "fork_before": {"state": "held", "ok": True},
        "fork_after": {"state": "held", "ok": True},
        "decision_boundary": {
            "module": "perf_pk_claim",
            "identity_bridge": "not_invoked_in_measurement_smoke",
            "promotion_integration": "blocked",
            "promotion_status": "BLOCKED",
            "reason": PC.SMOKE_CLAIM_NONCLAIM,
        },
        "sandbox": {
            "engine": "bwrap", "network": "available", "package_read_only": True,
            "answer_surface_coverage_gap": [],
            "required_tool_probes": [row["label"] for row in probes],
            "tool_probe_results": probes,
        },
        "expected_identities": [row["identity"] for row in rows],
        "completion": _completion(),
    }


def _write_read_only(path: Path, document: object) -> tuple[str, int]:
    payload = _canonical(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return _sha(payload), len(payload)


def _seal_results(run: Path, campaign: dict, rows: list[dict]) -> None:
    results_path = run / "perf_results.json"
    digest_path = run / "perf_results.digest.json"
    for path in (results_path, digest_path):
        if path.exists():
            path.chmod(0o644)
    result_sha, n_bytes = _write_read_only(results_path, rows)
    record = {
        "path": "perf_results.json", "sha256": result_sha, "n_bytes": n_bytes,
        "records": len(rows), "verified": True, "immutable": True,
    }
    digest_sha, _ = _write_read_only(digest_path, record)
    campaign["results"] = {**record, "digest_record_sha256": digest_sha}


def _write_campaign(run: Path, campaign: dict) -> None:
    manifest = run / "campaign_manifest.json"
    digest = run / "campaign_manifest.digest.json"
    if manifest.exists():
        manifest.chmod(0o644)
    if digest.exists():
        digest.chmod(0o644)
        digest.unlink()
    manifest.write_bytes(_canonical(campaign))
    PC.seal_campaign_manifest(manifest, campaign)


def _freeze_workload(run: Path, campaign: dict) -> None:
    root = run / "_frozen_workload"
    capsules_root = root / "capsules"
    rows = []
    for capsule in ("PK00_k8", "PK01_k16"):
        path = capsules_root / "_perf" / capsule
        path.mkdir(parents=True)
        (path / "capsule.yaml").write_text(f"name: {capsule}\n")
        (path / "capsule.interface.mlir").write_text("module {}\n")
        for entry in path.iterdir():
            entry.chmod(0o444)
        path.chmod(0o555)
        record = PR._exact_tree_record(path)
        rows.append({
            "family": "PK", "capsule": capsule,
            "snapshot_relative_path": f"capsules/_perf/{capsule}",
            "source_sha256": record["sha256"], "snapshot_sha256": record["sha256"],
            "n_files": record["n_files"], "n_bytes": record["n_bytes"],
            "performance": {"family": "PK", "claim": "PREDICTS"},
        })
    (capsules_root / "_perf").chmod(0o555)
    capsules_root.chmod(0o555)
    aggregate = PR._exact_tree_record(capsules_root)
    manifest = {
        "schema_version": 1, "target": "fixture",
        "counts": {"capsules": 2, "families": 1, "files": aggregate["n_files"],
                   "bytes": aggregate["n_bytes"]},
        "capsules_sha256": aggregate["sha256"], "capsules": rows,
    }
    manifest_path = root / "performance_corpus_manifest.json"
    manifest_sha, _ = _write_read_only(manifest_path, manifest)
    root.chmod(0o555)
    campaign.update({
        "workload_snapshot": str(root), "workload_manifest": str(manifest_path),
        "workload_manifest_sha256": manifest_sha, "workload_sha256": aggregate["sha256"],
    })


def _formal_descriptor(index: int, k: int) -> dict:
    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text())
    performance = copy.deepcopy(next(
        row for row in profile["sweeps"] if row["id"] == "PK")["base"]["performance"])
    return {
        "name": f"PK{index:02d}_k{k}", "kind": "model_slice", "label": "dev",
        "source_role": "derived_sweep",
        "inputs": [
            {"name": "W", "role": "weight", "shape": [k, 16], "dtype": "i8"},
            {"name": "A0", "role": "input", "shape": [16, k], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {
            "lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [],
            "output_dtype": "i32",
        }},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "performance": performance,
    }


def _formal_rows(descriptors: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for descriptor in descriptors:
        k = descriptor["inputs"][1]["shape"][1]
        for replicate in ("r000", "r001", "r002"):
            common = {
                "approach": "arm4", "correct": True, "tier_status": "pass",
                "grade_status": "pass", "numeric_status": "pass",
            }
            rows.extend((
                {**common, "identity": {
                    "family": "PK", "capsule": descriptor["name"],
                    "simulator": "spike", "replicate": replicate,
                 }, "tier": "L2", "purpose": "correctness_screen", "citable": False,
                 "cycles": None},
                {**common, "identity": {
                    "family": "PK", "capsule": descriptor["name"],
                    "simulator": "verilator", "replicate": replicate,
                 }, "tier": "L3", "purpose": "performance_certification", "citable": True,
                 "cycles": 5 * k + 100},
            ))
    return rows


def _freeze_formal_workload(run: Path, campaign: dict, descriptors: list[dict]) -> None:
    root = run / "_frozen_workload"
    capsules_root = root / "capsules"
    manifest_rows = []
    for descriptor in descriptors:
        capsule = descriptor["name"]
        path = capsules_root / "_perf" / capsule
        path.mkdir(parents=True)
        (path / "capsule.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
        (path / "capsule.interface.mlir").write_text("module {}\n")
        for entry in path.iterdir():
            entry.chmod(0o444)
        path.chmod(0o555)
        record = PR._exact_tree_record(path)
        manifest_rows.append({
            "family": "PK", "capsule": capsule,
            "source_relative_path": f"_perf/{capsule}",
            "snapshot_relative_path": f"capsules/_perf/{capsule}",
            "source_sha256": record["sha256"], "snapshot_sha256": record["sha256"],
            "n_files": record["n_files"], "n_bytes": record["n_bytes"],
            "performance": descriptor["performance"],
        })
    (capsules_root / "_perf").chmod(0o555)
    capsules_root.chmod(0o555)
    aggregate = PR._exact_tree_record(capsules_root)
    manifest = {
        "schema_version": 1, "target": "fixture",
        "counts": {"capsules": 4, "families": 1, "files": aggregate["n_files"],
                   "bytes": aggregate["n_bytes"]},
        "capsules_sha256": aggregate["sha256"], "capsules": manifest_rows,
    }
    manifest_path = root / "performance_corpus_manifest.json"
    manifest_sha, _ = _write_read_only(manifest_path, manifest)
    root.chmod(0o555)
    campaign.update({
        "workload_snapshot": str(root), "workload_manifest": str(manifest_path),
        "workload_manifest_sha256": manifest_sha, "workload_sha256": aggregate["sha256"],
    })


def _candidate_handoff(
        run: Path, campaign: dict, *, expected_rows: list[dict] | None = None,
        formal_claim: dict | None = None) -> SimpleNamespace:
    def file(path: Path, payload: bytes) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o444)
        return _sha(payload)

    record = run / "stage/performance_candidate.json"
    record_sha = file(record, b"{}\n")
    candidate = run / "stage/sealed_candidate/submission"
    candidate.mkdir(parents=True)
    file(candidate / "manifest.yaml", b"target: fixture\n")
    candidate.chmod(0o555)
    base = run / "stage/frozen_functional/submission"
    base.mkdir(parents=True)
    file(base / "manifest.yaml", b"target: fixture\n")
    base.chmod(0o555)
    bundle = run / "functional_bundle"
    host_package = bundle / "repo/out/host"
    host_package.mkdir(parents=True)
    file(host_package / "manifest.yaml", b"target: rvv\n")
    host_package.chmod(0o555)
    contract = bundle / "repo/merlin/contract"
    contract.mkdir(parents=True)
    file(contract / "schema.yaml", b"version: 1\n")
    bundle_manifest = bundle / "snapshot.json"
    bundle_sha = "b" * 64
    bundle_manifest_sha = file(bundle_manifest, _canonical({
        "version": 2, "content_sha256": bundle_sha, "n_files": 3, "n_bytes": 30,
        "grants": [],
    }))
    bundle.chmod(0o555)
    descriptor = run / "target_experiment.yaml"
    descriptor_sha = file(descriptor, b"target: fixture\n")
    prompt = run / "stage/prompt.txt"
    prompt_sha = file(prompt, b"Improve the exact Arm4 candidate.\n")
    transcript = run / "stage/transcript.jsonl"
    transcript_sha = file(transcript, b'{"type":"command"}\n')
    receipts = run / "stage/receipts.json"
    receipt_sha = file(receipts, b"[]\n")
    sentinel_source = bundle / "repo/merlin/contract/capsules/model/M2"
    sentinel_source.mkdir(parents=True)
    file(sentinel_source / "capsule.yaml", b"name: M2\nkind: model\n")
    for path in sorted(contract.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    contract.chmod(0o555)
    sentinel_sha = PR._exact_tree_record(sentinel_source)["sha256"]
    host = {
        "package": "out/host", "resolved_package": str(host_package),
        "package_sha256": "c" * 64, "target": "rvv", "run_id": "host0",
        "run_snapshot": {"path": str(bundle), "content_sha256": bundle_sha,
                         "n_files": 3, "n_bytes": 30, "version": 2},
    }
    probes = [{"label": "verilator", "command": "verilator --version", "returncode": 0}]
    expected_rows = expected_rows or _rows()
    formal_claim = formal_claim or {
        "schema_version": 1, "family": "PK", "claim": "PREDICTS",
        "status": "READY", "declaration": {}, "cohort": {},
        "expected_identities": [], "refusal_reasons": [],
    }
    handoff = SimpleNamespace(
        record_path=record, record_sha256=record_sha,
        candidate_path=candidate, candidate_sha256="e" * 64,
        candidate_initial_sha256="d" * 64,
        functional_run_id="arm4_functional_exact",
        functional_submission_sha256="d" * 64, functional_base_path=base,
        functional_bundle_snapshot_sha256=bundle_sha,
        functional_bundle_manifest=bundle_manifest,
        functional_bundle_manifest_sha256=bundle_manifest_sha,
        target_descriptor=descriptor, target_descriptor_sha256=descriptor_sha,
        corpus_root=Path(campaign["workload_snapshot"]),
        corpus_manifest=Path(campaign["workload_manifest"]),
        corpus_manifest_sha256=campaign["workload_manifest_sha256"],
        corpus_sha256=campaign["workload_sha256"], replicates=3,
        formal_replicate_identities=("r000", "r001", "r002"),
        formal_claim=formal_claim,
        smoke_replicates=1,
        expected_cells=tuple(row["identity"] for row in expected_rows),
        families=({"family": "PK", "claim": "PREDICTS"},), host_lane=host,
        e2e_sentinel={
            "capsule": "M2", "capsule_path": "/mounted/M2",
            "frozen_source_path": str(sentinel_source), "capsule_sha256": sentinel_sha,
            "required_lanes": ["on_mesh", "scalar_rvv_lane"],
            "required_tiers": ["L2", "L3"],
            "purpose": "functional_L2_L3_admission_not_performance_measurement",
        },
        prompt_sha256=prompt_sha, prompt_facts_sha256="f" * 64, prompt_path=prompt,
        transcript_path=transcript, transcript_sha256=transcript_sha,
        transcript_audit={"clean": True, "hits": [], "commands_seen": 1},
        receipt_path=receipts, receipt_sha256=receipt_sha,
        required_actions=("candidate-compile",),
        tool_evidence={"registry_sha256": "a" * 64, "round_receipts": [{}],
                       "tool_probe_results": probes, "tool_probe_recheck_results": probes},
        sandbox_evidence={
            plane: {"network": "available_not_an_isolation_claim",
                    "clear_environment": True, "policy_sha256": "a" * 64}
            for plane in ("outer_codex_control_plane", "inner_execution_plane")},
    )
    _HANDOFFS[str(record.resolve())] = handoff
    campaign.update({
        "snapshot": str(candidate), "snapshot_sha256": handoff.candidate_sha256,
        "candidate_stage": PC.candidate_handoff_record(handoff),
        "model_host_lane_snapshot": host,
        "full_model_admission": {
            "capsule": "M2", "kind": "model",
            "source": "functional_bundle_input_snapshot_v2", "source_sha256": sentinel_sha,
            "n_files": 1, "n_bytes": len(b"name: M2\nkind: model\n"),
            "lanes_required": ["on_mesh", "scalar_rvv_lane"],
            # A whole model emits exactly ONE execution tier -- the last declared tier the target
            # counts as RTL -- and the reporting gate re-derives it from these recorded inputs.
            "required_tiers": ["L3"],
            "tier_status": {"L3": "pass"},
            "tier_derivation": {"declared": ["L2", "L3"], "rtl_tiers": ["L3", "L4", "L5"],
                                "cert_tier": "L3",
                                "rule": "the last declared tier the target's capability manifest "
                                        "counts as RTL"},
            "grade_status": "pass", "numeric_status": "pass", "passed": True,
            "cycles_recorded": False,
            "role": "correctness_admission_not_performance_claim",
        },
        "frozen_contract": {
            "path": str(contract), "sha256": PR._exact_tree_record(contract)["sha256"],
            "source": "functional_bundle_input_snapshot_v2",
        },
        "candidate_before": {"state": "held", "ok": True,
                             "sha256": handoff.candidate_sha256},
        "candidate_after": {"state": "held", "ok": True,
                            "sha256": handoff.candidate_sha256},
    })
    return handoff


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run0"
    run.mkdir(parents=True)
    campaign = _campaign()
    _freeze_workload(run, campaign)
    _candidate_handoff(run, campaign)
    rows = _rows()
    _seal_results(run, campaign, rows)
    for offset in range(0, len(rows), 2):
        identity = rows[offset]["identity"]
        name = f"{identity['family']}__{identity['capsule']}__{identity['replicate']}.json"
        (run / name).write_bytes(_canonical(rows[offset:offset + 2]))
    _write_campaign(run, campaign)
    return run


def _formal_run(tmp_path: Path, *, refuted: bool = False) -> Path:
    run = tmp_path / ("formal_refuted" if refuted else "formal0")
    run.mkdir(parents=True)
    descriptors = [_formal_descriptor(index, k)
                   for index, k in enumerate((16, 32, 64, 128))]
    rows = _formal_rows(descriptors)
    if refuted:
        for row in rows:
            if (row["identity"]["capsule"] == "PK02_k64"
                    and row["identity"]["simulator"] == "verilator"):
                row["cycles"] += 300
    preflight = PK.preflight_pk_claim(descriptors)
    decision = PK.analyze_pk_claim(descriptors, rows)
    expected_status = "REFUTED" if refuted else "ESTABLISHED"
    assert preflight["status"] == "READY" and decision["status"] == expected_status
    campaign = _campaign()
    campaign.update({
        "claim_status": decision["status"], "experiment_mode": "formal_claim",
        "claim_launch_status": "GO", "claim_launch_blocker": None,
        "claim_preflight": preflight, "claim_decision": decision,
        "decision_boundary": {
            "module": "perf_pk_claim",
            "identity_bridge": "analyze_pk_claim(frozen_descriptors,sealed_result_rows)",
            "promotion_integration": "integrated",
            "promotion_status": "BLOCKED" if refuted else "PROMOTED",
            "reason": ("the predeclared PK quantitative decision was refuted" if refuted else
                       "the predeclared PK quantitative decision was established"),
        },
        "expected_identities": [row["identity"] for row in rows],
        "completion": {
            "expected": 24, "reported": 24, "correct": 24, "failed": 0, "missing": 0,
            "screen_expected": 12, "screen_passed": 12, "citable_expected": 12,
            "citable_measured": 12, "citable_passed": 12, "complete": True,
        },
    })
    _freeze_formal_workload(run, campaign, descriptors)
    _candidate_handoff(run, campaign, expected_rows=rows, formal_claim=preflight)
    _seal_results(run, campaign, rows)
    for offset in range(0, len(rows), 2):
        identity = rows[offset]["identity"]
        name = f"{identity['family']}__{identity['capsule']}__{identity['replicate']}.json"
        (run / name).write_bytes(_canonical(rows[offset:offset + 2]))
    _write_campaign(run, campaign)
    return run


def _replace_rows(run: Path, rows: list[dict]) -> None:
    campaign = json.loads((run / "campaign_manifest.json").read_text())
    _seal_results(run, campaign, rows)
    _write_campaign(run, campaign)


def test_complete_arm4_campaign_recomputes_all_exact_cell_metrics(tmp_path: Path) -> None:
    run = _run(tmp_path)
    campaign, rows, counts = PR.load_reportable_run(run)
    assert campaign["functional_run_id"] == "arm4_functional_exact"
    assert [PR.identity_tuple(row["identity"], owner="test result") for row in rows] == [
        ("PK", "PK00_k8", "spike", "r000"),
        ("PK", "PK00_k8", "verilator", "r000"),
        ("PK", "PK01_k16", "spike", "r000"),
        ("PK", "PK01_k16", "verilator", "r000"),
    ]
    assert counts == _completion()
    assert not ((run / "campaign_manifest.json").stat().st_mode & 0o222)
    assert not ((run / "campaign_manifest.digest.json").stat().st_mode & 0o222)


def test_writable_or_tampered_campaign_manifest_is_refused(tmp_path: Path) -> None:
    run = _run(tmp_path)
    manifest = run / "campaign_manifest.json"
    manifest.chmod(0o644)
    with pytest.raises(PR.ReportingGateError, match="writable"):
        PR.load_reportable_run(run)
    manifest.chmod(0o444)

    manifest.chmod(0o644)
    document = json.loads(manifest.read_text())
    document["claim_status"] = "ESTABLISHED"
    manifest.write_bytes(_canonical(document))
    manifest.chmod(0o444)
    with pytest.raises(PR.ReportingGateError, match="digest mismatch|byte count"):
        PR.load_reportable_run(run)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda doc: doc["candidate_stage"].update({"prompt_facts_sha256": "0" * 64}),
         "candidate-stage attribution disagrees"),
        (lambda doc: doc["model_host_lane_snapshot"].update({"package_sha256": "0" * 64}),
         "host lane differs"),
        (lambda doc: doc["frozen_contract"].update({"sha256": "0" * 64}),
         "contract digest changed"),
    ],
)
def test_candidate_host_and_contract_attribution_corruption_is_refused(
        tmp_path: Path, mutation, match: str) -> None:
    run = _run(tmp_path)
    campaign = json.loads((run / "campaign_manifest.json").read_text())
    mutation(campaign)
    _write_campaign(run, campaign)
    with pytest.raises(PR.ReportingGateError, match=match):
        PR.load_reportable_run(run)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda doc: doc["completion"].update({"expected": 0}), "vacuous|disagrees"),
        (lambda doc: doc.update({"measurement_status": "NO_GO"}), "not GO"),
        (lambda doc: doc.update({"claim_status": "ESTABLISHED"}), "non-claim boundary"),
        (lambda doc: doc.update({"claim_launch_status": "GO"}), "non-claim boundary"),
        (lambda doc: doc.update({"candidate_stage": {"status": "ready"}}), "candidate-stage"),
        (lambda doc: doc["sandbox"].update({"network": "unshared"}), "network available"),
    ],
)
def test_vacuous_incomplete_overclaiming_or_networkless_campaigns_are_refused(
        tmp_path: Path, mutation, match: str) -> None:
    run = _run(tmp_path)
    campaign = json.loads((run / "campaign_manifest.json").read_text())
    mutation(campaign)
    _write_campaign(run, campaign)
    with pytest.raises(PR.ReportingGateError, match=match):
        PR.load_reportable_run(run)


@pytest.mark.parametrize(
    "failure",
    ("spike_cycles", "mixed_policy", "wrong_order", "float_cycles", "grade_failure",
     "numeric_failure", "error", "failure_record"),
)
def test_mixed_or_misclassified_simulator_rows_are_refused(tmp_path: Path, failure: str) -> None:
    run = _run(tmp_path)
    rows = _rows()
    if failure == "spike_cycles":
        rows[0]["cycles"] = 7
    elif failure == "mixed_policy":
        rows[1]["citable"] = False
    elif failure == "float_cycles":
        rows[1]["cycles"] = 101.5
    elif failure == "grade_failure":
        rows[1]["grade_status"] = "fail"
    elif failure == "numeric_failure":
        rows[1]["numeric_status"] = "fail"
    elif failure == "error":
        rows[1]["error"] = "runner failed"
    elif failure == "failure_record":
        rows[1]["failure"] = {"plane": "numeric"}
    else:
        rows[0], rows[1] = rows[1], rows[0]
    _replace_rows(run, rows)
    with pytest.raises(PR.ReportingGateError, match="cycles|policy|sequence|completion"):
        PR.load_reportable_run(run)


def test_result_manifest_and_workload_corruption_are_each_detected(tmp_path: Path) -> None:
    run = _run(tmp_path)
    results = run / "perf_results.json"
    results.chmod(0o644)
    results.write_text("[]\n")
    results.chmod(0o444)
    with pytest.raises(PR.ReportingGateError, match="digest mismatch|byte count"):
        PR.load_reportable_run(run)

    run = _run(tmp_path / "manifest_case")
    manifest = run / "_frozen_workload" / "performance_corpus_manifest.json"
    manifest.chmod(0o644)
    manifest.write_text("{}\n")
    manifest.chmod(0o444)
    with pytest.raises(PR.ReportingGateError, match="manifest digest mismatch"):
        PR.load_reportable_run(run)

    run = _run(tmp_path / "capsule_case")
    capsule_file = (run / "_frozen_workload/capsules/_perf/PK00_k8/capsule.yaml")
    capsule_file.chmod(0o644)
    capsule_file.write_text("name: changed\n")
    capsule_file.chmod(0o444)
    with pytest.raises(PR.ReportingGateError, match="capsule digest changed"):
        PR.load_reportable_run(run)


def test_digest_record_contents_are_verified_not_only_its_hash(tmp_path: Path) -> None:
    run = _run(tmp_path)
    campaign = json.loads((run / "campaign_manifest.json").read_text())
    digest_path = run / "perf_results.digest.json"
    record = json.loads(digest_path.read_text())
    record["records"] = 999
    digest_path.chmod(0o644)
    digest_sha, _ = _write_read_only(digest_path, record)
    campaign["results"]["digest_record_sha256"] = digest_sha
    _write_campaign(run, campaign)
    with pytest.raises(PR.ReportingGateError, match="contents disagree"):
        PR.load_reportable_run(run)


def test_assembler_only_verifies_pair_files_and_never_rewrites_sealed_results(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path)
    before = (run / "perf_results.json").read_bytes()
    before_mode = os.stat(run / "perf_results.json").st_mode
    monkeypatch.setattr(AR.PB, "RUNS", tmp_path)
    assert AR.main(["--run-id", run.name]) == 0
    assert (run / "perf_results.json").read_bytes() == before
    assert os.stat(run / "perf_results.json").st_mode == before_mode

    pair = run / "PK__PK00_k8__r000.json"
    doc = json.loads(pair.read_text())
    doc[0]["correct"] = False
    pair.write_bytes(_canonical(doc))
    with pytest.raises(PR.ReportingGateError, match="exact sealed"):
        AR.main(["--run-id", run.name])


def test_report_states_claim_boundary_network_and_l3_only_timing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path)
    campaign, rows, counts = PR.load_reportable_run(run)
    text = GR.render_report("run0", campaign, rows, counts)
    assert "CLAIM NOT ESTABLISHED" in text
    assert "not a completed performance experiment" in text
    assert PC.SMOKE_CLAIM_NONCLAIM in campaign["claim_launch_blocker"]
    assert "arm4_functional_exact" in text and "d" * 64 in text
    assert "4/4" in text and "PK00_k8" in text and "r000" in text
    assert "Verilator L3" in text and "101" in text
    assert "deliberately omitted" in text and "Spike L2 provides no citable" in text
    assert "Network: available" in text and "networkless" not in text.lower()
    assert campaign["workload_manifest_sha256"] in text
    assert campaign["candidate_stage"]["prompt_facts_sha256"] in text
    assert campaign["frozen_contract"]["sha256"] in text
    assert campaign["results"]["digest_record_sha256"] in text
    reports = tmp_path / "reports"
    monkeypatch.setattr(GR.PB, "RUNS", tmp_path)
    monkeypatch.setattr(GR.PB, "REPORTS", reports)
    assert GR.main(["--run-id", run.name]) == 0
    assert (reports / "run0_arm4_performance.md").read_text() == text


def test_formal_pk_decision_is_recomputed_promoted_and_rendered_from_sealed_rows(
        tmp_path: Path) -> None:
    run = _formal_run(tmp_path)
    campaign, rows, counts = PR.load_reportable_run(run)
    assert counts["reported"] == 24
    assert campaign["claim_preflight"]["status"] == "READY"
    assert campaign["claim_decision"]["status"] == "ESTABLISHED"
    assert campaign["claim_decision"]["evidence"]["l2_cycles_consumed"] == 0
    assert campaign["claim_decision"]["evidence"]["l3_positive_cycle_rows_consumed"] == 12
    assert campaign["decision_boundary"]["promotion_status"] == "PROMOTED"
    text = GR.render_report(run.name, campaign, rows, counts)
    assert "PK CLAIM ESTABLISHED" in text and "frozen fixed-M/N K cohort" in text
    assert "PK quantitative decision" in text and "12** Verilator L3 rows" in text
    assert "`0` Spike L2 cycles consumed" in text
    assert "Network: available" in text and "networkless" not in text.lower()
    assert "Spike L2 provides no citable" in text


def test_formal_pk_refutation_is_reportable_but_never_promoted(tmp_path: Path) -> None:
    run = _formal_run(tmp_path, refuted=True)
    campaign, rows, counts = PR.load_reportable_run(run)
    assert campaign["claim_status"] == "REFUTED"
    assert campaign["decision_boundary"]["promotion_status"] == "BLOCKED"
    text = GR.render_report(run.name, campaign, rows, counts)
    assert "PK CLAIM REFUTED" in text
    assert "not promoted" in text


def test_formal_pk_decision_tamper_is_refused_even_when_campaign_is_resealed(
        tmp_path: Path) -> None:
    run = _formal_run(tmp_path)
    campaign = json.loads((run / "campaign_manifest.json").read_text())
    campaign["claim_decision"]["fit"]["rate_cycles_per_K_element"] = 999
    _write_campaign(run, campaign)
    with pytest.raises(PR.ReportingGateError, match="differs from exact-row recomputation"):
        PR.load_reportable_run(run)


def test_runner_formal_preflight_is_exact_and_refuses_wrong_replicates() -> None:
    descriptors = [_formal_descriptor(index, k)
                   for index, k in enumerate((16, 32, 64, 128))]
    capsules = tuple(SimpleNamespace(
        family="PK", capsule=descriptor["name"], descriptor=descriptor)
        for descriptor in descriptors)
    expected = tuple(
        PC.PerfCellIdentity("PK", descriptor["name"], simulator, replicate)
        for descriptor in descriptors
        for replicate in ("r000", "r001", "r002")
        for simulator in ("spike", "verilator"))
    preflight = RR._pk_preflight(capsules, 3, expected)
    assert preflight == PK.preflight_pk_claim(descriptors)
    assert RR.measurement_mode("claim", preflight)["claim_launch_status"] == "GO"
    assert RR._claim_decision(capsules, _formal_rows(descriptors))["status"] == "ESTABLISHED"
    with pytest.raises(RR.PC.CampaignGateError, match="predeclared 3 replicates"):
        RR._pk_preflight(capsules, 1, expected)


def test_runner_smoke_workers_sealing_and_report_flow_are_integrated(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real CLI orchestration while replacing only expensive simulator cell execution."""
    stage = tmp_path / "authoring"
    stage.mkdir()
    descriptors = [_formal_descriptor(index, k)
                   for index, k in enumerate((16, 32, 64, 128))]
    formal_rows = _formal_rows(descriptors)
    stage_campaign = _campaign()
    _freeze_formal_workload(stage, stage_campaign, descriptors)
    preflight = PK.preflight_pk_claim(descriptors)
    handoff = _candidate_handoff(
        stage, stage_campaign, expected_rows=formal_rows, formal_claim=preflight)
    corpus = PC.load_frozen_performance_corpus(
        handoff.corpus_root, manifest_sha256=handoff.corpus_manifest_sha256,
        capsules_sha256=handoff.corpus_sha256)
    sentinel_source = Path(handoff.e2e_sentinel["frozen_source_path"])
    sentinel_tree = PC._exact_tree_record(sentinel_source)
    sentinel = PC.FullModelSentinel(
        handoff.e2e_sentinel["capsule"], sentinel_source,
        yaml.safe_load((sentinel_source / "capsule.yaml").read_text()),
        sentinel_tree["sha256"], sentinel_tree["n_files"], sentinel_tree["n_bytes"])
    contract = Path(handoff.functional_bundle_manifest).parent / "repo/merlin/contract"
    candidate = RR.MeasurementCandidate(
        handoff, Path(handoff.candidate_path), handoff.candidate_sha256, corpus,
        Path(handoff.functional_base_path), contract, PC._exact_tree_record(contract)["sha256"],
        sentinel)
    functional = SimpleNamespace(
        run_id=handoff.functional_run_id, digest=handoff.functional_submission_sha256,
        public_capsules=2, hidden_capsules=1,
        model_host_lane_snapshot=handoff.host_lane)
    held = SimpleNamespace(
        ok=True, reason=None,
        to_dict=lambda: {"state": "held", "ok": True, "reason": None})
    fork = SimpleNamespace(to_dict=lambda: {"schema_version": 1})
    captured: dict[str, object] = {}

    monkeypatch.setattr(RR.PC, "inspect_functional_run", lambda *_args: functional)
    monkeypatch.setattr(RR, "load_measurement_candidate", lambda *_args: candidate)
    monkeypatch.setattr(RR.PC, "functional_fork", lambda _functional: fork)
    monkeypatch.setattr(RR.PC, "check_fork", lambda *_args: held)
    monkeypatch.setattr(RR.PC, "functional_host_lane",
                        lambda _functional: contextlib.nullcontext())
    monkeypatch.setattr(
        RR.PC, "run_tool_probes",
        lambda policy: [{"label": probe.label, "returncode": 0, "stdout": "ok", "stderr": ""}
                        for probe in policy.required_tools])

    def admission(_package, selected, _workspace, _timeout, _target, _contract, _sha):
        captured["sentinel"] = selected.capsule
        return {
            "capsule": selected.capsule, "kind": "model",
            "source": "functional_bundle_input_snapshot_v2",
            "source_sha256": selected.source_sha256,
            "n_files": selected.n_files, "n_bytes": selected.n_bytes,
            "lanes_required": ["on_mesh", "scalar_rvv_lane"],
            # A whole model emits exactly ONE execution tier -- the last declared tier the target
            # counts as RTL -- and the reporting gate re-derives it from these recorded inputs.
            "required_tiers": ["L3"],
            "tier_status": {"L3": "pass"},
            "tier_derivation": {"declared": ["L2", "L3"], "rtl_tiers": ["L3", "L4", "L5"],
                                "cert_tier": "L3",
                                "rule": "the last declared tier the target's capability manifest "
                                        "counts as RTL"},
            "grade_status": "pass", "numeric_status": "pass", "passed": True,
            "cycles_recorded": False,
            "role": "correctness_admission_not_performance_claim",
        }

    def cell(task):
        _package, member, replicate, *_rest = task
        rows = RR._result_rows((member.family, member.capsule, replicate))
        for row in rows:
            row.update({"correct": True, "tier_status": "pass", "grade_status": "pass",
                        "numeric_status": "pass"})
            if row["identity"]["simulator"] == "verilator":
                row["cycles"] = 100 + int(member.capsule.split("_k")[-1])
        return rows

    class RecordingExecutor:
        def __init__(self, *, max_workers, mp_context):
            captured["workers"] = max_workers
            captured["start_method"] = mp_context.get_start_method()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def map(function, tasks):
            captured["task_order"] = [
                (task[1].capsule, task[2]) for task in tasks]
            return [function(task) for task in tasks]

    monkeypatch.setattr(RR, "run_full_model_admission", admission)
    monkeypatch.setattr(RR, "_run_cell_task", cell)
    monkeypatch.setattr(RR.concurrent.futures, "ProcessPoolExecutor", RecordingExecutor)
    runs = tmp_path / "runs"
    monkeypatch.setattr(RR.PB, "RUNS", runs)
    run_id = "smoke_integrated"
    assert RR.main([
        "--functional-run-id", handoff.functional_run_id,
        "--functional-submission-sha256", handoff.functional_submission_sha256,
        "--candidate-record", str(handoff.record_path),
        "--mode", "measurement-smoke", "--replicates", "1", "--workers", "2",
        "--run-id", run_id,
    ]) == 0

    run = runs / run_id
    campaign, rows, counts = PR.load_reportable_run(run)
    report = GR.render_report(run_id, campaign, rows, counts)
    assert captured == {
        "sentinel": handoff.e2e_sentinel["capsule"], "workers": 2,
        "start_method": "spawn",
        "task_order": [(descriptor["name"], "r000") for descriptor in descriptors],
    }
    assert counts["reported"] == counts["expected"] == 8
    assert all(row["cycles"] is None for row in rows
               if row["identity"]["simulator"] == "spike")
    assert not ((run / "perf_results.json").stat().st_mode & 0o222)
    assert not ((run / "campaign_manifest.json").stat().st_mode & 0o222)
    assert "CLAIM NOT ESTABLISHED" in report
    assert "not a completed performance experiment" in report
    assert "Network: available" in report and "networkless" not in report.lower()


def test_legacy_cross_approach_renderers_refuse_arm4_campaign(tmp_path: Path) -> None:
    run = _run(tmp_path)
    with pytest.raises(PR.ReportingGateError, match="Arm4-only report"):
        PR.refuse_legacy_cross_approach(run, "legacy plot")


def test_legacy_scripts_install_the_arm4_refusal_guard() -> None:
    for name in ("gen_perf_plots.py", "build_artifact.py"):
        source = (SCRIPTS / name).read_text()
        assert "refuse_legacy_cross_approach" in source
