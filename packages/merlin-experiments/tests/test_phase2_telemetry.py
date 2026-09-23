"""Telemetry ownership and receipt versions, without agent or hardware execution."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2 import telemetry as T
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common.digest import sha256_file
from merlin.common.paths import module_source_path, python_import_roots


@pytest.fixture
def preflight(tmp_path):
    executable = tmp_path / "codex"
    executable.write_text("#!/bin/sh\nexit 99\n")
    executable.chmod(0o755)
    prices = tmp_path / "prices.yaml"
    prices.write_text("gpt-fixture: [5, 30, 0.5, 5]\n")
    authoring = tmp_path / "authoring.py"
    authoring.write_text("# actual authoring source stays distinct from telemetry\n")
    return T.prepare(model="gpt-fixture", authoring_stage=authoring, price_table=prices, codex_binary=executable)


def test_package_import_has_no_native_controller_dependencies(tmp_path):
    program = """
import importlib.abc,json,sys
sys.path[:0]=json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name in {'perf_agent_stage','perf_pk_claim','perf_campaign','perf_prompt','_pbcommon','_common',
                    'perf_model','perf_capsule_verdict','run_paired_perf_bench',
                    'heldout_gsim_qualification','produce_gsim_certificate'}:
            raise AssertionError('native dependency: '+name)
sys.meta_path.insert(0,NoNative())
from merlin_experiments.phase2 import telemetry
assert callable(telemetry.prepare) and callable(telemetry.collect_round) and callable(telemetry.finalize)
from merlin_experiments.phase2 import development_feedback, feedback_metrics, calibration, capsule_verdict
from merlin_experiments.phase2 import agent_workspace
from merlin_experiments.phase2 import measurement_support, gsim_workload
from merlin_experiments.phase2 import paired_measurement
from merlin_experiments.phase2 import candidate_record, candidate_verification
from merlin_experiments.phase2 import stage_inputs
from merlin_experiments.phase2 import emission_diagnostics
from merlin_experiments.phase2 import emission_analysis
from merlin_experiments.phase2 import paired_inputs, revealed_corpus
from merlin_experiments.phase2 import authoring, authoring_cli
from merlin_experiments.phase2 import gsim_certificate, heldout_qualification
from merlin_experiments.phase2 import functional_cohort, functional_coverage, functional_qualification, holdout_corpus
assert callable(gsim_certificate.capture_case) and callable(heldout_qualification.qualify_revealed_holdout)
assert callable(authoring.run_stage) and callable(authoring_cli.main)
assert callable(paired_inputs.load_paired_inputs) and callable(revealed_corpus.load_revealed_members)
assert callable(emission_analysis.analyze_whole_model_emission)
assert callable(emission_diagnostics.analyze_command_buffers)
assert callable(stage_inputs.prepare_prompt_inputs)
assert callable(candidate_record.validate_candidate_record)
assert callable(candidate_verification.verify_candidate_handoff)
assert callable(paired_measurement.execute_schedule) and callable(paired_measurement.run_execution)
assert callable(measurement_support.measurement_identity) and callable(gsim_workload.derive_workload)
assert callable(agent_workspace.inner_execution_policy) and callable(agent_workspace.verify_answer_free_agent_inputs)
assert callable(development_feedback.DevelopmentGsimFeedback)
assert callable(feedback_metrics.declared_capsule_macs)
assert callable(calibration.achievable_ceiling) and callable(capsule_verdict.capsule_verdict)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, json.dumps([str(path) for path in python_import_roots()])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preflight_has_distinct_native_renderer_and_packaged_telemetry(preflight):
    assert preflight["schema_version"] == 4
    assert preflight["source_policy_version"] == 4
    assert set(preflight["sources"]) == T.TREATMENT_SOURCES
    sources = preflight["sources"]
    assert sources["performance_authoring_stage"]["path"] != sources["performance_telemetry"]["path"]
    assert Path(sources["performance_telemetry"]["path"]) == module_source_path(T.__name__).resolve()
    assert sources["performance_package_sources"] == T._package_source_record()
    for name, source in sources.items():
        if name == "performance_package_sources":
            continue
        assert source["sha256"] == sha256_file(source["path"])
    identity = T.treatment_identity(preflight)
    assert identity["authoring_stage_sha256"] == sources["performance_authoring_stage"]["sha256"]


@pytest.mark.parametrize("missing", sorted(T.TREATMENT_SOURCES))
def test_each_current_source_role_is_required(preflight, missing):
    del preflight["sources"][missing]
    with pytest.raises(StageGateError, match="identity is incomplete"):
        T.treatment_identity(preflight)


def test_historical_exact_set_is_read_without_rewrite(preflight):
    old = copy.deepcopy(preflight)
    old["schema_version"] = 1
    del old["source_policy_version"]
    old["sources"] = {name: old["sources"][name] for name in T.LEGACY_TREATMENT_SOURCES}
    before = copy.deepcopy(old)
    assert set(T.treatment_identity(old)["telemetry_source_sha256"]) == T.LEGACY_TREATMENT_SOURCES
    assert old == before
    old["sources"]["unexpected"] = next(iter(old["sources"].values()))
    with pytest.raises(StageGateError, match="identity is incomplete"):
        T.treatment_identity(old)


def test_packaged_extraction_receipt_remains_readable_without_new_price_fields(preflight):
    old = copy.deepcopy(preflight)
    old.update(schema_version=2, source_policy_version=2)
    old.pop("accounting_policy")
    old["price_table"].pop("snapshot")
    old["sources"] = {name: old["sources"][name] for name in T.PACKAGED_TREATMENT_SOURCES}
    before = copy.deepcopy(old)
    assert set(T.treatment_identity(old)["telemetry_source_sha256"]) == T.PACKAGED_TREATMENT_SOURCES
    assert old == before


@pytest.mark.parametrize(
    "schema,policy", [(1, 2), (2, None), (3, 2), (True, None), (3, True), (4, 3), (4, True), (5, 5)]
)
def test_unsupported_source_policy_refuses(preflight, schema, policy):
    preflight.update(schema_version=schema, source_policy_version=policy)
    with pytest.raises(StageGateError, match="unsupported"):
        T.treatment_identity(preflight)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_source_receipt_decoding_never_reads_live_implementations(preflight, monkeypatch, version):
    old = copy.deepcopy(preflight)
    required = {
        1: T.LEGACY_TREATMENT_SOURCES,
        2: T.PACKAGED_TREATMENT_SOURCES,
        3: T.EXPLICIT_PRICE_TREATMENT_SOURCES,
        4: T.TREATMENT_SOURCES,
    }[version]
    old["schema_version"] = version
    if version == 1:
        old.pop("source_policy_version")
    else:
        old["source_policy_version"] = version
    old["sources"] = {name: old["sources"][name] for name in required}
    before = copy.deepcopy(old)
    monkeypatch.setattr(T, "_sha256_file", lambda *_: pytest.fail("historical decoder read live bytes"))
    monkeypatch.setattr(T.source_membership, "python_members", lambda *_a, **_k: pytest.fail("live walk"))
    assert set(T.treatment_identity(old)["telemetry_source_sha256"]) == required
    assert old == before


@pytest.mark.parametrize(
    "mutation",
    [
        "add",
        "remove",
        "change",
        "helper",
        "reporting",
        "statistics",
        "broker",
        "broker_policy",
        "corpus_feedback",
        "whole_model",
        "transcript_audit",
        "functional_inputs",
        "development_feedback",
        "agent_workspace",
        "measurement_support",
        "paired_measurement",
        "candidate_record",
        "candidate_verification",
        "stage_inputs",
        "emission_diagnostics",
        "emission_analysis",
        "paired_inputs",
        "revealed_corpus",
        "authoring",
        "authoring_cli",
        "gsim_certificate",
        "heldout_qualification",
        "functional_cohort",
        "functional_coverage",
        "functional_qualification",
        "holdout_corpus",
        "checkpoint_admission",
        "checkpoint_controller",
        "checkpoint_cli",
        "chia_launch",
        "paired_cli",
        "gsim_workload",
        "feedback_metrics",
        "calibration",
        "capsule_verdict",
    ],
)
def test_package_or_membership_helper_drift_refuses(preflight, tmp_path, monkeypatch, mutation):
    directory = tmp_path / "phase2"
    directory.mkdir()
    (directory / "__init__.py").write_text("# package\n")
    (directory / "corpus.py").write_text("# member\n")
    for owner in (
        "reporting",
        "statistics",
        "broker",
        "broker_policy",
        "corpus_feedback",
        "whole_model",
        "transcript_audit",
        "functional_inputs",
        "development_feedback",
        "agent_workspace",
        "measurement_support",
        "paired_measurement",
        "candidate_record",
        "candidate_verification",
        "stage_inputs",
        "emission_diagnostics",
        "emission_analysis",
        "paired_inputs",
        "revealed_corpus",
        "authoring",
        "authoring_cli",
        "gsim_certificate",
        "heldout_qualification",
        "functional_cohort",
        "functional_coverage",
        "functional_qualification",
        "holdout_corpus",
        "checkpoint_admission",
        "checkpoint_controller",
        "checkpoint_cli",
        "chia_launch",
        "paired_cli",
        "gsim_workload",
        "feedback_metrics",
        "calibration",
        "capsule_verdict",
    ):
        (directory / f"{owner}.py").write_bytes(module_source_path(f"merlin_experiments.phase2.{owner}").read_bytes())
    preflight["sources"]["performance_package_sources"] = T._package_source_record(directory)
    original = T.module_source_path
    monkeypatch.setattr(
        T,
        "module_source_path",
        lambda name: directory / "__init__.py" if name == "merlin_experiments.phase2" else original(name),
    )
    # The positive control reaches real AET snapshot construction with this exact closure.
    T._verified_snapshot(preflight, "gpt-fixture")
    if mutation == "add":
        (directory / "next_owner.py").write_text("# new source\n")
    elif mutation == "remove":
        (directory / "corpus.py").unlink()
    elif mutation == "change":
        (directory / "corpus.py").write_text("# changed source\n")
    elif mutation in {
        "reporting",
        "statistics",
        "broker",
        "broker_policy",
        "corpus_feedback",
        "whole_model",
        "transcript_audit",
        "functional_inputs",
        "development_feedback",
        "agent_workspace",
        "measurement_support",
        "paired_measurement",
        "candidate_record",
        "candidate_verification",
        "stage_inputs",
        "emission_diagnostics",
        "emission_analysis",
        "paired_inputs",
        "revealed_corpus",
        "authoring",
        "authoring_cli",
        "gsim_certificate",
        "heldout_qualification",
        "functional_cohort",
        "functional_coverage",
        "functional_qualification",
        "holdout_corpus",
        "checkpoint_admission",
        "checkpoint_controller",
        "checkpoint_cli",
        "chia_launch",
        "paired_cli",
        "gsim_workload",
        "feedback_metrics",
        "calibration",
        "capsule_verdict",
    }:
        owner = directory / f"{mutation}.py"
        owner.write_bytes(owner.read_bytes() + b"\n# changed authoritative source\n")
    else:
        helper = tmp_path / "source_membership.py"
        helper.write_text("# pinned helper\n")
        preflight["sources"]["python_source_membership"] = {"path": str(helper), "sha256": sha256_file(helper)}
        helper.write_text("# changed helper\n")
    with pytest.raises(StageGateError, match="implementation changed"):
        T._verified_snapshot(preflight, "gpt-fixture")


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/absolute.py",
        "nested/../alias.py",
        "double//slash.py",
        "__pycache__/hidden.py",
        "member.txt",
        "nul\0.py",
    ],
)
def test_escaping_or_unapproved_closure_members_refuse_without_live_reads(preflight, name):
    closure = preflight["sources"]["performance_package_sources"]
    closure["members"][name] = "a" * 64
    closure["sha256"] = T._sha256(T._canonical_json(closure["members"]))
    with pytest.raises(StageGateError, match="closure"):
        T.treatment_identity(preflight)


@pytest.mark.parametrize("mutation", ["missing-init", "extra-field", "wrong-digest", "relative-root", "bad-hash"])
def test_malformed_closure_refuses(preflight, mutation):
    closure = preflight["sources"]["performance_package_sources"]
    if mutation == "missing-init":
        closure["members"].pop("__init__.py")
    elif mutation == "extra-field":
        closure["unbound"] = True
    elif mutation == "wrong-digest":
        closure["sha256"] = "f" * 64
    elif mutation == "relative-root":
        closure["path"] = "relative"
    else:
        closure["members"]["corpus.py"] = False
    with pytest.raises(StageGateError, match="closure"):
        T.treatment_identity(preflight)


def test_renderer_and_executable_bindings_refuse_mismatch(preflight):
    for field in ("authoring_stage_sha256", "codex_binary_sha256"):
        with pytest.raises(StageGateError, match="identities differ"):
            T.treatment_identity(preflight, **{field: "0" * 64})


def test_extraction_keeps_legacy_declared_cache_defaults(tmp_path):
    prices = tmp_path / "prices.yaml"
    prices.write_text("fixture: [5, 30]\n")
    assert T._declared_price_rate(prices, "fixture-model") == (5, 30, 0.5, 6.25)


def _round_files(root):
    events = [
        {"type": "thread.started", "thread_id": "fixture"},
        {"type": "turn.started"},
        {"type": "item.started", "item": {"id": "one", "type": "command_execution", "command": "synthetic"}},
        {
            "type": "item.completed",
            "item": {"id": "one", "type": "command_execution", "command": "synthetic", "exit_code": 0},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 20000,
                "cached_input_tokens": 5000,
                "cache_write_input_tokens": 1000,
                "output_tokens": 4000,
            },
        },
    ]
    rounds = root / "rounds"
    rounds.mkdir(parents=True)
    raw = rounds / "round_00.codex_events.raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in events))
    (rounds / "round_00.codex_events.timestamped.jsonl").write_text(
        "".join(
            json.dumps({"seq": i, "arrived_at": f"2026-09-20T00:00:0{i}+00:00", "event": event}) + "\n"
            for i, event in enumerate(events, 1)
        )
    )
    for suffix in ("codex_stderr.log", "prompt.txt", "final.txt"):
        (rounds / f"round_00.{suffix}").write_text("fixture\n")
    (rounds / "round_00.codex_summary.json").write_text(
        json.dumps(
            {
                "billing_mode": "subscription_notional",
                "exit_code": 0,
                "usage_complete": True,
                "timed_out": False,
                "wall_s": 5.0,
            }
        )
    )
    return raw


def test_token_only_reader_never_consults_prices_and_default_stays_compatible(tmp_path, monkeypatch):
    from merlin.targetgen import experiment_tokens as ET

    raw = _round_files(tmp_path)
    monkeypatch.setattr(ET, "_bucket_rate", lambda model: (5e-6, 30e-6, 0.5e-6, 5e-6))
    default = ET.parse_agent_transcript(raw, driver="codex", model="gpt-fixture", billing_mode=ET.SUBSCRIPTION_NOTIONAL)
    assert default["subscription_notional_usd"] == 0.1975
    monkeypatch.setattr(ET, "_bucket_rate", lambda model: pytest.fail("token-only reader accessed ambient pricing"))
    tokens = ET.parse_agent_transcript(
        raw, driver="codex", model="gpt-fixture", billing_mode=ET.SUBSCRIPTION_NOTIONAL, include_cost=False
    )
    assert tokens["tokens_total"] == 24000
    assert "subscription_notional_usd" not in tokens and "estimated_cost_usd" not in tokens
    assert {key: default[key] for key in tokens} == tokens


def test_pinned_price_is_single_authority_and_sequential_runs_do_not_change_globals(tmp_path, monkeypatch, preflight):
    from aet.trajectory.price_snapshot import PriceSnapshot

    from merlin.targetgen import experiment_tokens as ET

    ambient = tmp_path / "ambient.yaml"
    ambient.write_text("gpt-fixture: [500, 3000, 50, 500]\n")
    monkeypatch.setenv("AET_PRICE_TABLE", str(ambient))
    cached = {"gpt-fixture": (99.0, 99.0, 99.0, 99.0)}
    monkeypatch.setattr(ET, "_OVERRIDES", cached)
    monkeypatch.setattr(ET, "_bucket_rate", lambda model: pytest.fail("P2 used ambient ET pricing"))
    monkeypatch.setattr(PriceSnapshot, "default_openai", lambda: pytest.fail("P2 used upstream bundled default prices"))
    first_plan = preflight
    explicit = tmp_path / "second-prices.yaml"
    explicit.write_text("gpt-fixture: [10, 60, 1, 10]\n")
    second_plan = T.prepare(
        model="gpt-fixture",
        authoring_stage=Path(first_plan["sources"]["performance_authoring_stage"]["path"]),
        price_table=explicit,
        codex_binary=first_plan["sources"]["codex_binary"]["path"],
    )
    before = dict(os.environ)
    observed = []
    for index, plan in enumerate((first_plan, second_plan)):
        stage = tmp_path / f"stage-{index}"
        _round_files(stage)
        evidence = T.collect_round(stage, 0, model="gpt-fixture", agent_exit_code=0, preflight_record=plan)
        result = T.finalize(
            stage,
            [{"round": 0, "telemetry": evidence}],
            model="gpt-fixture",
            target="synthetic",
            suite="fixture-suite",
            run_id=stage.name,
            preflight_record=plan,
        )
        trajectory = json.loads(Path(result["artifacts"]["trajectory"]["path"]).read_text())
        amount = result["accounting"]["subscription_notional_usd"]
        assert amount == evidence["accounting"]["subscription_notional_usd"]
        assert amount == round(trajectory["cost"]["value_usd"], 4)
        assert trajectory["cost"]["price_table_sha256"] == plan["price_table"]["snapshot"]["sha256"]
        assert result["accounting"]["estimated_cost_usd"] is None
        assert os.environ == before
        assert ET._OVERRIDES is cached
        T.verify_price_evidence(result, [{"round": 0, "telemetry": evidence}], plan)
        forged = copy.deepcopy(evidence)
        forged["price_snapshot_sha256"] = "0" * 64
        with pytest.raises(StageGateError, match="different price authority"):
            T.verify_price_evidence(result, [{"round": 0, "telemetry": forged}], plan)
        observed.append(amount)
    assert observed == [0.1975, 0.395]


@pytest.mark.parametrize("rates", ["[.nan, 1]", "[1, .inf]", "[-1, 1]", "[1, 2, -1, 1]"])
def test_invalid_normalized_rates_refuse(tmp_path, preflight, rates):
    prices = tmp_path / "invalid.yaml"
    prices.write_text(f"gpt-fixture: {rates}\n")
    with pytest.raises(StageGateError, match="finite|negative"):
        T.prepare(
            model="gpt-fixture",
            authoring_stage=Path(preflight["sources"]["performance_authoring_stage"]["path"]),
            price_table=prices,
            codex_binary=preflight["sources"]["codex_binary"]["path"],
        )


def test_explicit_authoring_source_symlink_is_refused(tmp_path, preflight):
    alias = tmp_path / "linked-authoring.py"
    alias.symlink_to(preflight["sources"]["performance_authoring_stage"]["path"])
    with pytest.raises(StageGateError, match="linked"):
        T.prepare(
            model="gpt-fixture",
            authoring_stage=alias,
            price_table=Path(preflight["price_table"]["path"]),
            codex_binary=preflight["sources"]["codex_binary"]["path"],
        )


@pytest.mark.parametrize("mutation", ["raw-price", "snapshot", "model", "source"])
def test_current_price_binding_refuses_drift_before_collecting(tmp_path, preflight, mutation):
    stage = tmp_path / "stage"
    _round_files(stage)
    model = "gpt-fixture"
    if mutation == "raw-price":
        Path(preflight["price_table"]["path"]).write_text("gpt-fixture: [1, 2]\n")
    elif mutation == "snapshot":
        preflight["price_table"]["snapshot"]["document"]["rates"]["gpt-fixture"][0] = 999
    elif mutation == "source":
        Path(preflight["sources"]["performance_authoring_stage"]["path"]).write_text("# drift\n")
    else:
        model = "different"
    with pytest.raises(StageGateError, match="price|implementation|model"):
        T.collect_round(stage, 0, model=model, agent_exit_code=0, preflight_record=preflight)
    assert not (stage / "telemetry").exists()


@pytest.mark.parametrize(
    "rate,expected",
    [
        ("[5, 30]", [5, 30, 0.5, 6.25]),
        ("[5, 30, 2]", [5, 30, 2, 6.25]),
        ("[5, 30, 2, 3]", [5, 30, 2, 3]),
        ("{input: 5, output: 30, cache_write: 4}", [5, 30, 0.5, 4]),
        ("{input: 5, output: 30, cache_creation: 3, cache_write: 4}", [5, 30, 0.5, 3]),
    ],
)
def test_legacy_price_syntax_normalizes_once_without_upstream_default_changes(tmp_path, rate, expected):
    prices = tmp_path / "legacy.yaml"
    prices.write_text(f"FiXtUrE: {rate}\n")
    document = T._price_snapshot_document(prices)
    assert document["rates"] == {"fixture": expected}
    assert T._snapshot(document).price_table().has_rate("fixture-model")


def _multi_model_round(tmp_path, preflight):
    price = tmp_path / "multi-model.yaml"
    price.write_text("gpt-fixture: [5, 30, .5, 5]\ngpt-other: [50, 300, 5, 50]\n")
    plan = T.prepare(
        model="gpt-fixture",
        authoring_stage=Path(preflight["sources"]["performance_authoring_stage"]["path"]),
        price_table=price,
        codex_binary=preflight["sources"]["codex_binary"]["path"],
    )
    stage = tmp_path / "multi-stage"
    raw = _round_files(stage)
    evidence = T.collect_round(stage, 0, model="gpt-fixture", agent_exit_code=0, preflight_record=plan)
    wrong = T._accounting(raw, model="gpt-other", snapshot=T._snapshot(plan["price_table"]["snapshot"]["document"]))
    assert wrong["cost_provenance"]["price_table_sha256"] == evidence["price_snapshot_sha256"]
    assert wrong["subscription_notional_usd"] != evidence["accounting"]["subscription_notional_usd"]
    return plan, stage, evidence, wrong


@pytest.mark.parametrize("mutation", ["other-model", "wrong-amount"])
def test_finalizer_rederives_each_round_in_same_multimodel_snapshot(tmp_path, preflight, mutation):
    plan, stage, evidence, wrong = _multi_model_round(tmp_path, preflight)
    if mutation == "wrong-amount":
        wrong["model"] = "gpt-fixture"
        wrong["cost_provenance"].update(model_requested="gpt-fixture", model_resolved="gpt-fixture")
    evidence["accounting"] = wrong
    with pytest.raises(StageGateError, match="raw evidence and pinned model/price"):
        T.finalize(
            stage,
            [{"round": 0, "telemetry": evidence}],
            model="gpt-fixture",
            target="synthetic",
            suite="fixture",
            run_id="multi",
            preflight_record=plan,
        )
    assert not (stage / "logs/metrics.jsonl").exists()


@pytest.mark.parametrize("which", ["round", "final"])
def test_sealed_verifier_rejects_another_model_inside_the_same_snapshot(tmp_path, preflight, which):
    plan, stage, evidence, wrong = _multi_model_round(tmp_path, preflight)
    result = T.finalize(
        stage,
        [{"round": 0, "telemetry": evidence}],
        model="gpt-fixture",
        target="synthetic",
        suite="fixture",
        run_id="multi",
        preflight_record=plan,
    )
    T.verify_price_evidence(result, [{"round": 0, "telemetry": evidence}], plan)
    (evidence if which == "round" else result)["accounting"] = wrong
    with pytest.raises(StageGateError, match="different price authority"):
        T.verify_price_evidence(result, [{"round": 0, "telemetry": evidence}], plan)
