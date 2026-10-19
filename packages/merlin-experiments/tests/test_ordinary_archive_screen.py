"""Original queue/admission controls; synthetic labels establish no hardware timing."""

import copy
import json

import pytest
from merlin_experiments.phase1 import annotated_spike_evidence as E
from merlin_experiments.phase2 import archive_spike_runtime as RT
from merlin_experiments.phase2 import ordinary_archive_screen as S
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_timing import archive as A
from merlin_timing import contracts as C
from merlin_timing import fitting as FIT
from test_ordinary_fast_feedback import cheap_inputs as cheap_inputs
from test_ordinary_fast_feedback import revisions
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs


def _dataset(domain):
    data = {
        "schema": C.DATASET,
        "profile": "phase1-public",
        "domain": domain,
        "feature_names": ["host_instructions"],
        "cases": [],
        "records": [],
    }
    for group in range(12):
        split = "train" if group < 4 else "calibration" if group < 6 else "test"
        for variant, count in enumerate((10, 20, 30, 40)):
            name = f"{group}-{variant}"
            case = {
                "case_id": name,
                "source_group_sha256": C.digest(["group", group]),
                "workload_sha256": "",
                "mechanism_family": ("conv", "matmul", "resident_chain")[group % 3],
                "split": split,
                "source": {
                    "visibility": "public",
                    "phase": "phase1",
                    "manifest_sha256": C.digest("manifest"),
                    "source_sha256": C.digest(["source", group]),
                },
                "elf_sha256": C.digest(name),
                "input_sha256": C.digest(["input", group]),
                "scope_sha256": domain["scope_sha256"],
                "device_sha256": domain["device_sha256"],
            }
            case["workload_sha256"] = C.workload(case)
            execution = C.execution_identity(case)
            data["cases"].append(case)
            data["records"].append(
                {
                    "record_id": name,
                    "case_id": name,
                    "features": {"host_instructions": count},
                    "feature_provenance": {
                        "engine_sha256": domain["feature_engine_sha256"],
                        "model_sha256": domain["target_model_sha256"],
                        "trace_sha256": C.digest([name, "trace"]),
                        "execution_sha256": execution,
                        "complete": True,
                    },
                    "label": {
                        "engine": "gsim",
                        "engine_sha256": domain["label_engine_sha256"],
                        "receipt_sha256": C.digest([name, "label"]),
                        "execution_sha256": execution,
                        "status": "pass",
                        "numeric": "pass",
                        "measured_now": True,
                        "cycles": 5 + 3 * count,
                    },
                    "event_report": None,
                }
            )
    return data


@pytest.fixture
def source(ordinary_inputs, tmp_path, monkeypatch):
    root = tmp_path / "archive-source"
    root.mkdir()
    files = {}
    for name in ("functional", "observer", "objdump", "library", "provider", "counter", "archive"):
        files[name] = root / name
        files[name].write_text("synthetic " + name)
    selection = {
        "schema": "merlin.kernel_call_timing.v1",
        "selection_sha256": C.digest("counter selection"),
        "counter_asm": "read %0, count",
        "boundary": "before entry through completion",
        "counter_semantics": {
            "unit": "cycles",
            "width_bits": 64,
            "wrap": "modulo",
            "clock": "synthetic",
            "source_pins": [{"path": str(files["counter"]), "sha256": sha256_file(files["counter"])}],
        },
        "max_delta_cycles": 100000,
    }
    domain = {name: C.digest(name) for name in C.DOMAIN_KEYS - {"label_engine", "target_model_sha256"}}
    domain.update(label_engine="gsim", scope_sha256=E.kernel_call_scope(selection))
    model = {
        "schema": E.MODEL_SCHEMA,
        "profile": "phase1-public",
        "domain": domain,
        "scope": {"kind": "counter_pc_pair"},
        "cpu": {"isa": "synthetic"},
        "libraries": [sha256_file(files[n]) for n in ("library", "provider")],
    }
    model_path, manifest = root / "model.json", root / "source.json"
    model_path.write_text(json.dumps(model))
    manifest.write_text(json.dumps({"sha256": domain["feature_engine_sha256"]}))
    runtime = RT.freeze_runtime(
        review_id="synthetic runtime control",
        functional_argv=(str(files["functional"]), "--isa=synthetic", "--extlib=" + str(files["library"])),
        observer=files["observer"],
        disassembler=files["objdump"],
        model_path=model_path,
        source_manifest=manifest,
        libraries=(files["library"], files["provider"]),
        dependencies=(),
        entry_symbol="entry",
        completion_symbol="complete",
        timing_selection=selection,
    )
    domain = dict(domain, target_model_sha256=sha256_file(model_path))
    dataset = _dataset(domain)
    observations = {
        "reports": {
            r["record_id"]: {"report": {"feature_counts": {**r["features"], "unfitted.fp": 0, "constant.issue": 2}}}
            for r in dataset["records"]
        }
    }
    binding_path, observations_path, screen_path = (
        root / n for n in ("binding.json", "observations.json", "screen.json")
    )
    binding_path.write_text(json.dumps({"domain": domain, "review": {"timing_selection": selection}}))
    observations_path.write_text(json.dumps(observations))
    screen_path.write_text(json.dumps(FIT.fit(dataset)))
    calls = []
    monkeypatch.setattr(
        A, "inspect_archive", lambda *a: calls.append("inspect") or {"export_scope": "training_eligible_public"}
    )
    monkeypatch.setattr(A, "prepare", lambda *a: calls.append("prepare") or {"synthetic": "plan"})
    monkeypatch.setattr(A, "attach", lambda *a: calls.append("attach") or copy.deepcopy(dataset))
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    arguments = dict(
        review_id="synthetic archive correspondence control",
        archive=files["archive"],
        archive_sha256=sha256_file(files["archive"]),
        binding_path=binding_path,
        observations_path=observations_path,
        screen_path=screen_path,
        runtime=runtime,
        expected_rtl_facts_sha256=sha256_file(policy.feedback_evaluator.rtl_facts_path),
    )
    return locals()


def test_archive_intake_reconstructs_public_data_and_all_held_gates_before_original_queue(source):
    policy = source["policy"]
    owner = S.issue_archive_screen(policy, **source["arguments"])
    assert source["calls"] == ["inspect", "prepare", "attach"]
    assert owner.summary()["metrics"]["eligible_held_pairs"] == 36
    assert owner.summary()["raw_activity_features_guarded"] == 3
    assert owner.summary()["event_engine_qualified"] is False
    before = policy.search_closure_sha256
    policy.install_screen(owner)
    owner.verify(policy)
    assert policy.search_closure_sha256 != before
    with pytest.raises(StageGateError, match="before queue admission"):
        S.issue_archive_screen(policy, **source["arguments"])


@pytest.mark.parametrize("damage", ["hidden", "held_error", "raw_held", "facts", "runtime", "private_profile"])
def test_archive_admission_cannot_hide_source_domain_or_held_failures(source, damage):
    data, arguments = source["dataset"], source["arguments"]
    if damage == "hidden":
        data["cases"][0]["source"]["visibility"] = "hidden"
    elif damage == "held_error":
        for case, record in zip(data["cases"], data["records"], strict=True):
            if case["split"] == "test":
                record["label"]["cycles"] = 150 - 3 * record["features"]["host_instructions"]
    elif damage == "raw_held":
        observations = source["observations"]
        observations["reports"][data["cases"][-1]["case_id"]]["report"]["feature_counts"]["unfitted.fp"] = 1
        source["observations_path"].write_text(json.dumps(observations))
    elif damage == "facts":
        arguments["expected_rtl_facts_sha256"] = "f" * 64
    elif damage == "runtime":
        source["files"]["observer"].write_text("changed")
    else:
        data["profile"] = "historical-private"
    with pytest.raises((ValueError, StageGateError)):
        S.issue_archive_screen(source["policy"], **arguments)


def test_unfitted_and_constant_activity_cannot_be_priced_as_free(source):
    owner = S.issue_archive_screen(source["policy"], **source["arguments"])
    ranges = json.loads(owner.ranges_json)
    assert (
        S._domain_reason({"feature_counts": {"host_instructions": 20, "unfitted.fp": 0, "constant.issue": 2}}, ranges)
        is None
    )
    assert "unfitted.fp" in S._domain_reason(
        {"feature_counts": {"host_instructions": 20, "unfitted.fp": 1, "constant.issue": 2}}, ranges
    )
    assert S._domain_reason({"feature_counts": {"host_instructions": 20}}, ranges)


def test_original_archive_source_drift_refuses_after_installation(source):
    owner = S.issue_archive_screen(source["policy"], **source["arguments"])
    source["files"]["archive"].write_text("changed original labels")
    with pytest.raises(StageGateError, match="evidence changed"):
        owner.verify(source["policy"])


def test_changed_runtime_cannot_replace_original_functional_library_or_flags(source):
    runtime = source["runtime"]
    scope = source["root"] / "scope.json"
    scope.write_text(json.dumps({"input_sha256": C.digest("inputs")}))
    original, annotated = runtime.commands(source["root"] / "own.elf", source["root"] / "report.json", scope)
    assert original[:-1] == list(runtime.functional_argv)
    assert annotated[0] == str(runtime.observer)
    assert [v for v in annotated if v.startswith("--extlib=")] == ["--extlib=" + str(p) for p in runtime.libraries]
    source["files"]["library"].write_text("foreign functional implementation")
    with pytest.raises(StageGateError, match="runtime source/tool/model changed"):
        runtime.commands(source["root"] / "own.elf", source["root"] / "report.json", scope)


def test_normal_cheap_queue_preserves_every_unknown_member_and_original_final_obligations(
    source, cheap_inputs, tmp_path
):
    policy = OS.prepare_ordinary_search_policy(**source["ordinary_inputs"].arguments)
    owner = S.issue_archive_screen(policy, **source["arguments"])
    policy.install_screen(owner)
    candidate, baseline = revisions(policy, tmp_path)
    result = policy.evaluate_retained(
        candidate, baseline, kind=FAST.KIND, work_root=tmp_path / "cheap-archive", call_index=0, timeout_s=10
    )
    assert result["schema"] == FAST.ARCHIVE_SCHEMA
    assert result["promotion"] == "SCREENING_ONLY"
    assert result["screening"]["promotion_authority"] == "NONE"
    assert len(result["screening"]["evidence"]) == len(policy.search_corpus.capsules)
    for cell in result["cells"]:
        for arm in ("baseline", "candidate"):
            assert cell[arm]["archive_counts"]["status"] == "UNKNOWN"
            assert cell[arm]["pending_tiers"] == ["L3"]
            assert cell[arm]["timing_cycles"] is None
    assert all(row["conservative_saved_cycles"] is None for row in result["screening"]["evidence"])
