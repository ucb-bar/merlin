"""Original source/protocol controls only: no simulator or large value allocation."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import public_compile_sources as SOURCES
from merlin_experiments.phase1.brokers import selfcheck as BROKER
from merlin_experiments.phase1.feedback import compile_only as ORIGINAL
from merlin_experiments.phase1.feedback import selfcheck as WORKER
from merlin_experiments.phase2 import ordinary_compile_only as QUEUED
from merlin_experiments.phase2.contracts import StageGateError, exact_tree_record
from merlin_experiments.phase2.search_inputs import SearchInputs

from merlin.targetgen import component_program


def capsule(tmp_path, *, size=3, label="public", graph=False):
    root = tmp_path / ("graph" if graph else "matmul")
    root.mkdir()
    program = {
        "inputs": [
            {"name": "A", "role": "input", "shape": [size, size + 1], "dtype": "operand"},
            {"name": "W", "role": "weight", "shape": [size + 1, size + 2], "dtype": "operand"},
        ],
        "nodes": [{"name": "P", "op": "matmul", "inputs": ["A", "W"]}],
        "outputs": [{"name": "Y", "value": "P"}],
    }
    if graph:
        program["nodes"].append({"name": "C", "op": "copy", "inputs": ["P"]})
        program["outputs"][0]["value"] = "C"
    typed, source = component_program.render(program, operand_dtype="i8", accumulator_dtype="i32")
    descriptor = {
        "name": root.name,
        "label": label,
        "kind": "isa",
        "linalg_mlir": "source.mlir",
        "component_program": typed,
        "operation": {"op": "component_program", "attributes": {"program": program}},
        "component_coverage": {"resource_boundaries": {"M": {"capacity_sha256": "f" * 64}}},
        "performance": {"arms": {"candidate": {"instruction_policy": {"prohibited_instruction_roles": []}}}},
    }
    (root / "capsule.yaml").write_text(json.dumps(descriptor))
    (root / "source.mlir").write_text(source)
    # No reference or data files exist. Discovery must work without them.
    return {**descriptor, "__dir__": str(root)}


def test_public_scale_and_frontier_cases_allocate_only_source_text(tmp_path):
    original = capsule(tmp_path, size=1 << 20)
    cases, manifest = SOURCES.derive([original], index_bits=64)
    assert [case.scale for case in cases] == list(SOURCES.SCALES)
    assert cases[-1].original_abi.inputs[0].shape[0] == (1 << 20) * 64
    assert max(len(case.source_ir) for case in cases) < 4096
    assert all(case.record()["derivation"]["tensor_values_allocated"] is False for case in cases)
    assert cases[0].source_ir == (Path(original["__dir__"]) / "source.mlir").read_text()
    assert manifest["numeric"] == manifest["rtl"] == "not_attempted"
    assert manifest["cases"][0]["derivation"]["resource_frontier"]
    assert not list(tmp_path.rglob("*.npz")) and not list(tmp_path.rglob("*.npy"))


def test_public_composition_stays_exact_and_hidden_source_is_not_opened(tmp_path):
    original = capsule(tmp_path, graph=True)
    hidden = {"name": "withheld", "label": "hidden", "__dir__": str(tmp_path / "absent-private")}
    cases, manifest = SOURCES.derive([original, hidden])
    assert len(cases) == 1 and cases[0].scale == 1
    assert cases[0].source_ir == (Path(original["__dir__"]) / "source.mlir").read_text()
    assert manifest["unavailable"] == []
    with pytest.raises(ValueError, match="non-public"):
        SOURCES.derive([original, hidden], selected="withheld")


def test_original_source_and_metadata_cannot_change_after_case_selection(tmp_path):
    original = capsule(tmp_path)
    cases, _ = SOURCES.derive([original])
    (Path(original["__dir__"]) / "source.mlir").write_text("mutated")
    with pytest.raises(ValueError, match="changed"):
        cases[0].verify()


def test_source_links_and_captured_values_are_unavailable(tmp_path):
    original = capsule(tmp_path)
    source = Path(original["__dir__"]) / "source.mlir"
    saved = source.read_text()
    source.unlink()
    source.symlink_to(tmp_path / "not-source")
    cases, manifest = SOURCES.derive([original])
    assert not cases and manifest["unavailable"]
    source.unlink()
    source.write_text('// prov.weights_file = "payload.safetensors"\n' + saved)
    assert not SOURCES.derive([original])[0]


def test_oversized_roster_never_keeps_half_issued_member(tmp_path, monkeypatch):
    original = capsule(tmp_path)
    monkeypatch.setattr(SOURCES, "MAX_CASES", 2)
    cases, manifest = SOURCES.derive([original])
    assert not cases and len(manifest["unavailable"]) == 1


def test_bound_compiler_snapshot_excludes_builds_and_is_immutable(tmp_path):
    source = tmp_path / "compiler"
    source.mkdir()
    (source / "main.py").write_text("return_value = 1\n")
    (source / "build").mkdir()
    (source / "build/huge.bin").write_bytes(b"ignored-generated-data")
    selected = ORIGINAL.snapshot_submission(source, tmp_path / "selected")
    assert not (selected / "build").exists()
    assert not (selected / "main.py").stat().st_mode & 0o222
    (source / "main.py").write_text("unfinished = [\n")
    assert (selected / "main.py").read_text() == "return_value = 1\n"


def test_original_compile_observer_consumes_linked_products_not_candidate_claims(tmp_path, monkeypatch):
    original = capsule(tmp_path)
    candidate, contract = tmp_path / "candidate", tmp_path / "contract"
    candidate.mkdir()
    contract.mkdir()
    (candidate / "compiler.py").write_text("fake_compile_success = True\n")
    (contract / "schema.json").write_text("{}")
    facts = tmp_path / "facts"
    facts.mkdir()
    fact_path = facts / "facts.json"
    fact_path.write_text('{"facts":{"target":"fixture"}}')
    import merlin.llvmlower.toolchain as tools
    import merlin.targetgen.native_model_execution as native

    monkeypatch.setattr(native, "_build_service_for", lambda target: object())
    tool = tmp_path / "readelf"
    tool.write_text("unused test executable")
    monkeypatch.setattr(tools, "readelf", lambda: tool)
    calls = []
    owner = object.__new__(ORIGINAL.CommonTransformService)
    owner.profile = {"sha256": "p" * 64, "host_index_bits": 64}
    owner._verify = lambda: None
    owner.compiler_imports = nullcontext

    def compile_source_only(**kwargs):
        assert ORIGINAL.P.active_package_executor() is not None
        assert "inputs" not in kwargs and "goldens" not in kwargs
        calls.append(kwargs)
        kwargs["output_root"].mkdir()
        (kwargs["output_root"] / "compile_only_result.json").write_text("{}")

    def verify(path, **kwargs):
        assert path.name == "compile_only_result.json"
        return {
            "compilation_status": "linked",
            "retained_entry": {"symbol": "kernel", "size_bytes": 17},
            "elf": {"sha256": "e" * 64},
            "instruction_policy": {"status": "accepted"},
            "products": {"kernel.o": {"sha256": "a" * 64}},
            "invocations": [{"sha256": "b" * 64}],
        }

    monkeypatch.setattr(ORIGINAL.CO, "compile_source_only", compile_source_only)
    monkeypatch.setattr(ORIGINAL.CO, "verify_compile_only_report", verify)
    result = ORIGINAL.observe(
        candidate=candidate,
        target_experiment=SimpleNamespace(target="fixture"),
        capsules=[original],
        contract=contract,
        facts_path=fact_path,
        common_service=owner,
        work_root=tmp_path / "products",
        timeout_s=30,
    )
    assert len(calls) == len(SOURCES.SCALES)
    assert result["candidate_sha256"] == exact_tree_record(candidate)["sha256"]
    assert result["all_pass"] is False and result["timing_cycles"] is None
    assert result["numeric"] == result["rtl"] == "not_attempted"
    assert all(row["elf_sha256"] == "e" * 64 and row["invocation_sha256s"] == ["b" * 64] for row in result["cases"])
    assert all(set(row["static_obligations"].values()) == {"UNKNOWN"} for row in result["cases"])


def test_original_compile_failure_cannot_credit_success_fields(tmp_path, monkeypatch):
    assert "compile_only" in BROKER.FORWARDED_PROBES
    assert BROKER.FORWARDED_PROBES["compile_only"] == "--compile-only"
    command = WORKER.worker_command(
        SimpleNamespace(descriptor=tmp_path / "descriptor", repo=tmp_path, readback_policy=None),
        tmp_path / "public",
        compiler_bundle_manifest=tmp_path / "original-bundle",
    )
    assert command[-2:] == ["--compiler-bundle-manifest", str(tmp_path / "original-bundle")]


def test_no_instruction_policy_is_not_implicit_permission(tmp_path):
    original = capsule(tmp_path)
    original.pop("performance")
    with pytest.raises(ValueError, match="explicit original"):
        ORIGINAL._roles([original])


def test_plain_profile_dict_cannot_select_compiler_import_authority(tmp_path):
    with pytest.raises(ValueError, match="original admitted"):
        ORIGINAL.observe(
            candidate=tmp_path,
            target_experiment=object(),
            capsules=[],
            contract=tmp_path,
            facts_path=tmp_path,
            common_service={"sha256": "p" * 64},
            work_root=tmp_path / "products",
            timeout_s=30,
        )


def test_compile_only_cost_can_never_rank_as_measured_champion(feedback):
    from merlin_experiments.phase2.search_coordinator import rank_measured_contenders
    from merlin_experiments.phase2.search_inputs import ORDINARY_KIND

    raw, _policy, revision = feedback
    result = rank_measured_contenders(
        [{"state": "completed", "kind": QUEUED.KIND, "result": raw, "source_sha256": revision.source_sha256}],
        roster=(("public", "member"),),
        corpus_sha256="c" * 64,
        configuration_sha256="r" * 64,
        timing_kind=ORDINARY_KIND,
    )
    assert result["ranked"] == []


@pytest.fixture
def feedback(monkeypatch):
    profile = {"sha256": "p" * 64}
    policy = SimpleNamespace(search_corpus=SimpleNamespace(capsules_sha256="c" * 64))
    revision = SimpleNamespace(source_sha256="s" * 64)
    monkeypatch.setattr(QUEUED, "selection", lambda _: profile)
    body = {
        "schema": ORIGINAL.SCHEMA,
        "candidate_sha256": revision.source_sha256,
        "compiler_tool_profile_sha256": profile["sha256"],
        "cases": [],
        "numeric": "not_attempted",
        "rtl": "not_attempted",
        "all_pass": False,
        "timing_cycles": None,
    }
    raw = {
        "schema": QUEUED.SCHEMA,
        "kind": QUEUED.KIND,
        "candidate_sha256": revision.source_sha256,
        "compiler_tool_profile_sha256": profile["sha256"],
        "tuning_corpus_sha256": "c" * 64,
        "feedback": body,
        "numeric": "not_attempted",
        "rtl": "not_attempted",
        "timing_cycles": None,
        "promotion": "NONE",
    }
    return raw, policy, revision


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_sha256", "z" * 64),
        ("numeric", "pass"),
        ("rtl", "pass"),
        ("timing_cycles", 10),
        ("promotion", "PROMOTE"),
        ("compiler_tool_profile_sha256", "z" * 64),
    ],
)
def test_queued_compile_only_refuses_false_authority(feedback, field, value):
    raw, policy, revision = feedback
    assert QUEUED.validate_feedback(raw, policy=policy, revision=revision) is raw
    raw[field] = value
    with pytest.raises(StageGateError):
        QUEUED.validate_feedback(raw, policy=policy, revision=revision)


def test_queued_compile_only_refuses_nested_numeric_credit(feedback):
    raw, policy, revision = feedback
    raw["feedback"]["cases"] = [{"numeric": "pass", "rtl": "not_attempted"}]
    with pytest.raises(StageGateError):
        QUEUED.validate_feedback(raw, policy=policy, revision=revision)


def test_queue_uses_compile_only_only_when_phase1_admitted(monkeypatch):
    inputs = object.__new__(SearchInputs)
    inputs.ordinary, inputs.policy = True, object()
    monkeypatch.setattr(QUEUED, "available", lambda _: False)
    assert QUEUED.KIND not in inputs.kinds
    monkeypatch.setattr(QUEUED, "available", lambda _: True)
    assert inputs.kinds[-1] == QUEUED.KIND
    assert inputs.timing_kind != QUEUED.KIND


def test_compile_only_uses_cpu_grant_without_waiting_for_busy_engine(tmp_path):
    from merlin_experiments.phase2 import evaluation_pool as EP
    from merlin_experiments.phase2.search_inputs import ORDINARY_KIND

    pool = object.__new__(EP.ComponentEvaluationPool)
    pool.inputs = SimpleNamespace(ordinary=True)
    pool.resources = SimpleNamespace(cpu_ids=(0, 1, 2), author_cores=0)
    pool.grant = EP.JobResources(1, 1024, 1024, 1)
    pool.capacity = EP.JobResources(3, 4096, 4096, 1)
    pool.beam = 3
    running = SimpleNamespace(kind=ORDINARY_KIND, state="cleanup_unknown", cpus=(0,))
    waiting = SimpleNamespace(kind=ORDINARY_KIND, state="queued", cpus=())
    compile_only = SimpleNamespace(kind=QUEUED.KIND, state="queued", cpus=())
    pool._jobs = {"active": running, "waiting": waiting, "compile": compile_only}
    pool._retained_output_bytes = lambda: 0
    started = []

    def start(job, cpus):
        job.state, job.cpus = "running", cpus
        started.append(job)

    pool._start = start
    pool._tick()
    assert started == [compile_only] and compile_only.cpus == (1,)
    assert waiting.state == "queued"
    assert EP._engine_slots(compile_only, pool.grant, ordinary=False) == 1
