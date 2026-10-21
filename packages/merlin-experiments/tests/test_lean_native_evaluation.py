"""The lean loop runs full numerics and invalidates cached control artifacts."""

import contextlib
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import lean_native_evaluation as L


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "compiler.py").write_text("original compiler")
    capsule, bundle = tmp_path / "capsule", tmp_path / "bundle"
    capsule.mkdir()
    bundle.mkdir()
    (capsule / "capsule.interface.mlir").write_text("original exact source")
    (capsule / "golden.npz").write_bytes(b"complete reference")
    (bundle / "inputs.npz").write_bytes(b"all saved operands")
    descriptor = tmp_path / "descriptor.yaml"
    contract = tmp_path / "selected.yaml"
    contract.write_text("target: fixture\n")
    descriptor.write_text("hardware_spec:\n  target_contract: " + str(contract) + "\n")
    facts = tmp_path / "facts.json"
    facts.write_text("{}")
    member = dict(
        id="case",
        capsule_dir=capsule,
        runtime_bundle=bundle,
        interface_mlir=capsule / "capsule.interface.mlir",
        target_descriptor=descriptor,
        rtl_facts=facts,
    )
    calls = []
    fail = {"numeric": False, "partial": False, "compile": False, "native_binding": False}
    monkeypatch.setattr(L, "shared_compute_pool", lambda **kw: contextlib.nullcontext())
    monkeypatch.setattr(L.PACKAGE, "load_package", lambda package: package)

    def compile(package, name, source, output, **kw):
        calls.append("compile")
        assert name == "emit_analysis_bundle" and source == member["interface_mlir"]
        output.write_text(
            json.dumps(
                dict(
                    kernel_abi=dict(kind="whole_program", outputs=["Y0"]),
                    tensors=dict(Y0=dict(shape=[2, 2], dtype="i8")),
                )
            )
        )
        return SimpleNamespace(
            stdout="module {}", stderr="bad compile" if fail["compile"] else "", returncode=1 if fail["compile"] else 0
        )

    def native(**kw):
        calls.append("native")
        assert kw["oracle_tier_ceiling"] == "L2" and kw["simulator"] == "spike"
        assert kw["source_intake_protocol"] == L.INTAKE and kw["complete_call_timing"] is False
        root = kw["out_dir"]
        root.mkdir()
        elf = root / "program.elf"
        elf.write_bytes(b"actual executable fixture")
        row = dict(
            status="fail" if fail["numeric"] else "pass",
            mismatch_count=int(fail["numeric"]),
            n_elements=3 if fail["partial"] else 4,
        )
        result = dict(numeric=dict(status=row["status"], policy="exact_int", per_output=dict(Y0=row)), elf=L._pin(elf))
        result["candidate"] = dict(
            command_buffer_sha256="changed"
            if fail["native_binding"]
            else hashlib.sha256(json.dumps(kw["command_buffer"], sort_keys=True).encode()).hexdigest(),
            lowered_mlir_sha256=hashlib.sha256(kw["lowered_mlir_text"].encode()).hexdigest(),
        )
        (root / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(L.PACKAGE, "run_entrypoint", compile)
    monkeypatch.setattr(L.NATIVE, "execute_candidate_model", native)
    return SimpleNamespace(member=member, package=package, root=tmp_path, calls=calls, fail=fail)


def test_direct_complete_output_path_needs_no_publication_or_source_owner(runtime):
    r = runtime
    result = L.evaluate_member(member=r.member, candidate_source=r.package, work_root=r.root / "job", timeout_s=10)
    assert r.calls == ["compile", "native"]
    assert result["status"] == "numeric_pass" and result["complete_outputs_checked"]
    assert result["qualification"] == "UNQUALIFIED" and result["source_effects"] == "PENDING"
    assert result["elf"] == L._pin(result["elf"]["path"])


@pytest.mark.parametrize("fault", ["compile", "numeric", "partial", "native_binding"])
def test_compile_failure_mismatch_or_incomplete_output_cannot_pass_or_observe_counts(runtime, fault):
    r = runtime
    r.fail[fault] = True
    counts = []
    result = L.evaluate_member(
        member=r.member,
        candidate_source=r.package,
        work_root=r.root / "job",
        timeout_s=10,
        count_runtime=object(),
        count_observer=lambda *a, **kw: counts.append(kw),
    )
    assert not result["complete_outputs_checked"] and result["status"] != "numeric_pass"
    assert counts == []


def test_count_failure_retains_the_completed_numerical_fact(runtime):
    r = runtime

    def count(*a, **kw):
        raise TimeoutError("count budget")

    result = L.evaluate_member(
        member=r.member,
        candidate_source=r.package,
        work_root=r.root / "job",
        timeout_s=10,
        count_runtime=object(),
        count_observer=count,
    )
    assert result["status"] == "numeric_pass" and result["complete_outputs_checked"]
    assert result["count_observation"]["status"] == "UNKNOWN"


def test_baseline_cached_once_and_recomputed_after_input_or_elf_mutation(runtime):
    r = runtime
    candidate = r.root / "changed-package"
    shutil.copytree(r.package, candidate)
    (candidate / "compiler.py").write_text("changed compiler")

    def evaluate(label):
        return L.evaluate_candidate(
            candidate_source=candidate,
            baseline_source=r.package,
            selected_members=(r.member,),
            work_root=r.root / label,
            timeout_s=10,
            baseline_cache_root=r.root / "cache",
        )

    first = evaluate("first")
    assert r.calls.count("native") == 2
    second = evaluate("second")
    assert second["members"][0]["baseline"]["baseline_cache_hit"] is True
    assert r.calls.count("native") == 3
    Path(first["members"][0]["baseline"]["elf"]["path"]).write_bytes(b"changed executable")
    third = evaluate("third")
    assert third["members"][0]["baseline"]["baseline_cache_hit"] is False
    assert r.calls.count("native") == 5
    fourth = evaluate("fourth")
    assert fourth["members"][0]["baseline"]["baseline_cache_hit"] is True
    (r.member["runtime_bundle"] / "inputs.npz").write_bytes(b"different operands")
    assert evaluate("fifth")["members"][0]["baseline"]["baseline_cache_hit"] is False
    assert r.calls.count("native") == 8


def test_frozen_count_model_identity_changes_cannot_reuse_the_old_baseline(runtime):
    r = runtime
    candidate = r.root / "changed-package"
    shutil.copytree(r.package, candidate)
    (candidate / "compiler.py").write_text("changed compiler")

    def count(*args, **kwargs):
        return dict(status="observed", estimated_cycles=10, complete_counts=True, timing_complete=True)

    def evaluate(label, model):
        return L.evaluate_candidate(
            candidate_source=candidate,
            baseline_source=r.package,
            selected_members=(r.member,),
            work_root=r.root / label,
            timeout_s=10,
            baseline_cache_root=r.root / "cache",
            count_runtime=object(),
            count_identity=dict(model_sha256=model),
            count_observer=count,
        )

    evaluate("first", "frozen-A")
    assert evaluate("second", "frozen-A")["members"][0]["baseline"]["baseline_cache_hit"] is True
    assert evaluate("third", "different-model")["members"][0]["baseline"]["baseline_cache_hit"] is False
    assert r.calls.count("native") == 5


def test_identical_source_reuses_one_control_observation_and_pins_its_origin(runtime):
    r = runtime
    candidate = r.root / "identical-package"
    shutil.copytree(r.package, candidate)
    counts = []

    def observe(*args, **kwargs):
        counts.append(kwargs)
        return dict(status="observed", estimated_cycles=10, complete_counts=True, timing_complete=True)

    result = L.evaluate_candidate(
        candidate_source=candidate,
        baseline_source=r.package,
        selected_members=(r.member,),
        work_root=r.root / "job",
        timeout_s=10,
        baseline_cache_root=r.root / "cache",
        count_runtime=object(),
        count_identity="frozen-count-runtime",
        count_observer=observe,
    )
    row = result["members"][0]
    assert r.calls == ["compile", "native"] and len(counts) == 1
    assert row["candidate"]["observation_origin"] == "identical_source_control_reuse"
    assert row["candidate"]["native_result"] == row["baseline"]["native_result"]
    assert row["candidate"]["count_observation"] == row["baseline"]["count_observation"]
    assert row["candidate"]["candidate_source"] == str(candidate)
    assert row["candidate"]["package_identity"] != row["baseline"]["package_identity"]
    assert L._source_bytes(row["candidate"]["package_identity"]) == L._source_bytes(row["baseline"]["package_identity"])


def test_byte_different_candidate_runs_independent_complete_outputs_and_counts(runtime):
    r = runtime
    candidate = r.root / "changed-package"
    shutil.copytree(r.package, candidate)
    (candidate / "compiler.py").write_text("different compiler")
    counts = []
    result = L.evaluate_candidate(
        candidate_source=candidate,
        baseline_source=r.package,
        selected_members=(r.member,),
        work_root=r.root / "job",
        timeout_s=10,
        baseline_cache_root=r.root / "cache",
        count_runtime=object(),
        count_identity="frozen-count-runtime",
        count_observer=lambda *a, **kw: counts.append(kw) or dict(status="observed"),
    )
    assert r.calls == ["compile", "native", "compile", "native"] and len(counts) == 2
    assert result["members"][0]["candidate"]["native_result"] != result["members"][0]["baseline"]["native_result"]
    assert result["members"][0]["candidate"].get("observation_origin") != "identical_source_control_reuse"


def test_complete_numerical_progress_is_visible_while_counts_are_pending(runtime):
    r = runtime
    seen = []

    def observe(record, **kwargs):
        private = json.loads((Path(record["work_root"]) / "evaluation.json").read_bytes())
        public = json.loads((r.root / "job/progress.json").read_bytes())
        assert private["phase"] == "observing_counts" and private["complete_outputs_checked"] is True
        assert private["count_observation"] is None
        assert public["status"] == "running"
        arm = public["members"][0]["baseline"]
        assert arm["phase"] == "observing_counts" and arm["complete_outputs_checked"] is True
        assert arm["count_status"] == "PENDING"
        assert set(arm) == {"phase", "status", "complete_outputs_checked", "count_status"}
        assert "native_result" not in json.dumps(public) and "numeric" not in arm
        seen.append(public)
        return dict(status="UNKNOWN")

    result = L.evaluate_candidate(
        candidate_source=r.package,
        baseline_source=r.package,
        selected_members=(r.member,),
        work_root=r.root / "job",
        timeout_s=10,
        baseline_cache_root=r.root / "cache",
        count_runtime=object(),
        count_identity="frozen-count-runtime",
        count_observer=observe,
    )
    assert len(seen) == 1 and result["all_complete_outputs_pass"] is True
    assert result["members"][0]["candidate"]["count_observation"]["status"] == "UNKNOWN"
    assert json.loads((r.root / "job/progress.json").read_bytes())["status"] == "finished"
