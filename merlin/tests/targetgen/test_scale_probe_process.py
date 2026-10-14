"""Real sealed-child overlap/custody controls; synthetic emission only."""

import json
import os
import subprocess
import time
from contextlib import contextmanager

import pytest
from merlin_experiments.phase0 import public_scale_sources as SOURCES
from merlin_experiments.phase1.feedback import scale_probe_child as S
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_profile_child import original  # noqa: F401 -- existing exact original-owner fixture

from merlin.common import bundle_snapshot as BW
from merlin.common import cpu_pool
from merlin.common.paths import repo_root

CONTROL = r'''
import json,os,sys,time
from pathlib import Path
from merlin.common import invocation_record
from merlin.llvmlower.device_ir_scalability import analyze_device_ir
from merlin_experiments.phase1.feedback import compile_only as C, scale_probe_child as S
def emit(candidate,source,output,**kwargs):
    barrier=output.parent/'overlap';barrier.mkdir(exist_ok=True)
    with S.cpu_pool.shared_compute_pool():
        (barrier/output.name).write_text(str(os.getpid()))
        deadline=time.monotonic()+5
        while len(list(barrier.iterdir()))<2:
            assert time.monotonic()<deadline, 'independent children were serialized or parent retained a slot'
            time.sleep(.01)
        generated=output/'generated';generated.mkdir()
        mlir=generated/'lowered.llvm.mlir';llvm=generated/'lowered.ll'
        with invocation_record.observe(output,stage='public_scale_source_lowering',argv=[sys.executable],
                inputs=(source,),outputs=(mlir,)) as emission:
            mlir.write_text('synthetic original emission')
            emission.complete(S.subprocess.CompletedProcess([],0))
        with invocation_record.observe(output,stage='public_scale_llvm_translation',argv=[sys.executable],
                inputs=(mlir,),outputs=(llvm,)) as translation:
            llvm.write_text('define void @entry() {\n  ret void\n}\n')
            translation.complete(S.subprocess.CompletedProcess([],0))
        analysis=analyze_device_ir(llvm)
        return {'lowering_status':'emitted','compilation_status':'not_attempted',
            'code_growth':{**analysis,'emitted_mlir_bytes':mlir.stat().st_size,
                'emitted_mlir_sha256':S.sha256_file(mlir)},
            'resource_budget':{'mode':'not_selected'},
            'static_obligations':{name:'UNKNOWN' for name in (
                'semantic_coverage','index_bounds','resource_legality','complete_output_coverage')},
            'emission_invocation_sha256':S.sha256_file(emission.path),
            'translation_invocation_sha256':S.sha256_file(translation.path)}
C._emit_scale_case=emit
raise SystemExit(S.main(sys.argv[1:]))
'''


@pytest.fixture
def setup(original, monkeypatch):  # noqa: F811
    import runpy

    member = runpy.run_path(str(repo_root() / "packages/merlin-experiments/tests/test_public_scale_probe.py"))["member"]
    cap, _ = member(original.private)
    cases, _ = SOURCES.derive([cap], index_bits=64, tile_edge=16)
    facts, contract = original.private / "facts.json", original.private / "contracts"
    facts.write_text(json.dumps({"facts": {"target": original.target.target, "arrays": [{"cols": 16}]}}))
    contract.mkdir()
    (contract / "contract.json").write_text("{}")
    output = original.private / "products"
    output.mkdir()
    (output / "sources").mkdir()
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) != 2:
        pytest.skip("two allowed CPUs are required for the real lease overlap control")
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(original.private / "slots"))
    monkeypatch.setenv("MERLIN_GRADE_WORKERS", "2")
    popen = subprocess.Popen

    def launch(command, **kwargs):
        assert cpu_pool._ACTIVE_LEASE.get() is None, "parent retained a compute lease before Popen"
        assert BW._ACTIVE_VERIFICATION.get() is None, "setup snapshot reuse crossed into child execution"
        index = command.index("-m")
        command = [*command[:index], "-c", CONTROL, *command[index + 2:]]
        return popen(command, **kwargs)

    monkeypatch.setattr(S.subprocess, "Popen", launch)
    original.capsules, original.cases = [cap], cases[:2]
    original.facts, original.contract, original.output = facts, contract, output
    return original


def emit(setup, *, timeout=20):
    return S.emit_cases(owner=setup.owner, candidate=setup.candidate, target_experiment=setup.target,
        capsules=setup.capsules, cases=setup.cases, contract=setup.contract, facts_path=setup.facts,
        work_root=setup.output, deadline=time.monotonic() + timeout, ir_budget=None, sandbox_inputs=setup.inputs)


def test_two_actual_sealed_children_overlap_with_independent_owners_and_ordered_results(setup):
    rows = emit(setup)
    assert len(rows) == 2 and all(row["lowering_status"] == "emitted" for row in rows)
    first, second = [row["preparation_process"] for row in rows]
    assert first["pid"] != second["pid"]
    assert max(first["started_monotonic"], second["started_monotonic"]) < min(
        first["finished_monotonic"], second["finished_monotonic"])
    assert [row["variant"] for row in rows] == ["scale1_tail+0", "scale8_tail+0"]
    assert all(row["numeric"] == row["rtl"] == "not_attempted" for row in rows)
    assert cpu_pool._ACTIVE_LEASE.get() is None


def test_child_source_mutation_refuses_before_aggregate_publication(setup, monkeypatch):
    original_popen = S.subprocess.Popen
    changed = False

    def launch(command, **kwargs):
        nonlocal changed
        if not changed:
            source = setup.output / "sources/case-000.mlir"
            source.write_text("changed original source")
            changed = True
        return original_popen(command, **kwargs)

    monkeypatch.setattr(S.subprocess, "Popen", launch)
    with pytest.raises(StageGateError, match="original input changed"):
        emit(setup)
    assert not (setup.output / "case-000/.scale_child_result.json").exists()
    assert not (setup.output / "public_compile_only.json").exists()


def test_unfinished_cases_remain_in_original_denominator_on_deadline(setup):
    rows = emit(setup, timeout=0.001)
    assert len(rows) == 2
    assert all(row["lowering_status"] == "not_attempted" for row in rows)
    assert all("budget exhausted" in row["reason"] for row in rows)


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_worker_selection_refuses(setup, monkeypatch, value):
    monkeypatch.setenv("MERLIN_GRADE_WORKERS", value)
    with pytest.raises(ValueError):
        S.workers()


def test_worker_selection_is_bounded_by_actual_shared_pool(setup, monkeypatch):
    monkeypatch.setenv("MERLIN_GRADE_WORKERS", "100")
    assert S.workers() == 2


def test_parent_reopens_products_before_accepting_child_completion(setup, monkeypatch):
    verify = S._verify_products

    def changed(root, row):
        (root / "generated/lowered.ll").write_text("changed completed output")
        return verify(root, row)

    monkeypatch.setattr(S, "_verify_products", changed)
    with pytest.raises(ValueError, match="product changed"):
        emit(setup)
    assert not (setup.output / "public_compile_only.json").exists()


def test_child_cannot_add_numerical_or_static_qualification_claims(tmp_path):
    with pytest.raises(StageGateError, match="qualification claims"):
        S._verify_products(tmp_path, {"numeric": "pass"})


def test_parent_batches_only_setup_and_final_join_with_independent_full_boundaries(setup, monkeypatch):
    digest, calls = BW.hashlib.file_digest, []

    def measured(stream, *args, **kwargs):
        if os.fspath(stream.name) == str(setup.frozen):
            calls.append(stream.name)
        return digest(stream, *args, **kwargs)

    monkeypatch.setattr(BW.hashlib, "file_digest", measured)
    rows = emit(setup)
    assert len(rows) == 2
    assert len(calls) == 4  # Setup pre/post, final join/namespace cleanup pre/post; no grade-wide reuse.
    assert BW._ACTIVE_VERIFICATION.get() is None


def test_namespace_exit_mutation_refuses_at_final_full_boundary(setup, monkeypatch):
    imports = type(setup.owner).compiler_imports

    @contextmanager
    def closing(owner):
        with imports(owner) as namespace:
            yield namespace
        if owner is setup.owner:
            path = setup.frozen
            before, raw = path.stat(), path.read_bytes()
            path.chmod(0o600)
            path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    monkeypatch.setattr(type(setup.owner), "compiler_imports", closing)
    with pytest.raises((ValueError, RuntimeError), match="snapshot|frozen|changed|digest"):
        emit(setup)
    assert not (setup.output / "public_compile_only.json").exists()
