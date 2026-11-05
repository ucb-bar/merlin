"""Real sealed process overlap/custody; synthetic commands grant no target proof."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
import yaml
from merlin_experiments.phase1.feedback import compiler_profile_process as E
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2.contracts import StageGateError
from test_compiler_profile_child import original  # noqa: F401

from merlin.common import cpu_pool
from merlin.targetgen import capsule_common as CC
from merlin.targetgen import package_runtime as P

BODY = """def lower_interface(pkg, interface, generated, **kwargs):
    import json,os,sys,time
    from pathlib import Path
    from merlin.common import invocation_record,cpu_pool
    generated=Path(generated)
    barrier=generated.parent/'overlap';barrier.mkdir(exist_ok=True)
    with cpu_pool.shared_compute_pool():
        (barrier/generated.name).write_text(str(os.getpid()))
        end=time.monotonic()+10
        while len(list(barrier.iterdir()))<2:
            if time.monotonic()>end: raise RuntimeError('workers did not overlap')
            time.sleep(.01)
        incoming=generated/'input.interface.mlir';incoming.write_bytes(Path(interface).read_bytes())
        for command in ('parse','lower_interface_to_target','emit_analysis_bundle'):
            result=invocation_record.run([sys.executable,'-c','pass',command],
                directory=generated,stage='original_profile_package',inputs=(incoming,),
                capture_output=True,text=True,timeout=5)
            assert result.returncode==0
        (generated/'lowered.target.mlir').write_text('module {}')
        buffer={'abi_version':'0.1','target':pkg.target,'tensors':{
            'Y':{'role':'output','shape':[1,1],'dtype':'i32'}},'commands':[]}
        (generated/'command_buffer.json').write_text(json.dumps(buffer))
        (generated/kwargs['artifact_name']).write_text('module {}')
    return buffer,'module {}'
"""

CONTROL = """import sys
from merlin.targetgen import capsule_common as CC
from merlin_experiments.phase1.feedback import compiler_profile_process as E
namespace={'__name__':CC.__name__}
exec(compile(BODY_TOKEN,CC.__file__,'exec'),namespace)
CC.lower_interface.__wrapped__=namespace['lower_interface']
raise SystemExit(E.main(sys.argv[1:]))
""".replace("BODY_TOKEN", repr(BODY))


@pytest.fixture
def setup(original, monkeypatch):  # noqa: F811
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": original.target.target,
        "language": "python",
        "authoring": {"mode": "hand_curated"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "tool.py"},
        "commands": {
            name: {"argv": ["python3", "{tool}", "{input_mlir}"]}
            for name in (
                "parse",
                "lower_interface_to_target",
                "emit_command_buffer",
                "lower_target_to_llvm",
                "emit_analysis_bundle",
            )
        },
    }
    (original.candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    original.pkg = P.load_package(original.candidate)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) != 2:
        pytest.skip("two allowed CPUs required")
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, cpus)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(original.private / "slots"))
    monkeypatch.setenv(E.ENV, "1")
    original.products = original.private / "products"
    original.products.mkdir()
    original.sources = []
    for n in range(2):
        source = original.private / f"input-{n}.mlir"
        source.write_text("module {}")
        original.sources.append(source)
    original.launched = []

    def command(spec, result, transport):
        return [sys.executable, "-B", "-c", CONTROL, "--spec", str(spec), "--result", str(result), *transport.argv]

    monkeypatch.setattr(E, "_command", command)
    real_popen = subprocess.Popen

    def launch(argv, **kwargs):
        assert not cpu_pool.current_thread_owns_cpu_lease()
        assert kwargs["start_new_session"] and kwargs["pass_fds"]
        process = real_popen(argv, **kwargs)
        original.launched.append(process)
        return process

    monkeypatch.setattr(E.subprocess, "Popen", launch)
    return original


def emit(setup):
    with R.original_compiler_execution(
        setup.owner,
        target_experiment=setup.target,
        candidate=setup.candidate,
        work_root=setup.private / "execution",
        sandbox_inputs=setup.inputs,
    ):
        executor = P.active_package_executor()

        def job(index):
            with executor.concurrent_entrypoint_scope_v1():
                return CC.lower_interface(
                    setup.pkg, setup.sources[index], setup.products / f"job-{index}", contract=None, timeout=10
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, job, index) for index in range(2)]
            return [future.result(timeout=40) for future in futures]


def test_two_actual_sealed_processes_preserve_parent_live_observer_and_all_commands(setup):
    rows = emit(setup)
    assert len(rows) == 2 and all(row[1] == "module {}" for row in rows)
    assert len(set(p.pid for p in setup.launched)) == 2 and all(p.poll() == 0 for p in setup.launched)
    for directory in sorted(setup.products.glob("job-*")):
        result = json.loads((directory / ".original-interface-child/result.json").read_text())
        assert result["status"] == "emitted" and result["original_compiler_child"]
        observed = list(directory.glob("emission-custody-*/invocations/*/invocation.json"))
        assert len(observed) == 1  # Parent owns the one actual live call, no child replay issuer.
        assert json.loads(observed[0].read_text())["status"] == "completed"
        records = [json.loads(p.read_text()) for p in directory.glob("invocations/*/invocation.json")]
        assert [p["argv"][-1] for p in sorted(records, key=lambda r: r["started_ns"])] == [
            "parse",
            "lower_interface_to_target",
            "emit_analysis_bundle",
        ]
        assert json.loads((directory / ".original-interface-child/process.json").read_text())["reaped"]
    assert P.active_package_executor() is None


def test_source_edit_after_sealing_refuses_and_reaps_every_started_child(setup, monkeypatch):
    launch = E.subprocess.Popen
    changed = False

    def mutate(command, **kwargs):
        nonlocal changed
        if not changed:
            setup.sources[0].write_text("module { changed }")
            changed = True
        return launch(command, **kwargs)

    monkeypatch.setattr(E.subprocess, "Popen", mutate)
    with pytest.raises((StageGateError, RuntimeError, ValueError)):
        emit(setup)
    assert all(p.poll() is not None for p in setup.launched)


def test_original_typed_plane_refusal_survives_process_boundary(setup, monkeypatch):
    body = """def lower_interface(pkg, interface, generated, **kwargs):
    from merlin.targetgen.package_runtime import CertFailure
    raise CertFailure('parse','tool_crash','selected original refusal')
"""
    control = CONTROL.replace(repr(BODY), repr(body))

    def command(spec, result, transport):
        return [sys.executable, "-B", "-c", control, "--spec", str(spec), "--result", str(result), *transport.argv]

    monkeypatch.setattr(E, "_command", command)
    with pytest.raises(P.CertFailure, match="selected original refusal") as error:
        emit(setup)
    assert error.value.plane == "parse" and error.value.category.value == "tool_crash"
    assert all(p.poll() == 0 for p in setup.launched)
    assert not list(setup.products.glob("job-*/lowered.llvm.mlir"))


@pytest.mark.parametrize("choice", ["-1", "2", "invalid"])
def test_invalid_process_selection_refuses(monkeypatch, choice):
    monkeypatch.setenv(E.ENV, choice)
    with pytest.raises(ValueError, match="requires 0 or 1"):
        E.enabled()


def test_default_keeps_existing_local_protocol(monkeypatch):
    monkeypatch.delenv(E.ENV, raising=False)
    assert not E.enabled()


def test_worker_output_mutation_and_missing_target_plane_refuse(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    for name, text in {
        "input.interface.mlir": "module {}",
        "lowered.target.mlir": "",
        "command_buffer.json": "{}",
        "lowered.llvm.mlir": "module {}",
    }.items():
        (tmp_path / name).write_text(text)
    job = {"generated": str(tmp_path), "artifact_name": "lowered.llvm.mlir", "source": E.CHILD._pin(source)}
    row = {"products": E._products(tmp_path, "lowered.llvm.mlir")}
    with pytest.raises(StageGateError, match="target plane"):
        E._read_products(job, row)
    (tmp_path / "lowered.llvm.mlir").write_text("different bytes")
    with pytest.raises(StageGateError, match="another source or product"):
        E._read_products(job, row)
