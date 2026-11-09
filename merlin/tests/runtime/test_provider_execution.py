"""Native recipe byte integrity and owned process lifetime are executable gates."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from merlin.compile import provider_execution as execution
from merlin.runtime.backends import device_simulator as ds


def bound(path):
    return {"path": str(path), "sha256": ds.sha(path)}


def script(path, body):
    path.write_text("#!" + sys.executable + "\n" + body + "\n")
    path.chmod(0o755)
    return path


def fixture(tmp_path, *, gate_pass=True, simulator_body=None):
    compiler = script(
        tmp_path / "compiler",
        "import sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'owned-object')",
    )
    simulator = script(tmp_path / "simulator", simulator_body or "print('OUT 1 2 3\\nDONE')")
    source = tmp_path / "harness.c"
    source.write_text("complete harness source")
    linker = tmp_path / "link.ld"
    linker.write_text("explicit linked address contract")
    checker = script(
        tmp_path / "checker.py",
        """import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--console',type=Path);p.add_argument('--output',type=Path);p.add_argument('--counter-semantics')
a=p.parse_args();a.output.mkdir()
complete=a.console.read_text().splitlines()[:2]==['OUT 1 2 3','DONE']
(a.output/'report.json').write_text(json.dumps({'pass':complete and GATE_PASS}))
""".replace("GATE_PASS", str(gate_pass)),
    )
    descriptor = {
        "available": True,
        "binary": str(simulator),
        "binary_sha256": ds.sha(simulator),
        "per_hart_isa": {"0": "rv64gcv_xunit", "1": "rv64gcv_xunit", "2": "rv64gcv"},
        "matrix_hart_ids": [0, 1],
        "memory_regions": [[0x80000000, 0x10000000]],
        "launch_args": ["--boot-device=explicit"],
        "counter_semantics": "instruction_count_not_modeled_cycles",
        "accuracy_2percent_certified": False,
    }
    descriptor_path = tmp_path / "simulator.json"
    descriptor_path.write_text(json.dumps(descriptor))
    work = tmp_path / "lowered"
    work.mkdir()
    obj = work / "model.o"
    obj.write_bytes(b"exact compiled object")
    ll = work / "model.ll"
    ll.write_text("define void @forward(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {\n ret void\n}\n")
    (work / "provider-workflow.json").write_text(
        json.dumps(
            {
                "status": "compiled",
                "object_sha256": ds.sha(obj),
                "llvm_sha256": ds.sha(ll),
                "input_sha256": {},
                "additional_objects": [],
            }
        )
    )
    plan = {
        "schema": "merlin.provider_execution.v1",
        "compiler": bound(compiler),
        "sources": [bound(source)],
        "objects": [],
        "linker_script": bound(linker),
        "compile_flags": [],
        "link_flags": [],
        "simulator": bound(descriptor_path),
        "isa": "rv64gcv",
        "harts": 3,
        "active_hart_ids": [0],
        "strict_postgate": {"python": bound(__import__("pathlib").Path(sys.executable)), "script": bound(checker)},
        "timeout_seconds": 5,
        "postgate_timeout_seconds": 5,
        "entry_abi": {"symbol": "forward", "argument_ranks": [1]},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    lowered = SimpleNamespace(workdir=work, riscv_obj=obj, ll_path=ll, stats={})
    return path, lowered, source, descriptor


def test_native_plan_preserves_three_harts_map_isa_false_qualifications_and_full_gate(tmp_path):
    plan, lowered, _, descriptor = fixture(tmp_path)
    report = execution.execute(lowered, plan)
    assert report["status"] == "passed_supplied_gate"
    assert report["hardware_contacted"] is False and report["timing_certification"] is False
    command = report["native"]["argv"]
    assert "-p3" in command and "--hart-isa=2:rv64gcv" in command
    assert "-m0x80000000:0x10000000" in command
    assert report["native"]["simulator"]["accuracy_2percent_certified"] is False
    assert report["counter_semantics"] == descriptor["counter_semantics"]
    assert lowered.stats["execution"].endswith("receipt.json")


def test_zero_guest_exit_is_not_a_numerical_pass(tmp_path):
    plan, lowered, _, _ = fixture(tmp_path, gate_pass=False)
    with pytest.raises(RuntimeError, match="postgate refused"):
        execution.execute(lowered, plan)
    saved = json.loads((lowered.workdir / "execution/receipt.json").read_text())
    assert saved["status"] == "failed" and saved["native"]["returncode"] == 0


def test_changed_harness_refuses_before_compiler_launch(tmp_path):
    plan, lowered, source, _ = fixture(tmp_path)
    source.write_text("changed after plan acceptance")
    with pytest.raises(ds.DeviceSimulatorError):
        execution.execute(lowered, plan)
    assert not (lowered.workdir / "execution/image.elf").exists()


def test_changed_compiled_object_refuses_before_harness_build(tmp_path):
    plan, lowered, _, _ = fixture(tmp_path)
    lowered.riscv_obj.write_bytes(b"changed object")
    with pytest.raises(ValueError, match="compiled object"):
        execution.execute(lowered, plan)


def test_unknown_workers_cannot_use_restricted_matrix_unit(tmp_path):
    plan, lowered, _, _ = fixture(tmp_path)
    data = json.loads(plan.read_text())
    data["active_hart_ids"] = [2]
    plan.write_text(json.dumps(data))
    with pytest.raises(ds.DeviceSimulatorError, match="lacks"):
        execution.execute(lowered, plan)


def test_bound_venv_interpreter_keeps_lexical_invocation_path(tmp_path):
    from pathlib import Path

    target = Path(sys.executable)
    alias = tmp_path / "venv-python"
    alias.symlink_to(target)
    assert execution._bound(bound(alias), tmp_path, {}) == alias


def test_wrong_entry_signature_refuses_before_harness_build(tmp_path):
    plan, lowered, _, _ = fixture(tmp_path)
    lowered.ll_path.write_text("define i64 @forward(ptr %0) {\n ret i64 0\n}\n")
    with pytest.raises(ValueError, match="entry"):
        execution.execute(lowered, plan)
    assert not (lowered.workdir / "execution/image.elf").exists()


def test_timeout_keeps_partial_output_and_reaps_owned_guest(tmp_path):
    sim = script(tmp_path / "sleep-sim", "import time\nprint('started',flush=True)\ntime.sleep(30)")
    elf = tmp_path / "elf"
    elf.write_bytes(b"image")
    result = ds.run(
        {"available": True, "binary": str(sim), "binary_sha256": ds.sha(sim)},
        elf,
        isa="rv64gcv",
        timeout=0.05,
        output=tmp_path / "native",
    )
    assert result["timed_out"] and result["status"] == "failed_guest_exit" and "started" in result["stdout"]
