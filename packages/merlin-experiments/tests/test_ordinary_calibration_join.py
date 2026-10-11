"""Original DF/paired schemas with harmless product/process fixtures; no engines."""

import json
import struct
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_ordinary_search_policy as NORMAL
from merlin_experiments.phase2 import ordinary_calibration as CAL
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import paired_measurement as PME
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.common import invocation_record as I

_ORIGINAL_RUN = PME.run_execution


def _elf():
    names, strings = b"\0.names\0.code\0.strings\0.symbols\0", b"\0entry\0"
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHIQQQIHHHHHH", header, 16, 2, 0, 1, 1024, 0, 64, 0, 64, 0, 0, 64, 5, 1)
    start = 64 + 5 * 64
    sections = [bytes(64), struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, start, len(names), 0, 0, 1, 0)]
    start += len(names)
    sections.append(struct.pack("<IIQQQQIIQQ", 8, 1, 4, 1024, start, 8, 0, 0, 1, 0))
    start += 8
    sections.append(struct.pack("<IIQQQQIIQQ", 14, 3, 0, 0, start, len(strings), 0, 0, 1, 0))
    start += len(strings)
    sections.append(struct.pack("<IIQQQQIIQQ", 23, 2, 0, 0, start, 24, 3, 0, 8, 24))
    return (
        bytes(header) + b"".join(sections) + names + b"opaque00" + strings + struct.pack("<IBBHQQ", 1, 2, 0, 2, 1024, 8)
    )


def _console():
    return "synthetic dispatch only; no simulator or physical qualification\n"


def _products(root):
    root.mkdir(parents=True, exist_ok=True)
    llvm = root / "lowered.llvm.mlir"
    llvm.write_text(
        "module { llvm.func @entry(%a: !llvm.ptr, %b: !llvm.ptr) {\n"
        "%v = llvm.load %a : !llvm.ptr -> i16\nllvm.store %v, %b : i16, !llvm.ptr\nllvm.return } }"
    )
    template, elf = root / "elf_fixture", root / FAST.PACKAGE_ELF_NAME
    template.write_bytes(_elf())
    # Real original invocation records include an ordinary failed link/probe
    # before its successful fallback. No compiler or engine is invoked here.
    I.run(
        [sys.executable, "-c", "raise SystemExit(1)"],
        directory=root,
        stage="elf",
        inputs=(llvm,),
        capture_output=True,
        text=True,
    )
    I.run(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())",
            str(template),
            str(elf),
        ],
        directory=root,
        stage="elf",
        inputs=(template, llvm),
        outputs=(elf,),
        capture_output=True,
        text=True,
    )
    with I.observe_call(
        root, stage="execution", function=_console, arguments={"simulator": "gsim"}, inputs=(elf,)
    ) as call:
        call.returned(stdout=_console())
    return elf


@pytest.fixture
def joined_inputs(tmp_path, monkeypatch):
    fixture = NORMAL.ordinary_inputs.__wrapped__(tmp_path, monkeypatch)
    attempts = []

    def engines(package, kernel, kernel_dir, runs, timeout, target, **kwargs):
        attempts.append(kwargs["measurement_pass"])
        elf = _products(runs)
        if len(attempts) == 1:
            return {
                "numeric": "pass",
                "gsim_qualification": {"admitted": False, "kind": "execution_missing"},
                "execution_outcome": {"gsim": {"tier_outcome": {"status": "unavailable"}}},
            }
        return {
            "status": "pass",
            "numeric": "pass",
            "per_sim": {"spike": {"correct": True}, "gsim": {"correct": True, "cycles": 10}},
            "gsim_qualification": {"admitted": True, "decision": kwargs["decision"].to_dict()},
            "gsim_execution": {"gsim": {"elf": str(elf), "elf_sha256": sha256_file(elf), "cycles": 10}},
        }

    monkeypatch.setattr(PME, "run_execution", _ORIGINAL_RUN)
    monkeypatch.setattr(PME, "_run_el4_compiler_engines", engines)
    monkeypatch.setattr(PME.PC, "package_sandbox_policy", lambda *args: None)
    monkeypatch.setattr(PME.PC, "boxed_entrypoints", lambda *args: nullcontext())
    monkeypatch.setattr(
        FAST,
        "get_backend",
        lambda target: SimpleNamespace(
            harness_build_recipe=lambda: SimpleNamespace(
                require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="entry")
            )
        ),
    )
    policy = OS.prepare_ordinary_search_policy(**fixture.arguments)
    member = policy.search_corpus.capsules[0]
    root = tmp_path / "timing"
    root.mkdir()
    raw = policy.feedback_evaluator._execute(
        arm="candidate",
        package=policy.search_baseline,
        package_sha256=policy.feedback_evaluator.baseline_sha256,
        member=member,
        decision=policy.feedback_evaluator.decisions[(member.family, member.capsule)],
        workspace=root / "candidate",
        timeout_s=10,
    )
    cheap = tmp_path / "cheap"
    _products(cheap)
    compiled = FAST.compiled_observation(cheap, target=policy.target_experiment.target)
    return SimpleNamespace(policy=policy, member=member, root=root, raw=raw, compiled=compiled, attempts=attempts)


def test_original_df_retry_and_paired_r000_observation_can_join_supported_cheap_elf(joined_inputs):
    fixture = joined_inputs
    assert fixture.compiled["status"] == "observed"
    assert fixture.raw["schema"] == "paired_arm4_raw_execution_v2"
    assert fixture.raw["execution"]["replicate"] == "r000"
    assert len(fixture.attempts) == 2
    assert fixture.policy.feedback_evaluator._redact_execution(
        fixture.raw,
        fixture.policy.feedback_evaluator.decisions[(fixture.member.family, fixture.member.capsule)],
        arm="candidate",
        family=fixture.member.family,
        capsule=fixture.member.capsule,
    )["correct"]
    record = json.loads((fixture.root / "candidate.attempts.json").read_text())
    assert record["schema"] == "performance-execution-attempts.v1"
    assert len(record["attempts"]) == 2
    program, attempt, process = CAL._join(
        fixture.root,
        {"family": fixture.member.family, "capsule": fixture.member.capsule},
        "candidate",
        fixture.compiled,
        fixture.policy.feedback_evaluator.baseline_sha256,
    )
    assert program == fixture.raw["measurement"]["gsim_execution"]["gsim"]["elf_sha256"]
    assert attempt[0] == str(fixture.root / "candidate.attempts.json")
    assert process and all(I.verify(Path(path))["stage"] == "execution" for path, _ in process)


@pytest.mark.parametrize("change", ["elf", "receipt"])
def test_selected_product_or_successful_dispatch_drift_still_refuses(joined_inputs, change):
    fixture = joined_inputs
    if change == "elf":
        Path(fixture.raw["measurement"]["gsim_execution"]["gsim"]["elf"]).write_bytes(b"changed")
    else:
        for path in fixture.root.rglob("invocation.json"):
            record = json.loads(path.read_text())
            if record["stage"] == "execution":
                Path(record["stdout"]["path"]).write_text("changed")
    with pytest.raises((ValueError, StageGateError), match="changed|escaped"):
        CAL._join(
            fixture.root,
            {"family": fixture.member.family, "capsule": fixture.member.capsule},
            "candidate",
            fixture.compiled,
            fixture.policy.feedback_evaluator.baseline_sha256,
        )
