"""The post-freeze whole-program GSIM runner preserves every evidence binding."""
from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load("run_whole_program_gsim_smoke")


class _Backend:
    def available(self, engine: str) -> bool:
        return engine == "gsim"

    def run_elf(self, elf: Path, *, simulator: str, timeout: int) -> str:
        assert elf.read_bytes() == b"bound elf"
        assert simulator == "gsim" and timeout == 17
        return "exact console"

    def parse_output(self, console: str):
        assert console == "exact console"
        return {"Y0": [[7]]}, {"cycles": 1234}


def test_main_executes_the_retained_candidate_and_writes_bound_result(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_sha256 = "a" * 64
    iteration = tmp_path / "iteration"
    retained = iteration / "compiler_scratch/candidate"
    retained.mkdir(parents=True)
    command_buffer = {
        "tensors": {"Y0": {"role": "output", "shape": [1, 1], "dtype": "i32"}},
        "commands": [],
    }
    command_buffer_text = json.dumps(command_buffer, indent=2) + "\n"
    (retained / "command_buffer.json").write_text(command_buffer_text, encoding="utf-8")
    lowered = "module {}\n"
    worker_result = iteration / "analysis_worker_result.json"
    worker_result.write_text(json.dumps({
        "analysis": {"candidate_sha256": candidate_sha256},
        "artifacts": {
            "candidate_sha256": candidate_sha256,
            "candidate_command_buffer_sha256": sha256(command_buffer_text.encode()).hexdigest(),
            "candidate_lowered_sha256": sha256(lowered.encode()).hexdigest(),
            "command_buffer": command_buffer,
            "lowered_text": lowered,
        },
    }), encoding="utf-8")
    capsule = tmp_path / "capsule.yaml"
    capsule.write_text("name: frozen-model\n", encoding="utf-8")

    monkeypatch.setattr(
        RUNNER.PGC,
        "_semantic_oracle",
        lambda manifest, cb: (
            dict(cb), {"Y0": [[7]]}, lambda observed: observed == {"Y0": [[7]]},
            {"kind": "frozen_model_test", "manifest": str(manifest)},
        ),
    )
    import merlin.targetgen.contract.compile as compile_module

    def compile_candidate(cb, llvm, workdir, *, target):
        assert cb == command_buffer and llvm == lowered and target == "test_target"
        elf = workdir / "program.elf"
        elf.write_bytes(b"bound elf")
        return elf

    monkeypatch.setattr(compile_module, "compile_lowered_to_elf", compile_candidate)
    from merlin.runtime.backends import base
    monkeypatch.setattr(base, "get_backend", lambda target: _Backend())

    output = tmp_path / "result.json"
    assert RUNNER.main([
        "--analysis-worker-result", str(worker_result),
        "--capsule-manifest", str(capsule),
        "--candidate-sha256", candidate_sha256,
        "--target", "test_target",
        "--workdir", str(tmp_path / "execution"),
        "--output", str(output),
        "--timeout", "17",
    ]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["candidate_sha256"] == candidate_sha256
    assert result["command_buffer_sha256"] == sha256(command_buffer_text.encode()).hexdigest()
    assert result["lowered_sha256"] == sha256(lowered.encode()).hexdigest()
    assert result["elf_sha256"] == sha256(b"bound elf").hexdigest()
    assert result["metrics"] == {"cycles": 1234}
    assert result["outputs"] == result["expected"] == {"Y0": [[7]]}
    assert (tmp_path / "execution/gsim_console.txt").read_text(encoding="utf-8") \
        == "exact console"
