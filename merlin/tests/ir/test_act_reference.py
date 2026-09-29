"""Plumbing tests only; fake executables do not qualify an ACT comparison."""

from __future__ import annotations

import hashlib
from pathlib import Path

from merlin.semantic_compiler.act_reference import parse_act_assembly, run_act_reference

ASSEMBLY = """import jax.numpy as jnp

def qkv(kernel, api):
    @kernel(hbm=64, input=[{'addr': 0, 'shape': (2, 2), 'dtype': jnp.bfloat16}], constant=[],
            output=[{'addr': 16, 'shape': (2, 2), 'dtype': jnp.bfloat16}])
    def qkv_():
        api.load_rm(n=2, addr_in=0, addr_out=0)
        api.store_rm(n=2, addr_in=0, addr_out=16)
    return qkv_
"""


def _backend(tmp_path: Path, body: str) -> tuple[Path, str]:
    path = tmp_path / "backend"
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_restricted_reader_never_executes_generated_python() -> None:
    parsed = parse_act_assembly(ASSEMBLY)
    assert parsed.kernel_name == "qkv"
    assert [instruction.name for instruction in parsed.instructions] == ["load_rm", "store_rm"]
    assert parsed.metadata["output"][0]["dtype"] == "jnp.bfloat16"
    poisoned = ASSEMBLY.replace("api.load_rm(n=2, addr_in=0, addr_out=0)", "__import__('os').system('false')")
    try:
        parse_act_assembly(poisoned)
    except ValueError as exc:
        assert "non-API call" in str(exc)
    else:
        raise AssertionError("executable Python was accepted")


def test_reference_runner_requires_exact_binary_and_fresh_parseable_output(tmp_path: Path) -> None:
    backend, digest = _backend(
        tmp_path, f"from pathlib import Path\nimport sys\nPath(sys.argv[4]).write_text({ASSEMBLY!r})\n",
    )
    root = tmp_path / "jobs"
    ready = run_act_reference(backend=backend, expected_backend_sha256=digest, hlo=b"HLO source A", artifact_root=root)
    assert ready.status == "candidate_imported" and ready.assembly is not None
    assert ready.input_sha256 == hashlib.sha256(b"HLO source A").hexdigest()
    assert ready.artifact_root is not None and (ready.artifact_root / "identity.json").is_file()
    changed = run_act_reference(
        backend=backend, expected_backend_sha256=digest, hlo=b"HLO source B", artifact_root=root,
    )
    assert changed.status == "candidate_imported"
    assert changed.artifact_root != ready.artifact_root
    assert changed.input_sha256 != ready.input_sha256
    wrong = run_act_reference(backend=backend, expected_backend_sha256="0" * 64, hlo=b"HLO", artifact_root=root)
    assert wrong.status == "tool_unavailable" and wrong.artifact_root is None


def test_reference_runner_rejects_no_output_stale_output_and_killed_worker(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "program.py").write_text(ASSEMBLY)
    backend, digest = _backend(tmp_path, "pass\n")
    empty = run_act_reference(backend=backend, expected_backend_sha256=digest, hlo=b"HLO", artifact_root=root)
    assert empty.status == "compile_error" and "no fresh output" in empty.reason
    backend, digest = _backend(tmp_path, "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    killed = run_act_reference(backend=backend, expected_backend_sha256=digest, hlo=b"HLO", artifact_root=root)
    assert killed.status == "compile_error" and "exited" in killed.reason
    missing = run_act_reference(
        backend=tmp_path / "missing", expected_backend_sha256=digest, hlo=b"HLO", artifact_root=root,
    )
    assert missing.status == "tool_unavailable"


def test_reference_runner_timeout_and_malformed_output(tmp_path: Path) -> None:
    backend, digest = _backend(tmp_path, "import time\ntime.sleep(3)\n")
    timed = run_act_reference(
        backend=backend, expected_backend_sha256=digest, hlo=b"HLO", artifact_root=tmp_path / "jobs",
        timeout_s=1,
    )
    assert timed.status == "search_timeout"
    backend, digest = _backend(tmp_path, "from pathlib import Path\nimport sys\nPath(sys.argv[4]).write_text('pass')\n")
    malformed = run_act_reference(
        backend=backend, expected_backend_sha256=digest, hlo=b"HLO", artifact_root=tmp_path / "jobs",
    )
    assert malformed.status == "unsupported_semantics"
