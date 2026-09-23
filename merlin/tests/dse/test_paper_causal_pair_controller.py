from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

from merlin.compare.paper_ablation_generator import produce_causal_pair, verify_causal_pair


def _path_sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"F\0" + f"0:{path.name}".encode() + b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> Path:
    source = tmp_path / "canonical.c"
    source.write_text(r'''
#include <stdio.h>
void runtime_dispatch(unsigned long value);
int main(void) {
  unsigned long value = 1;
  for (unsigned long index = 1; index < 5000000; ++index) {
    value = value * 33u + index;
    /* MERLIN_TYPED_TRANSFORM:runtime_dispatch_elimination_v1 */
  }
  puts("{\"schema_version\":1,\"kind\":\"merlin_continuous_session_completion_v1\","
       "\"status\":\"pass\","
       "\"output_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}");
  return 0;
}
''', encoding="utf-8")
    package = tmp_path / "dispatch.c"
    package.write_text(r'''
static volatile unsigned long sink;
__attribute__((noinline)) void runtime_dispatch(unsigned long value) { sink ^= value; }
''', encoding="utf-8")
    contract = {
        "schema_version": 1, "kind": "paper_causal_pair_contract_v1", "status": "ready",
        "binding_sha256": "a" * 64, "target": "unit-test",
        "intervention_id": "runtime_dispatch_elimination_v1",
        "canonical_source": {"path": source.name, "sha256": _path_sha(source)},
        "dispatch_package": {"path": package.name, "sha256": _path_sha(package)},
        "compiler_sha256": hashlib.sha256(Path("/usr/bin/cc").read_bytes()).hexdigest(),
        "objdump_sha256": hashlib.sha256(Path("/usr/bin/objdump").read_bytes()).hexdigest(),
        "timeout_seconds": 30, "warmup_iterations": 1, "measured_iterations": 3,
    }
    contract_path = tmp_path / "pair.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    return produce_causal_pair(contract_path, tmp_path / "pair-bundle")


def test_pair_controller_builds_one_typed_delta_and_replays_binary_dataflow(tmp_path):
    receipt = _fixture(tmp_path)

    result = verify_causal_pair(receipt, expected_binding_sha256="a" * 64)

    assert result["control_runtime_dispatch_calls"] == 2
    assert result["treatment_runtime_dispatch_calls"] == 1
    assert result["functional_stdout_sha256"]


def test_pair_verifier_rejects_sleep_comment_copy_style_forgery_with_refreshed_hashes(tmp_path):
    receipt_path = _fixture(tmp_path)
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    treatment = receipt_path.parent / receipt["arms"]["treatment"]["generated_source"]["path"]
    treatment.write_text(
        treatment.read_text(encoding="utf-8").replace(
            "runtime_dispatch(value);", "/* runtime_dispatch(value); */\nsystem(\"sleep 0.01\");", 1),
        encoding="utf-8")
    receipt["arms"]["treatment"]["generated_source"]["sha256"] = _path_sha(treatment)
    receipt_path.write_text(yaml.safe_dump(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="non-intervention source|source/config/build"):
        verify_causal_pair(receipt_path, expected_binding_sha256="a" * 64)


def test_pair_verifier_rejects_shell_artifact_even_after_refreshed_digest(tmp_path):
    receipt_path = _fixture(tmp_path)
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    executable = receipt_path.parent / receipt["arms"]["control"]["executable"]["path"]
    executable.write_text("#!/bin/sh\nexit 0\n# runtime_dispatch runtime_dispatch\n", encoding="utf-8")
    executable.chmod(0o755)
    receipt["arms"]["control"]["executable"]["sha256"] = _path_sha(executable)
    receipt_path.write_text(yaml.safe_dump(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="build replay|ELF|structural analyzer"):
        verify_causal_pair(receipt_path, expected_binding_sha256="a" * 64)
