"""A package's offload is read from what it emitted, and an emission that says nothing is named."""

from __future__ import annotations

import json
import subprocess
import sys

from merlin.common.paths import python_source_dir
from merlin.targetgen import offload_census as OC


def _buffer(commands=(), placement=None, declined=None) -> dict:
    buffer: dict = {
        "abi_version": "1",
        "target": "synthetic",
        "commands": list(commands),
        "tensors": {
            "A": {"shape": [4, 8], "dtype": "i8"},
            "B": {"shape": [8, 2], "dtype": "i8"},
            "C": {"shape": [4, 2], "dtype": "i32"},
        },
    }
    if placement is not None:
        buffer["params"] = {"lane_placement": placement}
    if declined is not None:
        buffer["declined"] = declined
    return buffer


def test_a_program_that_emits_nothing_and_says_nothing_is_silent() -> None:
    assert OC.program_row("p", _buffer())["outcome"] == "silent"


def test_a_decline_and_a_declared_host_program_are_not_silent() -> None:
    declined = OC.program_row("p", _buffer(declined={"reason": "rank 5 is not lowered"}))
    assert declined["outcome"] == "declined" and "rank 5" in declined["reason"]
    host = OC.program_row("p", _buffer(placement=[{"lane": "host", "family": "elementwise_map"}]))
    assert host["outcome"] == "host_only" and host["placement_declared"]


def test_work_on_a_unit_counts_and_its_placement_must_be_declared() -> None:
    matmul = {"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "B", "dst": "C"}}
    placed = OC.program_row(
        "p",
        _buffer(
            [matmul],
            placement=[{"lane": "mesh", "family": "contraction"}, {"lane": "host", "family": "elementwise_map"}],
        ),
    )
    assert placed["outcome"] == "offloaded" and placed["commands"] == 1
    assert placed["contraction_offload_fraction"] == 1.0
    bare = OC.program_row("p", _buffer([matmul]))
    assert bare["outcome"] == "offloaded" and not bare["placement_declared"]


def test_instruction_use_is_measured_against_the_derived_table_or_says_why_not(monkeypatch) -> None:
    from merlin.kernels.decode import rocc

    monkeypatch.setattr(rocc, "funct_table_for", lambda target: {})
    assert OC.instruction_use(["anything"], target="synthetic")["status"] == "unavailable"

    monkeypatch.setattr(
        rocc,
        "funct_table_for",
        lambda target: {"custom_opcode": "0x7b", "names": {"2": "LOAD", "3": "COMPUTE", "9": "SEQUENCER"}},
    )
    artifact = 'llvm.inline_asm asm_string = ".insn r 0x7b, 0x3, 2, x0, x10, x11"'
    use = OC.instruction_use([artifact], target="synthetic")
    assert use["status"] == "measured"
    assert [row["name"] for row in use["unused"]] == ["COMPUTE", "SEQUENCER"]


def test_full_grade_census_reuses_public_emissions_and_names_missing_artifacts(tmp_path) -> None:
    root = tmp_path / "graded"
    generated = root / "op_a" / "generated"
    generated.mkdir(parents=True)
    (root / "op_a" / "capsule_result.json").write_text('{"capsule":"op_a"}')
    (generated / "command_buffer.json").write_text(json.dumps(_buffer(
        [{"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "B", "dst": "C"}}],
        placement=[{"lane": "mesh", "family": "contraction"}],
    )))
    (generated / "lowered.llvm.mlir").write_text("module {}")
    (root / "op_b").mkdir()
    (root / "op_b" / "capsule_result.json").write_text('{"capsule":"op_b"}')
    report = OC.census_from_graded_artifacts(
        target="synthetic",
        capsules=[{"name": "op_a", "interface_mlir": "a.mlir"},
                  {"name": "op_b", "interface_mlir": "b.mlir"},
                  {"name": "capstone", "kind": "model"}],
        suite_runs=root,
    )
    assert report["execution"] == {
        "mode": "same_grade_artifacts", "programs_requested": 2, "programs_observed": 1,
    }
    assert report["outcomes"] == {"offloaded": 1, "error": 1}
    assert report["programs"][0]["placement_declared"]
    assert report["programs"][1]["program"] == "op_b"
    assert not report["all_accounted"]


def test_full_grade_census_refuses_indirect_candidate_output(tmp_path) -> None:
    root = tmp_path / "graded"
    generated = root / "op" / "generated"
    generated.mkdir(parents=True)
    (root / "op" / "capsule_result.json").write_text('{"capsule":"op"}')
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_buffer()))
    (generated / "command_buffer.json").symlink_to(outside)
    (generated / "lowered.llvm.mlir").write_text("module {}")
    report = OC.census_from_graded_artifacts(
        target="synthetic", capsules=[{"name": "op", "interface_mlir": "op.mlir"}], suite_runs=root,
    )
    assert report["outcomes"] == {"error": 1}
    assert not report["all_accounted"]


def test_installed_public_probes_do_not_require_legacy_package_shim(tmp_path) -> None:
    script = """
import importlib
import importlib.abc
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])

class HideCompatibilityShim(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "merlin.targetgen.oot_runner":
            raise ModuleNotFoundError("compatibility shim is outside the public projection")

sys.meta_path.insert(0, HideCompatibilityShim())
from merlin.targetgen import capsule_common, lowering_coverage, offload_census, package_runtime
for module_name in (
    "merlin.compile.mesh",
    "merlin.compile.mesh_backend",
    "merlin.compile_cli",
    "merlin.llvmlower.device_native",
    "merlin.llvmlower.group_offload",
    "merlin.llvmlower.device_build",
    "merlin.llvmlower.region_capsule",
    "merlin.llvmlower.whole_program",
    "merlin.llvmlower.exact_offload",
    "merlin.perf.whole_model_passes",
    "merlin.perf.whole_model_build",
    "merlin.perf.whole_model_replies",
):
    importlib.import_module(module_name)
assert offload_census.BackendDeclined is package_runtime.BackendDeclined
assert offload_census.CertFailure is package_runtime.CertFailure
assert lowering_coverage.BackendDeclined is package_runtime.BackendDeclined
assert lowering_coverage.CertFailure is package_runtime.CertFailure

missing_tool = Path(sys.argv[2]) / "absent-tool"
package = SimpleNamespace(tool=missing_tool)
# AET's failure-category enum is optional in the core-only installed suite.
# Keep this check on the package-runtime import and CertFailure identity.
capsule_common._cat = lambda name: name
try:
    capsule_common.run_entrypoints(
        package, sys.argv[2], {}, None, contract=None, timeout=1, fourth_output_name="unused"
    )
except package_runtime.CertFailure as error:
    assert "tool missing" in str(error)
else:
    raise AssertionError("missing tool did not refuse")
try:
    capsule_common.lower_interface(
        package, missing_tool, Path(sys.argv[2]) / "generated", contract=None, timeout=1
    )
except FileNotFoundError:
    pass
else:
    raise AssertionError("missing interface did not refuse")
assert "merlin.targetgen.oot_runner" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(python_source_dir()), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
