"""Host packages retain their payload and refusal rules without importing the driver."""

from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from merlin.compiler.backend.host import HostPackage, load_host_package
from merlin.compiler.backend.isa import split_isa
from merlin.mining import registry


def test_explicit_codegen_policy_is_retained_and_target_name_is_not_a_provider(tmp_path):
    from merlin.compiler.backend.host import resolve_features

    calls = []

    class Policy:
        API_VERSION = 1

        def microkernel_features(self, knobs, shapes=()):
            calls.append((knobs, shapes))
            return ["existing", "provider_point"]

        def shape_adapted_features(self, package, model_dir, *, harts):
            assert package.codegen_provider is self
            assert model_dir == "/selected/model" and harts == 3
            return ["provider_shaped_point"]

    policy = Policy()
    knobs = {"compiler_features": ["existing"], "microkernel": {"MR": 3, "NR": 8, "KC": 16}}
    assert resolve_features(knobs, {"target": "composite_cpu"}, shapes=("shape",), codegen_provider=policy) == [
        "existing",
        "provider_point",
    ]
    assert calls == [(knobs["microkernel"], ("shape",))]
    package = _package(tmp_path, schedule_file="schedule.mlir", **knobs)
    (package / "schedule.mlir").write_text("module {}")
    loaded = load_host_package(package, codegen_provider=policy)
    assert loaded.codegen_provider is policy
    assert loaded.compiler_features == ["existing", "provider_point"]
    from merlin.mining.apply import shape_adapted_features

    assert shape_adapted_features(loaded, "/selected/model", harts=3) == ["provider_shaped_point"]
    with pytest.raises(ValueError, match="versioned microkernel"):
        resolve_features(knobs, {}, codegen_provider=object())
    # Scalar/no-microkernel loading remains independent of any policy interface.
    assert resolve_features({"compiler_features": ["existing"]}, {}, codegen_provider=object()) == ["existing"]


def _package(tmp_path, **knobs):
    (tmp_path / "manifest.yaml").write_text(
        "target: rvv\nrun_id: hand_test\nfamily: vector_schedule\n"
        "authoring: {mode: hand_curated, author: human}\n"
        "outputs: {schedule: schedule.mlir, knobs: knobs.yaml}\n",
        encoding="utf-8",
    )
    (tmp_path / "knobs.yaml").write_text(yaml.safe_dump(knobs), encoding="utf-8")
    return tmp_path


def test_rvv_payload_and_compatibility_identity(tmp_path):
    schedule = "module { // frozen schedule payload\n}\n"
    package = _package(
        tmp_path,
        schedule_file="custom.mlir",
        cflags=["-march=rv64gcv", "-mabi=lp64d"],
        dtype_strategy="bf16_f32acc",
        compiler_features=["retained_feature"],
        op_match=[{"op": "linalg.matmul"}],
    )
    (package / "custom.mlir").write_text(schedule, encoding="utf-8")
    loaded = load_host_package(package)
    assert registry.RvvPackage is HostPackage
    assert registry.load_rvv_package is load_host_package
    assert isinstance(loaded, registry.RvvPackage)
    assert loaded.schedule_text == schedule
    assert loaded.cflags == ["-march=rv64gcv", "-mabi=lp64d"]
    assert loaded.compiler_features == ["retained_feature"]
    assert loaded.op_match == [{"op": "linalg.matmul"}]
    assert loaded.is_half_f32acc and not loaded.is_int8


def test_scalar_loading_does_not_import_orchestration(tmp_path):
    package = _package(tmp_path, backend="scalar", cflags=["-march=rv64gc"])
    code = """
import sys
from merlin.compiler.backend.host import load_host_package
package = load_host_package(sys.argv[1])
assert package.backend == 'scalar' and package.schedule_text == ''
assert not any(name == 'merlin.compile' or name.startswith('merlin.compile.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(package)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("march", ["rv64gcv", "rv64gc_zve32x", "rv64gc_zvl128b", "rv64gcv_zvfh"])
def test_scalar_refuses_vector_isa(tmp_path, march):
    with pytest.raises(ValueError, match="non-vector -march"):
        load_host_package(_package(tmp_path, backend="scalar", cflags=[f"-march={march}"]))


@pytest.mark.parametrize(
    "extra",
    [{"schedule_file": "schedule.mlir"}, {"op_match": [{"op": "linalg.matmul"}]}, {"compiler_features": ["x"]}],
)
def test_scalar_refuses_vector_codegen_selection(tmp_path, extra):
    with pytest.raises(ValueError, match="cannot select an RVV"):
        load_host_package(_package(tmp_path, backend="scalar", cflags=["-march=rv64gc"], **extra))


@pytest.mark.parametrize(
    ("isa", "xlen", "extensions"),
    [
        ("rv64gc_zba", "rv64", set("imafdc") | {"zicsr", "zifencei", "zba"}),
        ("RV32IMAC", "rv32", set("imac")),
        ("rv64imafdcv_zicsr_zifencei", "rv64", set("imafdcv") | {"zicsr", "zifencei"}),
        ("rv64imaczba", "rv64", set("imac") | {"zba"}),
    ],
)
def test_conservative_isa_grammar_is_retained(isa, xlen, extensions):
    assert split_isa(isa) == (xlen, extensions)


@pytest.mark.parametrize("isa", ["x86_64", "rv64", "rv64zba", "rv64i2p0", "rv64iz"])
def test_conservative_isa_grammar_refuses_unknown_base(isa):
    with pytest.raises(ValueError, match="unrecognized RISC-V ISA"):
        split_isa(isa)
