"""TargetGen toy_npu vertical slice: build, inspect, simulate. No external repos."""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import importlib.util
import json
from pathlib import Path

from merlin.common import schemas
from merlin.targetgen import pipeline
from merlin.validation.generated_target import check_generated_target

REPO = repo_root()
TOY_DOCS = REPO / "merlin/targets/toy_npu/docs"
TOY_EXAMPLES = REPO / "merlin/targets/toy_npu/examples"

REQUIRED_METRICS = {
    "cycles", "bytes_moved", "command_count", "pack_count",
    "resident_hits", "evictions", "accumulator_commits",
}


def _build(out: Path):
    return pipeline.build(
        target_name="toy_npu",
        source_dir=str(TOY_DOCS),
        examples_dir=str(TOY_EXAMPLES),
        out=out,
        emit=["xdsl", "mlir", "zephyr", "llvm-plan", "runtime"],
    )


def test_build_produces_valid_plans(tmp_path):
    result = _build(tmp_path / "merlin-target-toy-npu")
    assert result.schema_problems == []
    # Each plan validates against its schema.
    assert schemas.validate(result.plans["target_contract"], "target_contract") == []
    assert schemas.validate(result.plans["dialect_plan"], "dialect_plan") == []
    # toy_npu stays consistent with the in-tree contract.
    tc = result.plans["target_contract"]
    assert tc["name"] == "toy_npu"
    assert tc["capabilities"]["resident_storage_bytes"] == 131072
    assert tc["requires_human_review"] is False
    # Spec-mandated abstraction surface on the toy_npu contract.
    assert tc["features"] == ["resident_packed_tensor", "accumulator_commit",
                              "command_buffer", "metrics"]
    assert tc["ops"] == ["res_pack", "matmul", "commit", "evict"]
    assert tc["types"] == ["resident_tensor", "accumulator"]
    assert tc["runtime"]["backends"] == ["simulator", "zephyr"]


def test_evidence_and_concepts(tmp_path):
    result = _build(tmp_path / "repo")
    assert "resident_packed_tensor" in result.evidence_concepts
    assert (result.out / "docs/evidence_report.md").is_file()
    assert (result.out / "docs/evidence_index.yaml").is_file()


def test_generated_repo_passes_inspect(tmp_path):
    result = _build(tmp_path / "repo")
    assert check_generated_target(result.out) == []


def _load_module(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so annotation/type-hint resolution can find the module globals.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_generated_adapter_executes_real_math(tmp_path):
    result = _build(tmp_path / "repo")
    cb_path = result.out / "runtime/command_buffer/example_repeated_rhs.json"
    adapter_mod = _load_module(result.out / "runtime/adapter/adapter.py", "toy_adapter")
    run_dir = tmp_path / "run"
    res = adapter_mod.RuntimeAdapter().run_simulator(str(cb_path), out_dir=str(run_dir))

    # Real execution: simulator output equals the independent reference recomputation.
    assert res["correct"] is True
    assert REQUIRED_METRICS <= set(res["metrics"])
    # The example buffer packs once, reuses 4x, commits 4x, evicts once.
    assert res["metrics"]["pack_count"] == 1
    assert res["metrics"]["resident_hits"] == 4
    assert res["metrics"]["accumulator_commits"] == 4
    assert res["metrics"]["evictions"] == 1
    # Real artifacts are written.
    for f in ("simulator_output.json", "reference_output.json", "metrics.json", "trace.json"):
        assert (run_dir / f).is_file()
    outputs = json.loads((run_dir / "simulator_output.json").read_text())
    assert outputs and all(isinstance(v, list) for v in outputs.values())


def test_generated_xdsl_dialect_verifies_and_roundtrips(tmp_path):
    pytest = __import__("pytest")
    try:
        __import__("xdsl")
    except Exception:
        pytest.skip("xDSL not installed")
    result = _build(tmp_path / "repo")
    mod = _load_module(result.out / "xdsl/toynpu_dialect.py", "gen_toynpu_dialect")
    assert mod.HAS_XDSL
    m = mod.build_example()
    m.verify()
    mod.roundtrip(m).verify()


def test_generated_dialect_verifiers_share_maxpool_epilogue_vocabulary(tmp_path):
    """The authoring kits must recognize the fused-pooling capsules' public spelling."""
    result = _build(tmp_path / "repo")
    xdsl_text = (result.out / "xdsl/toynpu_dialect.py").read_text()
    assert '"maxpool"' in xdsl_text

    ops_cpp = next((result.out / "lib").rglob("*Ops.cpp"))
    assert '"maxpool"' in ops_cpp.read_text()


def test_contract_only_still_structurally_valid(tmp_path):
    result = pipeline.build("toy_npu", out=tmp_path / "co", emit=["contract-only"])
    assert result.schema_problems == []
    assert check_generated_target(result.out) == []


def test_build_is_deterministic(tmp_path):
    a = _build(tmp_path / "a")
    b = _build(tmp_path / "b")
    tc_a = (a.out / "contracts/target_contract.yaml").read_text()
    tc_b = (b.out / "contracts/target_contract.yaml").read_text()
    assert tc_a == tc_b
