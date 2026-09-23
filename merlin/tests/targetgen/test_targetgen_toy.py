"""TargetGen toy_npu vertical slice: build, inspect, simulate. No external repos."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from merlin.common import schemas
from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import pipeline
from merlin.validation.generated_target import check_generated_target

REPO = repo_root()
TOY_DOCS = REPO / "examples/toy_npu/target/docs"
TOY_EXAMPLES = REPO / "examples/toy_npu/target/examples"

REQUIRED_METRICS = {
    "cycles",
    "bytes_moved",
    "command_count",
    "pack_count",
    "resident_hits",
    "evictions",
    "accumulator_commits",
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
    assert tc["features"] == ["resident_packed_tensor", "accumulator_commit", "command_buffer", "metrics"]
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


def test_the_generated_runtime_layer_is_a_registrable_backend(tmp_path):
    """The generated repo's runtime layer is a ``Backend``, not a bespoke class nothing can reach.

    This replaces a test that asserted the old ``RuntimeAdapter`` "executes real math". It did — by
    calling ``merlin.runtime``'s own reference oracle and returning those numbers under the target's
    name. Two things were wrong with that as a claim about a generated package: the adapter was not a
    ``Backend`` and no ``plugin`` block named it, so nothing could reach it through ``get_backend``;
    and the numbers it produced were the oracle's, not the target's. What the generated layer owes is
    to be REACHABLE and to be HONEST about having nothing to run yet — which is what is asserted here.
    """
    from merlin.runtime.backends import base

    result = _build(tmp_path / "repo")
    assert result.plans["target_contract"]["plugin"]["backend"] == "backend", (
        "no plugin block means the package resolves and then raises KeyError from get_backend"
    )
    mod = _load_module(result.out / "backend/__init__.py", "gen_toy_backend")
    assert isinstance(mod, base.Backend), "the generated backend must satisfy the module-level protocol"
    assert mod.available() is False
    import pytest as _pytest

    with _pytest.raises(NotImplementedError, match="execution path"):
        mod.compile_command_buffer({}, tmp_path)
    with _pytest.raises(NotImplementedError, match="execution path"):
        mod.run_elf(tmp_path / "a.elf")
    declined = mod.run_command_buffer({"target": "toy_npu", "commands": []})
    assert declined["declined"]["reason"], "a decline must name what is missing, never be empty"
    assert declined["outputs"] == {}


def test_the_generated_backend_is_not_a_route_to_the_oracle(tmp_path):
    """MUTATION GUARD: re-introducing the reference/simulator import must fail here, loudly.

    The retired adapter's ``semantics.py`` did ``from merlin.runtime import reference_outputs``, which
    is why ``generate/runtime_adapter.py`` sits on the sandbox's oracle-callable deny list to this day.
    The replacement is granted to agents, so the absence of that route is load-bearing — and an absence
    that only a comment asserts is an absence that comes back.
    """
    import ast

    result = _build(tmp_path / "repo")
    source = (result.out / "backend/__init__.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    oracle = {n for n in imported if n.startswith("merlin.runtime.") and "backends.base" not in n}
    assert not oracle, f"the generated backend imports a route to the oracle: {sorted(oracle)}"


def test_the_generated_compiler_manifest_is_a_work_order_that_declines(tmp_path):
    """Absent, and saying so in the shared vocabulary — not a tool that exits 0 with no work."""
    import subprocess
    import sys

    import yaml

    from merlin.targetgen import package as pkg

    result = _build(tmp_path / "repo")
    cap = pkg.capability(result.out, "compiler")
    assert cap.provided is False and cap.source == "declared"
    manifest = yaml.safe_load((result.out / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "mlir_oot_target_backend"
    assert set(manifest["commands"]) >= {
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "lower_target_to_llvm",
    }
    # Invoked directly anyway, it answers in the contract's own vocabulary: a well-formed buffer
    # carrying `declined`. Exit 0 with an empty `commands` and no `declined` is SILENT_NO_WORK; a
    # non-zero exit is indistinguishable from a broken tool. Neither is honest.
    run = subprocess.run(
        [sys.executable, str(result.out / "tools/compile.py"), "g0.mlir"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0
    buffer = json.loads(run.stdout)
    assert schemas.validate(buffer, "command_buffer") == []
    assert buffer["commands"] == [] and buffer["declined"]["reason"]
    assert buffer["declined"]["op"] == "g0.mlir"


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
