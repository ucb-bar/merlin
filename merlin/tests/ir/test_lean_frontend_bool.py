"""Authoritative MLIR boundaries must reach the Lean frontend policy checker.

The exported mutation cases are accepted by the same torch-mlir parser and verifier used by the
authoritative exporter before the AST is passed to Lean. A duplicate-name case instead checks
compiler refusal. Python does not decide the repair shape.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import jsonschema
import pytest
from test_bool_cast_poison import _bool_cast_src

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
from merlin.verify.lean.check import (
    _FRONTEND_BOOL,
    _validate_receipt,
    check_frontend_bool_stage,
    check_stage,
    check_support,
    check_typed_stage,
    compose_chain,
)
from merlin.verify.lean.export import export_file
from merlin.verify.lean.stages import record_boundary
from merlin.xdsl_dialects._common import text as to_text


def _tools() -> tuple[Path, Path]:
    python = os.environ.get("MERLIN_COMPILER_PYTHON")
    if not python:
        pytest.skip("MERLIN_COMPILER_PYTHON is required for the authoritative torch-mlir verifier")
    lean = merlin_dir().parent / "out/build/lean/build/bin/merlin-lean"
    if not lean.is_file():
        pytest.skip("merlin-lean has not been built")
    return Path(python), lean


def _boundary(tmp_path: Path, before: str, after: str, *, expected: str) -> dict:
    python, lean = _tools()
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, source in (("before", before), ("after", after)):
        mlir = tmp_path / f"{name}.mlir"
        exported = tmp_path / f"{name}.json"
        mlir.write_text(source)
        # Verify through the compiler's own MLIR bindings, then export those very bytes.
        program = (
            "from pathlib import Path; from torch_mlir import ir; "
            "import sys; ctx=ir.Context(); ctx.allow_unregistered_dialects=True; "
            "module=ir.Module.parse(Path(sys.argv[1]).read_text(), ctx); "
            "assert module.operation.verify()"
        )
        subprocess.run([str(python), "-c", program, str(mlir)], check=True, capture_output=True, text=True)
        export_file(mlir, exported, backend="mlir", m2m_python=python)
        paths.append(exported)
    proc = subprocess.run(
        [str(lean), "check-frontend-bool", *(str(p) for p in paths)],
        check=False,
        capture_output=True,
        text=True,
        timeout=360,
    )
    result = json.loads(proc.stdout)
    assert result["checker"] == "frontend-bool"
    assert result["claim_type"] == "policy"
    assert result["policy"] == "frontend-float-to-bool"
    assert result["verdict"] == expected, result
    assert proc.returncode == (0 if expected == "verified" else 1)
    return result


def _synthetic(dtype: str = "f32", cast: str = "fptosi") -> tuple[str, str]:
    module = parse_mlir_text(_bool_cast_src(dtype).replace("arith.fptosi", f"arith.{cast}"))
    before = to_text(module, generic=True)
    assert fix_bool_fptosi(module) == 1
    module.verify()
    return before, to_text(module, generic=True)


def test_actual_smolvla_exported_program_is_accepted(tmp_path):
    _tools()
    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    before = to_text(module, generic=True)
    assert fix_bool_fptosi(module) == 1
    module.verify()
    _boundary(tmp_path, before, to_text(module, generic=True), expected="verified")


def test_synthetic_exported_program_and_valid_mutations(tmp_path):
    before, after = _synthetic()
    _boundary(tmp_path / "baseline", before, after, expected="verified")
    cmp_line = next(line for line in after.splitlines() if '"arith.cmpf"' in line)
    zero_line = next(line for line in after.splitlines() if '"arith.constant"' in line)
    cases = {
        "ordered_ne": after.replace("predicate = 13 : i64", "predicate = 6 : i64"),
        "wrong_predicate": after.replace("predicate = 13 : i64", "predicate = 8 : i64"),
        "fastmath": after.replace("#arith.fastmath<none>", "#arith.fastmath<nnan>"),
        "wrong_source": after.replace('"arith.cmpf"(%xv, %0)', '"arith.cmpf"(%0, %0)'),
        "negative_zero": after.replace("0.000000e+00 : f32", "-0.000000e+00 : f32"),
        "retained_cast": before,
        "wrong_use": after.replace('"linalg.yield"(%c) : (i1)', '"linalg.yield"(%o) : (i1)'),
        "extra_ssa_definition": after.replace(
            cmp_line,
            '      %extra = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32\n' + cmp_line,
        ),
        "bypassed_cmp": after.replace(
            cmp_line,
            '      %old = "arith.fptosi"(%xv) : (f32) -> i1\n' + cmp_line,
        ).replace('"linalg.yield"(%c) : (i1)', '"linalg.yield"(%old) : (i1)'),
        "wrong_zero_type": after.replace(
            zero_line,
            '      %half = "arith.truncf"(%xv) <{fastmath = #arith.fastmath<none>}> : (f32) -> f16\n'
            '      %0 = "arith.constant"() <{value = 0.000000e+00 : f16}> : () -> f16',
        )
        .replace('"arith.cmpf"(%xv, %0)', '"arith.cmpf"(%half, %0)')
        .replace(": (f32, f32) -> i1", ": (f16, f16) -> i1"),
    }
    for name, mutated in cases.items():
        assert mutated != after, name
        _boundary(tmp_path / name, before, mutated, expected="abstracted")


def test_every_site_is_checked_including_unused_unsigned_cast(tmp_path):
    source = _bool_cast_src().replace(
        "%c = arith.fptosi %xv : f32 to i1 ",
        "%c = arith.fptosi %xv : f32 to i1 %second = arith.fptoui %xv : f32 to i1 ",
    )
    module = parse_mlir_text(source)
    before = to_text(module, generic=True)
    retained = next(line for line in before.splitlines() if '"arith.fptoui"' in line)
    assert fix_bool_fptosi(module) == 2
    module.verify()
    after = to_text(module, generic=True)
    comparisons = [line for line in after.splitlines() if '"arith.cmpf"' in line]
    assert len(comparisons) == 2
    _boundary(tmp_path / "both", before, after, expected="verified")
    # A valid, unused retained cast must not disappear from the checker denominator.
    _boundary(tmp_path / "retained-unused", before, after.replace(comparisons[1], retained), expected="abstracted")


def test_unsigned_f32_site_is_covered_and_other_formats_are_explicitly_refused(tmp_path):
    before, after = _synthetic(cast="fptoui")
    _boundary(tmp_path / "f32_unsigned", before, after, expected="verified")
    for dtype in ("f16", "bf16", "f64"):
        for cast in ("fptosi", "fptoui"):
            before, after = _synthetic(dtype, cast)
            _boundary(tmp_path / f"{dtype}_{cast}", before, after, expected="abstracted")


def test_duplicate_ssa_definition_is_refused_by_compiler_verifier(tmp_path):
    python, _ = _tools()
    _, after = _synthetic()
    cmp_line = next(line for line in after.splitlines() if '"arith.cmpf"' in line)
    duplicated = after.replace(cmp_line, cmp_line + "\n" + cmp_line)
    path = tmp_path / "duplicate.mlir"
    path.write_text(duplicated)
    program = (
        "from pathlib import Path; from torch_mlir import ir; import sys; "
        "ctx=ir.Context(); ctx.allow_unregistered_dialects=True; "
        "module=ir.Module.parse(Path(sys.argv[1]).read_text(), ctx); "
        "assert module.operation.verify()"
    )
    proc = subprocess.run([str(python), "-c", program, str(path)], capture_output=True, text=True)
    assert proc.returncode != 0


def test_policy_receipt_replays_and_binds_a2_spec(tmp_path, monkeypatch):
    python, _ = _tools()
    before, after = _synthetic()
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    stage = record_boundary("prepare-fix-bool-fptosi", "merlin.llvmlower.passes_xdsl.fix_bool_fptosi", before, after)
    receipt = check_frontend_bool_stage(stage, m2m_python=python)
    assert receipt["verdict"] == "verified", receipt.get("reason")
    assert receipt["policy"] == "frontend-float-to-bool"
    assert receipt["comparison_model"]["status"] == "proposal; pending human review"
    assert len(receipt["comparison_model"]["proofs"]) == 3
    assert receipt["site_theorem"] == "MerlinVerify.Rules.FrontendBoolProgram.accepted_site_exec"
    assert check_stage(stage, m2m_python=python)["claim_type"] == "policy"
    receipt = check_typed_stage(stage, m2m_python=python)
    assert receipt["claim_type"] == "policy"
    assert (
        _validate_receipt(
            receipt, obligation=_FRONTEND_BOOL, function="forward", m2m_python=python, work=tmp_path / "replay"
        )
        == []
    )
    changed = {**receipt, "spec": {**receipt["spec"], "sha256": "0" * 64}}
    assert (
        "A2 specification changed"
        in _validate_receipt(
            changed, obligation=_FRONTEND_BOOL, function="forward", m2m_python=python, work=tmp_path / "bad"
        )[0]
    )
    changed_model = {**receipt, "comparison_model": {**receipt["comparison_model"], "host_bridge": "proved"}}
    assert _validate_receipt(
        changed_model, obligation=_FRONTEND_BOOL, function="forward", m2m_python=python, work=tmp_path / "bad-model"
    ) == ["proposed f32 comparison evidence changed"]
    support_dir = tmp_path / "support"
    check_support(stage / "before.mlir", "forward", support_dir, m2m_python=python)
    chain = compose_chain([stage], support_dir / "receipt.json", "forward", tmp_path / "chain.json", m2m_python=python)
    assert chain["links"][0]["claim_type"] == "policy"
    assert chain["links"][0]["policy"] == "frontend-float-to-bool"
    assert chain["refinement_verdict"] == "abstracted"
    assert "policy repair cannot be composed" in chain["reason"]
    assert chain["policy_composition"]["verdict"] == "verified"
    assert chain["policy_composition"]["anchor_before_sha256"] == receipt["before"]["sha256"]
    assert chain["policy_composition"]["suffix_boundaries"] == 0
    suffix = record_boundary("ordinary-suffix", "identity", after, after)
    assert check_typed_stage(suffix, m2m_python=python, index_bits=64)["verdict"] == "verified"
    composed = compose_chain(
        [stage, suffix], support_dir / "receipt.json", "forward", tmp_path / "suffix-chain.json", m2m_python=python
    )
    assert composed["policy_composition"]["verdict"] == "verified"
    assert composed["policy_composition"]["suffix_boundaries"] == 1
    assert composed["refinement_verdict"] == composed["verdict"] == "abstracted"
    assert check_typed_stage(suffix, m2m_python=python, index_bits=32)["verdict"] == "verified"
    wrong_domain = compose_chain(
        [stage, suffix], support_dir / "receipt.json", "forward", tmp_path / "wrong-domain.json", m2m_python=python
    )
    assert wrong_domain["policy_composition"]["verdict"] == "abstracted"
    assert "different index width" in wrong_domain["policy_composition"]["reason"]

    schema = json.loads((merlin_dir() / "contract/schemas/lean_chain.schema.json").read_text())
    for document in (chain, composed, wrong_domain):
        jsonschema.validate(document, schema)
