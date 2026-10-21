"""Bind conditional initialized W8A8 and suffix proofs without promoting their premises."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_lean_w8a8_chain import identity_suffix, report
from test_lean_w8a8_dynamic_policy import called_source, reused_dynamic_source
from test_lean_w8a8_module import source
from test_lean_w8a8_operand_paths import mixed_source
from test_lean_w8a8_receipts import stage_at

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import policy, policy_numerics
from merlin.verify.lean import stages as recording


def evidence(receipt, *, chain):
    assert receipt["verdict"] == "abstracted", receipt
    checked = report(receipt)
    numerical = checked["conditional_numerical"]
    module = policy_numerics.CHAIN_MODULE if chain else policy_numerics.QUALIFIED_MODULE
    assert numerical["checker"] == f"{module}.check"
    assert numerical["theorem"] == f"{module}.check_sound"
    assert set(numerical["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
    assert numerical["policy_reviewed"] is False
    assert numerical["primitive_model_laws_verified"] is False
    assert numerical["whole_pass_verified"] is False
    assert numerical["input_domain"] == "typed_initialized_inputs_with_finite_f32_values"
    assert numerical["floating_domain"] == "byte_extension_with_explicit_source_and_dynamic_primitive_laws"
    return numerical


@pytest.mark.parametrize("kind", ["raw", "empty", "mixed_left", "mixed_right", "reuse_left", "reuse_right", "called"])
def test_public_receipt_records_actual_initialized_policy_proof(tmp_path, lean_tools, kind):
    if kind in {"raw", "empty"}:
        before = source(2, 0 if kind == "empty" else 3, 4)
    elif kind.startswith("mixed"):
        before = mixed_source(kind == "mixed_left", False)
    elif kind.startswith("reuse"):
        before = reused_dynamic_source(kind == "reuse_left")
    else:
        before = called_source(3)
    stage = stage_at(tmp_path, before)
    scope = "captured" if kind.startswith("reuse") else "all"
    receipt = policy.check_w8a8_draft(stage, m2m_python=lean_tools.m2m_python, scope=scope)
    numerical = evidence(receipt, chain=False)
    assert numerical["conditional_initialized_policy_refinement"]
    assert numerical["boundary_qualification_passed"]
    assert numerical["policy_scope"] == scope
    assert len(numerical["functions"]) == (4 if kind == "called" else 1)


@pytest.mark.parametrize("preprocess", [False, True])
def test_dynamic_proof_composes_through_real_suffix(tmp_path, lean_tools, monkeypatch, preprocess):
    first = stage_at(tmp_path, source())
    if preprocess:
        monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "preprocessing"))
        preprocess_text((first / "after.mlir").read_text())
        suffix = recording.completed(tmp_path / "preprocessing")
        assert len(suffix) == 4
    else:
        suffix = []
    receipt = policy.check_w8a8_chain_draft(
        [first, *suffix], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    numerical = evidence(receipt, chain=True)
    assert numerical["conditional_initialized_policy_refinement"]
    assert numerical["suffix_checked"] and numerical["suffix_match"]
    assert len(numerical["suffix_boundaries"]) == len(suffix)


def test_unqualified_boundary_leaves_conditional_suffix_unmeasured(tmp_path, lean_tools):
    first = stage_at(tmp_path, source())
    (first / "after.mlir").write_bytes((first / "before.mlir").read_bytes())
    last = identity_suffix(tmp_path, first)
    receipt = policy.check_w8a8_chain_draft(
        [first, last], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    numerical = evidence(receipt, chain=True)
    assert not numerical["boundary_qualification_passed"]
    assert not numerical["conditional_initialized_policy_refinement"]
    assert not numerical["suffix_checked"] and numerical["suffix_match"] is None
    assert numerical["suffix_boundaries"] == [{"index": 0, "conditional_match": None, "status": "unmeasured"}]


@pytest.mark.parametrize("source_extension", [False, True])
def test_semantics_generic_suffix_accepts_renamed_captures(tmp_path, lean_tools, source_extension):
    from test_lean_w8a8_parametric_policy import renumber

    first = stage_at(tmp_path, reused_dynamic_source(True) if source_extension else source())
    receipt = policy.check_w8a8_draft(
        first, m2m_python=lean_tools.m2m_python, scope="captured" if source_extension else "all"
    )
    documents = report(receipt)["program_exports"]
    renamed = json.loads(Path(documents[1]["path"]).read_text())
    renumber(renamed)
    target = tmp_path / "renamed.json"
    target.write_text(json.dumps(renamed))
    checked = subprocess.run(
        [
            str(lean_tools.lean_exe),
            "check-w8a8-chain-draft",
            documents[0]["path"],
            documents[1]["path"],
            "64",
            "captured" if source_extension else "all",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 1, checked.stdout + checked.stderr
    numerical = json.loads(checked.stdout)["conditional_numerical"]
    assert numerical["conditional_initialized_policy_refinement"]
    assert numerical["suffix_checked"] and numerical["suffix_match"]
    assert numerical["policy_reviewed"] is False and numerical["whole_pass_verified"] is False


@pytest.mark.parametrize("source_extension", [False, True])
def test_typed_suffix_requires_interpretation_agreement(tmp_path, lean_tools, source_extension):
    from test_lean_splat_inputs import _source

    from merlin.llvmlower.splat_inputs import scalarize_splat_inputs

    before = reused_dynamic_source(True) if source_extension else source()
    helper = parse_mlir_text(_source(element="f32"))
    before.body.block.add_ops(op.clone() for op in helper.body.block.ops)
    first = stage_at(tmp_path, before)
    suffix = identity_suffix(tmp_path, first)
    changed = parse_mlir_text((suffix / "after.mlir").read_text())
    assert scalarize_splat_inputs(changed) == 1
    changed.verify()
    (suffix / "after.mlir").write_text(str(changed))
    receipt = policy.check_w8a8_chain_draft(
        [first, suffix],
        tmp_path / "result",
        m2m_python=lean_tools.m2m_python,
        scope="captured" if source_extension else "all",
    )
    numerical = evidence(receipt, chain=True)
    assert numerical["boundary_qualification_passed"]
    assert numerical["conditional_initialized_policy_refinement"] is not source_extension
    assert numerical["suffix_match"] is not source_extension
    documents = report(receipt)["program_exports"]
    diagnosed = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_dynamic_step.lean"),
            documents[1]["path"],
            documents[2]["path"],
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert diagnosed.returncode == 0, diagnosed.stdout + diagnosed.stderr
    steps = json.loads(diagnosed.stdout)
    assert steps["source_free"] is not source_extension and steps["target_free"] is not source_extension
    assert all(fn["frozen"] for fn in steps["functions"])
    assert all(fn["conditional"] is not source_extension for fn in steps["functions"])


@pytest.mark.parametrize("location", ["first", "middle", "last"])
def test_each_actual_suffix_boundary_is_checked(tmp_path, lean_tools, location):
    from xdsl.dialects.builtin import FloatAttr, f32

    first = stage_at(tmp_path, source())
    stages = [first]
    for name in ("first", "middle", "last"):
        stage = identity_suffix(tmp_path, stages[-1], name=name)
        if name == location:
            changed = parse_mlir_text((stage / "after.mlir").read_text())
            scale = next(
                op
                for op in changed.walk()
                if op.name == "linalg.generic" and any(body.name == "arith.select" for body in op.body.block.ops)
            )
            list(scale.body.block.ops)[2].properties["value"] = FloatAttr(1e-4, f32)
            changed.verify()
            (stage / "after.mlir").write_text(str(changed))
        stages.append(stage)
    receipt = policy.check_w8a8_chain_draft(stages, tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all")
    numerical = evidence(receipt, chain=True)
    assert numerical["boundary_qualification_passed"]
    assert not numerical["suffix_match"] and not numerical["conditional_initialized_policy_refinement"]
    assert [link["conditional_match"] for link in numerical["suffix_boundaries"]] == [
        name != location for name in ("first", "middle", "last")
    ]


@pytest.mark.parametrize(
    "change",
    [
        "checker",
        "theorem",
        "index_bits",
        "policy_scope",
        "input_domain",
        "floating_domain",
        "source_primitive_laws",
        "additional_dynamic_primitive_laws",
        "policy_reviewed",
        "primitive_model_laws_verified",
        "whole_pass_verified",
        "missing",
        "functions",
        "initialization",
        "qualification_type",
        "suffix_count",
        "suffix_order",
        "suffix_status",
        "suffix_summary",
    ],
)
def test_malformed_conditional_evidence_invalidates_attempt(tmp_path, lean_tools, monkeypatch, change):
    first = stage_at(tmp_path, source())
    last = identity_suffix(tmp_path, first)
    run = subprocess.run

    def corrupt(command, **kwargs):
        checked = run(command, **kwargs)
        if len(command) < 2 or command[1] != "check-w8a8-chain-draft":
            return checked
        result = json.loads(checked.stdout)
        numerical = result["conditional_numerical"]
        if change == "missing":
            del result["conditional_numerical"]
        elif change == "functions":
            numerical["functions"] = []
        elif change == "initialization":
            numerical["functions"][0]["source_initialized"] = False
        elif change == "qualification_type":
            numerical["conditional_initialized_policy_refinement"] = 1
        elif change == "suffix_count":
            numerical["suffix_boundaries"] = []
        elif change == "suffix_order":
            numerical["suffix_boundaries"][0]["index"] = 1
        elif change == "suffix_status":
            numerical["suffix_boundaries"][0]["status"] = "unmeasured"
        elif change == "suffix_summary":
            numerical["suffix_match"] = False
        elif type(numerical[change]) is bool:
            numerical[change] = True
        elif type(numerical[change]) is int:
            numerical[change] += 1
        else:
            numerical[change] += "_changed"
        return subprocess.CompletedProcess(command, checked.returncode, json.dumps(result), checked.stderr)

    monkeypatch.setattr(policy.subprocess, "run", corrupt)
    receipt = policy.check_w8a8_chain_draft(
        [first, last], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    assert receipt["verdict"] == "unmeasured", receipt
    assert "conditional W8A8" in receipt["reason"]
    assert "cert" not in receipt
