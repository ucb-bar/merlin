"""Actual-pass numerical-domain qualification, refusal and byte-bound replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from lean_w8a8_numerical_cases import normalization_source, reduction_source
from test_lean_w8a8_receipts import stage_at

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import interpreted, numerical_evidence, numerical_policy
from merlin.verify.lean.check import _file_ref

pytestmark = pytest.mark.usefixtures("lean_oracle")


def invoke(work, tools, before=None, *, stages=None, bounds=None, index_bits=64):
    work.mkdir(exist_ok=True)
    first = stage_at(work, normalization_source() if before is None else before) if stages is None else stages[0]
    contracts = work / "contracts.json"
    contracts.write_text(json.dumps(bounds or [{"function": "forward", "intervals": [None, None, None]}]))
    receipt = numerical_policy.check_w8a8_numerical_chain_draft(
        [first] if stages is None else stages,
        work / "receipt",
        input_contracts=contracts,
        m2m_python=tools.m2m_python,
        index_bits=index_bits,
        scope="captured",
    )
    return receipt, first


def evidence(receipt):
    import jsonschema

    assert receipt["verdict"] == "abstracted", receipt
    jsonschema.validate(
        receipt, json.loads((repo_root() / "merlin/contract/schemas/lean_receipt.schema.json").read_text())
    )
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["runtime_sha256"] == receipt["lean"]["checker_sha256"]
    assert set(report["proof_axioms"]) == set(numerical_evidence.ROOTS)
    assert all(
        set(axioms) <= {"propext", "Classical.choice", "Quot.sound"} for axioms in report["proof_axioms"].values()
    )
    candidate = report["numerical"]
    assert candidate["input_domain"] == numerical_evidence.INPUT_DOMAIN
    assert candidate["arithmetic_agreement_assumed"] is True
    for flag in (
        "whole_pass_verified",
        "policy_reviewed",
        "primitive_model_laws_verified",
        "concrete_context_inhabited",
    ):
        assert candidate[flag] is False
    return report


@pytest.mark.parametrize("rows,width,zero_width", [(2, 3, 64), (3, 7, 32), (1, 4, 64), (0, 3, 32)])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_normalization_quantization_and_actual_contraction_qualify(
    tmp_path, lean_tools, rows, width, zero_width, index_bits
):
    receipt, _ = invoke(
        tmp_path, lean_tools, normalization_source(rows=rows, width=width, zero_width=zero_width), index_bits=index_bits
    )
    candidate = evidence(receipt)["numerical"]
    assert candidate["structural_match"] and candidate["conditional_initialized_policy_refinement"]
    (function,) = candidate["functions"]
    assert function["domain_inhabited"] and function["source_initialized"] and function["reference_initialized"]
    for name in ("source_prefix", "reference_prefix"):
        assert function[name]["checked"] and function[name]["first_refusal"] is None
        assert len(function[name]["initialized_source_quantizers"]) == 1


@pytest.mark.parametrize(
    "parameters",
    [
        {"epsilon": "-1.0"},
        {"scale": "0.0"},
        {"scale": "-0.01"},
        {"scale": "1.401298464324817e-45"},
        {"scale": "0x7FC00000"},
        {"scale": "0x7F800000"},
    ],
)
def test_unjustified_quantizer_inputs_refuse_after_valid_export(tmp_path, lean_tools, parameters):
    receipt, _ = invoke(tmp_path, lean_tools, normalization_source(**parameters))
    candidate = evidence(receipt)["numerical"]
    assert receipt["exporter"]["round_trip"] == "identical"
    assert not candidate["conditional_initialized_policy_refinement"]
    (function,) = candidate["functions"]
    assert function["domain_inhabited"] and not function["source_initialized"]
    prefix = function["source_prefix"]
    assert prefix["checked"] and not prefix["initialized_source_quantizers"]
    assert prefix["first_refusal"]["operation"] == "quant_ext.quantize_per_tensor"


def test_poison_destination_cannot_supply_the_square_fact(tmp_path, lean_tools):
    before = str(normalization_source()).replace("arith.mulf %x, %x", "arith.mulf %x, %dest", 1)
    receipt, _ = invoke(tmp_path, lean_tools, parse_mlir_text(before))
    candidate = evidence(receipt)["numerical"]
    assert not candidate["functions"][0]["source_initialized"]
    assert not candidate["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("mutation", ["rounding", "epsilon", "scale", "quantizer_ssa", "broadcast"])
def test_valid_changed_targets_are_not_hidden_by_prefix_success(tmp_path, lean_tools, mutation):
    stage = stage_at(tmp_path, normalization_source(rows=3, width=3))
    text = (stage / "after.mlir").read_text()
    replacements = {
        "rounding": ("math.roundeven", "math.round"),
        "epsilon": ("1.000000e-05", "2.000000e-05"),
        "scale": ("1.000000e-02", "2.000000e-02"),
        "quantizer_ssa": ('"quant_ext.quantize_per_tensor"(%scaled,', '"quant_ext.quantize_per_tensor"(%square,'),
        "broadcast": ("affine_map<(d0, d1) -> (d1)>", "affine_map<(d0, d1) -> (d0)>"),
    }
    old, new = replacements[mutation]
    assert old in text
    (stage / "after.mlir").write_text(text.replace(old, new, 1))
    receipt, _ = invoke(tmp_path, lean_tools, stages=[stage])
    candidate = evidence(receipt)["numerical"]
    assert receipt["exporter"]["round_trip"] == "identical"
    assert candidate["functions"][0]["source_initialized"]
    assert candidate["functions"][0]["source_prefix"]["initialized_source_quantizers"]
    assert not candidate["structural_match"] and not candidate["conditional_initialized_policy_refinement"]


def test_nonzero_zero_point_is_not_silently_approved_as_calibrated_reuse(tmp_path, lean_tools):
    receipt, _ = invoke(tmp_path, lean_tools, normalization_source(zero=1))
    candidate = evidence(receipt)["numerical"]
    assert candidate["functions"][0]["source_initialized"]
    assert not candidate["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize(
    "shape,dimensions,variadic",
    [
        ((2, 3, 4), (0, 2), False),
        ((2, 3), (0, 1), True),
        ((2, 0, 3), (1,), True),
        ((2, 3), (), False),
    ],
)
def test_named_reduction_rule_is_independent_of_rank_and_split(tmp_path, lean_tools, shape, dimensions, variadic):
    before = reduction_source(shape, dimensions, variadic=variadic)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "before.mlir").write_text(before)
    (stage / "after.mlir").write_text(before)
    (stage / "meta.json").write_text(json.dumps({"boundary": "named-reduction-rule-probe"}))
    receipt, _ = invoke(tmp_path, lean_tools, stages=[stage], bounds=[{"function": "forward", "intervals": [None]}])
    candidate = evidence(receipt)["numerical"]
    assert candidate["conditional_initialized_policy_refinement"]
    assert receipt["before"]["sha256"] == receipt["after"]["sha256"]  # Identity evidence, no rewrite-site claim.


def test_reduction_dimension_mutation_is_checked_in_the_complete_target(tmp_path, lean_tools):
    stage = stage_at(tmp_path, normalization_source(rows=3, width=3))
    text = (stage / "after.mlir").read_text()
    assert "dimensions = [1]" in text
    (stage / "after.mlir").write_text(text.replace("dimensions = [1]", "dimensions = [0]", 1))
    receipt, _ = invoke(tmp_path, lean_tools, stages=[stage])
    candidate = evidence(receipt)["numerical"]
    assert receipt["exporter"]["round_trip"] == "identical"
    assert candidate["functions"][0]["source_initialized"]
    assert not candidate["structural_match"]


def test_candidate_receipt_replays_real_exports_and_suffix(tmp_path, lean_tools):
    stage = stage_at(tmp_path, normalization_source())
    suffix = tmp_path / "suffix"
    suffix.mkdir()
    source = (stage / "after.mlir").read_text()
    (suffix / "before.mlir").write_text(source)
    (suffix / "after.mlir").write_text(source.replace("%result", "%renamed_result"))
    (suffix / "meta.json").write_text(json.dumps({"boundary": "ssa-renaming"}))
    receipt, _ = invoke(tmp_path, lean_tools, stages=[stage, suffix])
    assert evidence(receipt)["numerical"]["conditional_initialized_policy_refinement"]
    replayed = numerical_policy.replay(
        tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python
    )
    assert evidence(replayed)["numerical"]["suffix_boundaries"] == [{"index": 0, "checked": True, "status": "checked"}]


def test_forged_green_partial_result_fails_recomputed_replay(tmp_path, lean_tools):
    receipt, _ = invoke(tmp_path, lean_tools, normalization_source(epsilon="-1.0"))
    report = evidence(receipt)
    forged = copy.deepcopy(report)
    numerical = forged["numerical"]
    numerical["boundary_qualification_passed"] = numerical["conditional_initialized_policy_refinement"] = True
    for function in numerical["functions"]:
        function["source_initialized"] = function["reference_initialized"] = True
    report_path = Path(receipt["cert"]["path"])
    report_path.write_text(json.dumps(forged))
    receipt["cert"] = _file_ref(report_path)
    path = tmp_path / "receipt/receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="recomputed numerical evidence"):
        numerical_policy.replay(path, tmp_path / "replay", m2m_python=lean_tools.m2m_python)


@pytest.mark.parametrize("changed", ["before", "after", "runtime"])
def test_replay_checks_claim_subjects_and_the_runtime_inventory(tmp_path, lean_tools, changed):
    receipt, _ = invoke(tmp_path, lean_tools)
    report = evidence(receipt)
    if changed in {"before", "after"}:
        other = tmp_path / "another-program.mlir"
        other.write_text("module { func.func @unrelated() { return } }")
        receipt[changed] = _file_ref(other)
        reason = "actual program binding"
    else:
        report["runtime"]["files"].pop()
        report_path = Path(receipt["cert"]["path"])
        report_path.write_text(json.dumps(report))
        receipt["cert"] = _file_ref(report_path)
        reason = "runtime inventory"
    path = tmp_path / "receipt/receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=reason):
        numerical_policy.replay(path, tmp_path / "replay", m2m_python=lean_tools.m2m_python)


@pytest.mark.parametrize("changed", ["contract", "export", "entry"])
def test_live_changes_invalidate_the_candidate_attempt(tmp_path, lean_tools, monkeypatch, changed):
    original_capture, original_run = interpreted.capture, interpreted.Runtime.run
    entry = tmp_path / "CopiedEntry.lean"
    entry.write_bytes((repo_root() / "merlin/lean/W8A8NumericalMain.lean").read_bytes())

    def capture(_entry, **kwargs):
        return original_capture(entry, **kwargs)

    def run(self, arguments, **kwargs):
        proc = original_run(self, arguments, **kwargs)
        path = {"contract": tmp_path / "contracts.json", "export": Path(arguments[0]), "entry": entry}[changed]
        path.chmod(0o644)
        path.write_text(path.read_text() + "\n")
        return proc

    monkeypatch.setattr(interpreted, "capture", capture)
    monkeypatch.setattr(interpreted.Runtime, "run", run)
    receipt, _ = invoke(tmp_path, lean_tools)
    assert receipt["verdict"] == "unmeasured" and "cert" not in receipt, receipt
    assert "changed" in receipt["reason"]


def test_input_contract_is_preserved_when_it_aliases_the_receipt(tmp_path):
    contracts = tmp_path / "receipt.json"
    contracts.write_text('[{"function":"forward","intervals":[]}]')
    original = contracts.read_bytes()
    with pytest.raises(ValueError, match="destination receipt file"):
        numerical_policy.check_w8a8_numerical_chain_draft(
            [],
            tmp_path,
            input_contracts=contracts,
            m2m_python=Path("unused-python"),
        )
    assert contracts.read_bytes() == original


def test_replay_preserves_its_original_receipt(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"retained":"original"}')
    original = receipt.read_bytes()
    with pytest.raises(ValueError, match="new destination receipt file"):
        numerical_policy.replay(receipt, tmp_path, m2m_python=Path("unused-python"))
    assert receipt.read_bytes() == original
