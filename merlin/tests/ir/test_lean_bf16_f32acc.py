"""The f32-accumulation contract checker on actual ``lower_bf16_matmul_f32acc`` output.

Every corrupted output keeps valid MLIR (the authoritative exporter parses it) and violates
exactly one contract clause; the checker must refuse each. Permitted variants (either flag
``none``, changed provenance) must be accepted, so refusal is not vacuous. Pass inputs outside
the contract expose pass defects: the checker refuses the pass's output for each.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lean_bf16_f32acc_cases import (
    STANDARD_BODY,
    bmm,
    line_with,
    matmul,
    operands,
    replace_once,
    reverse_reduction,
    run_pass,
)

from merlin.verify.lean.bf16_f32acc import CHECKER, THEOREM, run_checker
from merlin.verify.lean.export import export_file


def check(tools, work: Path, before: str, after: str) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in (("before", before), ("after", after)):
        source = work / f"{name}.mlir"
        source.write_text(text)
        export_file(source, work / f"{name}.json", backend="mlir", m2m_python=tools.m2m_python)
        paths.append(work / f"{name}.json")
    return run_checker(paths, index_bits=64, timeout_seconds=300)


@pytest.fixture(scope="module")
def actual():
    source = bmm()
    output, count = run_pass(source)
    assert count == 1
    return source, output


def test_actual_bf16_output_is_accepted_with_strict_qualification(lean_tools, tmp_path, actual):
    outcome = check(lean_tools, tmp_path, *actual)
    assert (outcome["checker"], outcome["theorem"]) == (CHECKER, THEOREM)
    assert outcome["conditional_match"] and outcome["qualification_passed"]
    assert outcome["candidates"] == 1
    assert outcome["sites"] == [{"mul": "contract", "add": "contract"}]
    assert outcome["verdict"] == "abstracted" and outcome["policy_reviewed"] is False
    assert outcome["pass_boundary"]["checked"]
    assert outcome["pass_boundary"]["semantic_match"]
    assert outcome["pass_boundary"]["site_qualification_passed"]
    assert outcome["pass_boundary"]["expected_sites"] == outcome["pass_boundary"]["covered_sites"] == 1


@pytest.mark.parametrize("fmt", ["bf16", "f16"])
def test_actual_named_matmul_output_is_accepted(lean_tools, tmp_path, fmt):
    source = matmul(fmt)
    output, count = run_pass(source)
    assert count == 1
    outcome = check(lean_tools, tmp_path, source, output)
    assert outcome["conditional_match"] and outcome["qualification_passed"], outcome["reason"]
    assert outcome["pass_boundary"]["checked"]


def test_actual_f16_batched_output_is_accepted(lean_tools, tmp_path):
    source = bmm("f16")
    outcome = check(lean_tools, tmp_path, source, run_pass(source)[0])
    assert outcome["conditional_match"] and outcome["qualification_passed"], outcome["reason"]


def _acc_line(text: str) -> str:
    return line_with(text, '"linalg.generic"', "iterator_type<reduction>")


def _second_extf(text: str) -> tuple[str, str]:
    lines = [line for line in text.splitlines() if '"arith.extf"' in line]
    assert len(lines) == 2
    return lines[0], lines[1]


def _swap_extf_operand(text: str) -> str:
    first, second = _second_extf(text)
    return replace_once(text, second, second.replace(f"({operands(second)[0]})", f"({operands(first)[0]})"))


def _uninitialized(text: str) -> str:
    acc = _acc_line(text)
    fill = line_with(text, '"linalg.fill"')
    return replace_once(text, acc, acc.replace(operands(acc)[2], operands(fill)[1]))


def _operands_swapped_in_add(text: str) -> str:
    add = line_with(text, '"arith.addf"')
    left, right = operands(add)
    return replace_once(text, add, add.replace(f"({left}, {right})", f"({right}, {left})"))


CORRUPTIONS = {
    "operand conversion: both factors read the left operand": _swap_extf_operand,
    "indexing maps: batches paired across the right operand": lambda t: replace_once(
        t,
        "(d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types",
        "(1 - d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types",
    ),
    "reduction order: K traversed in reverse": reverse_reduction,
    "accumulator initialization: -0.0": lambda t: replace_once(
        t, "<{value = 0.000000e+00 : f32}>", "<{value = -0.000000e+00 : f32}>"
    ),
    "accumulator initialization: 1.0": lambda t: replace_once(
        t, "<{value = 0.000000e+00 : f32}>", "<{value = 1.000000e+00 : f32}>"
    ),
    "accumulator initialization: uninitialized destination": _uninitialized,
    "final conversion: rounding toward zero": lambda t: replace_once(
        t,
        line_with(t, '"arith.truncf"'),
        line_with(t, '"arith.truncf"').replace(") :", ") <{roundingmode = 3 : i32}> :"),
    ),
    "final conversion: reads another batch's accumulator": lambda t: replace_once(
        t,
        "[affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]",
        "[affine_map<(d0, d1, d2) -> (1 - d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]",
    ),
    "fast-math: reassoc on the add": lambda t: replace_once(
        t, line_with(t, '"arith.addf"'), line_with(t, '"arith.addf"').replace("<contract>", "<reassoc,contract>")
    ),
    "fast-math: fast on the multiply": lambda t: replace_once(
        t, line_with(t, '"arith.mulf"'), line_with(t, '"arith.mulf"').replace("<contract>", "<fast>")
    ),
    "fast-math: nnan on the add": lambda t: replace_once(
        t, line_with(t, '"arith.addf"'), line_with(t, '"arith.addf"').replace("<contract>", "<nnan,contract>")
    ),
    # IEEE addition commutes, but the parametric contract fixes the accumulator as left addend.
    "accumulation order: product as left addend": _operands_swapped_in_add,
    "semantic attribute added to the accumulating generic": lambda t: replace_once(
        t, 'prov.transforms = "bf16_f32acc"}', 'prov.transforms = "bf16_f32acc", test.extra = 1 : i64}'
    ),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_corrupted_output_is_refused(lean_tools, tmp_path, actual, name):
    source, output = actual
    corrupted = CORRUPTIONS[name](output)
    assert corrupted != output
    outcome = check(lean_tools, tmp_path, source, corrupted)
    assert not outcome["conditional_match"], name
    assert not outcome["qualification_passed"]


def test_unconverted_site_is_refused(lean_tools, tmp_path, actual):
    source, _ = actual
    outcome = check(lean_tools, tmp_path, source, source)
    assert not outcome["conditional_match"]


PERMITTED = {
    "no contraction license": lambda t: t.replace("fastmath<contract>", "fastmath<none>"),
    "license on the add only": lambda t: replace_once(
        t, line_with(t, '"arith.mulf"'), line_with(t, '"arith.mulf"').replace("<contract>", "<none>")
    ),
    "changed provenance annotation": lambda t: replace_once(
        t, 'prov.transforms = "bf16_f32acc"', 'prov.transforms = "renamed"'
    ),
}


@pytest.mark.parametrize("name", sorted(PERMITTED))
def test_permitted_variant_is_accepted(lean_tools, tmp_path, actual, name):
    source, output = actual
    outcome = check(lean_tools, tmp_path, source, PERMITTED[name](output))
    assert outcome["conditional_match"], (name, outcome["reason"])


def test_pass_refuses_nonzero_initialization_and_checker_retains_counterexample(lean_tools, tmp_path):
    source = bmm(init="1.000000e+00")
    with pytest.raises(ValueError, match="unsupported contraction legality"):
        run_pass(source)
    # Retain the independent former bug shape: target zero accumulation for a
    # source initialized to one must still be refused by the checker.
    wrong, count = run_pass(bmm())
    assert count == 1
    outcome = check(lean_tools, tmp_path, source, wrong)
    assert not outcome["conditional_match"]
    assert "destination is not a +0.0 splat or fill" in outcome["reason"]


def test_pass_preserves_explicit_matmul_maps_and_checker_refuses_the_old_fault(lean_tools, tmp_path):
    source = matmul(m=4, n=4, depth=4, transposed=True)
    output, count = run_pass(source)
    assert count == 1 and "-> (d1, d2)>" in _acc_line(output)
    outcome = check(lean_tools, tmp_path / "correct", source, output)
    assert outcome["pass_boundary"]["positive_boundary_checked"]
    wrong = replace_once(output, "-> (d1, d2)>", "-> (d2, d1)>")
    outcome = check(lean_tools, tmp_path / "old-fault", source, wrong)
    assert not outcome["conditional_match"]


def test_pass_binds_generic_arithmetic_and_checker_retains_counterexample(lean_tools, tmp_path):
    body = STANDARD_BODY.replace('"arith.mulf"(%x, %y)', '"arith.mulf"(%x, %x)')
    source = bmm(body=body, m=3, n=5, depth=5)
    with pytest.raises(ValueError, match="unsupported contraction legality"):
        run_pass(source)
    wrong, count = run_pass(bmm(m=3, n=5, depth=5))
    assert count == 1
    outcome = check(lean_tools, tmp_path, source, wrong)
    assert not outcome["conditional_match"]
    assert "body is not yield(acc + a * b)" in outcome["reason"]


def test_pass_refuses_f32_operands_before_emitting_invalid_casts():
    source = matmul(operand="f32")
    with pytest.raises(ValueError, match="unsupported contraction legality"):
        run_pass(source)


def test_report_states_refusal_reason_for_outside_candidates(lean_tools, tmp_path):
    source = bmm(init="1.000000e+00")
    outcome = check(lean_tools, tmp_path, source, source)
    assert outcome["candidates"] == 1 and not outcome["conditional_match"]
    assert "@subject: linalg.generic defining" in outcome["reason"]
    assert "destination is not a +0.0" in outcome["reason"]


def _stage(
    root: Path,
    before: str,
    after: str,
    name: str = "stage",
    boundary: str = "prepare-bf16-matmul-f32acc",
    pass_: str = "merlin.llvmlower.passes_xdsl.lower_bf16_matmul_f32acc",
) -> Path:
    import hashlib
    import json

    stage = root / name
    stage.mkdir()
    (stage / "before.mlir").write_text(before)
    (stage / "after.mlir").write_text(after)
    digest = {key: hashlib.sha256((stage / f"{key}.mlir").read_bytes()).hexdigest() for key in ("before", "after")}
    (stage / "meta.json").write_text(
        json.dumps(
            {
                "schema": "merlin.lean.stage.v1",
                "boundary": boundary,
                "changed": before != after,
                "pass": pass_,
                "run": "test",
                "before_sha256": digest["before"],
                "after_sha256": digest["after"],
            }
        )
    )
    return stage


def test_receipt_binds_bytes_and_replays(lean_tools, tmp_path, actual):
    import json

    from merlin.verify.lean.bf16_f32acc import POLICY, check_stage, replay

    destination = tmp_path / "receipt"
    receipt = check_stage(_stage(tmp_path, *actual), destination, m2m_python=lean_tools.m2m_python)
    assert (receipt["schema"], receipt["claim_type"], receipt["policy"]) == ("merlin.lean.receipt.v1", "policy", POLICY)
    assert receipt["verdict"] == "abstracted" and receipt["exporter"]["round_trip"] == "identical"
    assert set(receipt["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["conditional_match"] and report["qualification_passed"]
    assert all(
        set(axioms) <= {"propext", "Classical.choice", "Quot.sound"}
        for axioms in report["contract_theorem_axioms"].values()
    )
    outcome = replay(destination / "receipt.json", m2m_python=lean_tools.m2m_python, work=tmp_path / "replay")
    assert outcome["fresh"] and outcome["matches_recorded"], outcome
    exported = Path(receipt["exports"]["after"]["path"])
    exported.write_text(exported.read_text() + " ")
    with pytest.raises(ValueError, match="bytes changed"):
        replay(destination / "receipt.json", m2m_python=lean_tools.m2m_python, work=tmp_path / "tampered")


RETAINED_RECEIPTS = ("SY_model_smolvla", "SY_model_smolvla-segment", "SY_model_tiny_llama")


@pytest.mark.slow
@pytest.mark.parametrize("name", RETAINED_RECEIPTS)
def test_retained_receipt_replays(lean_tools, tmp_path, name):
    from merlin.common.paths import artifacts_dir
    from merlin.verify.lean.bf16_f32acc import replay

    receipt = artifacts_dir() / "verification/merlin-passes/frontend/bf16-f32acc" / name / "receipt.json"
    if not receipt.is_file():
        pytest.skip(f"retained receipt {name} is not present in this checkout")
    outcome = replay(receipt, m2m_python=lean_tools.m2m_python, work=tmp_path / "replay")
    assert outcome["matches_recorded"], outcome["differences"]
    rechecked = outcome["rechecked"]
    assert rechecked["conditional_match"]
    if name.startswith("SY_model_smolvla"):
        assert rechecked["candidates"] == 24
        assert rechecked["sites"] == [{"mul": "contract", "add": "contract"}] * 24
    if "segment_match" in rechecked:
        assert rechecked["segment_match"]


def _handoff(root: Path, subject: Path) -> Path:
    import hashlib
    import json

    record = root / "handoff.json"
    data = subject.read_bytes()
    record.write_text(
        json.dumps(
            {
                "schema": "merlin.lean.handoff.v1",
                "path": str(subject.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    )
    return record


def test_segment_receipt_composes_the_suffix_and_handoff(lean_tools, tmp_path, actual):
    import json

    from merlin.verify.lean.bf16_f32acc import SEGMENT, check_stages, replay

    source, output = actual
    first = _stage(tmp_path, source, output, "s0")
    second = _stage(
        tmp_path, output, output, "s1", "prepare-fix-bool-sitofp", "merlin.llvmlower.passes_xdsl.fix_bool_sitofp"
    )
    destination = tmp_path / "receipt"
    receipt = check_stages(
        [first, second],
        destination,
        m2m_python=lean_tools.m2m_python,
        handoff=_handoff(tmp_path, second / "after.mlir"),
    )
    assert receipt["checker"] == f"{SEGMENT[1]}.check" and receipt["verdict"] == "abstracted"
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["segment_match"] and report["segment_qualified"]
    assert report["suffix_boundaries"] == [{"conditional_match": True, "index": 0, "reason": ""}]
    assert report["handoff"]["subject"]["sha256"] == receipt["after"]["sha256"]
    outcome = replay(destination / "receipt.json", m2m_python=lean_tools.m2m_python, work=tmp_path / "replay")
    assert outcome["fresh"] and outcome["matches_recorded"], outcome


def test_segment_refuses_a_suffix_that_undoes_the_policy(lean_tools, tmp_path, actual):
    import json

    from merlin.verify.lean.bf16_f32acc import check_stages

    source, output = actual
    first = _stage(tmp_path, source, output, "s0")
    second = _stage(tmp_path, output, source, "s1", "prepare-revert", "test.revert")
    receipt = check_stages([first, second], tmp_path / "receipt", m2m_python=lean_tools.m2m_python)
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["policy_boundary"]["conditional_match"] and not report["segment_match"]
    assert not report["suffix_boundaries"][0]["conditional_match"]
    assert receipt["reason"].startswith("refused:")


def test_segment_refuses_stages_that_do_not_join(lean_tools, tmp_path, actual):
    from merlin.verify.lean.bf16_f32acc import check_stages

    source, output = actual
    first = _stage(tmp_path, source, output, "s0")
    second = _stage(tmp_path, source, source, "s1", "prepare-other", "test.other")
    receipt = check_stages([first, second], tmp_path / "receipt", m2m_python=lean_tools.m2m_python)
    assert receipt["verdict"] == "unmeasured" and "byte-identical chain" in receipt["reason"]


def test_fusion_premise_counts_licensed_source_operations(lean_tools, tmp_path):
    source = bmm(body=STANDARD_BODY.replace("fastmath<none>", "fastmath<contract>"))
    outcome = check(lean_tools, tmp_path, source, source)
    assert outcome["fusion_premises"]["source_licensed_operations"] == 2
    assert outcome["fusion_premises"]["theorem"] == "MerlinVerify.Rules.Bf16F32AccFusion.fused_step_on_image"
    assert not outcome["conditional_match"]  # a pre-licensed body is outside the contract


def test_local_success_cannot_hide_an_unrelated_module_change(lean_tools, tmp_path, actual):
    source, output = actual
    # The source's old zero splat survives the pass. Its value is irrelevant to the new result,
    # but a certificate for this pass must still account for the complete bound output.
    changed = replace_once(output, "value = 0.000000e+00 : bf16", "value = 1.000000e+00 : bf16")
    outcome = check(lean_tools, tmp_path, source, changed)
    local = outcome["pass_boundary"]
    assert local["site_qualification_passed"]
    assert not local["semantic_match"] and not local["checked"]


def test_pass_local_coverage_includes_every_function(lean_tools, tmp_path):
    lines = bmm().splitlines()
    body = "\n".join(lines[1:-1])
    source = lines[0] + "\n" + body + "\n" + body.replace('sym_name = "subject"', 'sym_name = "second"')
    source += "\n" + lines[-1] + "\n"
    output, count = run_pass(source)
    assert count == 2
    outcome = check(lean_tools, tmp_path, source, output)
    local = outcome["pass_boundary"]
    assert local["checked"] and local["expected_sites"] == local["covered_sites"] == 2
    assert [site["function"] for site in local["site_obligations"]] == ["subject", "second"]
    assert all(site["qualification_passed"] for site in local["site_obligations"])


def test_review_publisher_reexecutes_a_forged_green_report(lean_tools, tmp_path, actual):
    import json

    from build_tools.scripts.prepare_bf16_review import evidence
    from merlin.verify.lean.bf16_f32acc import check_stage
    from merlin.verify.lean.check import _file_ref

    source, output = actual
    invalid = output.replace("fastmath<contract>", "fastmath<nnan>", 1)
    destination = tmp_path / "receipt"
    receipt = check_stage(_stage(tmp_path, source, invalid), destination, m2m_python=lean_tools.m2m_python)
    path = Path(receipt["cert"]["path"])
    report = json.loads(path.read_text())
    forged = report["pass_boundary"]
    assert not forged["checked"]
    forged.update(checked=True, semantic_match=True, site_qualification_passed=True)
    for site in forged["site_obligations"]:
        site["qualification_passed"] = True
    path.write_text(json.dumps(report))
    receipt["cert"] = _file_ref(path)
    (destination / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="replay refused"):
        evidence(destination / "receipt.json", lean_tools.m2m_python, tmp_path / "replay")
