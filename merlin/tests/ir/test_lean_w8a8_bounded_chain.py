"""Keep each function's declared input domain through actual W8A8 and suffix checks."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_bounded import contract, embedding_source
from test_lean_w8a8_dynamic_policy import reused_dynamic_source
from test_lean_w8a8_native_recipe import dynamic_matmul_reference
from test_lean_w8a8_views import compare_outputs, words

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import stages as recording

pytestmark = pytest.mark.usefixtures("lean_oracle")


def lowered(before, count=1):
    after = before.clone()
    assert lower_contraction_int8(after) == count
    return after


def check_chain(programs, contracts, work, tools, *, index_bits=64, scope="all"):
    documents = []
    for index, program in enumerate(programs):
        directory = work / f"program-{index}"
        directory.mkdir()
        documents.append(exported(program, directory, tools, generic=True, allow_unregistered=True))
    bounds = work / "input-contracts.json"
    bounds.write_text(json.dumps(contracts))
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_w8a8_bounded_chain.lean"),
            str(documents[0]),
            str(documents[1]),
            str(bounds),
            str(index_bits),
            scope,
            *map(str, documents[2:]),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    (work / "driver-stderr.txt").write_text(proc.stderr)
    if proc.returncode == 2:
        assert not proc.stdout, proc.stdout
        return None, documents, proc
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    (work / "bounded-chain-report.json").write_text(json.dumps(report, indent=2))
    module = "MerlinVerify.Check.PolicyW8A8BoundedChain"
    assert report["checker"] == f"{module}.check" and report["theorem"] == f"{module}.check_sound"
    assert report["verdict"] == "abstracted"
    assert report["index_bits"] == index_bits and report["policy_scope"] == scope
    assert report["input_domain"] == "finite_f32_and_declared_integer_intervals_per_function"
    assert report["floating_domain"] == "byte_extension_with_explicit_source_and_dynamic_primitive_laws"
    assert report["source_primitive_laws"] == 7 and report["additional_dynamic_primitive_laws"] == 3
    for key in ("policy_reviewed", "primitive_model_laws_verified", "whole_pass_verified"):
        assert report[key] is False
    links = report["suffix_boundaries"]
    assert len(links) == len(programs) - 2
    assert [link["index"] for link in links] == list(range(len(links)))
    boundary = report["boundary_qualification_passed"]
    assert report["suffix_checked"] is boundary
    for link in links:
        if boundary:
            assert type(link["conditional_match"]) is bool
            assert link["status"] == ("checked" if link["conditional_match"] else "refused")
        else:
            assert link["conditional_match"] is None and link["status"] == "unmeasured"
    matched = all(link["conditional_match"] for link in links) if boundary else None
    assert report["suffix_match"] is matched
    assert report["conditional_initialized_policy_refinement"] is (boundary and matched is True)
    return report, documents, proc


def preprocessed(after, work, monkeypatch):
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(work / "preprocessing"))
    preprocess_text(str(after))
    boundaries = recording.completed(work / "preprocessing")
    assert len(boundaries) == 4
    previous = str(after).encode()
    results = []
    for boundary in boundaries:
        assert (boundary / "before.mlir").read_bytes() == previous
        previous = (boundary / "after.mlir").read_bytes()
        results.append(parse_mlir_text(previous.decode()))
    return results


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("suffix", ["empty", "identity", "preprocess"])
def test_bounds_follow_actual_downstream_passes(tmp_path, lean_tools, monkeypatch, index_bits, suffix):
    before = embedding_source(token_view=True)
    after = lowered(before)
    rest = [] if suffix == "empty" else [after.clone()]
    if suffix == "preprocess":
        rest = preprocessed(after, tmp_path, monkeypatch)
    report, _, _ = check_chain([before, after, *rest], contract(7), tmp_path, lean_tools, index_bits=index_bits)
    assert report["conditional_initialized_policy_refinement"], report
    assert report["functions"][0]["input_intervals"] == contract(7)[0]["intervals"]


@pytest.mark.parametrize("change", ["unknown", "too_large", "empty", "arity", "floating"])
def test_unqualified_input_domains_cannot_acquire_checked_suffixes(tmp_path, lean_tools, change):
    before = embedding_source()
    after = lowered(before)
    bounds = contract(7)
    if change == "unknown":
        bounds[0]["intervals"][0] = None
    elif change == "too_large":
        bounds[0]["intervals"][0]["upper"] = 8
    elif change == "empty":
        bounds[0]["intervals"][0]["lower"] = 7
    elif change == "arity":
        bounds[0]["intervals"].pop()
    else:
        bounds[0]["intervals"][1] = {"lower": 0, "upper": 7}
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools)
    assert not report["boundary_qualification_passed"]
    assert not report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("swap", [False, True])
def test_function_names_bind_domains_across_the_whole_chain(tmp_path, lean_tools, swap):
    from xdsl.dialects.builtin import StringAttr

    before = embedding_source(vocabulary=7)
    other = embedding_source(vocabulary=11)
    fn = next(op for op in other.walk() if op.name == "func.func")
    fn.properties["sym_name"] = StringAttr("other")
    before.body.block.add_op(fn.clone())
    after = lowered(before, 2)
    first, second = (11, 7) if swap else (7, 11)
    bounds = contract(first) + [{"function": "other", "intervals": contract(second)[0]["intervals"]}]
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"] is not swap
    assert [fn["function"] for fn in report["functions"]] == ["forward", "other"]


def changed_initializer(after, *, live):
    from xdsl.dialects.builtin import FloatAttr, f32

    changed = after.clone()
    if live:
        select = next(op for op in changed.walk() if op.name == "arith.select")
        scale = select.operands[1].owner
    else:
        scale = next(op for op in changed.walk() if op.name == "arith.constant" and str(op.results[0].type) == "f32")
    assert scale.name == "arith.constant"
    scale.properties["value"] = FloatAttr(1.0, f32)
    return changed


@pytest.mark.parametrize("edge", [0, 1])
def test_changed_suffix_output_is_refused_at_the_actual_boundary(tmp_path, lean_tools, edge):
    before = embedding_source()
    after = lowered(before)
    changed = changed_initializer(after, live=True)
    rest = [changed, changed.clone()] if edge == 0 else [after.clone(), changed]
    report, _, _ = check_chain([before, after, *rest], contract(7), tmp_path, lean_tools)
    assert report["boundary_qualification_passed"]
    assert not report["conditional_initialized_policy_refinement"]
    assert not report["suffix_boundaries"][edge]["conditional_match"]


@pytest.mark.parametrize("edge", [0, 1])
def test_changed_unused_initializer_is_proved_unobservable(tmp_path, lean_tools, edge):
    before = embedding_source()
    after = lowered(before)
    changed = changed_initializer(after, live=False)
    rest = [changed, changed.clone()] if edge == 0 else [after.clone(), changed]
    report, _, _ = check_chain([before, after, *rest], contract(7), tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report


@pytest.mark.parametrize("live", [False, True])
def test_mutation_observability_is_confirmed_by_lean_and_native_execution(tmp_path, lean_tools, fidelity_tools, live):
    import numpy as np
    from test_lean_w8a8_byte_model import evaluate

    from merlin.verify.lean import differential as diff

    before = embedding_source()
    after = lowered(before)
    changed = changed_initializer(after, live=live)
    report, documents, _ = check_chain([before, after, changed], contract(7), tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"] is not live
    table = np.arange(21, dtype=np.float32).reshape((7, 3)) / np.float32(30)
    weights = np.full((3, 4), np.float32(0.1))
    indices = [1, 4]
    samples = [[[str(i) for i in indices], words(table), words(weights)]]
    expected = [words(dynamic_matmul_reference(table[indices], weights))]
    compare_outputs(documents[1], samples, expected, documents[1].parent, fidelity_tools)
    work = documents[2].parent
    proc, lean = evaluate(documents[2], samples, work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    inputs, outputs = diff.signature(json.loads(documents[2].read_text()), "forward")
    native = diff.execute_mlir(
        work / "actual.mlir", samples, work, fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    diff.assert_equal(lean, native["results"], outputs, samples)
    assert (native["results"] != expected) is live


def test_typed_splat_suffix_preserves_original_function_domains(tmp_path, lean_tools):
    from test_lean_splat_inputs import _source

    from merlin.llvmlower.splat_inputs import scalarize_splat_inputs

    before = embedding_source()
    helper = parse_mlir_text(_source(element="f32"))
    before.body.block.add_ops(op.clone() for op in helper.body.block.ops)
    after = lowered(before)
    final = after.clone()
    assert scalarize_splat_inputs(final) >= 1
    bounds = contract(7) + [{"function": "subject", "intervals": [None] * 4}]
    report, _, _ = check_chain([before, after, final], bounds, tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report


def test_suffix_cannot_hide_changed_top_level_operations(tmp_path, lean_tools):
    before = embedding_source()
    after = lowered(before)
    final = after.clone()
    extra = parse_mlir_text('"research.unverified"() : () -> ()')
    final.body.block.add_ops(op.clone() for op in extra.body.block.ops)
    report, _, _ = check_chain([before, after, final], contract(7), tmp_path, lean_tools)
    assert report["boundary_qualification_passed"]
    assert not report["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("integer_left", [False, True])
def test_captured_source_quantization_keeps_its_additive_interpretation(tmp_path, lean_tools, integer_left):
    before = reused_dynamic_source(integer_left)
    after = lowered(before)
    bounds = [{"function": "forward", "intervals": [None, None]}]
    report, _, _ = check_chain([before, after, after.clone()], bounds, tmp_path, lean_tools, scope="captured")
    assert report["conditional_initialized_policy_refinement"], report


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "negative", "float", "interval_key", "entry_key"])
def test_contract_protocol_rejects_ambiguous_or_malformed_domains(tmp_path, lean_tools, change):
    before = embedding_source()
    bounds = contract(7)
    if change == "duplicate":
        bounds += contract(7)
    elif change == "missing":
        bounds = []
    elif change == "extra":
        bounds += [{"function": "unrelated", "intervals": []}]
    elif change in {"negative", "float"}:
        bounds[0]["intervals"][0]["lower"] = -1 if change == "negative" else 0.5
    elif change == "interval_key":
        bounds[0]["intervals"][0]["trusted"] = True
    else:
        bounds[0]["policy_reviewed"] = True
    report, _, proc = check_chain([before, lowered(before)], bounds, tmp_path, lean_tools)
    assert report is None and proc.returncode == 2


def test_complete_preprocessing_suffix_matches_native_lean_and_recipe(
    tmp_path, lean_tools, fidelity_tools, monkeypatch
):
    import numpy as np

    before = embedding_source(token_view=True)
    after = lowered(before)
    rest = preprocessed(after, tmp_path, monkeypatch)
    report, documents, _ = check_chain([before, after, *rest], contract(7), tmp_path, lean_tools)
    assert report["conditional_initialized_policy_refinement"], report
    rng = np.random.default_rng(20261024)
    samples, expected = [], []
    for indices in ([0, 0], [6, 6], [1, 4]):
        table = rng.uniform(-5, 5, (7, 3)).astype(np.float32)
        weights = rng.uniform(-3, 3, (3, 4)).astype(np.float32)
        samples.append([[str(i) for i in indices], words(table), words(weights)])
        expected.append(words(dynamic_matmul_reference(table[indices], weights)))
    compare_outputs(documents[-1], samples, expected, documents[-1].parent, fidelity_tools)
