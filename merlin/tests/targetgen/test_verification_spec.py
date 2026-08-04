"""The agent-facing verification spec is DERIVED from the answer-free capsule declarations and carries no
golden. Hermetic: a synthetic corpus with a golden.yaml sitting beside the capsule proves the generator
never lets an output value into the spec. Plus a gated check that a real target's ops/dtypes/policy derive.
"""
from __future__ import annotations

import types

import pytest
import yaml

from merlin.targetgen import verification_spec as VS


def _synth_te(root):
    return types.SimpleNamespace(target="synth", capsule_corpus=root / "isa",
                                 corpus_siblings=lambda: [], isa_headers=["docs/isa.md"])


def test_spec_is_answer_free_even_with_a_golden_beside_the_capsule(tmp_path):
    cap = tmp_path / "isa" / "T0_matmul"
    cap.mkdir(parents=True)
    (cap / "capsule.yaml").write_text(yaml.safe_dump({
        "name": "T0_matmul", "label": "public",
        "inputs": [{"name": "A", "role": "input", "dtype": "i8"},
                   {"name": "W", "role": "weight", "dtype": "i8"}],
        "operation": {"op": "matmul", "attributes": {"output_dtype": "i32", "epilogue": ["relu"]}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": ["MVIN", "MATMUL", "MVOUT"]},
    }))
    # the ANSWER KEY sits right beside it — the generator must never read or echo it.
    (cap / "golden.yaml").write_text(yaml.safe_dump(
        {"golden_source": "x", "outputs": {"Y": [[424242, 424242]]}}))

    te = _synth_te(tmp_path)
    spec = VS.build_spec(te)
    md = VS.render_markdown(te)

    # the CONTRACT is present: op, dtypes, acceptance policy, epilogue, coverage
    assert "matmul" in spec["ops"]
    assert spec["ops"]["matmul"]["dtypes"] == ["i8 -> i32"]
    assert any("exact_int" in a for a in spec["ops"]["matmul"]["accept"])
    assert "relu" in spec["ops"]["matmul"]["epilogues"]
    assert "MVIN" in spec["ops"]["matmul"]["coverage"] and "MVOUT" in spec["ops"]["matmul"]["coverage"]
    # the ANSWER is absent: no golden output value, no outputs payload key
    assert "424242" not in md
    assert "outputs:" not in md
    # 'golden' appears only as the reframing prose ("no answer key / no stored golden"), never a value
    assert "no answer key" in md.lower()


def test_hidden_labelled_capsule_is_excluded(tmp_path):
    for name, label in (("P0", "public"), ("H0", "hidden")):
        d = tmp_path / "isa" / name
        d.mkdir(parents=True)
        (d / "capsule.yaml").write_text(yaml.safe_dump({
            "name": name, "label": label,
            "inputs": [{"name": "A", "dtype": "i8"}],
            "operation": {"op": "matmul", "attributes": {"output_dtype": "i32"}},
            "numeric_policy": {"compare": "exact_int"}}))
    spec = VS.build_spec(_synth_te(tmp_path))
    assert spec["n_capsules"] == 1                      # the hidden-labelled capsule is not counted


def test_real_target_ops_and_policy_derive():
    """A real target's spec derives its ops + acceptance policy from its committed corpus, no code change."""
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin.common.paths import merlin_dir
    p = merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    if not p.is_file():
        pytest.skip("gemmini descriptor absent")
    spec = VS.build_spec(load_target_experiment(p))
    if not spec["ops"]:
        pytest.skip("gemmini corpus not generated")
    assert "matmul" in spec["ops"]
    assert any("exact_int" in a for a in spec["ops"]["matmul"]["accept"])
    md = VS.render_markdown(load_target_experiment(p))
    assert "acceptance contract for `gemmini`" in md and "outputs:" not in md
