"""A refuted obligation becomes a witness the capsule bench can grade.

This closes the loop the PI review asked about: the corpus stops being only what an author thought to
write down. The solver picks the shape and the inputs, and the resulting witness is graded by the same
ladder as every derived one.

The emitted witness must be SCHEMA-VALID — a "counterexample capsule" the corpus cannot load is a
demo, not a result.
"""
from __future__ import annotations

import json

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.tools import find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra (xdsl + z3) and mlir-translate")


def _refuted_verdict():
    from merlin.verify.refine import validate_interface_module
    from merlin.xdsl_dialects.lowering import pipeline

    module = pipeline.lower_repeated_rhs_matmul(reuse=2, m=2, k=2, n=2).interface_module
    func = next(o for o in module.walk() if o.name == "func.func")
    ops = list(func.body.block.ops)
    commits = [o for o in ops if o.name == "interface.commit"]
    matmuls = [o for o in ops if o.name == "interface.matmul"]
    commits[1].operands[0] = matmuls[0].results[0]
    return validate_interface_module(module)


@pytest.fixture(scope="module")
def witness(tmp_path_factory):
    from merlin.verify.witness import emit_witness

    v = _refuted_verdict()
    assert v.refuted, v.status
    return emit_witness(v, name="CE_test", dest=tmp_path_factory.mktemp("ce"),
                        obligation="commit reads its own accumulator",
                        producing_pass="merlin-materialize-interface")


def test_witness_is_schema_valid(witness):
    from merlin.common.yaml import load_yaml
    from merlin.targetgen.contract.schemas import validate_capsule

    capsule = load_yaml(witness / "capsule.yaml")
    validate_capsule(capsule)
    assert capsule["source_role"] == "smt_counterexample"


def test_witness_uses_the_corpus_leaf_convention(witness):
    """Leaf tensors are materialized deterministically BY NAME, so encoder-internal symbol names
    would not line up with a golden."""
    from merlin.common.yaml import load_yaml

    names = {i["name"] for i in load_yaml(witness / "capsule.yaml")["inputs"]}
    assert names == {"W", "A0", "A1"}


def test_counterexample_values_are_carried(witness):
    """The values are the point: they are what the degenerate default stimulus would never produce."""
    values = json.loads((witness / "counterexample_inputs.json").read_text())
    assert set(values) == {"W", "A0", "A1"}
    flat = [v for t in values.values() for row in t for v in row]
    assert all(-128 <= v <= 127 for v in flat), "values must be representable as i8"
    assert len(set(flat)) > 1, "a constant counterexample would be a degenerate witness"


def test_a_partial_model_is_refused():
    """A missing element must raise, not silently yield a smaller tensor."""
    from merlin.verify.witness import parse_model

    with pytest.raises(ValueError):
        parse_model({"arg0_0_0_1": 1, "arg0_1_1_2": 2})  # (0,1) and (1,0) absent


def test_only_a_refuted_verdict_yields_a_witness(tmp_path):
    from merlin.verify.smt_export import Verdict
    from merlin.verify.witness import emit_witness

    with pytest.raises(ValueError):
        emit_witness(Verdict("unsat"), name="x", dest=tmp_path, obligation="o", producing_pass="p")


def test_witness_lands_in_the_directory_it_was_named(tmp_path):
    """Regression: a loop variable shadowed the `name` parameter, so the witness was written into a
    directory named after the last activation tensor instead of the witness's own name. The returned
    path looked plausible in isolation; only listing the tree showed it."""
    from merlin.verify.witness import emit_witness

    out = emit_witness(_refuted_verdict(), name="CE_named", dest=tmp_path,
                       obligation="o", producing_pass="p")
    assert out.name == "CE_named"
    assert {p.name for p in tmp_path.iterdir()} == {"CE_named"}
