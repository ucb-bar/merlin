"""A matching signature alone does not prove captured argument identity."""
import hashlib
import inspect
from pathlib import Path

import pytest

from merlin.frontends.argument_identity import ArgumentIdentityStage, replay_argument_identity
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text as module_text

SOURCE = '''builtin.module {
  func.func @forward(%a: tensor<2xi8>, %b: tensor<2xi8>) -> tensor<2xi8> {
    func.return %a : tensor<2xi8>
  }
}'''


def _noop(module):
    return None


def _replace_function(module):
    old = next(op for op in module.body.block.ops if op.name == "func.func")
    module.body.block.insert_op_before(old.clone(), old)
    module.body.block.erase_op(old)


def _pins(callback):
    path = Path(inspect.getsourcefile(callback))
    return {path: hashlib.sha256(path.read_bytes()).hexdigest()}


def _run(stages=(), **changes):
    args = dict(raw_text=SOURCE, source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
                normalized_sha256=hashlib.sha256(module_text(parse_mlir_text(SOURCE)).encode()).hexdigest(),
                entry="forward", stages=stages, source_pins=_pins(_noop))
    args.update(changes)
    return replay_argument_identity(**args)


def test_replay_preserves_actual_arguments_across_explicit_serialization_boundary():
    proof = _run((ArgumentIdentityStage(_noop), ArgumentIdentityStage(_noop, reparse_before=True)))
    assert proof.to_evidence()["argument_index_map"] == [0, 1]
    assert proof.argument_types == ("tensor<2xi8>", "tensor<2xi8>")
    assert proof.to_evidence()["stages"][1]["reparse_before"] is True


def test_same_signature_replacement_is_not_argument_identity():
    with pytest.raises(ValueError, match="identity"):
        _run((ArgumentIdentityStage(_replace_function),))


@pytest.mark.parametrize("change", [
    {"source_sha256": "0" * 64}, {"normalized_sha256": "0" * 64}, {"entry": "missing"},
])
def test_wrong_source_endpoint_or_entry_refuses(change):
    with pytest.raises(ValueError):
        _run(**change)


def test_missing_or_stale_callback_source_pin_refuses():
    with pytest.raises(ValueError, match="pinned host implementation"):
        _run((ArgumentIdentityStage(_noop),), source_pins={})
    with pytest.raises(ValueError, match="source pins"):
        _run((ArgumentIdentityStage(_noop),), source_pins={next(iter(_pins(_noop))): "0" * 64})
