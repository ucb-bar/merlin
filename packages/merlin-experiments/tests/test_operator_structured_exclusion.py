"""Complete registered reductions use one explicit public/private grammar."""

import hashlib
import json

import pytest
import test_connected_integer_bounds as B
import test_operator_exclusion_issuer as H
import test_operator_mixed_exclusion as MIX
import test_operator_private_exclusion as PUB
from merlin_experiments.phase0 import operator_exclusion as E
from merlin_experiments.phase0 import operator_exclusion_intake as I
from merlin_experiments.phase0 import operator_streaming_exclusion_issuer as S
from merlin_experiments.phase0 import operator_structured_exclusion as R
from merlin_experiments.phase0 import operator_typed_exclusion as T

LIMIT = 1 << 20


def reduction():
    return b'''module { func.func @reduce(%x: tensor<2x3xi32>, %z: tensor<2xi32>) -> tensor<2xi32> {
%r = linalg.reduce ins(%x : tensor<2x3xi32>) outs(%z : tensor<2xi32>) dimensions = [1]
(%v: i32, %a: i32) {
  %sum = arith.addi %v, %a : i32
  linalg.yield %sum : i32
}
return %r : tensor<2xi32>
} }'''


def test_registered_reduction_is_explicit_and_original_contractions_are_unchanged():
    raw = reduction()
    with pytest.raises(ValueError, match='unsupported named'):
        T.signatures(raw, max_bytes=LIMIT)
    assert len(R.signatures(raw, max_bytes=LIMIT)) == 1
    assert R.signatures(B.program().encode(), max_bytes=LIMIT) == T.signatures(B.program().encode(), max_bytes=LIMIT)
    assert len(R.signatures(MIX.interface(), max_bytes=LIMIT)) == 2
    assert R.signatures(raw.replace(b'@reduce', b'@renamed'), max_bytes=LIMIT) == R.signatures(raw, max_bytes=LIMIT)


@pytest.mark.parametrize('before,after', [
    (b'arith.addi', b'arith.muli'),
    (b'2x3xi32', b'2x4xi32'),
    (b'xi32', b'xf32'),
])
def test_semantic_mutations_never_keep_the_original_reduction_hash(before, after):
    raw = reduction()
    try:
        changed = R.signatures(raw.replace(before, after), max_bytes=LIMIT)
    except Exception:
        return
    assert changed != R.signatures(raw, max_bytes=LIMIT)


@pytest.mark.parametrize('before,after', [
    (b'dimensions = [1]', b'dimensions = [0, 0]'),
    (b'dimensions = [1]', b'dimensions = [3]'),
    (b'dimensions = [1]', b'dimensions = []'),
    (b'%v: i32, %a: i32', b'%v: i32, %a: i8'),
    (b'%sum : i32\n}', b'%v, %a : i32, i32\n}'),
    (b'arith.addi %v, %a : i32', b'"unknown.effect"(%v, %a) : (i32, i32) -> i32'),
    (b'%sum = arith.addi %v, %a : i32', b'%sum = arith.addi %v, %outer : i32'),
    (b' linalg.yield', b' linalg.unknown'),
])
def test_incomplete_dimensions_abi_ownership_and_unknown_effects_refuse(before, after):
    raw = reduction()
    assert before in raw
    with pytest.raises(Exception):
        R.signatures(raw.replace(before, after), max_bytes=LIMIT)


def test_v4_streamed_issuance_and_public_reader_use_the_same_checked_grammar(tmp_path, monkeypatch):
    source = tmp_path/'original.mlir'
    source.write_bytes(reduction())
    selected = {str(source): dict(sha256=H.pin(source)['sha256'], read_path=str(source), alias=None)}
    monkeypatch.setattr(I, 'resolve_streaming_v2', lambda *_: selected)
    private = tmp_path/'private'
    private.mkdir(mode=0o700)
    destination = private/'selection.json'
    live, receipt = S.issue(tmp_path/'pointer.json', 'a'*64, destination,
        forbidden_roots=[tmp_path/'public'], source_mode=R.SCHEMA,
        max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT, max_source_bytes=LIMIT, max_total_source_bytes=LIMIT)
    data = json.loads(destination.read_bytes())
    assert data['schema'] == E.SCHEMA_V4
    assert data['excluded_typed_ir_sha256'] == list(R.signatures(reduction(), max_bytes=LIMIT))
    assert receipt['signature_schema'] == R.SCHEMA
    assert destination.stat().st_mode & 0o777 == 0o600
    # The same live reader accepts unrelated checked public sources, while an
    # actual matching contraction must still be rejected by its versioned gate.
    cap, raw = PUB.H.capsule(PUB.original())
    directory = PUB.writer(tmp_path/'public', cap=cap, source=raw)
    live.check_member(directory)
    data['excluded_typed_ir_sha256'] = list(R.signatures(raw.encode(), max_bytes=LIMIT))
    destination.write_bytes(json.dumps(data).encode())
    with pytest.raises(E.PublicationRefusal, match=E.REFUSAL):
        E.select(destination, hashlib.sha256(destination.read_bytes()).hexdigest()).check_member(directory)
