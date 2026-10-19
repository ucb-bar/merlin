"""No unregistered semantics or loop-coordinate ownership is inferred."""

import pytest
import test_connected_integer_bounds as B
import test_operator_structured_exclusion as RED
from merlin_experiments.phase0 import operator_registered_exclusion as G
from merlin_experiments.phase0 import operator_structured_exclusion as OLD

LIMIT = 1 << 20


def indexed(dimension=0):
    return B.program().replace(
        '%p = arith.muli', f'%coordinate = linalg.index {dimension} : index\n            %p = arith.muli').encode()


def test_registered_iterator_coordinate_is_owned_and_old_version_refusal_stays():
    raw = indexed()
    with pytest.raises(ValueError):
        OLD.signatures(raw, max_bytes=LIMIT)
    assert G.signatures(raw, max_bytes=LIMIT)
    assert G.signatures(indexed(1), max_bytes=LIMIT) != G.signatures(raw, max_bytes=LIMIT)
    assert G.signatures(B.program().encode(), max_bytes=LIMIT) == OLD.signatures(B.program().encode(), max_bytes=LIMIT)
    assert G.signatures(RED.reduction(), max_bytes=LIMIT) == OLD.signatures(RED.reduction(), max_bytes=LIMIT)


@pytest.mark.parametrize('mutation', [
    lambda s: s.replace(b'linalg.index 0', b'linalg.index -1'),
    lambda s: s.replace(b'linalg.index 0', b'linalg.index 3'),
    lambda s: s.replace(b'%coordinate = linalg.index 0 : index',
                        b'%coordinate = "linalg.index"() <{dim = 0 : i64}> {semantic = true} : () -> index'),
    lambda s: s.replace(b'linalg.index 0 : index', b'"unknown.index"() : () -> index'),
    lambda s: s.replace(b'%coordinate = linalg.index 0 : index', b'').replace(
        b'%z = arith.constant', b'%coordinate = linalg.index 0 : index\n %z = arith.constant'),
])
def test_bad_dimension_type_effect_or_parent_never_becomes_a_checked_coordinate(mutation):
    with pytest.raises(Exception):
        G.signatures(mutation(indexed()), max_bytes=LIMIT)


def test_registered_parallel_tensor_operation_uses_actual_maps_iterators_and_body():
    raw = b'''module { func.func @add(%a: tensor<2x3xi32>,
%b: tensor<2x3xi32>, %i: tensor<2x3xi32>) -> tensor<2x3xi32> {
%r = linalg.add ins(%a, %b : tensor<2x3xi32>, tensor<2x3xi32>) outs(%i : tensor<2x3xi32>) -> tensor<2x3xi32>
return %r : tensor<2x3xi32>
} }'''
    with pytest.raises(ValueError):
        OLD.signatures(raw, max_bytes=LIMIT)
    assert G.signatures(raw, max_bytes=LIMIT) == ()  # Explicit registered parallel iteration; no reduction.
    with pytest.raises(Exception):
        G.signatures(raw.replace(b'linalg.add', b'linalg.unknown'), max_bytes=LIMIT)


def test_registered_named_reduction_hashes_actual_indexing_iteration_and_scalar_body():
    raw = b'''module { func.func @pool(%a: tensor<1x1x4x4xf32>,
%k: tensor<2x2xf32>, %i: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
%r = linalg.pooling_nchw_max {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>}
ins(%a, %k : tensor<1x1x4x4xf32>, tensor<2x2xf32>) outs(%i : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
return %r : tensor<1x1x3x3xf32>
} }'''
    assert len(G.signatures(raw, max_bytes=LIMIT)) == 1
    changed = raw.replace(b'3x3xf32', b'2x2xf32').replace(b'strides = dense<1>', b'strides = dense<2>')
    assert G.signatures(changed, max_bytes=LIMIT) != G.signatures(raw, max_bytes=LIMIT)
    with pytest.raises(Exception):
        G.signatures(raw.replace(b'tensor<1x1x4x4xf32>', b'tensor<1x1x2x2xf32>'), max_bytes=LIMIT)


def test_v5_private_stream_and_live_public_reader_select_the_same_registered_checker(tmp_path, monkeypatch):
    import hashlib
    import json

    import test_operator_exclusion_issuer as H
    import test_operator_private_exclusion as PUB
    from merlin_experiments.phase0 import operator_exclusion as E
    from merlin_experiments.phase0 import operator_exclusion_intake as I
    from merlin_experiments.phase0 import operator_streaming_exclusion_issuer as S
    from merlin_experiments.phase0 import operator_windowed_ir as W

    source = tmp_path/'original.mlir'
    source.write_bytes(indexed())
    selected = {str(source): dict(sha256=H.pin(source)['sha256'], read_path=str(source), alias=None)}
    monkeypatch.setattr(I, 'resolve_streaming_v2', lambda *_: selected)
    private = tmp_path/'private'
    private.mkdir(mode=0o700)
    path = private/'selection.json'
    live, receipt = S.issue(tmp_path/'pointer.json', 'a'*64, path,
        forbidden_roots=[tmp_path/'public'], source_mode=G.SCHEMA, input_mode=W.SCHEMA,
        max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT, max_source_bytes=LIMIT, max_total_source_bytes=LIMIT)
    data = json.loads(path.read_bytes())
    assert data['schema'] == E.SCHEMA_V5 and receipt['signature_schema'] == G.SCHEMA
    assert set(data['excluded_typed_ir_sha256']) == set(G.signatures(indexed(), max_bytes=LIMIT))
    cap, raw = PUB.H.capsule(PUB.original())
    directory = PUB.writer(tmp_path/'public', cap=cap, source=raw)
    live.check_member(directory)
    data['excluded_typed_ir_sha256'] = list(G.signatures(raw.encode(), max_bytes=LIMIT))
    path.write_bytes(json.dumps(data).encode())
    with pytest.raises(E.PublicationRefusal, match=E.REFUSAL):
        E.select(path, hashlib.sha256(path.read_bytes()).hexdigest()).check_member(directory)
