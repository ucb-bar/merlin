"""Capture dependencies, storage, dominance and effects are never approximated."""

import io

import pytest
import test_connected_integer_bounds as B
from merlin_experiments.phase0 import operator_capture_exclusion as C
from merlin_experiments.phase0 import operator_dense_projection as P
from merlin_experiments.phase0 import operator_initializer_identity as ID
from merlin_experiments.phase0 import operator_registered_exclusion as OLD
from merlin_experiments.phase0 import operator_windowed_ir as W

from merlin.common import mlir_query as Q

LIMIT = 1 << 20


def source(producer='%c = arith.constant 2 : i32', scalar=''):
    return B.program().replace('%z =', producer + '\n %z =').replace(
        '%sum = arith.addi %acc, %p', scalar + '\n %biased = arith.addi %p, %c : i32\n'
        ' %sum = arith.addi %acc, %biased').encode()


def signature(raw):
    return C.signatures(raw, max_bytes=LIMIT)


def test_old_versions_stay_unchanged_and_capture_graph_has_explicit_version():
    raw = source()
    with pytest.raises(ValueError, match='unowned scalar'):
        OLD.signatures(raw, max_bytes=LIMIT)
    assert signature(raw)
    ordinary = B.program().encode()
    assert signature(ordinary) == OLD.signatures(ordinary, max_bytes=LIMIT)
    for mutation in [raw.replace(b'constant 2', b'constant 3'),
                     raw.replace(b'%biased = arith.addi', b'%biased = arith.subi'),
                     raw.replace(b'%c : i32', b'%p : i32')]:
        assert signature(mutation) != signature(raw)
    assert signature(raw.replace(b'%c', b'%renamed_capture')) == signature(raw)


@pytest.mark.parametrize('literal', ['[1, 2, 3, 4]', '"0x01000000020000000300000004000000"'])
def test_tensor_extract_binds_full_original_storage_and_public_private_readers_agree(literal):
    raw = source(f'%tensor = arith.constant dense<{literal}> : tensor<4xi32>\n'
                 '%index = arith.constant 0 : index',
                 '%c = tensor.extract %tensor[%index] : tensor<4xi32>')
    metadata, compact, records = P.scan(io.BytesIO(raw), max_input_bytes=LIMIT,
                                      max_compact_bytes=LIMIT, chunk_bytes=7)
    module, _ = W.parse(io.BytesIO(compact), max_window_bytes=LIMIT, max_total_bytes=LIMIT)
    ids = ID.identities(module, records, max_decoded_bytes=LIMIT)
    assert metadata['source_bytes'] == len(raw)
    assert C.module_signatures(module, initializers=ids) == signature(raw)
    assert C.module_signatures(Q.parse(raw.decode())) == signature(raw)
    assert signature(raw.replace(b'constant 0 : index', b'constant 1 : index')) != signature(raw)
    changed = raw.replace(b'[1, 2, 3, 4]', b'[1, 2, 3, 5]').replace(
        b'04000000', b'05000000')
    assert signature(changed) != signature(raw)


def test_semantically_identical_splat_array_and_hex_storage_share_identity():
    outputs = []
    for literal in ['2', '[2, 2, 2, 2]', '"0x02000000020000000200000002000000"']:
        outputs.append(signature(source(f'%tensor = arith.constant dense<{literal}> : tensor<4xi32>\n'
            '%index = arith.constant 0 : index', '%c = tensor.extract %tensor[%index] : tensor<4xi32>')))
    assert outputs[0] == outputs[1] == outputs[2]


def test_exact_producer_chain_and_function_argument_position_are_bound():
    raw = source('%one = arith.constant 1 : i32\n %c = arith.addi %one, %one : i32')
    assert signature(raw) != signature(raw.replace(b'%c = arith.addi', b'%c = arith.subi'))
    captured = source('').replace(b'%w: tensor<3x2xi8>', b'%w: tensor<3x2xi8>, %c: i32, %other: i32')
    assert signature(captured) != signature(captured.replace(b'%p, %c :', b'%p, %other :'))


@pytest.mark.parametrize('mutation', [
    lambda raw: raw.replace(b'%c = arith.constant 2 : i32', b'%c = "unknown.value"() : () -> i32'),
    lambda raw: raw.replace(b'%c = arith.constant 2 : i32', b'%c = "arith.unknown"() : () -> i32'),
    lambda raw: raw.replace(b'%c = arith.constant 2 : i32', b'%c = func.call @external() : () -> i32'),
    lambda raw: raw.replace(b'%c = arith.constant 2 : i32', b'%c = arith.addi %later, %later : i32\n'
                           b'%later = arith.constant 2 : i32'),
    lambda raw: raw.replace(b'%c = arith.constant 2 : i32', b'').replace(
        b'%empty = tensor.empty()', b'%c = arith.constant 2 : i32\n %empty = tensor.empty()'),
    lambda raw: raw.replace(b'%biased = arith.addi %p, %c', b'%biased = arith.addi %future, %c').replace(
        b'linalg.yield %sum', b'%future = arith.constant 1 : i32\n linalg.yield %sum'),
])
def test_unknown_effects_forward_and_nondominating_values_refuse(mutation):
    with pytest.raises(Exception):
        signature(mutation(source()))


def test_undefined_tensor_and_foreign_block_captures_refuse():
    raw = source('%tensor = tensor.empty() : tensor<4xi32>\n %index = arith.constant 0 : index',
                 '%c = tensor.extract %tensor[%index] : tensor<4xi32>')
    with pytest.raises(ValueError, match='unknown or impure'):
        signature(raw)
    module = Q.parse(source().decode())
    root = next(op for op in module.walk() if op.name == 'linalg.generic')
    argument = root.regions[0].block.args[0]
    foreign = next(op for op in module.walk() if op.name == 'arith.constant')
    use = next(op for op in root.walk() if op.name == 'arith.addi')
    use.operands = (argument, foreign.results[0])
    with pytest.raises(Exception):
        C.module_signatures(module)


def test_captured_producer_attributes_and_unowned_resources_refuse_or_change_identity():
    raw = source('%one = arith.constant 1 : i32\n %c = arith.addi %one, %one : i32')
    module = Q.parse(raw.decode())
    owner = next(op for op in module.walk() if op.name == 'arith.addi')
    from xdsl.dialects.builtin import StringAttr
    before = C.module_signatures(module)
    owner.attributes['checked_semantics'] = StringAttr('different')
    assert C.module_signatures(module) != before
    resource = source('%tensor = arith.constant dense_resource<external> : tensor<4xi32>\n'
                      '%index = arith.constant 0 : index',
                      '%c = tensor.extract %tensor[%index] : tensor<4xi32>')
    with pytest.raises(ValueError, match='original initializer identity'):
        signature(resource)


def test_explicit_v6_issuer_and_live_public_reader_use_the_same_grammar(tmp_path, monkeypatch):
    import hashlib
    import json

    import test_operator_exclusion_issuer as H
    import test_operator_private_exclusion as PUB
    from merlin_experiments.phase0 import operator_exclusion as E
    from merlin_experiments.phase0 import operator_exclusion_intake as I
    from merlin_experiments.phase0 import operator_streaming_exclusion_issuer as S

    original = tmp_path/'original.mlir'
    original.write_bytes(source())
    selected = {str(original): dict(sha256=H.pin(original)['sha256'], read_path=str(original), alias=None)}
    monkeypatch.setattr(I, 'resolve_streaming_v2', lambda *_: selected)
    private = tmp_path/'private'
    private.mkdir(mode=0o700)
    path = private/'selection.json'
    live, receipt = S.issue(tmp_path/'pointer.json', 'a'*64, path,
        forbidden_roots=[tmp_path/'public'], source_mode=C.SCHEMA, input_mode=W.SCHEMA,
        max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT, max_source_bytes=LIMIT, max_total_source_bytes=LIMIT)
    data = json.loads(path.read_bytes())
    assert data['schema'] == E.SCHEMA_V6 and receipt['signature_schema'] == C.SCHEMA
    assert set(data['excluded_typed_ir_sha256']) == set(signature(source()))
    cap, raw = PUB.H.capsule(PUB.original())
    directory = PUB.writer(tmp_path/'public', cap=cap, source=raw)
    live.check_member(directory)
    data['excluded_typed_ir_sha256'] = list(signature(raw.encode()))
    path.write_bytes(json.dumps(data).encode())
    with pytest.raises(E.PublicationRefusal, match=E.REFUSAL):
        E.select(path, hashlib.sha256(path.read_bytes()).hexdigest()).check_member(directory)
