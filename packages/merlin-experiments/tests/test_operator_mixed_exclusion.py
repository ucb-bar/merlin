"""Complete restricted interface value algebra, with no protected source inputs."""

import hashlib
import json

import pytest
import test_connected_integer_bounds as B
import test_operator_private_exclusion as H
from merlin_experiments.phase0 import operator_exclusion as E
from merlin_experiments.phase0 import operator_mixed_exclusion as M
from merlin_experiments.phase0 import operator_typed_exclusion as T

from merlin.targetgen.contract.matmul_interface import emit_interface_mlir

LIMIT = 1 << 20


def interface(m=2, k=3, n=2):
    return emit_interface_mlir(lhs='A', weight='W', out='Y', M=m, K=k, N=n, epilogue=[],
                               output_dtype='i32', target='synthetic', operand_dtype='i8', acc_dtype='i32').encode()


def test_explicit_interface_geometry_and_original_integer_generic_canonical_hash():
    source = interface()
    assert M.raw_interface_geometry(source) == dict(M=2, N=2, K=3)
    images = M.signatures(source, max_bytes=LIMIT)
    assert len(images) == 2 and set(T.signatures(B.program().encode(), max_bytes=LIMIT)) <= set(images)
    assert M.signatures(B.program().encode(), max_bytes=LIMIT) == T.signatures(B.program().encode(), max_bytes=LIMIT)
    renamed = source.replace(b'%A', b'%other_lhs').replace(b'%W', b'%other_rhs')
    renamed = renamed.replace(b'name = "A"', b'name = "other_lhs"').replace(b'name = "W"', b'name = "other_rhs"')
    assert M.signatures(renamed, max_bytes=LIMIT) == images
    assert M.signatures(source.replace(b'synthetic', b'another_target'), max_bytes=LIMIT) == images


def test_changed_complete_shapes_change_both_original_canonical_images():
    old = set(M.signatures(interface(), max_bytes=LIMIT))
    for new in [interface(m=3), interface(n=4), interface(k=5)]:
        assert not old & set(M.signatures(new, max_bytes=LIMIT))


@pytest.mark.parametrize('mutation', [
    lambda s: s.replace(b'xi8>', b'xi16>'),
    lambda s: s.replace(b'output_dtype = "i32"', b'output_dtype = "i8"'),
    lambda s: s.replace(b'epilogue = []', b'epilogue = ["relu"]'),
    lambda s: s.replace(b'epilogue = []', b'epilogue = [], acc_scale = 1.0 : f32'),
    lambda s: s.replace(b'merlin_iface.matmul %A, %W_res', b'merlin_iface.matmul %A, %W_res {init = 0 : i32}'),
    lambda s: s.replace(b'merlin_iface.matmul %A, %W_res', b'merlin_iface.matmul %A, %W_res {init = 1 : i32}'),
    lambda s: s.replace(b'merlin_iface.matmul %A, %W_res', b'merlin_iface.matmul %A, %W_res {accumulate = true}'),
    lambda s: s.replace(b'merlin_iface.resident_pack %W', b'merlin_iface.resident_pack %A'),
    lambda s: s.replace(b'merlin_iface.matmul %A, %W_res', b'merlin_iface.matmul %A, %A'),
    lambda s: s.replace(b'layout = "packed_rhs"', b'layout = "unreviewed_layout"'),
    lambda s: s.replace(b'name = "Y"', b'name = "A"'),
    lambda s: s.replace(b'tensor<2x2xi32>', b'tensor<3x2xi32>'),
    lambda s: s.replace(b'!merlin_iface.acc<i32>', b'!merlin_iface.acc<i64>'),
    lambda s: s.replace(b'  merlin_iface.evict',
                        b'  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()\n  merlin_iface.evict'),
    lambda s: s.replace(b'merlin_iface.version = "0.1"', b'merlin_iface.version = "other"'),
])
def test_dtypes_readouts_initialization_continuation_extra_effects_and_ownership_refuse(mutation):
    with pytest.raises(Exception):
        M.signatures(mutation(interface()), max_bytes=LIMIT)


def test_v3_public_reader_uses_same_semantic_images_and_old_v2_linalg_behavior_is_preserved(tmp_path):
    cap, source = H.H.capsule(H.original())
    directory = H.writer(tmp_path/'public', cap=cap, source=source)
    private_hashes = M.signatures(interface(), max_bytes=LIMIT)
    assert set(private_hashes) & set(T.signatures(source.encode(), max_bytes=LIMIT))
    path = tmp_path/'selection.json'
    raw = json.dumps(dict(schema=E.SCHEMA_V3, excluded_source_sha256=[], excluded_contraction_sha256=[],
                          excluded_typed_ir_sha256=list(private_hashes))).encode()
    path.write_bytes(raw)
    path.chmod(0o600)
    with pytest.raises(E.PublicationRefusal, match=E.REFUSAL):
        E.select(path, hashlib.sha256(raw).hexdigest()).check_member(directory)
    # Existing V2 projections keep their exact historical meaning.
    raw = json.dumps(dict(schema=E.SCHEMA_V2, excluded_source_sha256=[], excluded_contraction_sha256=[],
                          excluded_typed_ir_sha256=list(T.signatures(source.encode(), max_bytes=LIMIT)))).encode()
    path.write_bytes(raw)
    with pytest.raises(E.PublicationRefusal, match=E.REFUSAL):
        E.select(path, hashlib.sha256(raw).hexdigest()).check_member(directory)
