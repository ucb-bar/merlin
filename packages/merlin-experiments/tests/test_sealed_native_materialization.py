"""Synthetic byte plumbing never supplies a real sealed source execution owner."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import test_public_integer_capture as F
from merlin_experiments.phase0 import sealed_native_materialization as M

from merlin.targetgen import capsule_source as S
from merlin.targetgen import native_reference_composition as C
from merlin.targetgen.application_inventory import verify_capture_receipt


def setup(tmp_path, monkeypatch):
    root = tmp_path/'original'
    root.mkdir()
    (root/'stages').mkdir()
    selection = F.capture(root/'stages/member_0')
    stage = root/'stages/member_0'
    receipt = json.loads((stage/'capture_receipt.json').read_bytes())
    receipt['artifacts'].pop('golden.json')
    raw = json.dumps(receipt).encode()
    (stage/'capture_receipt.json').write_bytes(raw)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}
    attestation = dict(selection=dict(sha256='a'*64),
        capture=dict(kind='session', capture_path=str(root), capture_tree_sha256='b'*64,
                     programs=[dict(name='member_0')]))
    calls = []
    monkeypatch.setattr(M.N, 'read', lambda *args, **kwargs: calls.append((args, kwargs)))
    destination = tmp_path/'composed'
    return stage, selection, attestation, destination, hashes, calls


def materialize(attestation, destination, **kwargs):
    return M.materialize(attestation, program='member_0', source_member='member_0.py',
                         destination=destination, max_payload_bytes=1<<20, **kwargs)


def test_complete_reference_added_under_new_schema_without_changing_originals(tmp_path, monkeypatch):
    stage, selection, attestation, dest, hashes, calls = setup(tmp_path, monkeypatch)
    result = materialize(attestation, dest)
    assert len(calls) == 1
    assert result['added_reference_members'] == ['golden.json']
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()} == hashes
    receipt = json.loads((dest/'capture_receipt.json').read_bytes())
    original = C.validate(receipt)
    assert original == json.loads((stage/'capture_receipt.json').read_bytes())
    assert (dest/'model.mlir').read_bytes() == (stage/'model.mlir').read_bytes()
    checked = verify_capture_receipt(dest/'model.mlir')
    assert checked['status'] == 'verified_materialized' and checked['source_closure_verified'] is False
    selection.update(path=str(dest/'model.mlir'), receipt_sha256=result['receipt_sha256'],
                     workload_role='iteration', loader_path=None, loader_sha256=None)
    artifact = S.materialized_model_artifacts(selection)
    assert artifact.golden[1] == [-16777217, 16777217]


@pytest.mark.parametrize('mutation', [
    lambda d: d['reference_composition']['original_receipt'].update(sha256='c'*64),
    lambda d: d['reference_composition']['original_receipt'].update(json_utf8='{}'),
    lambda d: d['artifacts'].pop('meta.json'),
    lambda d: d['artifacts']['inputs.npz'].update(sha256='c'*64),
    lambda d: d['reference_composition']['added_reference_members'].append('inputs.npz'),
    lambda d: d['reference_composition'].update(saved_verdict='verified'),
    lambda d: d['source'].update(path='different source'),
])
def test_base_membership_metadata_and_complete_original_receipt_are_fixed(tmp_path, monkeypatch, mutation):
    _, _, attestation, dest, _, _ = setup(tmp_path, monkeypatch)
    materialize(attestation, dest)
    document = json.loads((dest/'capture_receipt.json').read_bytes())
    mutation(document)
    with pytest.raises((KeyError, TypeError, ValueError)):
        C.validate(document)
    (dest/'capture_receipt.json').write_text(json.dumps(document))
    assert verify_capture_receipt(dest/'model.mlir')['status'] == 'unverified'


def test_original_seal_reference_and_source_binding_recheck_are_separate(tmp_path, monkeypatch):
    stage, _, attestation, dest, _, _ = setup(tmp_path, monkeypatch)
    materialize(attestation, dest)
    public = tmp_path/'public'
    public.mkdir()
    (public/'source-capture-receipt.json').write_bytes((dest/'capture_receipt.json').read_bytes())
    (public/'source-capture-golden.json').write_bytes((stage/'golden.json').read_bytes())
    M.verify_against_original(public, attestation, program='member_0', root=stage)
    for change in ['source', 'reference']:
        changed = copy.deepcopy(attestation)
        if change == 'source':
            changed['selection']['sha256'] = 'd'*64
        else:
            (public/'source-capture-golden.json').write_text('[[[7]], [1,2], [[-2,3]]]')
        with pytest.raises(ValueError, match='original'):
            M.verify_against_original(public, changed, program='member_0', root=stage)


def test_public_payload_refusal_precedes_original_array_loading_or_destination(tmp_path, monkeypatch):
    _, _, attestation, dest, _, calls = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='bound'):
        M.materialize(attestation, program='member_0', source_member='member_0.py',
                      destination=dest, max_payload_bytes=8)
    assert len(calls) == 1 and not dest.exists()  # Mocked original reader; real reader bounds before arrays.
    monkeypatch.setattr(M.N, 'read', lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError('unverified seal')))
    with pytest.raises(ValueError, match='unverified seal'):
        materialize(attestation, dest)
    assert not dest.exists()


def test_original_metadata_owns_same_process_catalog_without_fabricating_producer_artifact(tmp_path, monkeypatch):
    stage, _, attestation, dest, _, _ = setup(tmp_path, monkeypatch)
    raw = b'{"source": "synthetic same-process catalog"}'
    (stage/'pytorch-opset.json').write_bytes(raw)
    pointer = dict(path='pytorch-opset.json', sha256=hashlib.sha256(raw).hexdigest())
    metadata = json.loads((stage/'meta.json').read_bytes())
    metadata['framework_catalog'] = pointer
    (stage/'meta.json').write_text(json.dumps(metadata))
    receipt = json.loads((stage/'capture_receipt.json').read_bytes())
    payload = (stage/'meta.json').read_bytes()
    receipt['artifacts']['meta.json'] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (stage/'capture_receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(M.N, 'read', lambda *_a, **_k: SimpleNamespace(meta=dict(framework_catalog=pointer)))
    result = materialize(attestation, dest)
    assert result['metadata_owned_members'] == ['pytorch-opset.json']
    assert (dest/'pytorch-opset.json').read_bytes() == raw
    composed = json.loads((dest/'capture_receipt.json').read_bytes())
    assert 'pytorch-opset.json' not in composed['artifacts']  # Preserve the original indirect metadata ownership.
    assert C.validate(composed)['artifacts']['meta.json'] == receipt['artifacts']['meta.json']
    (stage/'pytorch-opset.json').write_text('changed')
    with pytest.raises(ValueError, match='catalog changed'):
        materialize(attestation, tmp_path/'refused')
