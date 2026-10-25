"""A metadata correction cannot rebind a different program or numerical policy."""
import copy
import hashlib
import json

import pytest
import yaml
from merlin_experiments.phase1 import joint_conv_metadata_revision as C
from merlin_experiments.phase1 import joint_source_revision as R
from merlin_experiments.phase2.contracts import document_sha256

SOURCE = (
    'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "test", '
    'merlin_iface.abi_version = "0.1"} {\n'
    '  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<1x5x5x4xi8>\n'
    '  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<36x8xi8>\n'
    '  %R = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} '
    ': (tensor<36x8xi8>) -> !merlin_iface.resident\n'
    '  %Y = merlin_iface.conv2d %X, %R {name = "Y", kernel = [3, 3, 4, 8], stride = [1, 1], '
    'padding = [0, 0, 0, 0], dilation = [1, 1], epilogue = [], output_dtype = "i8", layout = "nhwc"} '
    ': (tensor<1x5x5x4xi8>, !merlin_iface.resident) -> tensor<9x8xi8>\n'
    '  merlin_iface.evict %R : (!merlin_iface.resident) -> ()\n'
    '}\n'
)


def fixture(tmp_path):
    old = dict(name='conv', kind='layer', interface_mlir='source.mlir',
               operation={'op': 'conv2d', 'attributes': {'ci': 64}},
               numeric_policy={'compare': 'exact_int'}, required_oracle_tiers=['L0', 'L1', 'L2'])
    new = copy.deepcopy(old)
    new['operation']['attributes']['ci'] = 4
    evidence = dict(schema='merlin.typed_convolution_channel_metadata_review.v1', name='conv')
    for role, document in [('before', old), ('after', new)]:
        directory = tmp_path/role
        directory.mkdir()
        path = directory/'capsule.yaml'
        path.write_text(yaml.safe_dump(document))
        (directory/'source.mlir').write_text(SOURCE)
        evidence[role] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    members = {'conv': dict(capsule_sha256=document_sha256(old), contractions=[[9, 36, 8]], logical_read_bytes=388)}
    return evidence, members, old, new


def rewrite(evidence, role, document=None, source=None):
    from pathlib import Path
    path = Path(evidence[role]['path'])
    if document is not None:
        path.write_text(yaml.safe_dump(document))
        evidence[role]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if source is not None:
        (path.parent/'source.mlir').write_text(source)


def test_only_typed_duplicate_metadata_is_rebound(tmp_path):
    evidence, members, old, new = fixture(tmp_path)
    assert C.verify(evidence, members=members) == ('conv', document_sha256(new))
    pin_path = tmp_path/'review.json'
    pin_path.write_text(json.dumps(evidence))
    initial = dict(members=copy.deepcopy(members), runtime='same')
    current = copy.deepcopy(initial)
    current['members']['conv']['capsule_sha256'] = document_sha256(new)
    current['source_binding_revisions'] = [dict(schema='merlin.typed_convolution_channel_metadata_revision.v1',
        review_id='typed-channel-only', evidence=dict(path=str(pin_path),
            sha256=hashlib.sha256(pin_path.read_bytes()).hexdigest()))]
    assert R.normalized(initial, current) == initial
    current['members']['conv']['logical_read_bytes'] += 1
    with pytest.raises(ValueError, match='unreviewed member'):
        R.normalized(initial, current)


@pytest.mark.parametrize('mutation', ['wrong_channel', 'numerics', 'tiers', 'source', 'geometry', 'extra_review_field'])
def test_changed_computation_or_contract_is_refused(tmp_path, mutation):
    evidence, members, old, new = fixture(tmp_path)
    if mutation == 'wrong_channel':
        new['operation']['attributes']['ci'] = 8
    elif mutation == 'numerics':
        new['numeric_policy']['compare'] = 'tolerance_float'
    elif mutation == 'tiers':
        new['required_oracle_tiers'] = ['L2']
    elif mutation == 'source':
        rewrite(evidence, 'after', source=SOURCE+'\n')
    elif mutation == 'geometry':
        members['conv']['contractions'] = [[9, 12, 8]]
    else:
        evidence['qualification_credit'] = True
    rewrite(evidence, 'after', new)
    with pytest.raises(ValueError):
        C.verify(evidence, members=members)
