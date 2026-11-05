"""Prepared model consumers must not combine current metadata with original payloads."""
import json
import struct

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.llvmlower.weights_pack import weight_files_beside
from merlin.xdsl_dialects.lowering.output_fusion import Values


def payload(path, data):
    array = np.asarray(data, dtype=np.int8)
    raw = array.tobytes()
    header = json.dumps({'w': {'dtype': 'I8', 'shape': list(array.shape), 'data_offsets': [0, len(raw)]}}).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + raw)


def test_actual_source_values_use_prepared_pair_despite_earlier_stale_alias(tmp_path):
    model = tmp_path / 'model.mlir'
    model.write_text('''builtin.module {
      func.func @forward(%w: tensor<2x2xi8>) -> tensor<2x2xi8> {
        func.return %w : tensor<2x2xi8>
      }
    }''')
    payload(tmp_path / 'capsule.weights.safetensors', [[-9, -9, -9, -9]])
    payload(tmp_path / 'weights.safetensors', [[1, 2], [3, 4]])
    (tmp_path / 'weights.safetensors.manifest.json').write_text(json.dumps({
        '0': {'kind': 'param', 'weight': 'w'}}))
    module = mq.parse(model.read_text())
    function = next(mq.walk(module, 'func.func'))
    assert Values(model).array(function.body.block.args[0]).tolist() == [[1, 2], [3, 4]]


def test_noncanonical_capture_pairs_match_basename_and_reject_ambiguity(tmp_path):
    (tmp_path / 'b.safetensors.manifest.json').write_text('{}')
    (tmp_path / 'b.safetensors').write_bytes(b'current')
    (tmp_path / 'a.safetensors').write_bytes(b'stale')
    assert weight_files_beside(tmp_path)[1].name == 'b.safetensors'
    (tmp_path / 'a.safetensors.manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='ambiguous'):
        weight_files_beside(tmp_path)


def test_partial_canonical_bundle_cannot_fall_back_to_old_source(tmp_path):
    (tmp_path / 'model.mlir').write_text('builtin.module {}')
    (tmp_path / 'weights.safetensors.manifest.json').write_text('{}')
    (tmp_path / 'capsule.weights.safetensors').write_bytes(b'old')
    with pytest.raises(ValueError, match='incomplete'):
        weight_files_beside(tmp_path)


def test_legacy_unique_pair_still_resolves(tmp_path):
    (tmp_path / 'source.manifest.json').write_text('{}')
    (tmp_path / 'frozen.safetensors').write_bytes(b'one')
    assert weight_files_beside(tmp_path)[1].name == 'frozen.safetensors'
