"""Reject unqualified or unknown capture records before considering payload reuse."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'tools/tiny_full')]
with patch.dict(os.environ, dict(MERLIN_ROOT=str(ROOT), TINY_WORK_ROOT=str(ROOT), ATLAS_NPU_MODEL_ROOT=str(ROOT))):
    spec = importlib.util.spec_from_file_location('qualification_reuse', ROOT/'tools/tiny_full/d1-qualify.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)


class QualificationReuse(unittest.TestCase):
    def test_schema_and_exactness_required(self):
        record = dict(schema='atlas.d1.stream_original_execution.v1', exact=True, full_arena_exact=True,
                      model_commit=q.MODEL_PIN, accelerated=False, native_dispatch_enabled=False)
        for field, value in [('schema', 'foreign'), ('exact', False), ('exact', 1), ('full_arena_exact', None)]:
            bad = dict(record, **{field: value})
            bundle = Mock()
            bundle.pins.read.return_value = json.dumps(bad).encode()
            with self.subTest(field=field, value=value), self.assertRaises(q.NotOriginalCapture):
                q.from_capture(dict(record=str(ROOT/'record.json'), root=str(ROOT)), bundle, {}, None, {}, {})
            bundle.payloads.assert_not_called()


if __name__ == '__main__':
    unittest.main()
