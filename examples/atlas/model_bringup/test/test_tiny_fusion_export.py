"""Immutable export admission, using synthetic transport records, without executing arithmetic."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parents[1]
MODEL_ROOT = os.environ.get('ATLAS_NPU_MODEL_ROOT')
MERLIN_ROOT = os.environ.get('MERLIN_ROOT')
CONFIGURED = unittest.skipUnless(MODEL_ROOT and MERLIN_ROOT, 'set ATLAS_NPU_MODEL_ROOT and MERLIN_ROOT to run fusion export tests')
if MODEL_ROOT and MERLIN_ROOT:
    MODEL_ROOT = Path(MODEL_ROOT).absolute()
    sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss'), str(MODEL_ROOT),
                    str(ROOT / 'tools/tiny_full'), str(Path(MERLIN_ROOT) / 'examples/atlas/comparisons')]
    import numpy as np
    from npu_model.configs import isa_definition
    from npu_model.util.converter import input_to_program
    from tiny_full.fusion import prepare_projection
    from tiny_full.fusion_export import NATIVE_MODEL, ORIGINAL_MODEL, projection_kernel
    from attention_export import Capture as FileCapture, projection_activation
    import kernel_library  # noqa: E402,F401
    from atlas_execution import Executor
    import tiny_fp8_pack


class Capture:
    def __init__(self):
        self.payloads = {}
        self.pins = {}
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def pin(self, path, expected_digest=None):
        self.pins[path] = expected_digest if expected_digest is not None else hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return Path(path)

    def save(self, name, raw, shape=None, dtype='uint8'):
        self.payloads[name] = bytes(raw)
        return dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    dtype=dtype, shape=shape or [len(raw)])

    def raw(self, descriptor):
        return self.payloads[descriptor['file']]

    def array(self, descriptor):
        return np.frombuffer(self.raw(descriptor), dtype=descriptor['dtype']).reshape(descriptor['shape'])


class Assets:
    def put(self, category, raw, suffix='.bin'):
        return category + '/' + hashlib.sha256(raw).hexdigest() + suffix


def fixture(role=5):
    m, k, n, column, width = 128, 2048, 1536, 4096, 5632
    if role != 5:
        m, n, column, width = 32, 2048, 0, 2048
    capture = Capture()
    a, b, operand = bytes(m*k), bytes(k*n), bytes(2*m*n)
    prepared = prepare_projection(m, k, n, -12, role, column, width, a, b, operand, -5 if role == 5 else 0)
    source = prepared['source'].encode()
    words = list(input_to_program(io.StringIO(source.decode())).assemble())
    program = struct.pack('<' + 'I'*len(words), *words)
    final = bytearray(prepared['initial'])
    offset = prepared['output_offset']
    for span in prepared['written_spans']:
        final[offset + span['offset']:offset + span['offset'] + span['bytes']] = bytes(span['bytes'])
    output = bytes(final[offset:offset + prepared['output_bytes']])
    requested = dict(load_bytes=prepared['dma_load_bytes'], store_bytes=prepared['dma_store_bytes'])
    binding = Executor(MODEL_ROOT).binding()
    sources = {str(MODEL_ROOT/name): sha for name, sha in binding['source_sha256'].items()}
    record = dict(program_sha256=hashlib.sha256(program).hexdigest(), model_commit='47b8599d8ffbe5ddd96059ffe2b5be2427a410f5', numerics='rtl', accelerated=False,
        dram_base=0x90000000, output_scale=prepared['output_scale'], output_offset=offset,
        output_bytes=len(output), epilogue=json.loads(json.dumps(prepared['epilogue'])),
        written_spans=prepared['written_spans'], requested_dma=requested,
        model_sources_before=sources, model_sources_after=dict(sources),
        harness_sources_before={'fixture': 'synthetic'}, harness_sources_after={'fixture': 'synthetic'},
        execution=dict(program_sha256=hashlib.sha256(program).hexdigest(), observed=requested))
    for key, raw in dict(a_fp8=a, b_fp8=b, operand_tiles=operand, program_source=source,
                         program_binary=program, arena_initial=prepared['initial'], arena_expected=bytes(final),
                         output_tiled=output).items():
        record[key] = capture.save(key, raw)
    record['output_bf16'] = capture.save('logical', bytes(2*m*n), [m, n], 'uint16')
    if role == 5:
        record['output_fp8'] = capture.save('fp8', bytes(m*n))
    panel = dict(launch=record, physical_mkn=[m,k,n], logical_mkn=[m,k,n], columns=[column,column+n],
                 A_scale=-6, B_scale=-6, rescale=-12, expected_bf16=record['output_bf16'])
    arguments = dict(step=0, layer=0, role=role, site='synthetic', site_index=role, rows=m, width=width, index=0)
    return capture, panel, arguments


@CONFIGURED
class FusionExport(unittest.TestCase):
    def test_extent_dma_and_written_holes_remain_distinct(self):
        capture, panel, arguments = fixture()
        item = projection_kernel(capture, panel, Assets(), input_to_program, **arguments)
        self.assertGreater(item['output_extent_bytes'], item['requested_dma']['store_bytes'])
        self.assertEqual(item['output_column_tile'], 128)
        self.assertEqual(item['operand_offset'], item['output_offset'] + item['output_extent_bytes'] + 256)
        self.assertEqual(item['fusion_role'], 5)

    def test_bf16_residual_extent(self):
        capture, panel, arguments = fixture(3)
        item = projection_kernel(capture, panel, Assets(), input_to_program, **arguments)
        self.assertEqual(item['output_extent_bytes'], item['requested_dma']['store_bytes'])
        self.assertEqual(item['fusion_role'], 3)

    def test_changed_hole_and_graph_reply_are_rejected(self):
        capture, panel, arguments = fixture()
        record = panel['launch']
        output = bytearray(capture.raw(record['output_tiled']))
        output[0] ^= 1
        final = bytearray(capture.raw(record['arena_expected']))
        final[record['output_offset']] ^= 1
        capture.payloads['output_tiled'], capture.payloads['arena_expected'] = bytes(output), bytes(final)
        with self.assertRaisesRegex(ValueError, 'unwritten output hole'):
            projection_kernel(capture, panel, Assets(), input_to_program, **arguments)
        capture, panel, arguments = fixture(3)
        capture.payloads['logical'] = b'\x80\x3f' + capture.payloads['logical'][2:]
        with self.assertRaisesRegex(ValueError, 'logical output'):
            projection_kernel(capture, panel, Assets(), input_to_program, **arguments)

    def test_execution_dma_and_model_bookends_are_rejected(self):
        capture, panel, arguments = fixture(3)
        for field in ('dma', 'bookend', 'accelerated'):
            bad = copy.deepcopy(panel)
            if field == 'dma':
                bad['launch']['execution']['observed']['store_bytes'] += 32
            elif field == 'bookend':
                bad['launch']['model_sources_after']['fixture'] = 'changed'
            else:
                bad['launch']['accelerated'] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                projection_kernel(capture, bad, Assets(), input_to_program, **arguments)

    def test_historical_helper_archive_coexists_with_current_import_pin(self):
        capture, panel, arguments = fixture(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live, old = root/'helper.py', root/'old-helper.py'
            live.write_bytes(b'current implementation\n')
            old.write_bytes(b'captured implementation\n')
            old_sha = hashlib.sha256(old.read_bytes()).hexdigest()
            current_sha = hashlib.sha256(live.read_bytes()).hexdigest()
            archive = root/'archive.json'
            archive.write_text(json.dumps(dict(schema='atlas.capture_source_archive.v1', sources=[
                dict(original_path=str(live), file=old.name, sha256=old_sha)])))
            report = root/'capture.json'
            report.write_text(json.dumps(dict(source_records=[dict(path=str(live), sha256=current_sha)])))
            real = FileCapture(report, hashlib.sha256(report.read_bytes()).hexdigest(), archive)
            for name, raw in capture.payloads.items():
                (root/name).write_bytes(raw)
            for field in ('harness_sources_before', 'harness_sources_after'):
                panel['launch'][field] = {str(live): old_sha}
            projection_kernel(real, panel, Assets(), input_to_program, **arguments)
            self.assertEqual(real.pins[str(live)], current_sha)
            self.assertEqual(real.pins[str(old)], old_sha)

    def test_model_relocation_authenticates_returned_paths_against_original_head(self):
        capture, panel, arguments = fixture(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, historical = MODEL_ROOT, root/'old-model'
            record = panel['launch']
            relocated = {str(historical/Path(name).relative_to(selected)): sha
                         for name, sha in record['model_sources_before'].items()}
            record['model_sources_before'], record['model_sources_after'] = relocated, dict(relocated)
            helper = root/'helper.py'
            helper.write_bytes(b'captured helper\n')
            for field in ('harness_sources_before', 'harness_sources_after'):
                record[field] = {str(helper): hashlib.sha256(helper.read_bytes()).hexdigest()}
            report = root/'capture.json'
            report.write_text(json.dumps(dict(source_records=[])))
            real = FileCapture(report, hashlib.sha256(report.read_bytes()).hexdigest(),
                               relocations=[str(historical)+'='+str(selected)])
            for name, raw in capture.payloads.items():
                (root/name).write_bytes(raw)
            projection_kernel(real, panel, Assets(), input_to_program, **arguments)
            self.assertTrue(all(str(selected/Path(name).relative_to(historical)) in real.pins
                                for name in relocated))


@CONFIGURED
class NativeExport(unittest.TestCase):
    """Exercise only the exporter/proof protocol; arithmetic and proof checks have separate gates."""
    @staticmethod
    def native_fixture():
        capture, panel, arguments = fixture(3)
        record = panel['launch']
        record.update(model_commit=NATIVE_MODEL, base_model_commit=ORIGINAL_MODEL, accelerated=True,
            native_dispatch_enabled=True, requires_original_backfill=True, qualification_candidate=False,
            native_kind_sha256='synthetic-kind', native_admission=dict(path='fixture', sha256='synthetic'),
            program_sha256=record['execution']['program_sha256'])
        return capture, panel, arguments

    class Checker:
        def verify(self, root, record):
            return dict(status='PASS_NATIVE_RECORD_ADMITTED_REQUIRES_ORIGINAL47_BACKFILL',
                execution_model=NATIVE_MODEL, reference_model=ORIGINAL_MODEL, requires_original_backfill=True,
                original_final_graph_numerics_qualified=False, native_library_loaded_by_checker=False,
                model_instruction_execution_by_checker=False, native_kind_sha256=record['native_kind_sha256'],
                native_admission=record['native_admission'], program_sha256=record['program_sha256'],
                record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                source_records=[dict(path='fixture', sha256='synthetic')])

    def test_native_requires_explicit_checker_and_original_backfill_metadata(self):
        capture, panel, arguments = self.native_fixture()
        with self.assertRaisesRegex(ValueError, 'explicit admitted provenance'):
            projection_kernel(capture, panel, Assets(), input_to_program, **arguments)
        item = projection_kernel(capture, panel, Assets(), input_to_program, native_checker=self.Checker(), **arguments)
        self.assertEqual(item['model_commit'], NATIVE_MODEL)
        self.assertEqual(item['base_model_commit'], ORIGINAL_MODEL)
        self.assertTrue(item['requires_original_backfill'])
        self.assertFalse(item['original_final_graph_numerics_qualified'])

    def test_misbound_proof_and_retagged_native_record_are_rejected(self):
        capture, panel, arguments = self.native_fixture()
        proof = self.Checker().verify(capture.root, panel['launch'])
        class WrongProof:
            def verify(self, root, record):
                return proof
        panel['launch']['output_scale'] += 1
        with self.assertRaisesRegex(ValueError, 'proof differs'):
            projection_kernel(capture, panel, Assets(), input_to_program, native_checker=WrongProof(), **arguments)
        panel['launch'].update(model_commit=ORIGINAL_MODEL, accelerated=False)
        with self.assertRaisesRegex(ValueError, 'original model or an explicit native proof'):
            projection_kernel(capture, panel, Assets(), input_to_program, native_checker=self.Checker(), **arguments)

    def test_fully_cleared_flags_cannot_relabel_native_provenance_or_sources(self):
        capture, panel, arguments = self.native_fixture()
        record = panel['launch']
        record.update(model_commit=ORIGINAL_MODEL, accelerated=False, native_dispatch_enabled=False,
                      requires_original_backfill=False)
        with self.assertRaisesRegex(ValueError, 'cannot be relabeled'):
            projection_kernel(capture, panel, Assets(), input_to_program, **arguments)
        for key in ('native_kind_sha256', 'native_admission', 'base_model_commit', 'qualification_candidate'):
            record.pop(key)
        record['model_sources_before'] = {str(Path(__file__).absolute()): 'foreign model source'}
        record['model_sources_after'] = dict(record['model_sources_before'])
        with self.assertRaisesRegex(ValueError, 'outside selected original clone'):
            projection_kernel(capture, panel, Assets(), input_to_program, **arguments)


@CONFIGURED
class ProjectionActivation(unittest.TestCase):
    class Encoder:
        @staticmethod
        def encode(value):
            return tiny_fp8_pack.encode(value)

    def fixture(self):
        capture = Capture()
        logical = np.full((1, 2048), 0x3f80, np.uint16)
        physical = np.zeros((32, 2048), np.uint16)
        physical[:1] = logical
        physical[1, 0] = 0x4280  # 64.0 controls the physical max-abs scan.
        descriptor = capture.save('physical', physical.tobytes(), list(physical.shape), 'uint16')
        return capture, dict(A_physical_bf16=descriptor), logical, physical

    def test_physical_scale_scan_then_zero_inactive_fp8_rows(self):
        capture, call, logical, physical = self.fixture()
        actual, encoded, scale = projection_activation(capture, call, logical, 3, self.Encoder(),
                                                        require_physical=True)
        self.assertTrue(np.array_equal(actual, physical))
        self.assertEqual(scale, -2)
        self.assertTrue(np.all(encoded[0] == 0x48))  # 1.0 / 2**-2 = FP8 4.0.
        self.assertFalse(encoded[1:].any())
        _, legacy, legacy_scale = projection_activation(capture, {}, logical, 3, self.Encoder())
        self.assertEqual(legacy_scale, -8)
        self.assertTrue(np.all(legacy[0] == 0x78))

    def test_nan_padding_and_missing_or_wrong_physical_extent_are_rejected(self):
        capture, call, logical, physical = self.fixture()
        physical[1, 1] = 0x7fc1
        capture.payloads['physical'] = physical.tobytes()
        with self.assertRaises(ValueError):
            projection_activation(capture, call, logical, 3, self.Encoder(), require_physical=True)
        with self.assertRaisesRegex(ValueError, 'complete physical'):
            projection_activation(capture, {}, logical, 3, self.Encoder(), require_physical=True)
        physical[0, 0] = 0x4000
        capture.payloads['physical'] = physical.tobytes()
        with self.assertRaisesRegex(ValueError, 'logical projection'):
            projection_activation(capture, call, logical, 3, self.Encoder(), require_physical=True)


if __name__ == '__main__':
    unittest.main()
