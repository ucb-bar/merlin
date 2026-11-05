"""Frozen full-arena qualifier with explicit tracked attention-v2 provenance admission."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
ATLAS_ROOT = HERE.parents[1]
MERLIN_ROOT = Path(os.environ['MERLIN_ROOT']).absolute()
ROOT = Path(os.environ['TINY_WORK_ROOT']).absolute()
ATLAS_HELPERS = ATLAS_ROOT/'tools/tiny_full'
MERLIN_HELPERS = MERLIN_ROOT/'examples/atlas/comparisons'
MODEL_ROOT = Path(os.environ['ATLAS_NPU_MODEL_ROOT']).absolute()
spec = importlib.util.spec_from_file_location('d1_v2_support', ATLAS_HELPERS / 'd1-v2-support.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
q = s.load('d1-qualify')
import kernel_library  # noqa: E402  (d1-qualify put tools/ on sys.path)
legacy_from_capture = q.from_capture
legacy_harness = q.harness
legacy_current_report = q.current_report
legacy_bundle = q.Bundle


class Bundle(legacy_bundle):
    def __init__(self, assets, smoke=False):
        super().__init__(assets,smoke)
        if any(item.get('attention_version') != 2 or item.get('kernel_schema') != 'atlas.tiny.attention_launch.v2'
               for item in self.launches if item.get('kind') == 'attention'):
            raise ValueError('v2 qualification requires typed v2 attention launches')


def harness():
    return dict(sorted({**legacy_harness(), **s.adapter_sources()}.items()))


def attention_model_inventory(record, model_sources):
    """The unit supplies inventory names only; current original HEAD authenticates every byte."""

    before = record.get('model_sources_before')
    if (not isinstance(before, dict) or len(before) < 45
            or before != record.get('model_sources_after')
            or any(model_sources.get(path) != digest for path,digest in before.items())):
        raise ValueError('attention-v2 original45 HEAD-authenticated model inventory differs')
    return before


def from_capture(mapping, bundle, item, model, model_sources, harness_sources):
    raw = bundle.pins.read(s.safe(mapping['record']))
    record = json.loads(raw)
    if record.get('schema') != 'atlas.tiny.attention_launch.v2':
        return legacy_from_capture(mapping, bundle, item, model, model_sources, harness_sources)
    if (record.get('kind') != 'attention' or record.get('model_commit') != q.MODEL_PIN
            or record.get('model_root') != str(MODEL_ROOT) or record.get('numerics') != 'rtl'
            or record.get('accelerated') is not False or record.get('native_dispatch_enabled') is not False
            or record.get('status') != 'PASS' or record.get('runtime_patched') is not False
            or record.get('mode') != 'functional' or record.get('source_bookends_verified') is not True
            or record.get('source_and_binary_exact') is not True or record.get('halt_exact') is not True
            or record.get('outside_written_spans_preserved') is not True
            or any(key in record for key in ('native_backend', 'native_kind', 'native_kind_sha256', 'native_admission',
                'base_model_commit', 'qualification_candidate', 'requires_original_backfill'))):
        raise ValueError('attention-v2 is not an actual tracked original47 functional execution')
    attention_model_inventory(record, model_sources)
    closure = record.get('harness_sources_before')
    required = [ATLAS_ROOT / kernel_library.relative_path(name) for name in
                ('tiny_full/attention_v2_worker.py', 'atlas_execution.py', 'atlas_path_guard.py',
                 'atlas_attention_v2_program.py', 'atlas_program_checks.py')]
    if (not isinstance(closure, dict) or not closure or closure != record.get('harness_sources_after')
            or set(closure) != {str(path) for path in required}):
        raise ValueError('tracked attention-v2 helper source closure differs')
    bundle.pins.bind(closure)
    expected_implementations = [dict(path=path, sha256=digest, bytes=s.safe(path).stat().st_size)
                                for path,digest in sorted(closure.items())]
    if record.get('implementation_sources') != expected_implementations:
        raise ValueError('attention-v2 implementation source inventory differs')
    # Exporter admission independently regenerated config, exact staging, assembly,
    # logical FP8 output, and sparse-store policy. Qualifier rejoins all five bytestrings.
    if (item.get('attention_version') != 2 or item.get('kernel_schema') != record['schema']
            or any(item.get(key) != record.get(key) for key in ('config', 'offsets', 'step', 'layer', 'site',
                'output_layout', 'output_dtype', 'output_scale', 'output_extent_bytes', 'written_spans'))):
        raise ValueError('attention-v2 typed final contract differs from actual execution')
    payloads = bundle.payloads(item)
    captured = {}
    root = s.safe(mapping['root'])
    for key,expected in zip(('program_source', 'program_binary', 'arena_initial', 'output_tiled', 'arena_expected'), payloads):
        desc = record[key]
        path = Path(desc['file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('capture member escapes capture root')
        path = s.safe(root/path)
        actual = bundle.pins.read(path)
        captured[str(path)] = s.sha(actual)
        if actual != expected or desc['bytes'] != len(actual) or desc['sha256'] != s.sha(actual):
            raise ValueError('attention-v2 exact source, program, output, or full arena differs')
    source,binary,initial,expected,final = payloads
    program,digest = model.assemble(source.decode())
    if digest != s.sha(binary) or b''.join(word.to_bytes(4,'little') for word in program.assemble()) != binary:
        raise ValueError('original47 assembly differs from exact capture binary')
    if (record.get('program_sha256') != s.sha(binary) or record.get('arena_bytes') != len(initial)
            or any(record.get(key) != item[key] for key in ('dram_base', 'output_offset', 'requested_dma'))
            or record.get('output_bytes') != len(expected)
            or not q.execution_matches(record.get('execution', {}), item, binary)
            or record['execution'].get('observed', {}).get('native_mxu0_calls', 0) != 0):
        raise ValueError('attention-v2 actual execution program, halt, DMA, or extent differs')
    expected_execution_helpers = {str(path): closure[str(path)] for path in required
                                  if path.name in ('atlas_execution.py', 'atlas_path_guard.py')}
    if record['execution'].get('implementation_source_sha256') != expected_execution_helpers:
        raise ValueError('actual Executor implementation source binding differs')
    execution_model = record['execution'].get('model', {})
    if (execution_model.get('revision') != q.MODEL_PIN or execution_model.get('root') != str(MODEL_ROOT)
            or execution_model.get('numerics') != 'rtl'
            or execution_model.get('source_sha256') != {str(Path(path).relative_to(MODEL_ROOT)): value
                                                        for path,value in record['model_sources_before'].items()}):
        raise ValueError('actual Executor metadata lacks original45 source binding')
    return dict(status='PASS', exact=True, full_arena_exact=True, accelerated=False, native_dispatch_enabled=False,
        model_pin=q.MODEL_PIN, model_root=str(MODEL_ROOT), numerics='rtl', **q.identity(item,payloads),
        execution=record['execution'], model_sources_before=model_sources, model_sources_after=model_sources,
        harness_sources_before=harness_sources, harness_sources_after=harness_sources,
        reused_capture_harness_sources=closure, reused_capture_record=str(s.safe(mapping['record'])),
        reused_capture_payload_sha256=captured, reused_capture_record_sha256=s.sha(raw),
        method='reused_original_capture_execution', new_timing=False,
        reused_capture_model_sources=record['model_sources_before'], reused_capture_root=str(root), original_attention_v2=True)


def current_report(path, bundle, item, model_sources, harness_sources):
    report = legacy_current_report(path,bundle,item,model_sources,harness_sources)
    if (report.get('method') == 'reused_original_capture_execution' or report.get('original_attention_v2')) and not report.get('reused_capture_record'):
        raise ValueError('reused execution receipt lacks actual capture pointer')
    if report.get('reused_capture_record'):
        record = json.loads(bundle.pins.read(report['reused_capture_record']))
        if record.get('schema') == 'atlas.tiny.attention_launch.v2':
            mapping = dict(record=report['reused_capture_record'], root=report.get('reused_capture_root', ''))
            if not mapping['root']:
                raise ValueError('attention-v2 reuse root missing')
            expected = from_capture(mapping,bundle,item,q.load_model(),model_sources,harness_sources)
            if any(report.get(key) != value for key,value in expected.items()):
                raise ValueError('attention-v2 resume execution or provenance differs')
    return report


# Frozen qualifier subprocesses re-enter this adapter. Source pins include both implementations.
q.Bundle = Bundle
q.from_capture = from_capture
q.harness = harness
q.current_report = current_report
q.__file__ = __file__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--index', type=int)
    parser.add_argument('--coverage-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse-records')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not s.safe(args.out).is_relative_to(ROOT/'d1-v2-qualification'):
        raise ValueError('qualification output must be below D/d1-v2-qualification')
    return q.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
