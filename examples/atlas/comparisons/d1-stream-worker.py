"""Execute one immutable capture projection on original47 without copying arena payloads."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
MERLIN_ROOT = HERE.parents[2]
ATLAS_ROOT = Path(os.environ['ATLAS_ROOT']).absolute()
ROOT = Path(os.environ['TINY_WORK_ROOT']).absolute()
ATLAS_HELPERS = ATLAS_ROOT/'tools/tiny_full'
MERLIN_HELPERS = MERLIN_ROOT/'examples/atlas/comparisons'
MODEL_ROOT = Path(os.environ['ATLAS_NPU_MODEL_ROOT']).absolute()
PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'
NATIVE = '70e2e7ceb81dac86210c8b3cca54024909c57d14'
WORKER = None
RECORD_KEYS = ('kind', 'm', 'k', 'n', 'logical_rows', 'role', 'column', 'width', 'rescale', 'a_scale', 'b_scale',
    'output_scale', 'dram_base', 'output_offset', 'output_bytes', 'arena_bytes', 'requested_dma', 'dma_store_bytes',
    'epilogue', 'extras', 'written_spans', 'output_dtype', 'output_row_stride', 'output_column_tile', 'a_fp8', 'b_fp8',
    'operand_tiles', 'program_sha256', 'step', 'site', 'panel', 'layer', 'name', 'output_bf16', 'output_fp8',
    'preclip_bf16_tiles', 'program_source', 'program_binary', 'arena_initial', 'arena_expected', 'output_tiled')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(value):
    path = Path(value).absolute()
    if ('..' in path.parts or '\\' in str(path) or any(ord(c) < 32 or ord(c) == 127 for c in str(path))
            or any(word in part.lower() for part in path.parts for word in ('hammer', 'vlsi'))
            or not any(path.is_relative_to(root) for root in (ROOT, ATLAS_ROOT, MERLIN_ROOT, MODEL_ROOT))):
        raise ValueError('stream path must be physical and beneath D')
    for prefix in (*reversed(path.parents), path):
        try:
            mode = prefix.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError('stream path contains a symbolic link')
    return path


def save(path, value):
    path = safe(path)
    temporary = safe(path.with_name(path.name + '.tmp-' + str(os.getpid())))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def bind(values):
    if not values:
        raise ValueError('empty source closure')
    for filename, expected in values.items():
        if sha(safe(filename).read_bytes()) != expected:
            raise ValueError('source or payload identity changed: ' + filename)


def load_original():
    global WORKER
    if WORKER is not None:
        return WORKER
    os.environ.update(ATLAS_NPU_MODEL_ROOT=str(MODEL_ROOT), ATLAS_NPU_MODEL_PIN=PIN,
                      ATLAS_NPU_MODEL_NATIVE='0', NPU_MODEL_RTL_NATIVE_MXU0='0')
    os.environ.pop('NPU_MODEL_RTL_NATIVE_BUILD_DIR', None)
    spec = importlib.util.spec_from_file_location('d1_stream_original_worker', safe(MERLIN_HELPERS / 'd1-worker.py'))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.model.safe_path = safe
    # Both shape families must be present before freezing. The generator imports
    # them lazily, so a decode-first freeze would otherwise omit prefill sources.
    for name in ('atlas_path_guard', 'atlas_program_reuse_tiny', 'atlas_program_reuse_tiny_breadth'):
        __import__(name)
    WORKER = worker
    return worker


def closure(worker):
    values = worker.harness_sources()
    for name in ('d1-stream-worker.py', 'd1-stream-original.py'):
        path = safe(MERLIN_HELPERS/name)
        values[str(path)] = sha(path.read_bytes())
    return dict(sorted(values.items()))


def source_state():
    worker = load_original()
    return dict(harness=closure(worker), model=worker.model.source_binding())


def verify_freeze(freeze):
    bind(freeze['harness'])
    bind(freeze['model'])
    current = source_state()
    if current != freeze:
        raise ValueError('original model or imported source closure differs from stream freeze')
    return WORKER


def stamp(path):
    value = safe(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


class Inputs:
    def __init__(self, root, record_path):
        self.root, self.path = safe(root), safe(record_path)
        if not self.path.is_relative_to(self.root/'calls') or self.path.name != 'record.json':
            raise ValueError('projection input must be calls/*/record.json')
        self.stat = stamp(self.path)
        self.raw = self.path.read_bytes()
        self.record = json.loads(self.raw)
        self.pins = {str(self.path): sha(self.raw)}
        record = self.record
        if (record.get('kind') != 'projection' or record.get('model_commit') != NATIVE
                or record.get('base_model_commit') != PIN or record.get('numerics') != 'rtl'
                or any(record.get(key) is not True for key in ('accelerated', 'native_dispatch_enabled', 'requires_original_backfill'))
                or record.get('qualification_candidate') is not False):
            raise ValueError('expected completed native70/base47 projection input')

    def read(self, desc):
        member = Path(desc['file'])
        if member.is_absolute() or '..' in member.parts or not member.parts:
            raise ValueError('capture member escapes root')
        path = safe(self.root/member)
        raw = path.read_bytes()
        digest = sha(raw)
        if type(desc.get('bytes')) is not int or len(raw) != desc['bytes'] or digest != desc['sha256']:
            raise ValueError('capture payload identity differs: ' + str(path))
        if self.pins.setdefault(str(path), digest) != digest:
            raise ValueError('capture payload changed during projection')
        return raw

    def verify(self):
        bind(self.pins)
        if stamp(self.path) != self.stat or self.path.read_bytes() != self.raw:
            raise ValueError('capture record identity changed during projection')


def prepare(inputs, worker):
    record = inputs.record
    prepared = worker.prepare_projection(record['m'], record['k'], record['n'], record['rescale'],
        record['role'], record['column'], record['width'], inputs.read(record['a_fp8']), inputs.read(record['b_fp8']),
        inputs.read(record['operand_tiles']) if record.get('operand_tiles') is not None else None, record['output_scale'])
    if prepared['source'].encode() != inputs.read(record['program_source']) or prepared['initial'] != inputs.read(record['arena_initial']):
        raise ValueError('independent original producer source or full initial arena differs')
    program, digest = worker.model.assemble(prepared['source'])
    binary = inputs.read(record['program_binary'])
    if binary != b''.join(word.to_bytes(4, 'little') for word in program.assemble()) or sha(binary) != digest or digest != record['program_sha256']:
        raise ValueError('physical source and binary differ')
    for key in ('output_offset', 'output_bytes', 'dma_store_bytes', 'written_spans', 'epilogue', 'extras',
                'output_dtype', 'output_row_stride', 'output_column_tile'):
        if json.loads(json.dumps(prepared[key])) != record[key]:
            raise ValueError('producer contract differs: ' + key)
    requested = dict(load_bytes=prepared['dma_load_bytes'], store_bytes=prepared['dma_store_bytes'])
    if (requested != record['requested_dma'] or record['arena_bytes'] != len(prepared['initial'])
            or record['dram_base'] != 0x90000000 or type(record['logical_rows']) is not int
            or not 1 <= record['logical_rows'] <= record['m']):
        raise ValueError('producer geometry, DMA, arena extent, or base differs')
    return prepared, binary


def check_result(inputs, worker, prepared, arena, output, execution, clipping, preclip):
    record = inputs.record
    offset, extent = prepared['output_offset'], prepared['output_bytes']
    if (arena != inputs.read(record['arena_expected']) or output != inputs.read(record['output_tiled'])
            or len(arena) != len(prepared['initial']) or output != arena[offset:offset+extent]):
        raise ValueError('original47 whole final arena or output differs')
    preserved = bytearray(prepared['initial'])
    compact = bytearray()
    cursor = 0
    for span in prepared['written_spans']:
        start, size = span['offset'], span['bytes']
        if start < cursor or size <= 0 or start+size > extent:
            raise ValueError('invalid written spans')
        preserved[offset+start:offset+start+size] = arena[offset+start:offset+start+size]
        compact.extend(output[start:start+size])
        cursor = start+size
    if bytes(preserved) != arena or sum(s['bytes'] for s in prepared['written_spans']) != prepared['dma_store_bytes']:
        raise ValueError('original47 changed an input, guard, or strided hole')
    if (execution['program_sha256'] != record['program_sha256'] or execution['program_words']*4 != record['program_binary']['bytes']
            or execution['halt_pc'] != execution['program_words']-1 or execution['instructions'] <= 0
            or any(execution['observed'].get(key) != value for key, value in record['requested_dma'].items())
            or any('native' in key.lower() for key in execution['observed'])):
        raise ValueError('actual original47 execution program, ECALL, DMA, or native flags differ')
    m, n, rows = record['m'], record['n'], record['logical_rows']
    compact = bytes(compact)
    if record['role'] == 5:
        if compact != inputs.read(record['output_fp8']) or preclip != inputs.read(record['preclip_bf16_tiles']):
            raise ValueError('original47 FP8 output or raw preclip trace differs')
        raw = worker.unpack_fp8_tiles(compact, m, n)
        tensor = worker.model.torch.from_numpy(raw.copy()).view(worker.model.torch.float8_e4m3fn).float()
        words = (tensor*2.0**record['output_scale']).to(worker.model.torch.bfloat16).view(worker.model.torch.uint16).numpy()
        clipping.update(fp8_nan_encodings=int(((raw & 127) == 127).sum()),
                        emitted_nonfinite=int((~worker.np.isfinite(tensor.numpy())).sum()))
        if (clipping.get('values') != m*n or clipping != record['output_quantization']
                or execution['observed'].get('fp8_output') != clipping):
            raise ValueError('original47 actual preclip counters differ')
        if record['preclip_bf16_tiles'].get('shape') != [m*n] or record['preclip_bf16_tiles'].get('dtype') != 'uint16':
            raise ValueError('preclip descriptor type or shape differs')
    else:
        words = worker.unpack_bf16_tiles(compact, m, n)
        if clipping is not None or record.get('output_quantization') is not None:
            raise ValueError('unexpected FP8 counters for BF16 projection')
    if (record['output_bf16'].get('shape') != [rows, n] or record['output_bf16'].get('dtype') != 'uint16'
            or words[:rows].astype('<u2').tobytes() != inputs.read(record['output_bf16'])):
        raise ValueError('original47 logical BF16 output differs')


def run(root, record_path, output, freeze):
    output = safe(output)
    if not output.is_relative_to(ROOT/'d1-stream-original') or output.is_relative_to(safe(root)):
        raise ValueError('original execution output must be outside capture beneath D/d1-stream-original')
    if output.exists():
        raise ValueError('fresh original execution receipt already exists; audit it instead')
    worker = verify_freeze(freeze)
    inputs = Inputs(root, record_path)
    prepared, binary = prepare(inputs, worker)
    before = source_state()
    executed = worker.model.execute(prepared['source'], prepared['initial'], prepared['output_offset'], prepared['output_bytes'],
        max_instructions=20_000_000, clip_exponent=inputs.record['output_scale'] if inputs.record['role'] == 5 else None)
    execution = executed.metadata()
    clipping = execution['observed'].get('fp8_output')
    check_result(inputs, worker, prepared, executed.arena, executed.output, execution, clipping,
                 b''.join(worker.model.CLIP_TILES))
    after = source_state()
    if before != after or before != freeze:
        raise ValueError('actual original47 source bookends differ')
    inputs.verify()
    # A fresh whitelist prevents native-only provenance and execution metadata
    # from leaking into an original execution record. Descriptors refer only to
    # capture payloads compared byte-for-byte with this actual execution above.
    record = {key: inputs.record[key] for key in RECORD_KEYS if key in inputs.record}
    record.update(schema='atlas.d1.stream_original_execution.v1', model_commit=PIN, numerics='rtl', accelerated=False,
        native_dispatch_enabled=False, execution=execution, output_quantization=clipping,
        model_sources_before=before['model'], model_sources_after=after['model'],
        harness_sources_before=before['harness'], harness_sources_after=after['harness'],
        implementation_sources=[dict(path=p, sha256=s, bytes=safe(p).stat().st_size) for p,s in before['harness'].items()],
        method='executed_original47_on_speculative_capture_inputs', speculative_capture_input_numerical_evidence=True,
        final_full_source_capture_replay_closure_required=True, overall_acceptance=False,
        source_capture_root=str(inputs.root), source_capture_record=str(inputs.path), source_capture_record_sha256=sha(inputs.raw),
        source_capture_record_stat=inputs.stat, capture_payload_sha256=inputs.pins,
        exact=True, full_arena_exact=True, completed_unix=time.time())
    save(output, record)
    return record


def audit(root, record_path, output, freeze):
    worker = verify_freeze(freeze)
    inputs = Inputs(root, record_path)
    result = json.loads(safe(output).read_bytes())
    if (result.get('schema') != 'atlas.d1.stream_original_execution.v1' or result.get('model_commit') != PIN
            or result.get('numerics') != 'rtl' or result.get('accelerated') is not False
            or result.get('native_dispatch_enabled') is not False or result.get('exact') is not True
            or result.get('full_arena_exact') is not True or result.get('method') != 'executed_original47_on_speculative_capture_inputs'
            or result.get('source_capture_root') != str(inputs.root) or result.get('source_capture_record') != str(inputs.path)
            or result.get('source_capture_record_sha256') != sha(inputs.raw) or result.get('source_capture_record_stat') != inputs.stat
            or any(result.get(prefix+'_sources_'+end) != freeze[key] for prefix,key in (('model','model'),('harness','harness')) for end in ('before','after'))
            or any(key.startswith('native_') and key != 'native_dispatch_enabled' for key in result)):
        raise ValueError('resume original47 execution receipt identity or source closure differs')
    prepared, _ = prepare(inputs, worker)
    if any(result.get(key) != inputs.record.get(key) for key in RECORD_KEYS):
        raise ValueError('resume geometry, contract, or payload descriptor differs')
    expected_implementations = [dict(path=p, sha256=s, bytes=safe(p).stat().st_size) for p,s in freeze['harness'].items()]
    if (result.get('implementation_sources') != expected_implementations
            or result.get('speculative_capture_input_numerical_evidence') is not True
            or result.get('final_full_source_capture_replay_closure_required') is not True
            or result.get('overall_acceptance') is not False):
        raise ValueError('resume implementation inventory or evidence scope differs')
    # Verify every referenced artifact, including ancillary FP8/BF16 payloads.
    for key,value in result.items():
        if isinstance(value, dict) and 'file' in value:
            if value != inputs.record.get(key):
                raise ValueError('resume descriptor differs from capture: '+key)
            inputs.read(value)
    check_result(inputs, worker, prepared, inputs.read(result['arena_expected']), inputs.read(result['output_tiled']),
                 result['execution'], result['output_quantization'],
                 inputs.read(result['preclip_bf16_tiles']) if result['role'] == 5 else b'')
    if result.get('capture_payload_sha256') != inputs.pins:
        raise ValueError('resume entire capture payload inventory differs')
    inputs.verify()
    verify_freeze(freeze)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'record', 'out', 'freeze'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    freeze = json.loads(safe(args.freeze).read_bytes())
    result = (audit if args.audit_only else run)(args.root, args.record, args.out, freeze)
    print(json.dumps(dict(status='PASS', record=args.out, instructions=result['execution']['instructions'])), flush=True)
