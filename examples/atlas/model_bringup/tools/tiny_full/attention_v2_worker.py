"""Explicit original47 instruction execution for versioned resident-KV attention.

The JSONL process is separate from legacy/native projection workers. Preparation
assembles but does not execute; functional and Core modes use the same raw binary.
"""
from collections import Counter
from dataclasses import asdict, fields, replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
TOOLS = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(TOOLS))
import numpy as np
import kernel_library  # noqa: E402,F401
import atlas_attention_v2_program as generator
from atlas_execution import Execution, Executor, MODEL_PIN
from atlas_path_guard import checked
from atlas_program_checks import check_matrix_registers

SCHEMA = 'atlas.tiny.attention_launch.v2'
GENERATOR_SHA = '16a6e96e5cbb801d7fe58a53c37e15d18fd098a2e72daae16abd337b3c97a774'
GENERATOR_UPSTREAM = '58feac951e5b71eff7e441a7648023d191a15984'
ARENA_BASE = 0x90000000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def config_from(value):
    require(isinstance(value, dict) and set(value) == {field.name for field in fields(generator.AttentionV2Config)},
            'v2 requires the complete explicit configuration')
    require(all(type(value[name]) is bool for name in ('output_fp8', 'packed_decode')), 'v2 Boolean field required')
    require(type(value['scale']) in (int, float) and (value['query_start'] is None or type(value['query_start']) is int),
            'v2 scalar configuration types differ')
    return generator.AttentionV2Config(**value)


def member(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'input member escaped root')
    return checked(root/path)


def read_descriptor(root, desc):
    require(isinstance(desc, dict) and type(desc.get('bytes')) is int, 'invalid input descriptor')
    raw = member(root, desc['file']).read_bytes()
    require(len(raw) == desc['bytes'] and digest(raw) == desc['sha256'], 'input descriptor identity differs')
    return raw


def output_spans(config):
    head_bytes = 2048 if config.output_fp8 else 4096
    if config.decode_packed:
        return [dict(offset=head*head_bytes+part*1024, bytes=32)
                for head in range(config.query_heads) for part in range(2 if config.output_fp8 else 4)]
    return [dict(offset=0, bytes=config.query_tiles*config.query_heads*head_bytes)]


def dma_counts(config, offsets):
    q = (config.kv_heads if config.decode_packed else config.query_tiles*config.query_heads)*4096
    kv = 2*config.kv_heads*config.key_tiles*4096
    tail = sum(offsets[key] for key in ('descriptor_bytes', 'key_count_bytes', 'mask_bytes'))
    return dict(load_bytes=q+kv+tail, store_bytes=sum(span['bytes'] for span in output_spans(config)))


def prepare(request, root):
    require(request.get('schema') == SCHEMA and request.get('command') == 'attention_v2', 'explicit attention-v2 request required')
    config = config_from(request['config'])
    tensors, normal, normal_origin = {}, {}, {}
    for name, heads, rows in (('q', config.query_heads, config.query_rows), ('k', config.kv_heads, config.keys),
                              ('v', config.kv_heads, config.keys)):
        desc = request[name+'_bf16']
        require(desc.get('dtype') == 'uint16' and desc.get('shape') == [heads, rows, 64], 'logical BF16 input shape/dtype differs')
        raw = read_descriptor(root, desc)
        require(len(raw) == heads*rows*128, 'logical BF16 input byte count differs')
        tensors[name] = np.frombuffer(raw, dtype='<u2').reshape(heads, rows, 64).copy()
        source_key = name+'_source_tiled' if name+'_source_tiled' in request else name+'_tiled'
        if name+'_source_tiled' in request and name+'_tiled' in request:
            require(request[name+'_source_tiled'] == request[name+'_tiled'], 'conflicting normal C source descriptors')
        if source_key in request:
            normal[name] = read_descriptor(root, request[source_key])
            require(len(normal[name]) == ((rows+31)//32)*heads*4096, 'normal C input extent differs')
            require(np.array_equal(generator.unpack_tiles(normal[name], heads, rows), tensors[name]), 'normal C input changed logical BF16 bits')
            if name != 'q':
                require(normal[name] == generator.pack_tiles(tensors[name]), 'future KV padding is not zero')
            normal_origin[name] = 'supplied_normal_c_tiled'
        else:
            normal[name] = generator.pack_tiles(tensors[name])
            normal_origin[name] = 'derived_from_logical_bf16'
    masks = None
    if 'mask_bools' in request:
        desc = request['mask_bools']
        raw = read_descriptor(root, desc)
        require(desc.get('dtype') == 'uint8' and desc.get('shape') == [config.query_rows, config.keys]
                and len(raw) == config.query_rows*config.keys and all(x in (0, 1) for x in raw), 'invalid explicit mask')
        padded = np.zeros((config.query_tiles*32, config.key_tiles*32), dtype=bool)
        padded[:config.query_rows, :config.keys] = np.frombuffer(raw, dtype=np.uint8).reshape(config.query_rows, config.keys)
        masks = [[padded[q*32:q*32+32, k*32:k*32+32] for k in range(config.key_tiles)] for q in range(config.query_tiles)]
    require(config.mask != 'arbitrary' or masks is not None, 'arbitrary mask requires mask_bools descriptor')
    initial, offsets = generator.stage_v2(config, tensors['q'], tensors['k'], tensors['v'], masks)
    staged = dict(q=initial[offsets['q']:offsets['k']], k=initial[offsets['k']:offsets['v']],
                  v=initial[offsets['v']:offsets['descriptors']])
    if config.decode_packed:
        packed = bytearray(config.kv_heads*4096)
        group = config.query_heads//config.kv_heads
        for head in range(config.query_heads):
            kv, row = divmod(head, group)
            for part in range(4):
                start = head*4096+part*1024
                dest = kv*4096+part*1024+row*32
                packed[dest:dest+32] = normal['q'][start:start+32]
        require(bytes(packed) == staged['q'], 'packed decode differs from exact normal-C row slices')
    else:
        require(np.array_equal(generator.unpack_tiles(staged['q'], config.query_heads, config.query_rows), tensors['q']),
                'staged Q changed logical BF16 bits')
    source = generator.assembly_v2(config, offsets, dram_base=ARENA_BASE)
    generator.check_program_v2(source)
    check_matrix_registers(source)
    return dict(config=config, initial=initial, offsets=offsets, source=source, staged=staged, normal=normal,
                normal_origin=normal_origin, spans=output_spans(config), requested_dma=dma_counts(config, offsets))


def validate_arena(initial, final, offset, extent, spans):
    require(len(initial) == len(final), 'final arena size differs')
    wanted = bytearray(initial)
    end = 0
    for span in spans:
        at, size = span['offset'], span['bytes']
        require(type(at) is int and type(size) is int and size > 0 and end <= at and at+size <= extent, 'invalid output spans')
        wanted[offset+at:offset+at+size] = final[offset+at:offset+at+size]
        end = at+size
    require(bytes(wanted) == final, 'input, static tail, guard, or output hole changed')


class QuietLogger:
    def log_cycle(self, *args, **kwargs): pass
    def log_insn(self, *args, **kwargs): pass
    def log_retire(self, *args, **kwargs): pass
    def log_stage_start(self, *args, **kwargs): pass
    def log_stage_end(self, *args, **kwargs): pass
    def log_arch_value(self, *args, **kwargs): pass
    def log_commit(self, *args, **kwargs): pass


def harness_sources():
    paths = {checked(__file__)}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).absolute()
            if path.is_relative_to(TOOLS):
                paths.add(checked(path))
    return {str(path): digest(path.read_bytes()) for path in sorted(paths)}


class Worker:
    def __init__(self, model_root):
        require(all(os.environ.get(name, '0') == '0' for name in ('ATLAS_NPU_MODEL_NATIVE', 'NPU_MODEL_RTL_NATIVE_MXU0')),
                'attention v2 requires original47 with native execution disabled')
        require(os.environ.get('ATLAS_NPU_MODEL_PIN', MODEL_PIN) == MODEL_PIN, 'foreign model pin selected')
        path = checked(generator.__file__)
        require(path == kernel_library.module_path('atlas_attention_v2_program.py') and digest(path.read_bytes()) == GENERATOR_SHA,
                'attention-v2 generator differs from exact upstream source')
        model_root = checked(model_root)
        require(Path(subprocess.check_output(['git', '-C', str(model_root), 'rev-parse', '--show-toplevel'],
            text=True).strip()) == model_root, 'original47 model Git root differs')
        self.executor = Executor(model_root)
        from npu_model.hardware import Core
        from npu_model.hardware.dma import DmaExecutionUnit
        self.core_type, self.dma_type = Core, DmaExecutionUnit

    def model_sources(self):
        binding = self.executor.binding()
        return {str(checked(self.executor.root/name)): sha for name, sha in binding['source_sha256'].items()}

    def timed(self, source, initial, offset, extent, cap):
        require(type(cap) is int and cap > 0, 'positive Core cycle bound required')
        executor = self.executor
        program, sha = executor.assemble(source)
        program.dram_base = ARENA_BASE
        config = executor.config()
        config.arch_state_config = replace(config.arch_state_config, numerics='rtl', randomize_init=True, init_seed=42)
        program.memory_regions = [(0, executor.torch.frombuffer(bytearray(initial), dtype=executor.torch.uint8))]
        core = self.core_type(config=config, logger=QuietLogger())
        started, observed = time.perf_counter(), Counter()
        try:
            core.load_program(program)
            state = core.arch_state
            dma = next(unit for unit in core.exus if isinstance(unit, self.dma_type))
            while not core.is_finished() and core.cycle_count < cap:
                core.tick()
                event = dma.last_cycle
                if event.a:
                    observed['store_bytes' if event.a[2] else 'load_bytes'] += 32
                if core.stall_reason:
                    observed['stall_'+core.stall_reason] += 1
            require(core.is_finished(), 'Core cycle bound reached')
            require(state.halted and state.halt_reason == 'ecall', 'Core did not halt by ECALL')
            require(not any(unit.has_in_flight for unit in core.exus) and not any(state.check_flag(ch) for ch in range(8)),
                    'Core halted before engines and DMA drained')
            final = bytes(state.read_dram(ARENA_BASE, len(initial)).tolist())
            binding = executor._bookend()
            return Execution(final, final[offset:offset+extent], time.perf_counter()-started, core.total_completed,
                len(program), sha, dict(observed), state.execute_pc, binding, dict(executor.implementation_sources), core.cycle_count)
        finally:
            core.close()

    def run(self, request, prepare_only=False):
        require(all(os.environ.get(name, '0') == '0' for name in ('ATLAS_NPU_MODEL_NATIVE', 'NPU_MODEL_RTL_NATIVE_MXU0')),
                'attention v2 requires native execution disabled')
        root, out = checked(request['root']), checked(request['out'])
        require(out != root and out.is_relative_to(root), 'output must be below input root')
        require(not out.exists(), 'fresh output directory required')
        mode = request.get('mode', 'functional')
        require(mode in ('functional', 'timed'), 'unknown attention execution mode')
        prepared = prepare(request, root)
        c, offsets, source, initial = (prepared[key] for key in ('config', 'offsets', 'source', 'initial'))
        program, sha = self.executor.assemble(source)
        binary = b''.join(struct.pack('<I', word) for word in program.assemble())
        require(digest(binary) == sha and len(binary)+20 <= 16384, 'raw attention binary identity or runtime IMEM extent differs')
        before, helpers_before = self.model_sources(), harness_sources()
        out.mkdir(parents=True, exist_ok=False)

        def desc(name, raw, shape=None, dtype='uint8'):
            (out/name).write_bytes(raw)
            return dict(file=str((out/name).relative_to(root)), bytes=len(raw), sha256=digest(raw),
                        shape=shape if shape is not None else [len(raw)], dtype=dtype)

        record = dict(schema=SCHEMA, kind='attention', config=asdict(c), offsets=offsets, mode=mode,
            status='PREPARED_NO_EXECUTION' if prepare_only else 'PASS',
            model_commit=MODEL_PIN, model_root=str(self.executor.root), numerics='rtl', accelerated=False,
            native_dispatch_enabled=False, runtime_patched=False, dram_base=ARENA_BASE,
            generator=dict(path=str(checked(generator.__file__)), sha256=GENERATOR_SHA, upstream_commit=GENERATOR_UPSTREAM),
            input_layout='bf16_tiled_packed_decode' if c.decode_packed else 'bf16_tiled',
            normal_input_layout='bf16_tiled', normal_input_origin=prepared['normal_origin'],
            output_layout='fp8_tiled' if c.output_fp8 else 'bf16_tiled', output_dtype='fp8' if c.output_fp8 else 'bf16',
            output_scale=c.o_exponent if c.output_fp8 else 0, output_offset=offsets['output'],
            output_bytes=offsets['output_bytes'], output_extent_bytes=offsets['output_bytes'], arena_bytes=len(initial),
            written_spans=prepared['spans'], requested_dma=prepared['requested_dma'], program_sha256=sha,
            source_and_binary_exact=True)
        for key in ('step', 'layer', 'site', 'name'):
            if key in request:
                record[key] = request[key]
        if 'mask_bools' in request:
            record['mask_bools'] = request['mask_bools']
        for name in ('q', 'k', 'v'):
            record[name+'_bf16'] = request[name+'_bf16']
            record[name+'_source_tiled'] = (request.get(name+'_source_tiled') or request.get(name+'_tiled')
                                           or desc(name+'-normal-tiled.bin', prepared['normal'][name]))
            record['staged_'+name] = desc(name+'-staged-tiled.bin', prepared['staged'][name])
        record['program_source'] = desc('program.S', source.encode())
        record['program_binary'] = desc('program.bin', binary)
        record['arena_initial'] = desc('arena-initial.bin', initial)
        if not prepare_only:
            expected = read_descriptor(root, request['expected_arena']) if 'expected_arena' in request else None
            if mode == 'functional':
                executed = self.executor.execute(source, initial, offsets['output'], offsets['output_bytes'],
                    expected_arena=expected, requested_dma=prepared['requested_dma'], program_sha256=sha,
                    max_instructions=request.get('max_instructions', 20_000_000), dram_base=ARENA_BASE)
            else:
                executed = self.timed(source, initial, offsets['output'], offsets['output_bytes'], request.get('max_cycles', 100_000_000))
            require(executed.program_sha256 == sha and executed.program_words == len(binary)//4
                    and executed.halt_pc == len(binary)//4-1, 'executed raw program or halt PC differs')
            require(all(executed.observed.get(key) == value for key, value in prepared['requested_dma'].items()), 'observed DMA differs')
            require(expected is None or executed.arena == expected, 'complete arena differs from independent expected')
            validate_arena(initial, executed.arena, offsets['output'], offsets['output_bytes'], prepared['spans'])
            output = executed.output
            logical = generator.unpack_output_v2(c, output)
            compact = b''.join(output[span['offset']:span['offset']+span['bytes']] for span in prepared['spans'])
            record.update(execution=executed.metadata(), independent_reference_exact=expected is not None,
                          expected_arena_reference=request.get('expected_arena'), outside_written_spans_preserved=True,
                          halt_exact=True, engine_drain_verified=mode == 'timed', halt_and_drain_exact=True if mode == 'timed' else None)
            record['arena_expected'] = desc('arena-expected.bin', executed.arena)
            record['output_tiled'] = desc('output-tiled.bin', output)
            if c.output_fp8:
                torch = self.executor.torch
                decoded = torch.from_numpy(logical.copy()).view(torch.float8_e4m3fn).float()
                logical_bf16 = (decoded*2.0**c.o_exponent).to(torch.bfloat16).view(torch.uint16).numpy()
                record['output_fp8'] = desc('output-fp8-compact.bin', compact)
                record['output_fp8_logical'] = desc('output-fp8-logical.bin', logical.tobytes(), list(logical.shape))
                def counts(raw):
                    value = np.frombuffer(raw, dtype=np.uint8)
                    return dict(values=len(value), nan_encodings=int(((value & 127) == 127).sum()),
                                at_finite_limit=int(((value & 127) == 126).sum()))
                record['output_quantization'] = dict(method='emitted_fp8_classification_only', scale_exponent=c.o_exponent,
                    physical=counts(output), logical=counts(logical.tobytes()), written=counts(compact),
                    preclip_saturation=None, preclip_note='Independent primitive-reference statistics belong to the unit/capture audit; at_finite_limit is not a clipping count.')
            else:
                logical_bf16 = logical
            record['output_bf16'] = desc('output-bf16.bin', logical_bf16.astype('<u2').tobytes(), list(logical_bf16.shape), 'uint16')
        for key in ('q_bf16', 'k_bf16', 'v_bf16', 'q_source_tiled', 'k_source_tiled', 'v_source_tiled',
                    'q_tiled', 'k_tiled', 'v_tiled', 'mask_bools'):
            if key in request:
                read_descriptor(root, request[key])
        if not prepare_only and 'expected_arena' in request:
            read_descriptor(root, request['expected_arena'])
        after, helpers_after = self.model_sources(), harness_sources()
        require(before == after and helpers_before == helpers_after, 'model or actual helper source bookends changed')
        record.update(model_sources_before=before, model_sources_after=after, harness_sources_before=helpers_before,
                      harness_sources_after=helpers_after, source_bookends_verified=True,
                      implementation_sources=[dict(path=path, sha256=value, bytes=checked(path).stat().st_size)
                                              for path, value in helpers_before.items()])
        (out/'record.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=checked, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--request', type=checked)
    choice.add_argument('--jsonl', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    worker = Worker(args.model_root)
    if args.request:
        record = worker.run(json.loads(args.request.read_bytes()), args.prepare_only)
        print(json.dumps(dict(status='ok', record=record)), flush=True)
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('command') == 'close':
                print(json.dumps(dict(status='closed')), flush=True)
                break
            print(json.dumps(dict(status='ok', record=worker.run(request, args.prepare_only))), flush=True)
        except BaseException as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(status='error', error=str(error), error_type=type(error).__name__)), flush=True)


if __name__ == '__main__':
    main()
