"""Persistent JSONL attention capture worker using the selected RTL model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import struct
import sys
import traceback

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import numpy as np
import kernel_library  # noqa: E402,F401
from atlas_attention_program import AttentionConfig, assembly, pack_tiles, stage, unpack_tiles
from atlas_execution import Executor, checked, sha


def dma_counts(config, arena, offsets):
    """Count generator traffic from its staged visibility descriptors."""
    loads = offsets['descriptor_bytes'] + config.query_heads * config.query_tiles * 4096
    for index in range(config.query_tiles * config.key_tiles):
        pointer, skip = struct.unpack_from('<II', arena, offsets['descriptors'] + index * 8)
        if not skip:
            loads += config.query_heads * (8192 + (4096 if pointer else 0))
    return dict(load_bytes=loads, store_bytes=config.query_heads * config.query_tiles * 4096)


class Worker:
    def __init__(self, model_root):
        self.executor = Executor(model_root)
        tools = Path(__file__).absolute().parents[1]
        self.sources = []
        for path in (Path(__file__), *(kernel_library.module_path(name) for name in
                     ('atlas_execution.py', 'atlas_path_guard.py', 'atlas_attention_program.py'))):
            path = checked(path)
            raw = path.read_bytes()
            self.sources.append(dict(path=str(path), sha256=sha(raw), bytes=len(raw)))

    def bookend(self):
        if any(sha(checked(item['path']).read_bytes()) != item['sha256'] for item in self.sources):
            raise ValueError('worker implementation source bookend changed')

    def run(self, request):
        self.bookend()
        root, out = checked(request['root']), checked(request['out'])
        if not out.is_relative_to(root) or out == root:
            raise ValueError('worker output must be beneath the capture directory')
        out.mkdir(parents=True, exist_ok=False)

        def descriptor(path, data, shape=None, dtype='uint8'):
            path.write_bytes(data)
            return dict(file=str(path.relative_to(root)), shape=shape or [len(data)], dtype=dtype,
                        sha256=sha(data), bytes=len(data))

        def read(desc):
            relative = Path(desc['file'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('worker input must be relative to capture root')
            path = checked(root / relative)
            data = path.read_bytes()
            if (desc['dtype'] != 'uint16' or len(data) != desc['bytes'] or sha(data) != desc['sha256']
                    or any(type(size) is not int or size <= 0 for size in desc['shape'])
                    or len(data) != int(np.prod(desc['shape'])) * 2):
                raise ValueError('worker input identity differs')
            return np.frombuffer(data, dtype='<u2').reshape(desc['shape']).copy()

        config = AttentionConfig(**request['config'])
        q, k, v = [read(request[role + '_bf16']) for role in ('q', 'k', 'v')]
        arena, offsets = stage(config, q, k, v)
        source = assembly(config, offsets)
        program, digest = self.executor.assemble(source)
        requested = dma_counts(config, arena, offsets)
        executed = self.executor.execute(source, arena, offsets['output'], offsets['output_bytes'],
                                         requested_dma=requested, program_sha256=digest)
        output = unpack_tiles(executed.output, config.query_heads, config.query_rows)
        self.bookend()
        record = dict(kind='attention', config=asdict(config), offsets=offsets, dram_base=0x90000000,
                      output_offset=offsets['output'], output_bytes=offsets['output_bytes'], arena_bytes=len(arena),
                      output_layout='bf16_tiled', model_commit=executed.model['revision'], numerics='rtl',
                      execution=executed.metadata(), implementation_sources=self.sources,
                      program_sha256=digest, requested_dma=requested)
        artifacts = dict(program_source=('program.S', source.encode()),
                         program_binary=('program.bin', b''.join(word.to_bytes(4, 'little') for word in program.assemble())),
                         arena_initial=('arena-initial.bin', arena), arena_expected=('arena-expected.bin', executed.arena),
                         output_tiled=('output-tiled.bin', executed.output))
        for role, value in zip(('q', 'k', 'v'), (q, k, v)):
            record[role + '_bf16'] = request[role + '_bf16']
            artifacts[role + '_tiled'] = (role + '-tiled.bin', pack_tiles(value))
        for name, (filename, data) in artifacts.items():
            record[name] = descriptor(out / filename, data)
        record['output_bf16'] = descriptor(out / 'output-bf16.bin', output.tobytes(), list(output.shape), 'uint16')
        (out / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=checked, required=True)
    args = parser.parse_args()
    worker = Worker(args.model_root)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('command') == 'close':
                print(json.dumps(dict(status='closed')), flush=True)
                break
            print(json.dumps(dict(status='ok', record=worker.run(request))), flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(status='error', error=str(error))), flush=True)


if __name__ == '__main__':
    main()
