"""Compile admitted TinyLlama physical-shape templates; no workload execution."""

from merlin.common.jsonio import write_pretty_json
from merlin.common.digest import sha256_bytes
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import struct
import sys
from source_relocation import physical_path as safe
from atlas_provider_files import PRODUCER_FILE, TILE_FILE


def sha(path):
    return sha256_bytes(safe(path).read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('partition', 'producer', 'model-root', 'out'):
        p.add_argument('--' + name, type=safe, required=True)
    p.add_argument('--producer-sha256', required=True)
    a = p.parse_args()
    if a.out.exists() or sha(a.producer) != a.producer_sha256:
        raise ValueError('fresh output/pinned producer required')
    pins = {str(path): sha(path) for path in [a.partition, a.producer, Path(__file__).absolute()]}
    partition = json.loads(a.partition.read_text())
    if partition['status'] != 'declared_partition_only' or len(partition['contractions']) != 155:
        raise ValueError('whole inventory differs')
    sys.path.insert(0, str(a.producer.parent))
    producer = importlib.import_module(PRODUCER_FILE[:-3])
    code = producer.code_sources()
    pins.update({str(a.producer.parent / name): digest for name, digest in code.items()})
    tile = importlib.import_module(TILE_FILE[:-3])
    model, torch, assemble, revision = tile.configure_model(a.model_root)
    before = tile.model_binding(a.model_root)
    a.out.mkdir(parents=True)
    records = []
    for shape in partition['unique_physical_template_shapes']:
        m, k, n = shape
        plan = producer.resource_plan(m, k, n)
        input_bytes = m * k + k * n
        offset = input_bytes + 256
        source = producer.assembly(m, k, n, offset, 0, resident_b=plan['resident_b'], pretranspose_resident_b=plan['resident_b'],
            resident_a=plan['resident_a'])
        words = assemble(io.StringIO(source)).assemble()
        producer.admit_program(words)
        locations = [i for i, w in enumerate(words) if w == 0x3f80025f]
        if len(locations) != 1:
            raise ValueError('unique canonical rescale word missing')
        patch = locations[0]
        for exp in (-126, -1, 0, 127):
            value = (exp + 127) << 7
            changed = source.replace('vli.all m4, 16256', 'vli.all m4, ' + str(value))
            expected = assemble(io.StringIO(changed)).assemble()
            actual = list(words)
            actual[patch] = ((exp + 127) << 23) | 0x25f
            if actual != expected:
                raise ValueError('patch differs from full reassembly')
        directory = a.out / ('-'.join(map(str, shape)))
        directory.mkdir()
        (directory / 'program.S').write_text(source)
        (directory / 'program.bin').write_bytes(struct.pack('<' + 'I' * len(words), *words))
        records.append({'shape_mkn': shape, 'program': {'path': str(directory / 'program.bin'), 'sha256': sha(
            directory / 'program.bin')},
                        'source': {'path': str(directory / 'program.S'), 'sha256': sha(directory / 'program.S')},
            'program_words': len(
            words),
                        'patch_word_index': patch, 'base_rescale_exponent': 0, 'patch_reassembly_exponents': [-126, -1, 0, 127],
                        'memory': {'dram_base': 1 << 32, 'input_bytes': input_bytes, 'output_offset': offset,
            'output_bytes': m * n * 2,
            'total_dram_bytes': input_bytes + m * n * 2 + 512}, 'resource_admission': plan})
    after = tile.model_binding(a.model_root)
    bookends = before == after and all(sha(Path(path)) == digest for path, digest in pins.items())
    if not bookends:
        raise ValueError('model/producer/partition source changed')
    record = {'status': 'passed_compile_only',
        'scope': 'ten physical-shape templates for declared full TinyLlama partition, not whole-model compilation/execution',
            'command': sys.argv, 'partition': {'path': str(a.partition), 'sha256': sha(a.partition)}, 'templates': records,
        'code_before': code, 'code_after': producer.code_sources(),
            'source_inputs': pins, 'model_binding_before': before, 'model_binding_after': after,
        'source_bookends_equal': bookends,
            'not_claimed': ['full model compiled', 'device kernel execution', 'decode execution', 'host runtime integration',
        'native quality passing']}
    write_pretty_json(a.out / 'receipt.json', record, sort_keys=False)
    print(json.dumps({'status': record['status'], 'templates': len(records)}))
if __name__ == '__main__':
    main()
