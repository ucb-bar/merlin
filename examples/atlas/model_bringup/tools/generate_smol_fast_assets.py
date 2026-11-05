#!/usr/bin/env python3
"""Bind Smol programs to an admitted ELF, retaining complete fusion variant keys."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).absolute().parent / 'host_iss'))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_program import assembly, _reuse_module
from atlas_weight_source import WeightSource
from chipyard.arena import relocate_assembly as old_relocate, DEFAULT_ARENA
from elf_image import Image
from npu_model.util.converter import input_to_program
from smol_fusion_assets import initial_dram, regenerate, read_record
from smol_weight_relocate import relocate_assembly


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(source):
    words = input_to_program(io.StringIO(source)).assemble()
    return struct.pack('<' + 'I' * len(words), *words)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_for(row, read, weight_source=None):
    if row.get('fusion'):
        initial = initial_dram(row, read)
        bias = read_record(row['fusion']['bias'], read)
        return regenerate(row, initial, bias_bf16=bias, weight_source=weight_source).source
    m, k, n = row['physical_mkn']
    reuse = row['recipe'] == 'traffic-reuse-v1'
    rb = not reuse and k * n <= 262144
    ra = not reuse and not rb and m * k <= 262144
    source = assembly(m, k, n, row['output_offset'], row['output_rescale_exponent'], traffic_reuse=reuse,
                      resident_b=rb, resident_a=ra, pretranspose_resident_b=rb, weight_source=weight_source)
    if weight_source:
        return relocate_assembly(source, DEFAULT_ARENA, row['dram_base'], weight_source=weight_source)
    return old_relocate(source, DEFAULT_ARENA, row['dram_base'])


def generate(bundle, elf, out):
    def read(name):
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Require a relative bundle member')
        return checked(bundle / path).read_bytes()

    image = Image(elf.read_bytes())
    contract = json.loads(read('host/smol-full-contract.json'))
    require(contract.get('schema') == 'atlas.smol_full_contract.v1', 'Require a full Smol contract')
    require(len(contract['kernels']) == 2715, 'Require the complete 2715-panel Smol plan')
    defaults, programs, rows, tables = {}, {}, [], []
    lines = ['#include "smol_fast_assets.h"']
    images = ['.section .rodata.smol_fast,"a",@progbits']
    symbols = {value['address']: key for key, value in image.symbols.items() if key.startswith('smol_full_weights_')}
    for index, item in enumerate(contract['kernels']):
        m, k, n = item['physical_mkn']
        if index % 100 == 0:
            print('shape', index, m, k, n, flush=True)
        # The complete key binds epilogues, fixed FP8 scales, extras, and scratch.
        key = (m, k, n, item['output_offset'], item['output_rescale_exponent'], item['recipe'],
               json.dumps(item.get('fusion'), sort_keys=True))
        if key not in defaults:
            source = source_for(item, read)
            defaults[key] = source.encode(), encoded(source)
        old_source, old_raw = defaults[key]
        require(old_source == read(item['source']), f'Default source differs at panel {index}')
        require(old_raw == read(item['original']), f'Default program differs at panel {index}')
        # RV64 smol_policy_panel layout: 144-byte stride, weight pointer at 8, and weight extent at 112.
        panel = image.bytes_at(image.symbols['smol_full_panels']['address'] + 144 * index, 144)
        base, size = struct.unpack_from('<Q', panel, 8)[0], struct.unpack_from('<Q', panel, 112)[0]
        require(size == k * n, f'Panel ABI differs at panel {index}')
        symbol = symbols[base]
        require(image.bytes_at(base, size) == read(item['weights']), f'Linked weights differ at panel {index}')
        weights = WeightSource(base, size, item['dram_base'])
        weights.validate(k, n)
        variant = key + (base,)
        if variant not in programs:
            source = source_for(item, read, weights)
            raw = encoded(source)
            require(len(raw) + 20 <= 131072, f'Program exceeds IMEM at panel {index}')
            number = len(programs)
            name = f'program-{number:04d}'
            (out / (name + '.S')).write_text(source)
            (out / (name + '.bin')).write_bytes(raw)
            programs[variant] = number, source, raw
            program_symbol = f'smol_fast_program_{number:04d}'
            images += ['.balign 4', '.global ' + program_symbol, '.type ' + program_symbol + ',@object',
                       program_symbol + ':', '.incbin ' + json.dumps(str(out / (name + '.bin'))),
                       '.size ' + program_symbol + ',.-' + program_symbol]
            lines.append('extern const uint32_t ' + program_symbol + '[];')
        number, source, raw = programs[variant]
        program_symbol = f'smol_fast_program_{number:04d}'
        tables.append(f'{{{program_symbol},{len(raw)//4},UINT64_C({base})}}')
        rows.append(dict(index=index, variant=number, weight_symbol=symbol, program_symbol=program_symbol,
                         source_sha256=sha(source.encode()), program_sha256=sha(raw),
                         default_source_sha256=sha(old_source), default_program_sha256=sha(old_raw),
                         original_source=item['source'], original_program=item['original'], fusion=item.get('fusion'),
                         recipe=_reuse_module(m, k, n) if item['recipe'] == 'traffic-reuse-v1' else 'base',
                         weight_source=weights.metadata((15, 20))))
    header = ('#ifndef SMOL_FAST_ASSETS_H\n#define SMOL_FAST_ASSETS_H\n#include <stdint.h>\n#include <stddef.h>\n'
              'typedef struct { const uint32_t *program; size_t program_words; uintptr_t encoded_weight_base; } '
              'smol_fast_panel;\nextern const smol_fast_panel smol_fast_panels[2715];\n#endif\n')
    (out / 'smol_fast_assets.h').write_text(header)
    (out / 'smol_fast_assets.c').write_text('\n'.join(lines) + '\nconst smol_fast_panel smol_fast_panels[2715]={\n'
                                           + ',\n'.join(tables) + '\n};\n')
    (out / 'smol_fast_images.S').write_text('\n'.join(images) + '\n')
    result = dict(status='GENERATED', input_elf=str(elf), input_elf_sha256=sha(image.raw),
                  default_source_exact=len(rows), default_program_exact=len(rows), distinct_default_variants=len(defaults),
                  distinct_fast_variants=len(programs), fused_launches=sum(bool(row['fusion']) for row in rows),
                  owner_contract_sha256=sha(read('host/smol-full-contract.json')), launches=rows)
    (out / 'relocations.json').write_text(json.dumps(result, indent=2) + '\n')
    print('generated', len(rows), 'variants', len(programs), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('elf', 'bundle', 'out'):
        parser.add_argument('--' + name, type=checked, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    generate(args.bundle, args.elf, args.out)


if __name__ == '__main__':
    main()
