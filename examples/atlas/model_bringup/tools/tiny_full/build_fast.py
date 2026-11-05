"""Build the TinyLlama fast-transport SIMS host from a built default SIMS output.

Fast projection programs read FP8 weights in place from the linked weight symbols. Generation starts from a
placeholder base and is repeated against the linked ELF until every weight base is stable.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_execution import Executor
from atlas_path_guard import checked
from atlas_resnet_input import relocate_assembly as relocate_weight_source
from atlas_tooling import sha256
from atlas_weight_source import WeightSource
from attention_bundle import AttentionBundle, require
from check_programs import projection_source
from chipyard.arena import DEFAULT_ARENA, relocate_assembly
from elf_image import Image

HERE = Path(__file__).absolute().parent
PLACEHOLDER_BASE = 0xb0000000
HEADER = ['#ifndef TINY_FAST_ASSETS_H', '#define TINY_FAST_ASSETS_H', '#include <stddef.h>', '#include <stdint.h>',
          'typedef struct { const uint32_t *program; size_t program_words; uintptr_t encoded_weight_base; } tiny_fast_panel;',
          '#include "tiny_full_assets.h"', 'extern const tiny_fast_panel tiny_fast_panels[TINY_FULL_KERNELS];', '#endif']


def encoded(executor, source):
    program, _ = executor.assemble(source)
    return b''.join(word.to_bytes(4, 'little') for word in program.assemble())


def validate_attention_initials(image, items, read):
    """Admit only immutable linked templates whose zero output and guards match the bundle."""
    table = image.symbols['tiny_full_attention_panels']
    require(table['size'] == 176 * len(items), 'attention initial admission table size differs')

    def readonly(address, size):
        return any(s['flags'] == 4 and s['address'] <= address
                   and address + size <= s['address'] + s['file_bytes'] for s in image.segments)

    require(readonly(table['address'], table['size']), 'attention initial admission table is writable')
    for index, item in enumerate(items):
        panel = image.bytes_at(table['address'] + 176 * index, 176)
        offset, extent, total = struct.unpack_from('<QQQ', panel, 120)
        pointer = struct.unpack_from('<Q', panel, 160)[0]
        symbol = image.symbols[f'tiny_full_attention_initial_{index}']
        require(item.get('attention_version') == 2 and offset == item['output_offset']
                and extent == item['output_extent_bytes'] and total == item['total_dram_bytes']
                and offset >= 256 and total == offset + extent + 256
                and pointer == symbol['address'] and total == symbol['size']
                and readonly(pointer, total), 'attention initial admission linked extent or placement differs')
        initial = image.bytes_at(pointer, total)
        require(initial == read(item['initial']), 'attention initial admission linked payload differs')
        require(initial[offset:offset + extent] == bytes(extent)
                and initial[offset - 256:offset] == b'\xa5' * 256
                and initial[offset + extent:] == b'\x5a' * 256,
                'attention initial admission zero output or guards differ')
    return len(items)


def generate(bundle, executor, default_host, out, image=None):
    """Write fast programs, their table, and images; return each launch's linked weight base."""
    out.mkdir(exist_ok=True)
    defaults, programs, bases, table = set(), {}, [], []
    admitted = bundle.data.get('attention_version', 1) == 2
    if admitted and image is not None:
        validate_attention_initials(image, bundle.data['attention_kernels'], bundle.read)
    header = HEADER[:-1] + [f'#define TINY_FAST_ATTENTION_INITIALS_ADMITTED {int(admitted)}', '#endif']
    lines = ['#include "fast-assets.h"']
    images = (default_host / 'images.S').read_text().replace('.balign 8', '.balign 64')
    for index, item in enumerate(bundle.data['projection_kernels']):
        m, k, n = item['physical_mkn']
        key = (m, k, n, item['output_offset'], item['rescale'], json.dumps(item['epilogue'], sort_keys=True))
        if key not in defaults:
            default = relocate_assembly(projection_source(item), DEFAULT_ARENA, item['dram_base'])
            require(default.encode() == bundle.read(item['source'])
                    and encoded(executor, default) == bundle.read(item['original']), 'default program differs: ' + str(index))
            defaults.add(key)
        base = image.symbols[f'tiny_full_weights_{index}']['address'] if image else PLACEHOLDER_BASE
        weights = WeightSource(base, k * n, item['dram_base'])
        weights.validate(k, n)
        if key + (base,) not in programs:
            source = relocate_weight_source(projection_source(item, weight_source=weights), item['dram_base'],
                                            arena_bytes=item['total_dram_bytes'], weight_source=weights)
            raw = encoded(executor, source)
            require(len(raw) + 20 <= 131072, 'fast program exceeds physical IMEM')
            programs[key + (base,)] = source.encode(), raw
        source, raw = programs[key + (base,)]
        (out / f'program-{index:04d}.S').write_bytes(source)
        (out / f'program-{index:04d}.bin').write_bytes(raw)
        symbol = f'tiny_fast_program_{index}'
        lines.append(f'extern const uint32_t {symbol}[];')
        images += ('\n.section .rodata,"a",@progbits\n.balign 4\n.global ' + symbol + '\n.type ' + symbol + ',@object\n'
                   + symbol + ':\n.incbin ' + json.dumps(str(out / f'program-{index:04d}.bin')) + '\n.size '
                   + symbol + ',.-' + symbol + '\n')
        table.append(f'{{{symbol},{len(raw) // 4},UINT64_C({base})}}')
        bases.append(base)
    lines.append('const tiny_fast_panel tiny_fast_panels[TINY_FULL_KERNELS]={' + ',\n'.join(table) + '};')
    (out / 'fast-assets.h').write_text('\n'.join(header) + '\n')
    (out / 'fast-assets.c').write_text('\n'.join(lines) + '\n')
    (out / 'fast-images.S').write_text(images)
    return bases


def build(args, generated):
    manifest = (args.default / 'sources.cmake').read_text().replace(str(args.default / 'host/images.S'),
                                                                    str(generated / 'fast-images.S'))
    manifest += ('\nlist(APPEND TL_SOURCES "' + str(generated / 'fast-assets.c') + '")\n'
                 + 'list(APPEND TL_INCLUDES "' + str(generated) + '")\n'
                 + f'list(APPEND TL_DEFINITIONS ATLAS_FAST_MODE=1 TINY_FAST_HOST_LEVEL={args.host_level})\n')
    path = args.out / 'fast.cmake'
    path.write_text(manifest)
    env = dict(os.environ, PATH=str(args.toolchain_bin) + os.pathsep + os.environ.get('PATH', ''),
               TMPDIR=str(args.out / 'tmp'), PYTHONDONTWRITEBYTECODE='1')
    (args.out / 'tmp').mkdir(exist_ok=True)
    platform = args.platform_root
    for command in ([args.cmake, '-S', str(platform), '-B', str(args.out / 'build'),
                     '-DCMAKE_TOOLCHAIN_FILE=' + str(platform / 'riscv-gcc.cmake'), '-DCMAKE_BUILD_TYPE=Release',
                     '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DTL_MANIFEST=' + str(path), '-DPLATFORM=SIMS',
                     '-DPENGUIN26_EXIT_TOHOST=OFF'],
                    [args.cmake, '--build', str(args.out / 'build'), '--target', 'atlas-tinyllama', '--parallel', str(args.jobs)]):
        subprocess.run(command, env=env, check=True)
    return args.out / 'build/atlas-tinyllama.elf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'model-root', 'default', 'toolchain-bin', 'out', 'platform-root'):
        parser.add_argument('--' + name, type=checked, required=True)
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--host-level', type=int, choices=(0, 1, 2), default=2,
                        help='0: no host optimizations; 1: admitted attention templates; 2: plus coalesced tiled copies')
    args = parser.parse_args()
    require(args.jobs > 0 and not args.out.exists(), 'positive jobs and a fresh output directory required')
    bundle = AttentionBundle(args.assets)
    record = json.loads((args.default / 'build.json').read_text())
    require(record['variant'] == 'sims' and record['assets_sha256sums_sha256'] == bundle.checksums_sha256
            and sha256(Path(record['elf'])) == record['elf_sha256'], 'built default SIMS output for these assets required')
    executor, generated = Executor(args.model_root), args.out / 'generated'
    args.out.mkdir(parents=True)
    bases = generate(bundle, executor, args.default / 'host', generated)
    for _ in range(3):
        elf = build(args, generated)
        linked = generate(bundle, executor, args.default / 'host', generated, Image(elf.read_bytes()))
        if linked == bases:
            break
        bases = linked
    else:
        raise ValueError('linked weight bases did not stabilize')
    print(elf)
    print(sha256(elf))


if __name__ == '__main__':
    main()
