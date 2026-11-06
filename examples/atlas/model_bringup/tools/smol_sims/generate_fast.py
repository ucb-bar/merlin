#!/usr/bin/env python3
"""Generate fast-mode Smol projection programs that read immutable B weights in place from a linked SIMS ELF.

Each default projection program is regenerated and checked against the bundle, then reissued with a WeightSource
bound to the weight symbol's linked address. The output directory holds program-NNNN.S/.bin, the C panel table
smol_fast_assets.c, the incbin images smol_fast_images.S, and relocations.json. Bind to a default SIMS ELF built from
the same package: its fixed immutable assets segment gives every phase variant the same weight addresses, and the
runtime rejects a stale table before START.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
TOOLS = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(TOOLS), str(TOOLS/'host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_program import assembly, _reuse_module
from atlas_weight_source import WeightSource
from chipyard.arena import relocate_assembly as default_relocate
from elf_image import Image
from smol_fusion_assets import profile as fusion_profile
from smol_model_binding import configure_model
from smol_weight_relocate import DEFAULT_ARENA, relocate_assembly

PANEL_BYTES = 144
IMEM_BYTES = 131072


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=checked, required=True, help='default standalone Smol bundle')
    p.add_argument('--elf', type=checked, required=True, help='default SIMS ELF linked from that bundle')
    p.add_argument('--model-root', type=checked, required=True, help='pinned npu_model checkout')
    p.add_argument('--out', type=checked, required=True)
    a = p.parse_args()
    if a.out.exists(): raise ValueError('Refuse existing fast asset directory')
    if fusion_profile(json.loads((a.bundle/'provenance.json').read_bytes())) != 'none':
        raise ValueError('Fast mode supports only the default profile')
    _, _, input_to_program, _ = configure_model(a.model_root)
    def encode(source):
        words = input_to_program(io.StringIO(source)).assemble()
        return struct.pack('<'+'I'*len(words), *words)
    image = Image(a.elf.read_bytes())
    contract_raw = (a.bundle/'host/smol-full-contract.json').read_bytes()
    contract = json.loads(contract_raw)
    weights = {v['address']: k for k, v in image.symbols.items() if k.startswith('smol_full_weights_')}
    panels = image.symbols['smol_full_panels']['address']
    a.out.mkdir(parents=True)
    defaults, programs, rows, tables = {}, {}, [], []
    externs, images = ['#include "smol_fast_assets.h"'], ['.section .rodata.smol_fast,"a",@progbits']
    for i, item in enumerate(contract['kernels']):
        m, k, n = item['physical_mkn']
        key = (m, k, n, item['output_offset'], item['output_rescale_exponent'])
        flags = dict(traffic_reuse=item['recipe'] == 'traffic-reuse-v1', resident_b=item['recipe'] == 'baseline-v1',
                     pretranspose_resident_b=item['recipe'] == 'baseline-v1')
        if key not in defaults:
            source = default_relocate(assembly(*key, **flags), DEFAULT_ARENA, item['dram_base'])
            defaults[key] = source.encode(), encode(source)
        default_source, default_program = defaults[key]
        if (default_source != (a.bundle/item['source']).read_bytes()
                or default_program != (a.bundle/item['original']).read_bytes()):
            raise ValueError(f'Launch {i} default program differs from the bundle')
        panel = image.bytes_at(panels + PANEL_BYTES*i, PANEL_BYTES)
        base, size = struct.unpack_from('<Q', panel, 8)[0], struct.unpack_from('<Q', panel, 112)[0]
        if size != k*n or image.bytes_at(base, size) != (a.bundle/item['weights']).read_bytes():
            raise ValueError(f'Launch {i} linked weights differ from the bundle')
        ws = WeightSource(base, size, item['dram_base'])
        ws.validate(k, n)
        variant = key + (base,)
        if variant not in programs:
            source = assembly(*key, **flags, weight_source=ws)
            source = relocate_assembly(source, DEFAULT_ARENA, item['dram_base'], weight_source=ws)
            raw = encode(source)
            if len(raw) + 20 > IMEM_BYTES: raise ValueError(f'Launch {i} fast program exceeds IMEM')
            v = len(programs)
            name, symbol = f'program-{v:04d}', f'smol_fast_program_{v:04d}'
            (a.out/(name+'.S')).write_text(source)
            (a.out/(name+'.bin')).write_bytes(raw)
            programs[variant] = v, source, raw
            images += ['.balign 4', '.global '+symbol, '.type '+symbol+',@object', symbol+':',
                       '.incbin '+json.dumps(str(a.out/(name+'.bin'))), '.size '+symbol+',.-'+symbol]
            externs.append('extern const uint32_t '+symbol+'[];')
        v, source, raw = programs[variant]
        symbol = f'smol_fast_program_{v:04d}'
        tables.append(f'{{{symbol},{len(raw)//4},UINT64_C({base})}}')
        rows.append(dict(index=i, variant=v, weight_symbol=weights[base], program_symbol=symbol,
                         source_sha256=sha(source.encode()), program_sha256=sha(raw),
                         default_program_sha256=sha(default_program),
                         recipe=_reuse_module(m, k, n) if flags['traffic_reuse'] else 'base',
                         weight_source=ws.metadata((15, 20))))
    count = len(contract['kernels'])
    table = f'const smol_fast_panel smol_fast_panels[{count}]={{\n'+',\n'.join(tables)+'\n};\n'
    (a.out/'smol_fast_assets.c').write_text('\n'.join(externs)+'\n'+table)
    (a.out/'smol_fast_images.S').write_text('\n'.join(images)+'\n')
    (a.out/'relocations.json').write_text(json.dumps(dict(status='GENERATED', input_elf_sha256=sha(image.raw),
        default_source_exact=len(rows), default_program_exact=len(rows), distinct_default_variants=len(defaults),
        distinct_fast_variants=len(programs), owner_contract_sha256=sha(contract_raw), launches=rows), indent=2)+'\n')
    print(json.dumps(dict(launches=len(rows), default_variants=len(defaults), fast_variants=len(programs))))


if __name__ == '__main__':
    main()
