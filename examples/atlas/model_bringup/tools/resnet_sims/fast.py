#!/usr/bin/env python3
"""Bind the ResNet50 immutable-weight fast programs to one platform's linked ELF.

Every run starts from a fresh package copy and programs generated against
placeholder weight bases, or the bases of an archived --seed-relocations
record. The package is then rebuilt and regenerated against the linked B
symbols until the ELF is stable. The bound layout is a fixed point of the link,
and a link can have more than one: linker relaxation near the global pointer
couples program lengths to the text size. Seeding reproduces an archived
binding exactly.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).absolute().parent
sys.path.insert(0, str(PACKAGE.parent))
import kernel_library  # noqa: E402,F401
from atlas_epilogue import Epilogue, Step, TileInput
from atlas_path_guard import checked
from atlas_program import assembly, legacy_epilogue_assembly
from atlas_resnet_input import DirectInput, relocate_assembly
from atlas_resnet_pool import MaxPool
from atlas_weight_source import WeightSource
from host_iss.elf_image import Image
from resnet_sims.build import PLATFORMS, build, copy_tree, prepare_platform_root

ARENA_BASE = 0x90000000
SEED_BASE = 0x84000000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source(launch, weight_source=None):
    """Emit one launch with the capture's archived x30/x31 hook scratch."""
    if launch.get('max_pool'):
        return MaxPool(**launch['max_pool']).assembly(ARENA_BASE)
    ep = launch['epilogue']
    if ep:
        steps = tuple(Step(s['op'], TileInput(**s['operand']) if s['operand'] else None) for s in ep['steps'])
        ep = Epilogue(**dict(ep, scratch_registers=tuple(ep.get('scratch_registers', (30, 31))), steps=steps))
    direct = DirectInput(**launch['direct_input']) if launch.get('direct_input') else None
    args = (launch['m'], launch['k'], launch['n'], launch['output_offset'], launch['a_scale_exponent']+launch['b_scale_exponent'])
    text = (legacy_epilogue_assembly(*args, epilogue=ep, direct_input=direct, weight_source=weight_source) if ep
            else assembly(*args, traffic_reuse=True, direct_input=direct, weight_source=weight_source))
    return relocate_assembly(text, ARENA_BASE, arena_bytes=launch['arena_bytes'], weight_source=weight_source)


def words(text, input_to_program):
    program = input_to_program(io.StringIO(text)).assemble()
    return struct.pack('<'+'I'*len(program), *program)


def generate(model, package, generated, input_to_program, elf=None, bases=None):
    """Write fast programs and the fast tables for the weight bases linked in elf, or for seed bases."""
    initial_symbols = json.loads((package/'package.json').read_text())['initial_symbols']
    image = Image(elf.read_bytes()) if elf else None
    generated.mkdir(exist_ok=True)
    rows = []
    for launch, symbol in zip(model['launches'], initial_symbols, strict=True):
        index = launch['index']
        baseline = source(launch)
        if (baseline.encode() != checked(launch['source']).read_bytes()
                or words(baseline, input_to_program) != checked(launch['program']).read_bytes()):
            raise ValueError(f'default program {index} differs from the capture')
        base = (image.symbols[symbol]['address']+launch['weight_offset'] if image
                else bases[index] if bases else SEED_BASE+64*index)
        ws = WeightSource(base, launch['k']*launch['n'], ARENA_BASE) if launch['k'] else None
        text = source(launch, ws)
        raw = words(text, input_to_program)
        if len(raw)+20 > 131072: raise ValueError(f'fast program {index} exceeds IMEM')
        (generated/f'program-{index:02}.S').write_text(text)
        (package/'data'/f'fast-program-{index:02}.bin').write_bytes(raw)
        rows.append(dict(index=index, initial_symbol=symbol, weight_offset=launch['weight_offset'], weight_base=base,
                         weight_bytes=ws.size_bytes if ws else 0, program_sha256=sha(raw), source_sha256=sha(text.encode()), words=len(raw)//4))
    write_tables(package, rows)
    (generated/'relocations.json').write_text(json.dumps(dict(elf_sha256=sha(image.raw) if image else None, launches=rows), indent=2)+'\n')
    return rows


def write_tables(package, rows):
    assets = package/'assets.c'
    text = re.sub(r'(m2_fast_program_(\d+),)\d+,',
                  lambda m: m[1]+str((package/'data'/f'fast-program-{int(m[2]):02}.bin').stat().st_size)+',', assets.read_text())
    begin = text.index('const m2_fast_launch m2_fast_launches[] = {')
    end = text.index('\n};', begin)+3
    table = [f"  {{{r['initial_symbol']}+{r['weight_offset']},{r['weight_offset']},{r['weight_bytes']},0x{r['weight_base']:x}UL}},"
             for r in rows]
    assets.write_text(text[:begin]+'\n'.join(['const m2_fast_launch m2_fast_launches[] = {', *table, '};'])+text[end:])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--package', type=checked, required=True, help='Directory written by package.py; left unchanged')
    p.add_argument('--model', type=checked, required=True, help='The captured model.json that package.py consumed')
    p.add_argument('--npu-model', type=checked, required=True, help='npu_model checkout providing the assembler')
    p.add_argument('--platform-root', type=checked, required=True, help='board-support source tree with the penguin26 platform')
    p.add_argument('--toolchain', type=checked, required=True, help='RISC-V GCC toolchain bin directory')
    p.add_argument('--work', type=checked, required=True, help='Fresh work directory outside the inputs')
    p.add_argument('--platform', choices=PLATFORMS, required=True)
    p.add_argument('--seed-relocations', type=checked, help='relocations.json of an archived binding to reproduce')
    p.add_argument('--attempts', type=int, default=6)
    a = p.parse_args()
    sys.path.insert(0, str(a.npu_model))
    from npu_model.util.converter import input_to_program
    model_raw = a.model.read_bytes()
    if sha(model_raw) != json.loads((a.package/'package.json').read_text())['model_sha256']:
        raise ValueError('model differs from the packaged capture')
    model = json.loads(model_raw)
    if a.work.exists(): raise ValueError('work directory must not already exist')
    a.work.mkdir(parents=True)
    package, generated = a.work/'package', a.work/'generated'
    copy_tree(a.package, package)
    platform_root = prepare_platform_root(a.platform_root, a.work/'platform_root')
    seed = json.loads(a.seed_relocations.read_text())['launches'] if a.seed_relocations else None
    bases = [row['weight_base'] if 'weight_base' in row else row['weight_source']['base'] for row in seed] if seed else None
    rows = generate(model, package, generated, input_to_program, bases=bases)
    if seed and any(row['program_sha256'] != old['program_sha256'] for row, old in zip(rows, seed, strict=True)):
        raise ValueError('seed relocations do not reproduce their archived programs')
    for attempt in range(1, a.attempts+1):
        elf = build(package, platform_root, a.work, a.platform, a.toolchain, fast=True)
        before = sha(elf.read_bytes())
        generate(model, package, generated, input_to_program, elf)
        elf = build(package, platform_root, a.work, a.platform, a.toolchain, fast=True)
        if sha(elf.read_bytes()) == before:
            print(json.dumps(dict(platform=a.platform, attempts=attempt, elf=str(elf), elf_sha256=before)))
            return
    raise RuntimeError('weight binding did not reach a stable link')


if __name__ == '__main__':
    main()
