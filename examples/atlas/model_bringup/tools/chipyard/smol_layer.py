"""Build a fixed live selected Smol layer from checksum-bound retained sources.

This builder preserves the retained nonlinear C and math support. Only raw
programs, capture checking, and host reporting change. It never runs a device.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools/host_iss'))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_program import assembly, dma_load_bytes
from kernel_library import code_sources
from elf_image import Image, mmio_sites
from bundle import ChecksumBundle
from firesim_target import package as package_firesim
from arena import add_arguments, audit, relocate_assembly, window, DEFAULT_ARENA

ROLES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
RETAINED = ('smol_vlm_block_prototype', 'host_matrix_fp8pack', 'smol_host_ops', 'smol_host_rope',
            'smol_host_softmax', 'smol_host_attention', 'smol_host_controls', 'smol_selected_workload')
HEADERS = ('smol_host_bits.h', 'smol_selected_assets.h', 'smol_selected_workload.h',
           'host_matrix_fp8pack.h', 'smol_vlm_block_prototype.h', 'smol_host_ops.h',
           'smol_host_rope.h', 'smol_host_softmax.h', 'smol_host_attention.h', 'smol_host_controls.h')


def sha(raw): return hashlib.sha256(raw).hexdigest()
def write_json(path, value): checked(path).write_text(json.dumps(value, indent=2) + '\n')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('expected exactly one patch target: ' + old)
    return source.replace(old, new)


def build(args):
    out = checked(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source, data, objects = (out / n for n in ('source', 'data', 'objects'))
    for p in (source, data, objects): p.mkdir()
    pins = {}
    def read(path, digest=None):
        path = checked(path)
        raw = path.read_bytes()
        observed = sha(raw)
        if digest is not None and observed != digest: raise ValueError('changed input: ' + str(path))
        if str(path) in pins and pins[str(path)] != observed: raise ValueError('changed input during build')
        pins[str(path)] = observed
        return raw
    def copy(path, dest, digest=None):
        raw = read(path, digest); checked(dest).write_bytes(raw); return raw
    receipt = dict(status='FAIL_BUILD', full_soc_executed=False, host_elf_executed=False,
                   scope='one live selected layer with retained C nonlinear operations and seven raw reuse projections',
                   commands=[])
    started = time.monotonic()
    try:
        read(__file__)
        read(Path(__file__).with_name("arena.py"))
        arena_base = DEFAULT_ARENA if args.arena_base is None else args.arena_base
        window(arena_base, 3625472)
        if args.target == 'firesim': read(Path(__file__).with_name('firesim_target.py'))
        for name, digest in code_sources(traffic_reuse=True).items():
            read(ROOT/kernel_library.relative_path(name), digest)
        for name in ('atlas_path_guard.py','atlas_tooling.py','host_iss/elf_image.py','host_iss/bundle.py'):
            read(ROOT/kernel_library.relative_path(name))
        git = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'],
                             check=True, capture_output=True, text=True)
        if git.stdout: raise ValueError('commit all source changes before build')
        receipt['source_commit'] = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        bundle = ChecksumBundle(checked(args.bundle))
        contract_raw = bundle.read('host/smol-epoch3-contract.json')
        contract = json.loads(contract_raw)
        old_build = json.loads(bundle.read('host/build.json'))
        old_pins = {r['path']: r['sha256'] for r in old_build['source_inputs']}
        results = json.loads(read(args.results))
        retained_programs = None
        if args.program_build:
            retained_programs = json.loads(read(args.program_build))
            if (retained_programs['status'] != 'PASS_BUILD_STATIC_ONLY'
                    or retained_programs['source_bookends_equal'] is not True
                    or len(retained_programs['captures']) != 7):
                raise ValueError('successful seven-projection program build required')
            receipt['retained_program_build'] = str(args.program_build)
        frozen = json.loads(read(results['frozen_manifest']['path'], results['frozen_manifest']['sha256']))
        if frozen['artifacts'].get(str(checked(args.bundle) / 'host/smol-epoch3-contract.json')) != sha(contract_raw):
            raise ValueError('selected contract is not bound by frozen qualification')
        families = {}
        for pair in results['pairs']:
            if pair['status'] != 'QUALIFIED': raise ValueError('unqualified shape')
            for family in pair['family']: families[family['ordinal']] = (pair, family)
        if sorted(families) != list(range(7)): raise ValueError('seven captured launches required')
        retained = checked(args.retained)
        def retained_copy(relative, target):
            p = retained / relative
            return copy(p, target, old_pins[str(p)])
        for name in RETAINED:
            retained_copy('source/' + name + '.c', source / (name + '.c'))
        for name in HEADERS:
            if name == 'smol_host_bits.h':
                copy(ROOT / 'runtime' / name, source / name)
            else:
                retained_copy('source/' + name, source / name)
        retained_copy('source/atlas_resnet_graph_math.c', source/'atlas_resnet_graph_math.c')
        for name in ('rv64_support.c', 'expf_bf16_values.inc'):
            retained_copy('elf/support/' + name, source / name)
        (source / 'include').mkdir()
        for name in ('math.h', 'string.h', 'fenv.h'):
            retained_copy('elf/support/include/' + name, source / 'include' / name)
        for name in ('atlas_host.c', 'atlas_host.h', 'atlas_host_layout.h', 'atlas_host_check.h',
                     'smol_block_raw_adapter.c', 'smol_block_raw_adapter.h', 'smol_block_capture.h'):
            copy(ROOT / 'runtime' / name, source / name)
        copy(ROOT / 'tools/chipyard/smol_layer_host.c', source / 'smol_layer_host.c')
        for name in ('start.S', 'link.ld'): retained_copy('elf/' + name, source / name)
        if arena_base != DEFAULT_ARENA:
            header = source / 'atlas_host_layout.h'
            header.write_text(replace_once(header.read_text(), '#define ATLAS_DRAM_BASE 0x100000000',
                              f'#define ATLAS_DRAM_BASE {arena_base:#x}'))
            link = source / 'link.ld'
            link_text = replace_once(link.read_text(), '.atlas_dram 0x100000000', f'.atlas_dram {arena_base:#x}')
            link.write_text(replace_once(link_text, '__atlas_dram_end <= 0x200000000',
                            f'__atlas_dram_end <= {((arena_base >> 32) + 1) << 32:#x}'))
        # Retain immutable nonlinear/final-reference data and parameter declarations.
        for name, asset in contract['assets'].items():
            raw = bundle.read('data/' + name + '.bin')
            if sha(raw) != asset['file']['sha256']: raise ValueError('nonlinear reference changed')
            (data / (name + '.bin')).write_bytes(raw)
        assets = retained_copy('source/smol_selected_assets.c', source / 'smol_selected_assets.c').decode()
        model_root = checked(args.model_root)
        model_pins = {p:d for p,d in frozen['sources'].items() if Path(p).is_relative_to(model_root)}
        if not model_pins: raise ValueError('frozen model sources missing')
        for p,digest in model_pins.items(): read(p,digest)
        sys.path.insert(0,str(model_root))
        from npu_model.configs import isa_definition  # Register assembler metadata.
        from npu_model.util.converter import input_to_program as assemble
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        rows, captures, image_lines = [], [], ['.section .rodata.smol_assets,"a",@progbits']
        def blob(label, name, raw):
            (data / name).write_bytes(raw)
            image_lines.extend(['.balign 64', '.global ' + label, label + ':', '.incbin "data/' + name + '"'])
        for name in contract['assets']:
            blob('smol_asset_' + name, name + '.bin', (data / (name + '.bin')).read_bytes())
        for index, projection in enumerate(contract['ordered_projections']):
            if projection['index'] != index or projection['role'] != ROLES[index]:
                raise ValueError('model order changed')
            pair, family = families[index]
            m, k, n = pair['shape']
            exponent = projection['activation_scale'] + projection['weight_scale']
            if exponent != family['rescale_exponent']: raise ValueError('rescale mismatch')
            initial = bundle.read(f'kernels/{index:02d}/initial-dram.bin')
            expected_output = bundle.read(f'kernels/{index:02d}/expected-output.bin')
            if sha(initial) != family['input_sha256'] or sha(expected_output) != family['expected_sha256']:
                raise ValueError('capture differs from qualified numerical family')
            offset = projection['output_offset']
            if (m, k, n) != (192, projection['input_width'], projection['output_width']):
                raise ValueError('shape mismatch')
            if offset != m*k+k*n+256 or len(initial) != offset+2*m*n+256: raise ValueError('DRAM layout mismatch')
            if len(expected_output) != 2*m*n: raise ValueError('physical output extent mismatch')
            expected = initial[:offset] + expected_output + initial[offset+len(expected_output):]
            if retained_programs:
                capture = retained_programs['captures'][index]
                if (capture['index'] != index or capture['role'] != ROLES[index]
                        or capture['shape'] != [m,k,n] or capture['exponent'] != exponent):
                    raise ValueError('retained program geometry differs')
                previous = args.program_build.parent / 'data'
                raw = read(previous / f'program_{index}.bin', capture['program_sha256'])
                words = [w[0] for w in struct.iter_unpack('<I', raw)]
                asm_path = previous / f'program_{index}.S'
                asm = read(asm_path, retained_programs['generated_inputs'][str(asm_path)]).decode()
                if b''.join(struct.pack('<I',w & 0xffffffff) for w in assemble(io.StringIO(asm)).assemble()) != raw:
                    raise ValueError('retained assembly and binary differ')
            else:
                asm = assembly(m, k, n, offset, exponent, traffic_reuse=True)
                words = assemble(io.StringIO(asm)).assemble()
                raw = b''.join(struct.pack('<I', w & 0xffffffff) for w in words)
            if sha(raw) != family['proofs']['candidate']['program_sha256']:
                raise ValueError('reuse generator differs from receipt')
            if arena_base != DEFAULT_ARENA:
                asm = relocate_assembly(asm, DEFAULT_ARENA, arena_base)
                words = assemble(io.StringIO(asm)).assemble()
                raw = b''.join(struct.pack('<I', w & 0xffffffff) for w in words)
            if len(words) > 8192 or words[-1] != 0x73:
                raise ValueError('program capacity/terminal failed')
            rescale = [i for i, word in enumerate(words) if word == ((exponent+127)<<23)|0x25f]
            if len(rescale) != 1: raise ValueError('rescale literal must be unique')
            weights = initial[m*k:m*k+k*n]
            if sha(weights) != projection['weights']['sha256']: raise ValueError('retained weights changed')
            blob(f'smol_asset_program_{index}', f'program_{index}.bin', raw)
            blob(f'smol_asset_weights_{index}', f'weights_{index}.bin', weights)
            blob(f'smol_expected_{index}', f'expected_{index}.bin', expected)
            (data / f'program_{index}.S').write_text(asm)
            (data / f'initial_{index}.bin').write_bytes(initial)
            (data / f'output_{index}.bin').write_bytes(expected_output)
            rows.append('{(const uint32_t *)smol_asset_program_%d,smol_asset_weights_%d,'
                        '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}' %
                        (index,index,len(words),rescale[0],k,n,m*k,k*n,2*m*n,offset,len(initial),
                         projection['activation_scale'],projection['weight_scale']))
            captures.append(dict(index=index, role=ROLES[index], shape=[m,k,n], exponent=exponent,
                rescale_word=rescale[0], program_words=len(words), wrapped_words=len(words)+5,
                program_sha256=sha(raw), initial_sha256=sha(initial), expected_sha256=sha(expected),
                expected_output_sha256=sha(expected_output), dram_base=arena_base,
                dram_bytes=len(initial), output_offset=offset,
                requested_dma=dict(load_bytes=dma_load_bytes(m,k,n,traffic_reuse=True),store_bytes=2*m*n)))
        start = assets.index('const smol_raw_kernel smol_kernels[7]={')
        end = assets.index('};', start)+2
        assets = assets[:start] + 'const smol_raw_kernel smol_kernels[7]={\n' + ',\n'.join(rows) + '\n};' + assets[end:]
        assets += '\n' + '\n'.join(f'extern const uint8_t smol_expected_{i}[];' for i in range(7))
        assets += ('\nconst uint8_t *const smol_capture_expected[7]={'
                   + ','.join(f'smol_expected_{i}' for i in range(7)) + '};\n')
        (source / 'smol_selected_assets.c').write_text(assets)
        (source / 'images.S').write_text('\n'.join(image_lines)+'\n')
        # Freeze all generated build inputs before invoking the compiler.
        generated = {}
        for directory in (source, source/'include', data):
            for p in directory.iterdir():
                if p.is_file(): generated[str(p)] = sha(p.read_bytes())
        write_json(out/'frozen.json', dict(source_commit=receipt['source_commit'], external_inputs=pins,
            generated_inputs=generated, captures=captures, original_contract_sha256=sha(contract_raw),
            bundle_sha256sums_sha256=bundle.checksums_sha256))
        compiler, linker = checked(args.compiler), checked(args.linker)
        read(compiler); read(linker)
        flags = [str(compiler),'--target=riscv64-unknown-elf','-march=rv64gc','-mabi=lp64d','-mcmodel=medany',
                 '-std=c11','-O2','-Wall','-Wextra','-Werror','-ffp-contract=off','-fno-fast-math','-ffreestanding',
                 '-fno-builtin','-fno-stack-protector','-fno-pic','-fno-vectorize','-fno-slp-vectorize',
                 '-DATLAS_SMOL_CAPTURE_CHECK=1','-I'+str(source/'include'),'-I'+str(source)]
        compiled = []
        for name in (*RETAINED, 'atlas_host', 'smol_block_raw_adapter', 'smol_selected_assets',
                     'smol_layer_host', 'rv64_support', 'start', 'images'):
            src = source/(name+('.S' if name in ('start','images') else '.c'))
            obj = objects/(name+'.o'); compiled.append(str(obj))
            cmd = flags+['-c',str(src),'-o',str(obj)]
            receipt['commands'].append(cmd)
            run = subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=300)
            (objects/(name+'.log')).write_text(run.stdout+run.stderr)
            if run.returncode: raise RuntimeError('compile failed: '+name+'\n'+run.stderr)
        cmd = [str(linker),'-flavor','gnu','-m','elf64lriscv',*compiled,'-T',str(source/'link.ld'),
               '--no-relax','-Map',str(out/'host.map'),'-o',str(out/'host.elf')]
        receipt['commands'].append(cmd)
        run = subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=300)
        (out/'link.log').write_text(run.stdout+run.stderr)
        if run.returncode: raise RuntimeError('link failed: '+run.stderr)
        image = Image((out/'host.elf').read_bytes())
        if image.entry != 0x80000000: raise ValueError('unexpected entry')
        arena = image.symbols['__atlas_dram_start']['address']
        extent = image.symbols['__atlas_dram_end']['address']-arena
        if arena != arena_base or extent != 3625472: raise ValueError('arena mismatch')
        if image.symbols['__stack_top']['address'] >= 0x90000000: raise ValueError('host memory overflow')
        for c in captures:
            i = c['index']
            for symbol, filename in ((f'smol_asset_program_{i}',f'program_{i}.bin'),
                                     (f'smol_expected_{i}',f'expected_{i}.bin')):
                raw=(data/filename).read_bytes()
                if image.bytes_at(image.symbols[symbol]['address'],len(raw)) != raw:
                    raise ValueError('ELF asset differs')
        for name,asset in contract['assets'].items():
            raw=(data/(name+'.bin')).read_bytes()
            if image.bytes_at(image.symbols['smol_asset_'+name]['address'],len(raw)) != raw:
                raise ValueError('ELF layer asset differs')
        dram_audit = audit(out, args.dram_base, args.dram_bytes)
        bundle.verify_bookends()
        for p,digest in {**pins,**generated}.items():
            if sha(checked(p).read_bytes()) != digest: raise ValueError('source bookend changed')
        if args.target == 'firesim': package_firesim(out, 'atlas-smol-selected-layer', args.dram_base, args.dram_bytes)
        receipt.update(status='PASS_BUILD_STATIC_ONLY', elf_sha256=sha(image.raw), captures=captures,
            target=args.target, physical_dram=dram_audit,
            source_bookends_equal=True, sources=pins, generated_inputs=generated,
            mmio_sites=mmio_sites(image), static=dict(imem_words=32768, software_program_words=8192,
              max_wrapped_words=max(c['wrapped_words'] for c in captures), dram_base=arena, dram_capacity=extent,
              host_stack_top=image.symbols['__stack_top']['address'], live_nonlinear_C_retained=True,
              full_physical_output_inputs_guards_checked=True, final_output_key_value_checks_retained=True))
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started
        write_json(out/'build.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','elf_sha256','source_commit','wall_seconds')}))


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out','retained','bundle','results','model-root','compiler','linker'):
        p.add_argument('--'+name,type=checked,required=True)
    p.add_argument('--target', choices=('sim', 'firesim'), default='sim')
    p.add_argument('--program-build', type=checked,
                   help='retain hash-bound programs from a previous build when generator HEAD has advanced')
    add_arguments(p)
    return p.parse_args()

if __name__=='__main__': build(arguments())
