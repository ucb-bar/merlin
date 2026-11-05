"""Build the smallest complete captured-weight TinyLlama block, without simulation.

One logical token, no past cache, all twelve physical M32 projection panels.
Pinned nonlinear host C stays live, with packed weights in disjoint LOADMEM regions.
"""
import argparse
import ctypes
import io
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_program import assembly, dma_load_bytes
from kernel_library import code_sources
from atlas_tooling import sha, write_json as write
from elf_image import Image, mmio_sites
from arena import add_arguments, audit, relocate_assembly, window, DEFAULT_ARENA


# These bind the qualified retained nonlinear implementation and captured TinyLlama
# recipe, including original weights and geometry, rather than a local output path.
RETAINED_SHA256 = 'c86e2880bb861308c9d3c405e429d3acdd1c4e53157408b3d987a0df02e64bb2'
RECIPE_SHA256 = 'bba8a77999e68517574958bf6aee20537c674e52d31af228b28b563240a471f2'


def loadmem_image(initial, activation_bytes):
    """Poison only the LOADMEM activation prefix; keep live model captures separate."""
    if type(activation_bytes) is not int or not 0 < activation_bytes <= len(initial):
        raise ValueError('activation prefix must be a positive byte count within the image')
    return b'\x96' * activation_bytes + initial[activation_bytes:]


def build(a):
    arena_base = DEFAULT_ARENA if a.arena_base is None else a.arena_base
    window(arena_base, 1)
    out = checked(a.out)
    out.mkdir(exist_ok=False)
    src = out / 'source'
    src.mkdir()
    (src / 'include').mkdir()
    data = out / 'data'
    data.mkdir()
    obj = out / 'objects'
    obj.mkdir()
    pins = {}
    build_context = {}

    def read(p):
        p = checked(p)
        b = p.read_bytes()
        d = sha(b)
        if str(p) in pins and pins[str(p)] != d:
            raise ValueError(f'source changed during build: {p}')
        pins[str(p)] = d
        return b

    def cp(p, q):
        raw = read(p)
        if 'retained' in build_context and Path(p).is_relative_to(build_context['retained']):
            relative = str(Path(p).relative_to(build_context['retained']))
            if sha(raw) != build_context['outputs'][relative]:
                raise ValueError(f'retained source hash mismatch: {relative}')
        q.write_bytes(raw)

    receipt = {
        'status': 'FAIL_BUILD',
        'base_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
    }
    try:
        read(__file__)
        for module in ('arena', 'elf_image', 'atlas_tooling', 'atlas_path_guard'):
            read(Path(sys.modules[module].__file__))
        for n, d in code_sources(traffic_reuse=True).items():
            if sha(read(ROOT / kernel_library.relative_path(n))) != d:
                raise ValueError(f'program source hash mismatch: {n}')
        retained = checked(a.retained)
        retained_raw = read(retained / 'build.json')
        if sha(retained_raw) != RETAINED_SHA256:
            raise ValueError('retained build does not match the qualified TinyLlama implementation')
        retained_outputs = json.loads(retained_raw)['outputs_sha256']
        build_context.update(retained=retained, outputs=retained_outputs)
        recipe_raw = read(a.recipe)
        if sha(recipe_raw) != RECIPE_SHA256:
            raise ValueError('recipe does not match the qualified TinyLlama capture')
        recipe = json.loads(recipe_raw)
        read(Path(shutil.which('cc')).resolve())
        names = ('tiny_block', 'llama_host_attention', 'llama_host_layout', 'llama_rotary_table',
                 'llama_host_bmm', 'host_matrix_fp8pack', 'smol_host_ops', 'smol_host_softmax')
        for stem in names:
            for ext in ('.c', '.h'):
                cp(retained / (stem + ext), src / (stem + ext))
        for n in ('atlas_resnet_graph_math.c', 'expf_bf16_values.inc', 'rv64_support.c'):
            cp(retained / n, src / n)
        for n in ('math.h', 'string.h', 'fenv.h'):
            cp(retained / 'include' / n, src / 'include' / n)
        for n in ('atlas_host.c', 'atlas_host.h', 'atlas_host_layout.h', 'atlas_host_check.h',
                  'tiny_projection.h', 'smol_host_bits.h', 'atlas_numeric.h'):
            cp(ROOT / 'runtime' / n, src / n)
        cp(ROOT / 'tools/chipyard/tiny_block_host.c', src / 'block_target.c')
        cp(ROOT / 'tools/chipyard/tiny_block_ref.c', src / 'reference.c')
        cp(ROOT / 'runtime/host_templates/tiny_start.S', src / 'start.S')
        # Specialize retained prefill128/decode1 geometry to position 0 with no past.
        t = (src / 'tiny_block.c').read_text()
        t = t.replace('128 *', '1 *').replace('[128]', '[1]').replace('32 * 128 * 128', '32')
        t = (t.replace('3014656', '16384').replace('32 * 1 * 128', '32').replace('softmax[129]', 'softmax[1]')
             .replace('w->softmax, 129', 'w->softmax, 1').replace('4 * 129 * 64', '256'))
        t = t.replace('(rows != 128 && rows != 1)', 'rows != 1')
        t = t.replace('|| (rows == 1 && (!past_keys || !past_values))', '|| (past_keys || past_values)')
        t = t.replace('|| (rows == 128 && (past_keys || past_values))', '')
        t = t.replace('(int64_t)(rows == 128 ? i : 128)', '(int64_t)i').replace('rows == 1 ? 128 : 0', '0')
        (src / 'tiny_block.c').write_text(t)
        # Use shared union conversions for the retained Llama bit copies.
        layout = (src / 'llama_host_layout.c').read_text()
        old = """    uint32_t bits = (uint32_t)value << 16;
    float output;
    memcpy(&output, &bits, sizeof(output));
    return output;"""
        if layout.count(old) != 1:
            raise ValueError('expected exactly one retained BF16-to-float helper')
        layout = layout.replace(old, '    return bf16_to_float(value);')
        old = """    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000))
        return (uint16_t)((bits >> 16) | UINT32_C(0x40));
    return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16) & 1u)) >> 16);"""
        if layout.count(old) != 1:
            raise ValueError('expected exactly one retained float-to-BF16 helper')
        layout = layout.replace(old, '    return float_to_bf16(value);')
        (src / 'llama_host_layout.c').write_text(
            layout.replace('#include <string.h>', '#include <string.h>\n#include "smol_host_bits.h"'))
        # The aligned one-token fast path skips bounds checks on 31 zero rows.
        # Generic validation and the unaligned fallback remain unchanged.
        pack = (src / 'host_matrix_fp8pack.c').read_text()
        fast = """    if (rows == 1 && physical_rows == 32 && columns == physical_columns
        && tile_rows == 32 && tile_columns == 32 && !((uintptr_t)output & 7u)) {
        typedef uint64_t tiny_pack_word __attribute__((__may_alias__));
        for (size_t c = 0; c < columns; c += 32) {
            uint8_t *tile = output + c * 32;
            for (size_t j = 0; j < 32; ++j) tile[j] = pack_bf16(input[c+j], scale);
            tiny_pack_word *zero = (tiny_pack_word *)(tile + 32);
            for (size_t j = 0; j < 124; j += 4) {
                zero[j] = 0; zero[j+1] = 0; zero[j+2] = 0; zero[j+3] = 0;
            }
        }
        return 0;
    }
"""
        if pack.count('    size_t cursor = 0;') != 1:
            raise ValueError('expected exactly one retained pack cursor')
        (src / 'host_matrix_fp8pack.c').write_text(
            pack.replace('    size_t cursor = 0;', fast + '    size_t cursor = 0;'))
        # Native retained host C surrounds an independent scalar FP8/BF16 oracle.
        cmd = ['cc', '-shared', '-fPIC', '-O2', '-std=c11', '-ffp-contract=off', '-fno-fast-math',
               '-DSMOL_HOST_EXCLUDE_MIXED_RESIDUAL', '-I' + str(src),
               *[str(src / (n + '.c')) for n in names], str(src / 'reference.c'),
               '-lm', '-o', str(out / 'reference.so')]
        p = subprocess.run(cmd, capture_output=True, text=True)
        (out / 'native.log').write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            raise ValueError(f'compiler or linker failed: {p.stderr}')
        lib = ctypes.CDLL(str(out / 'reference.so'))
        P = ctypes.c_void_p
        Z = ctypes.c_size_t
        lib.host_matrix_scale_bf16.argtypes = [P, Z, Z, P]
        lib.host_matrix_pack_bf16_at_scale.argtypes = [P, Z, Z, Z, Z, Z, Z, ctypes.c_int, P, Z]
        lib.tiny_reference.argtypes = [P, P, Z, Z, ctypes.c_int, P]
        lib.tiny_block_workspace_bytes.restype = Z
        lib.tiny_block_run.argtypes = [Z, P, P, P, ctypes.c_float, P, P, P, P, P, P, P, Z, P, P, P]

        def datum(name):
            r = recipe['data'][name]
            raw = read(r['path'])
            if sha(raw) != r['sha256']:
                raise ValueError(f'recipe data hash mismatch: {name}')
            return raw

        # Use the first original prompt token and unchanged norms/RoPE/weights.
        raws = {n: datum(n) for n in ('step0-entry', 'input-norm', 'post-norm', 'cosine', 'sine')}
        raws['step0-entry'] = raws['step0-entry'][:4096]
        buffers = {n: ctypes.create_string_buffer(b) for n, b in raws.items()}
        rows = []
        failures = []
        offset = 0
        projections = recipe['projections'][:12]
        import numpy as np
        sys.path.insert(0, str(checked(a.model_root)))
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from npu_model.util.converter import input_to_program

        def assemble(s):
            return b''.join(struct.pack('<I', x & 0xffffffff)
                            for x in input_to_program(io.StringIO(s)).assemble())

        CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, P, ctypes.c_uint, P, Z, Z, Z, P)

        def project(ctx, logical, input, logical_rows, k, width, output):
            nonlocal offset
            try:
                if logical_rows != 1:
                    raise ValueError('projection requires one logical row')
                scale = ctypes.c_int()
                if lib.host_matrix_scale_bf16(input, 1, k, ctypes.byref(scale)) != 0:
                    raise ValueError('activation scale computation failed')
                col = 0
                while col < width:
                    i = len(rows)
                    p = projections[i]
                    n = p['n']
                    if (p['logical'], p['k'], p['column'], p['width']) != (logical, k, col, width):
                        raise ValueError(f'projection geometry mismatch for panel {i}')
                    weights = read(p['weights']['path'])
                    if sha(weights) != p['weights']['sha256']:
                        raise ValueError(f'weight hash mismatch for panel {i}')
                    ar = ctypes.create_string_buffer(32 * k)
                    if lib.host_matrix_pack_bf16_at_scale(input, 1, k, 32, k, 32, 32, scale.value, ar, 32 * k) != 0:
                        raise ValueError('activation packing failed')
                    wr = ctypes.create_string_buffer(weights)
                    result = (ctypes.c_uint16 * n)()
                    exponent = scale.value + p['rescale_exponent'] - p['a_scale']
                    lib.tiny_reference(ar, wr, k, n, exponent, result)
                    physical = np.zeros((n // 32, 2, 32, 16), dtype='<u2')
                    r = np.ctypeslib.as_array(result)
                    physical[:, :, 0, :] = r.reshape(n // 32, 2, 16)
                    expected = physical.tobytes()
                    oo = 32 * k + k * n + 256
                    total = oo + len(expected) + 256
                    initial = ar.raw[:32 * k] + weights + b'\xa5' * 256 + b'\xcd' * len(expected) + b'\x5a' * 256
                    preload = loadmem_image(initial, 32 * k)
                    offset = (offset + 63) // 64 * 64
                    window(arena_base, offset + total)
                    original_asm = assembly(32, k, n, oo, exponent, traffic_reuse=True)
                    asm = relocate_assembly(original_asm, DEFAULT_ARENA, arena_base + offset, force=True)
                    program = assemble(asm)
                    if len(program) // 4 + 5 > 32768:
                        raise ValueError('wrapped program exceeds IMEM capacity')
                    # Model inputs retain live A; only LOADMEM starts with poison.
                    for name, b in [('initial', initial), ('loadmem', preload), ('expected', expected),
                                    ('program', program), ('weights', weights),
                                    ('activation', ar.raw[:32 * k]), ('program.S', asm.encode())]:
                        (data / f'{i}-{name}.bin').write_bytes(b)
                    row = dict(
                        index=i, role=('q', 'k', 'v', 'o', 'gate', 'up', 'down')[logical], shape=[32, k, n],
                        logical_rows=1, column=col, width=width, logical=logical, a_scale=scale.value,
                        exponent=exponent, dram_offset=offset, dram_base=arena_base + offset,
                        output_offset=oo, total_bytes=total, program_words=len(program) // 4,
                        program_kind='qualified_reuse_with_scalar_DMA_relocation', program_sha256=sha(program),
                        initial_sha256=sha(initial), loadmem_sha256=sha(preload), expected_sha256=sha(expected),
                        weights_sha256=sha(weights), activation_sha256=sha(ar.raw[:32 * k]),
                        requested_dma=dict(load_bytes=dma_load_bytes(32, k, n, traffic_reuse=True),
                                           store_bytes=len(expected)))
                    rows.append(row)
                    ctypes.memmove(output + col * 2, result, n * 2)
                    col += n
                    offset += total
                return 0
            except Exception as e:
                failures.append(repr(e))
                return -1
        callback = CALLBACK(project)
        workspace = ctypes.create_string_buffer(lib.tiny_block_workspace_bytes())
        ko = ctypes.create_string_buffer(512)
        vo = ctypes.create_string_buffer(512)
        bo = ctypes.create_string_buffer(4096)
        epsilon = recipe['parameters']['epsilon']
        code = lib.tiny_block_run(
            1, buffers['step0-entry'], buffers['input-norm'], buffers['post-norm'], epsilon,
            buffers['cosine'], buffers['sine'], None, None, callback, None, workspace, len(workspace), ko, vo, bo)
        if code != 0:
            raise ValueError(f'native Tiny block failed: code={code}, projection failures={failures}')
        if len(rows) != 12:
            raise ValueError('expected twelve physical projection panels')
        raws.update({'output': bo.raw[:4096], 'keys': ko.raw[:512], 'values': vo.raw[:512]})
        header = [
            '#include "tiny_projection.h"', 'extern const tiny_panel tiny_panels[12];',
            'extern const uint8_t *const tiny_panel_expected[12];',
            'extern const uint16_t *const tiny_entry[1],*const tiny_expected_output[1],'
            '*const tiny_expected_keys[1],*const tiny_expected_values[1];',
            'extern const uint16_t *const tiny_input_norm,*const tiny_post_norm,*const tiny_cosine,*const tiny_sine;',
            'extern const float tiny_epsilon;',
        ]
        (src / 'tiny_bundle_assets.h').write_text('\n'.join(header) + '\n')
        images = ['.section .rodata.assets,"a",@progbits']
        decl = []

        def blob(label, path, section=None):
            if section:
                images.append(section)
            images.extend(['.balign 64', f'.global {label}', f'{label}:', f'.incbin "{path}"'])
            decl.append(f'extern const unsigned char {label}[];')
        for name, raw in raws.items():
            path = data / (name + '.bin')
            path.write_bytes(raw)
            blob('asset_' + name.replace('-', '_'), path)
        for r in rows:
            i = r['index']
            blob(f'program_{i}', data / f'{i}-program.bin')
            blob(f'expected_{i}', data / f'{i}-expected.bin')
        for r in rows:
            i = r['index']
            blob(f'arena_{i}', data / f'{i}-loadmem.bin', f'.section .arena{i},"aw",@progbits')
        images.append('.section .note.GNU-stack,"",@progbits')
        (src / 'images.S').write_text('\n'.join(images) + '\n')
        text = '#include "tiny_bundle_assets.h"\n' + '\n'.join(decl) + '\nconst tiny_panel tiny_panels[12]={\n'
        for r in rows:
            i = r['index']
            k, n = r['shape'][1:]
            vals = [1, 32, k, n, r['column'], r['width'], r['logical'], r['a_scale']]
            text += ('{' + ','.join(map(str, vals))
                     + f",arena_{i}+{32 * k},{k * n},(const uint32_t*)program_{i},"
                     f"{r['program_words']},{r['output_offset']},{r['total_bytes']},"
                     f"{r['requested_dma']['load_bytes']},{2 * 32 * n}" + '},\n')
        text += ('};\nconst uint8_t *const tiny_panel_expected[12]={'
                 + ','.join(f'expected_{i}' for i in range(12)) + '};\n')
        for sym, n in [('tiny_entry', 'step0-entry'), ('tiny_expected_output', 'output'),
                       ('tiny_expected_keys', 'keys'), ('tiny_expected_values', 'values')]:
            text += f"const uint16_t *const {sym}[1]={{(const uint16_t*)asset_{n.replace('-', '_')}}};\n"
        for sym, n in [('tiny_input_norm', 'input-norm'), ('tiny_post_norm', 'post-norm'),
                       ('tiny_cosine', 'cosine'), ('tiny_sine', 'sine')]:
            text += f"const uint16_t *const {sym}=(const uint16_t*)asset_{n.replace('-', '_')};\n"
        text += f'const float tiny_epsilon={float(epsilon).hex()}f;\n'
        (src / 'tiny_bundle_assets.c').write_text(text)
        link = read(ROOT / 'runtime/host_templates/tiny_link.ld').decode()
        link = link.replace(
            'MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M }',
            f'MEMORY {{ RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M\n'
            f' DRAM (rw) : ORIGIN = 0x{arena_base:x}, LENGTH = 128M }}')
        sections = '\n'.join(f" .arena{i} 0x{r['dram_base']:x} : {{ *(.arena{i}) }} > DRAM"
                             for i, r in enumerate(rows))
        link = link.replace(' /DISCARD/', sections + '\n /DISCARD/')
        (src / 'link.ld').write_text(link)
        compiler = checked(a.compiler)
        linker = checked(a.linker)
        read(compiler)
        read(linker)
        flags = ['--target=riscv64-unknown-elf', '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany',
                 '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off', '-fno-fast-math',
                 '-ffreestanding', '-DATLAS_FAST_CHECK=1', '-DSMOL_HOST_EXCLUDE_MIXED_RESIDUAL',
                 '-I' + str(src / 'include'), '-I' + str(src)]
        resource = subprocess.check_output([str(compiler), '-print-resource-dir'], text=True).strip()
        flags += ['-nostdinc', '-isystem', resource + '/include']
        objects = []
        commands = []
        for name in (*names, 'rv64_support', 'atlas_host', 'block_target', 'tiny_bundle_assets', 'images', 'start'):
            s = src / (name + ('.S' if name in ('images', 'start') else '.c'))
            o = obj / (name + '.o')
            objects.append(str(o))
            cmd = [str(compiler), *flags, '-c', str(s), '-o', str(o)]
            commands.append(cmd)
            p = subprocess.run(cmd, capture_output=True, text=True)
            (obj / (name + '.log')).write_text(p.stdout + p.stderr)
            if p.returncode != 0:
                raise ValueError(f'compiler or linker failed: {p.stderr}')
        cmd = [str(linker), '-flavor', 'gnu', '-m', 'elf64lriscv', '--threads=1', '--no-relax',
               '-T', str(src / 'link.ld'), '-Map', str(out / 'host.map'), *objects, '-o', str(out / 'host.elf')]
        commands.append(cmd)
        p = subprocess.run(cmd, capture_output=True, text=True)
        (out / 'link.log').write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            raise ValueError(f'compiler or linker failed: {p.stderr}')
        image = Image((out / 'host.elf').read_bytes())
        if image.entry != 0x80000000:
            raise ValueError('unexpected ELF entry point')
        for r in rows:
            i = r['index']
            if image.bytes_at(r['dram_base'], r['total_bytes']) != (data / f'{i}-loadmem.bin').read_bytes():
                raise ValueError(f'LOADMEM image mismatch for panel {i}')
            program_address = image.symbols[f'program_{i}']['address']
            if image.bytes_at(program_address, r['program_words'] * 4) != (data / f'{i}-program.bin').read_bytes():
                raise ValueError(f'program image mismatch for panel {i}')
        if image.symbols['__stack_top']['address'] >= 0x88000000:
            raise ValueError('host stack exceeds retained RAM bounds')
        for p, d in pins.items():
            if sha(checked(p).read_bytes()) != d:
                raise ValueError(f'source changed during build: {p}')
        memory_audit = audit(out, a.dram_base, a.dram_bytes)
        generated = {str(p): sha(p.read_bytes()) for folder in (src, src / 'include', data)
                     for p in folder.iterdir() if p.is_file()}
        receipt.update(
            status='PASS_BUILD_STATIC_ONLY', arena_base=arena_base, dram_base=a.dram_base, dram_bytes=a.dram_bytes,
            memory_audit=memory_audit, elf=str(out / 'host.elf'), elf_sha256=sha(image.raw), sources=pins,
            generated_inputs=generated, captures=rows, commands=commands, mmio_sites=mmio_sites(image),
            workspace_bytes=len(workspace), prefill_tokens=1, decode_tokens=0,
            final_buffers={n: dict(bytes=len(raws[n]), sha256=sha(raws[n])) for n in ('output', 'keys', 'values')},
            static=dict(loadmem_arenas=12, dram_start=arena_base, dram_end=arena_base + offset, fast_check=True,
                        bit_copy_fix=True, A_preload_poison=0x96,
                        host_stack_top=image.symbols['__stack_top']['address'],
                        max_wrapped_words=max(r['program_words'] + 5 for r in rows)))
    except Exception as e:
        import traceback
        receipt['error'] = repr(e)
        receipt['traceback'] = traceback.format_exc()
        raise
    finally:
        write(out / 'build.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'elf_sha256', 'workspace_bytes')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for n in ('out', 'retained', 'recipe', 'model-root', 'compiler', 'linker'):
        p.add_argument('--' + n, type=Path, required=True)
    add_arguments(p)
    build(p.parse_args())
