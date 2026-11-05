"""Build a qualified TinyLlama full model or complete layer-0 block for SIMS/CHIP."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).absolute().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha256
from attention_bundle import AttentionBundle, require

HERE = Path(__file__).absolute().parent
MERLIN_UNITS = ('llama_host_attention.c', 'llama_host_layout.c', 'llama_rotary_table.c',
                'llama_host_bmm.c', 'host_matrix_fp8pack.c', 'smol_host_ops.c', 'smol_host_softmax.c')
MERLIN_HEADERS = ('llama_host_attention.h', 'llama_host_layout.h', 'llama_rotary_table.h', 'llama_host_bmm.h',
                  'host_matrix_fp8pack.h', 'smol_host_ops.h', 'smol_host_bits.h', 'smol_host_softmax.h',
                  'smol_host_controls.h', 'smol_host_rope.h', 'tiny_block.h')


def build(args):
    require(not args.out.exists(), 'fresh attention build output directory required')
    bundle = AttentionBundle(args.assets)
    roots = dict(atlas=ROOT, merlin=args.merlin.parents[2], model=getattr(args, 'model_root', None),
                 data=getattr(args, 'receipt_data_root', None))
    relocations = []
    for value in getattr(args, 'receipt_relocate', []):
        old, separator, new = value.partition('=')
        require(separator, 'receipt relocation requires OLD=NEW')
        relocations.append((checked(old), checked(new)))
    bundle.qualify(args.qualification, args.qualification_sha256, roots=roots, relocations=relocations)
    pins = {}

    def pin(path, expected=None):
        path = checked(path)
        actual = sha256(path)
        require(expected is None or actual == expected, 'attention build source differs: ' + str(path))
        require(str(path) not in pins or pins[str(path)] == actual, 'attention build source changed')
        pins[str(path)] = actual
        return path

    def compiler_path(value):
        # GCC emits lexical ../ segments in its installed resource paths.
        return checked(os.path.normpath(value))

    platform = args.platform_root
    for name in ('CMakeLists.txt',):
        pin(platform / name)
    sources = [HERE / 'tinyllama_main.c', HERE / 'tiny_full_model.c', HERE / 'tiny_full_projection.c',
               ROOT / 'runtime/tiny_attention.c', ROOT / 'runtime/atlas_host.c']
    sources += [args.out / 'host/tiny_full_assets.c', args.out / 'host/images.S', platform / 'tinyllama_math.c']
    sources += [args.merlin / name for name in MERLIN_UNITS]
    for path in sources:
        if args.out not in path.parents:
            pin(path)
    for name in MERLIN_HEADERS:
        pin(args.merlin / name)
    for path in (HERE / 'tiny_full_model.h', HERE / 'tiny_full_tiles.h', HERE / 'tiny_full_projection.h', ROOT / 'runtime/tiny_projection.h',
                 ROOT / 'runtime/tiny_attention.h', ROOT / 'runtime/atlas_host.h', ROOT / 'runtime/atlas_host_layout.h',
                 ROOT / 'runtime/atlas_host_check.h', ROOT / 'runtime/atlas_graph_memref.h', ROOT / 'runtime/atlas_numeric.h',
                 HERE / 'attention_bundle.py', HERE / 'attention_export.py', Path(__file__), args.qualification):
        pin(path)
    compiler = pin(args.toolchain_bin / 'riscv64-unknown-elf-gcc')
    for name in ('g++', 'ar', 'ld', 'as'):
        pin(args.toolchain_bin / ('riscv64-unknown-elf-' + name))
    args.out.mkdir(parents=True)
    host = args.out / 'host'
    # Generate the tables from admitted metadata; never compile caller-supplied C.
    from attention_export import emit_c
    emit_c(SimpleNamespace(root=bundle.root, sums={}, path=bundle.path), bundle.data, host=host)
    for name in ('tiny_full_assets.c', 'tiny_full_assets.h', 'images.S'):
        pin(host / name)
    includes = [host, HERE, ROOT / 'runtime', args.merlin, platform]

    def setting(name, values):
        return 'set(' + name + '\n' + ''.join('  ' + json.dumps(str(value)) + '\n' for value in values) + ')\n'

    manifest = args.out / 'sources.cmake'
    manifest.write_text(setting('TL_SOURCES', sources) + setting('TL_INCLUDES', includes)
                        + setting('TL_ASSEMBLY', [host / 'images.S']) + setting('TL_BLOBS', [host])
                        + setting('TL_DEFINITIONS', ['TINYLLAMA_STOP_AFTER=0']))
    pin(manifest)
    env = dict(os.environ, PATH=str(args.toolchain_bin) + os.pathsep + os.environ.get('PATH', ''),
               TMPDIR=str(args.out / 'tmp'), PYTHONDONTWRITEBYTECODE='1')
    for name in ('CFLAGS', 'CXXFLAGS', 'ASMFLAGS', 'LDFLAGS'):
        env.pop(name, None)
    (args.out / 'tmp').mkdir()
    commands = []

    def run(command, label):
        commands.append(command)
        with (args.out / (label + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=args.out, env=env, stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, label + ' failed; see ' + str(args.out / (label + '.log')))

    configuration = ['-DPLATFORM=' + args.variant.upper(), '-DPENGUIN26_EXIT_TOHOST=' +
                     ('ON' if args.variant == 'chip' else 'OFF')]
    if args.variant == 'chip':
        configuration += ['-DPENGUIN26_UART_BAUDRATE=3686400']
    run([args.cmake, '-S', str(platform), '-B', str(args.out / 'build'),
         '-DCMAKE_TOOLCHAIN_FILE=' + str(platform / 'riscv-gcc.cmake'), '-DCMAKE_BUILD_TYPE=Release',
         '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DTL_MANIFEST=' + str(manifest), *configuration], 'configure')
    # Ask the selected compiler for all actual includes before compiling, including
    # platform and toolchain headers, so post-build bookends cover dependencies.
    entries = json.loads((args.out / 'build/compile_commands.json').read_text())
    for index, entry in enumerate(entries):
        command = shlex.split(entry['command'])
        output_index = command.index('-o')
        del command[output_index:output_index + 2]
        command.remove('-c')
        depfile = args.out / ('dependencies-' + str(index) + '.d')
        run([*command, '-M', '-MF', str(depfile)], 'dependencies-' + str(index))
        dependencies = depfile.read_text().replace('\\\n', ' ').split(':', 1)[1]
        for value in shlex.split(dependencies):
            pin(compiler_path(value))
    for name in ('libgcc.a', 'libc.a', 'libm.a', 'libnosys.a'):
        path = subprocess.check_output([str(compiler), '-march=rv64gcv_zfh', '-mabi=lp64d',
                                        '-print-file-name=' + name], env=env, text=True).strip()
        pin(compiler_path(path))
    run([args.cmake, '--build', str(args.out / 'build'), '--target', 'atlas-tinyllama',
         '--parallel', str(args.jobs), '--verbose'], 'build')
    bundle.verify_bookends()
    require(all(sha256(checked(path)) == expected for path, expected in pins.items()), 'attention build bookend changed')
    elf = args.out / 'build/atlas-tinyllama.elf'
    elf_sha256 = sha256(elf)
    require(args.expected_elf_sha256 is None or elf_sha256 == args.expected_elf_sha256,
            'attention ELF differs from requested byte identity')
    # Admit concrete ELF load ranges against the explicit exported physical range.
    from elf_image import Image
    image = Image(elf.read_bytes())
    placement = bundle.data['placement']
    loads = []
    for segment in image.segments:
        start, end = segment['physical'], segment['physical'] + segment['memory_bytes']
        require(placement['dram_base'] <= start <= end <= placement['dram_base'] + placement['dram_bytes'],
                'attention ELF exceeds physical DRAM placement')
        require(end <= placement['arena_base'] or start >= placement['arena_base'] + bundle.arena_bytes,
                'attention ELF overlaps DMA arena')
        require(end <= placement['arena_base'] if segment['flags'] & 2 else start >= 0x90800000,
                'attention writable or read-only placement differs')
        loads.append(dict(start=start, end=end, flags=segment['flags']))
    require(len(loads) == 2, 'attention ELF must retain the two-LOAD layout')
    record = dict(schema='atlas.tiny_attention_host_build.v1', status='PASS_BUILT_ATTENTION_HOST_UNEXECUTED',
                  variant=args.variant, elf=str(elf), elf_sha256=elf_sha256, elf_bytes=elf.stat().st_size,
                  qualification_sha256=args.qualification_sha256, assets_sha256sums_sha256=bundle.checksums_sha256,
                  scope=bundle.scope, layer_count=bundle.data['layer_count'],
                  launches=bundle.data['kernel_count'], projection_launches=bundle.data['projection_kernel_count'],
                  attention_launches=bundle.data['attention_kernel_count'], final_buffers=len(bundle.data['reference_buffers']),
                  placement=placement, load_ranges=loads, source_sha256=pins, source_bookends_verified=True,
                  commands=commands, device_execution=False)
    (args.out / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(elf)
    print(elf_sha256)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'qualification', 'merlin', 'toolchain-bin', 'out', 'platform-root'):
        parser.add_argument('--' + name, type=checked, required=True)
    parser.add_argument('--model-root', type=checked)
    parser.add_argument('--receipt-data-root', type=checked)
    parser.add_argument('--receipt-relocate', action='append', default=[])
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--variant', choices=('sims', 'chip'), required=True)
    parser.add_argument('--expected-elf-sha256')
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    require(args.jobs > 0, 'jobs must be positive')
    build(args)


if __name__ == '__main__':
    main()
