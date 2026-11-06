#!/usr/bin/env python3
"""Rebuild an admitted Smol SIMS recipe with the maintained raw runtime.

The compile database and link command supply immutable host and platform inputs.
Fast mode binds regenerated programs to the exact linked weight addresses. The
normal build stays the default and can be checked against an expected ELF hash.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked


def digest(path):
    with checked(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform-root', type=checked, required=True, help='Board-support tree for the target platform')
    parser.add_argument('--build-dir', type=checked, required=True, help='Directory containing compile_commands.json')
    parser.add_argument('--target-dir', type=checked, required=True, help='Target working directory recorded in the database')
    parser.add_argument('--link-command', type=checked, required=True)
    parser.add_argument('--out', type=checked, required=True)
    parser.add_argument('--generated', type=checked, help='Output of generate_smol_fast_assets.py')
    parser.add_argument('--expect-sha256')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    if args.fast and args.generated is None:
        parser.error('--fast requires --generated')
    if not args.platform_root.is_dir():
        parser.error('--platform-root must be a directory')
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / 'objects'
    work.mkdir(exist_ok=True)
    runtime = Path(__file__).absolute().parent.parent / 'runtime'
    commands_file = checked(args.build_dir / 'compile_commands.json')
    commands = json.loads(commands_file.read_text())
    mapping, records, invocations = {}, [], []
    start = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(work))
    # The large immutable asset object never needs a persistent copy on disk.
    fd = os.memfd_create('smol-immutable-assets')
    asset_obj = f'/proc/{os.getpid()}/fd/{fd}'

    def validate_platform(argv):
        for value in argv:
            if '/platform/penguin26' in value or '/glossy/' in value:
                if str(args.platform_root) + '/' not in value:
                    raise ValueError('Recipe board input is outside --platform-root')

    def run(argv, label):
        validate_platform(argv)
        invocations.append(argv)
        with (args.out / (label + '.log')).open('w') as log:
            subprocess.run(argv, check=True, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)

    try:
        # Stage only the runtime being replaced. Quoted provider headers remain
        # the admitted generated package's headers, except for the context ABI.
        raw_header = runtime / 'smol_policy_raw.h'
        (work / raw_header.name).write_bytes(raw_header.read_bytes())
        for index, row in enumerate(commands):
            if Path(row['directory']) != args.target_dir:
                continue
            argv = shlex.split(row['command'])
            source = checked(row['file'])
            original = str(args.target_dir / argv[argv.index('-o') + 1])
            obj = asset_obj if source.name == 'assets.S' else str(work / f'{index:02d}.o')
            mapping[original] = obj
            src = source
            if source.name == 'smol_policy_raw.c':
                src = work / source.name
                src.write_bytes((runtime / source.name).read_bytes())
            elif args.fast and source.name == 'main.c':
                text = source.read_text()
                # RV64 context sizes track the generated ABI in build_smol_full_host.py.
                old = '_Static_assert(sizeof(smol_policy_raw_context)==184,"context ABI");'
                new = '_Static_assert(sizeof(smol_policy_raw_context)==232,"fast context ABI");'
                if text.count(old) != 1:
                    raise ValueError('Unexpected generated host context ABI assertion')
                src = work / source.name
                src.write_text(text.replace(old, new))
            argv[argv.index('-o') + 1] = obj
            argv[argv.index('-c') + 1] = str(src)
            if args.fast:
                argv[1:1] = ['-DATLAS_FAST_MODE=1', '-I' + str(args.generated)]
                if source.suffix == '.c':
                    argv[1:1] = ['-include', str(work / raw_header.name)]
            records.append(dict(source=str(src), source_sha256=digest(src), command=argv, object=obj))
            run(argv, f'compile-{index:02d}')
        if not mapping:
            raise ValueError('No compilation units matched the explicit target directory')
        extras = []
        if args.fast:
            for name, template in [('smol_fast_assets.c', 'smol_policy_raw.c'), ('smol_fast_images.S', 'assets.S')]:
                argv = list(next(row for row in records if Path(row['source']).name == template)['command'])
                src, obj = checked(args.generated / name), str(work / (name + '.o'))
                argv[argv.index('-c') + 1], argv[argv.index('-o') + 1] = str(src), obj
                records.append(dict(source=str(src), source_sha256=digest(src), command=argv, object=obj))
                run(argv, name)
                extras.append(obj)
        link = []
        for arg in shlex.split(args.link_command.read_text()):
            if arg.endswith('.obj'):
                arg = mapping[str(args.target_dir / arg)]
            elif arg.endswith('.a'):
                arg = str(checked(os.path.abspath(args.target_dir / arg)))
            elif arg.startswith('-Wl,-Map,'):
                arg = '-Wl,-Map,' + str(args.out / 'link.map')
            link.append(arg)
        elf = args.out / 'smolvla.elf'
        link[link.index('-o') + 1] = str(elf)
        link[link.index('-o'):link.index('-o')] = extras
        run(link, 'link')
    finally:
        os.close(fd)
    actual = digest(elf)
    result = dict(status='BUILT', platform_root=str(args.platform_root), fast=args.fast, elf=str(elf), elf_sha256=actual, compile=records,
                  link_command=link, compile_commands_sha256=digest(commands_file),
                  link_recipe_sha256=digest(args.link_command), elapsed_seconds=time.monotonic() - start,
                  runtime_header_sha256=digest(raw_header), memfd_assets=True)
    if args.expect_sha256:
        result['expected_sha256'] = args.expect_sha256
        result['status'] = 'PASS_BYTE_IDENTICAL' if actual == args.expect_sha256 else 'FAIL_IDENTITY'
    (args.out / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    for obj in work.iterdir():
        if obj.suffix == '.o':
            obj.unlink()
    print(result['status'], actual, flush=True)
    return 0 if result['status'] != 'FAIL_IDENTITY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
