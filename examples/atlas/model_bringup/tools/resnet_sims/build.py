#!/usr/bin/env python3
"""Cross-compile a ResNet50 package into SIMS, CHIP, or QEMU host ELFs; nothing is executed."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked

PLATFORMS = {
    'sims': ['-DPLATFORM=SIMS', '-DPENGUIN26_EXIT_TOHOST=OFF', '-DRESNET50_QEMU=OFF'],
    'chip': ['-DPLATFORM=CHIP', '-DPENGUIN26_EXIT_TOHOST=ON', '-DPENGUIN26_UART_BAUDRATE=3686400', '-DRESNET50_QEMU=OFF'],
    'qemu': ['-DPLATFORM=SIMS', '-DPENGUIN26_EXIT_TOHOST=OFF', '-DRESNET50_QEMU=ON'],
}
HOOK = '''
if (PENGUIN26_TESTS_ATLAS_RESNET50_DIR)
  add_subdirectory(${PENGUIN26_TESTS_ATLAS_RESNET50_DIR} atlas_resnet50)
endif()
'''


def copy_tree(source, target):
    target.mkdir()
    with os.scandir(source) as entries:
        for entry in entries:
            if entry.name == '.git' or entry.name.startswith('build'):
                continue
            if entry.is_symlink(): raise ValueError('board-support symlinks are unsupported')
            destination = target/entry.name
            if entry.is_dir(follow_symlinks=False): copy_tree(Path(entry.path), destination)
            elif entry.is_file(follow_symlinks=False): shutil.copy2(entry.path, destination)
            else: raise ValueError('Unsupported board-support entry')


def prepare_platform_root(source, target):
    """Copy a board-support tree and attach only the ResNet50 test directory."""
    copy_tree(checked(source), target)
    # The delivery directory may carry machine-local smoke includes; ResNet50 does not use it.
    unused = target/'penguin26-tests/atlas_delivery/CMakeLists.txt'
    if unused.exists(): unused.write_text('# Unused delivery targets omitted for this ResNet50-only build.\n')
    tests = target/'penguin26-tests/CMakeLists.txt'
    text = tests.read_text()
    if 'PENGUIN26_TESTS_ATLAS_RESNET50_DIR' not in text: tests.write_text(text.rstrip()+'\n'+HOOK)
    return target


def build(package, platform_root, work, platform, toolchain, fast=False):
    """Configure and build one variant under work; returns the ELF path."""
    mode = 'fast' if fast else 'default'
    out = work/f'build-{mode}-{platform}'
    (work/'logs').mkdir(exist_ok=True)
    env = dict(os.environ, PATH=f'{toolchain}{os.pathsep}{os.environ["PATH"]}')
    configure = ['cmake', '-S', str(platform_root), '-B', str(out), f'-DCMAKE_TOOLCHAIN_FILE={platform_root}/riscv-gcc.cmake',
                 '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCHIP=penguin26', '-DRVV_TYPE=1',
                 '-DUSE_FREERTOS=OFF', '-DPENGUIN26_TESTS_ATLAS_COMPILED_DIR=', '-DPENGUIN26_TESTS_ATLAS_DELIVERY_DIR=',
                 '-DPENGUIN26_TESTS_EXTERNAL_OBJECTS=', f'-DPENGUIN26_TESTS_ATLAS_DIR={work}/disabled-atlas-tests',
                 f'-DPENGUIN26_TESTS_ATLAS_RESNET50_DIR={package}', '-DRESNET50_VERIFY=ON', *PLATFORMS[platform],
                 f'-DRESNET50_FAST={"ON" if fast else "OFF"}']
    make = ['cmake', '--build', str(out), '--target', 'atlas-resnet50', '--parallel', os.environ.get('RESNET50_BUILD_JOBS', '2')]
    for name, command in (('configure', configure), ('build', make)):
        with open(work/'logs'/f'{name}-{mode}-{platform}.log', 'w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out/'penguin26-tests/atlas_resnet50/atlas-resnet50.elf'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=checked, required=True, help='Directory written by package.py')
    p.add_argument('--platform-root', type=checked, required=True, help='board-support source tree with the penguin26 platform')
    p.add_argument('--toolchain', type=checked, required=True, help='RISC-V GCC toolchain bin directory')
    p.add_argument('--work', type=checked, required=True, help='Fresh work directory outside the inputs')
    p.add_argument('--platform', choices=PLATFORMS, nargs='+', default=list(PLATFORMS))
    a = p.parse_args()
    if a.work.exists(): raise ValueError('work directory must not already exist')
    a.work.mkdir(parents=True)
    platform_root = prepare_platform_root(a.platform_root, a.work/'platform_root')
    for platform in a.platform:
        print(platform, build(a.package, platform_root, a.work, platform, a.toolchain))


if __name__ == '__main__':
    main()
