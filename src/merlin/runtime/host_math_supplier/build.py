"""Build the policy-selected supplier: python build.py --out OUT."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('host_math', ROOT / '__init__.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
FLAGS = ['-O2', '-std=gnu11', '-fno-fast-math', '-frounding-math', '-ffp-contract=off',
         '-fno-strict-aliasing', '-fno-math-errno', '-fno-builtin']


def run(command):
    try:
        return subprocess.check_output(['nice', '-n', '10', *map(str, command)],
            env=dict(os.environ, OMP_NUM_THREADS='1'), stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as error:
        print(error.output, flush=True)
        raise


def build(target, out, policy_path):
    policy = api.load_policy(policy_path, api.FUNCTIONS)
    out = out.resolve() / target
    out.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get('RISCV_PREFIX', 'riscv64-unknown-elf-') if target == 'rv64' else ''
    cc, ar, nm = (prefix+'gcc', prefix+'ar', prefix+'nm') if target == 'rv64' else ('cc', 'ar', 'nm')
    flags = FLAGS + (['-march=rv64gc', '-mabi=lp64d'] if target == 'rv64' else ['-march=x86-64-v3', '-m64', '-fPIC'])
    if target == 'rv64':
        cc = os.environ.get('RISCV_CC', cc)
    wrapper = out / 'wrapper.c'
    text = '#include "host_math.h"\n#include <stdint.h>\n#include <string.h>\n'
    text += '_Static_assert(sizeof(float)==4 && sizeof(double)==8,"IEEE formats required");\n'
    text += """
int merlin_host_check_fp_env(void) {
#if defined(__x86_64__)
    unsigned mxcsr; unsigned short control;
    __asm__ volatile("stmxcsr %0" : "=m"(mxcsr));
    __asm__ volatile("fnstcw %0" : "=m"(control));
    return (mxcsr & 0xe040U) == 0 && (control & 0xc00U) == 0;
#elif defined(__riscv)
    unsigned long frm;
    __asm__ volatile("frrm %0" : "=r"(frm));
    return frm == 0;
#else
#error Unsupported host-math target
#endif
}
"""
    for suffix, scalar, word, size, mask, inf, nan in [
        ('32', 'float', 'uint32_t', 4, '0x7fffffffU', '0x7f800000U', '0x7fc00000U'),
        ('64', 'double', 'uint64_t', 8, '0x7fffffffffffffffULL', '0x7ff0000000000000ULL', '0x7ff8000000000000ULL')]:
        text += f'static {word} bits{suffix}({scalar} x){{{word} u;memcpy(&u,&x,{size});return u;}}\n'
        text += f'static {scalar} canonical{suffix}({scalar} x){{{word} u=bits{suffix}(x);if((u&{mask})>{inf})u={nan};memcpy(&x,&u,{size});return x;}}\n'
    for name, entry in policy['functions'].items():
        scalar, word, suffix, size = ('double', 'uint64_t', '64', 8) if name in api.FUNCTIONS64 else ('float', 'uint32_t', '32', 4)
        argc = api.arity(name)
        params = ','.join([scalar] * argc)
        scalar_args = ','.join(f'{scalar} x{i}' for i in range(argc))
        word_args = ','.join(f'{word} u{i}' for i in range(argc))
        call_args = ','.join(f'x{i}' for i in range(argc))
        conversions = ''.join(f'{scalar} x{i};memcpy(&x{i},&u{i},{size});' for i in range(argc))
        text += f'extern {scalar} {entry["implementation"]}({params});\n'
        text += f'{scalar} merlin_host_{name}({scalar_args}){{return canonical{suffix}({entry["implementation"]}({call_args}));}}\n'
        text += f'{word} merlin_host_bits_{name}({word_args}){{{conversions}return bits{suffix}(merlin_host_{name}({call_args}));}}\n'
    wrapper.write_text(text)
    sources = [(ROOT if name == 'sqrtf' else ROOT/'third_party/core-math')/v['source'] for name, v in policy['functions'].items()] + [wrapper] + [ROOT / 'round_support.c']
    objects = []
    for src in sources:
        obj = out / (src.stem+'.o')
        extra = ['-include', ROOT/'supplier_support.h'] if src.is_relative_to(ROOT/'third_party/core-math') else []
        run([cc, *flags, *extra, '-I', ROOT, '-c', src, '-o', obj])
        objects.append(obj)
    archive = out / 'libmerlin-host-math.a'
    archive.unlink(missing_ok=True)
    run([ar, 'rcsD', archive, *objects])
    symbols = run([nm, '-g', archive])
    (out/'symbols.txt').write_text(symbols)
    receipt = {'schema':'merlin.host_math.build.v1','target':target,
        'complete':set(policy['functions']) == set(api.FUNCTIONS),
        'functions':list(policy['functions']), 'compiler':run([cc,'--version']).splitlines()[0],
        'flags':flags, 'policy_sha256':api.sha256(policy_path), 'header_sha256':api.sha256(api.HEADER_PATH),
        'wrapper_sha256':api.sha256(wrapper), 'source_sha256':{s.name:api.sha256(s) for s in sources},
        'support_sources_sha256': policy['support_sources'],
        'support_header_sha256':api.sha256(ROOT/'supplier_support.h'),
        'archive_sha256':api.sha256(archive)}
    if target == 'x86':
        so = out / 'libmerlin-host-math.so'
        run([cc, '-shared', '-Wl,--no-undefined', *objects, '-o', so])
        receipt['shared_sha256'] = api.sha256(so)
    (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--policy', type=Path, default=api.POLICY_PATH)
    parser.add_argument('--target', choices=['both','rv64','x86'], default='both')
    args = parser.parse_args()
    api.load_policy(args.policy, api.FUNCTIONS)
    for target in (['rv64','x86'] if args.target == 'both' else [args.target]):
        build(target,args.out,args.policy)
