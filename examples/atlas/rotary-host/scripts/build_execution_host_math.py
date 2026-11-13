"""Build a separate native CORE-MATH archive; never modify the RV64 supplier."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/execution-host-math'
REVISION = 'a055b9b9a86b857e49573e339cf68306ad92c831'
PRODUCTION_SHA = '0626df7b1ce6426945820fd03dbd6752931c7594676c94f2043d5673588d178b'
INV_FREQ_SHA = '734f9e6b629f7baccc4de81b7e8d2982035ee1b81f73f158e7d33a4dcf13fa0e'
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, label):
    actual = ['nice', '-n', '10', *map(str, command)]
    result = subprocess.run(actual, capture_output=True, text=True, env=ENV)
    with (OUT / 'commands.jsonl').open('a') as log:
        log.write(json.dumps({'label': label, 'command': actual, 'returncode': result.returncode,
                              'stdout': result.stdout, 'stderr': result.stderr,
                              'OMP_NUM_THREADS': '1'}) + '\n')
    if result.returncode:
        raise RuntimeError(f'{label} failed: {result.stderr}')
    return result


def main():
    assert platform.machine() == 'x86_64', 'this execution archive requires x86_64'
    OUT.mkdir(parents=True, exist_ok=True)
    vendor = ROOT / 'third_party/core-math'
    production = ROOT / 'out/core-math/libcore-math-rotary.a'
    production_receipt = ROOT / 'out/core-math/build-receipt.json'
    pinned = json.loads(production_receipt.read_text())
    assert pinned['revision'] == (vendor / 'REVISION').read_text().strip() == REVISION
    assert digest(production) == pinned['archive_sha256'] == PRODUCTION_SHA
    for name in ('sinf.c', 'cosf.c'):
        assert digest(vendor / name) == pinned['source_sha256'][name], name
    wrapper = OUT / 'wrapper.c'
    wrapper.write_text('extern float cr_sinf(float);\nextern float cr_cosf(float);\nfloat sinf(float x) { return cr_sinf(x); }\nfloat cosf(float x) { return cr_cosf(x); }\n')
    assert digest(wrapper) == pinned['source_sha256']['wrapper.c']
    cc = shutil.which(os.environ.get('CC', 'cc'))
    assert cc, 'native compiler unavailable'
    cc = str(Path(cc).resolve())
    flags = ['-march=x86-64-v3', '-m64', '-fPIC', '-O2', '-std=c11', '-fno-fast-math',
             '-frounding-math', '-ffp-contract=off', '-fno-strict-aliasing', '-fno-math-errno']
    objects = []
    for source in (vendor / 'sinf.c', vendor / 'cosf.c', wrapper):
        obj = OUT / (source.stem + '.o')
        run([cc, *flags, '-c', source, '-o', obj], 'compile ' + source.name)
        objects.append(obj)
    archive = OUT / 'libcore-math-execution-host.a'
    archive.unlink(missing_ok=True)
    run(['ar', 'rcsD', archive, *objects], 'archive')
    nm = run(['nm', '-g', archive], 'archive symbols').stdout
    (OUT / 'symbols.txt').write_text(nm)
    assembly = run(['objdump', '-d', archive], 'archive disassembly').stdout
    (OUT / 'disassembly.txt').write_text(assembly)
    fused = [line.strip() for line in assembly.splitlines()
             if re.search(r'\bv(?:f|fn)m(?:add|sub)\w*', line)]
    assert fused, 'explicit source FMA must remain fused'
    probe = OUT / 'execution_host_probe.c'
    probe.write_text('''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#if !defined(__x86_64__)
#error native x86_64 execution host required
#endif
static float (*volatile sine)(float)=sinf;
static float (*volatile cosine)(float)=cosf;
static uint32_t word(float x) {uint32_t b; memcpy(&b,&x,4); return b;}
int main(int argc,char **argv) {
 if(argc!=2)return 1;
 unsigned mxcsr; unsigned short cw;
 __asm__ volatile("stmxcsr %0":"=m"(mxcsr));
 __asm__ volatile("fnstcw %0":"=m"(cw));
 if((mxcsr & 0x6000u) || (cw & 0xc00u))return 2;
 FILE *in=fopen(argv[1],"rb"); float freq[32];
 if(!in || fread(freq,4,32,in)!=32)return 3;
 if(fgetc(in)!=EOF)return 4;
 fclose(in);
 for(int p=0;p<2048;p++)for(int j=0;j<32;j++) {
  volatile float angle=(float)p*freq[j];
  volatile float s=sine(angle),c=cosine(angle);
  volatile float ss=s*1.0f,cc=c*1.0f;
  uint32_t words[3]={word(angle),word(ss),word(cc)};
  if(fwrite(words,4,3,stdout)!=3)return 5;
 }
 return fclose(stdout)!=0;
}
''')
    executable = OUT / 'execution_host_probe'
    link_flags = ['-fno-builtin-sinf', '-fno-builtin-cosf', '-Wl,-Map=' + str(OUT / 'execution_host_probe.map')]
    for symbol in ('sinf', 'cosf', 'cr_sinf', 'cr_cosf'):
        link_flags.append('-Wl,--trace-symbol=' + symbol)
    linked = run([cc, *flags, *link_flags, probe, archive, '-o', executable], 'link probe')
    trace = linked.stdout + linked.stderr
    (OUT / 'link-symbol-trace.txt').write_text(trace)
    for symbol in ('sinf', 'cosf', 'cr_sinf', 'cr_cosf'):
        assert re.search(re.escape(str(archive)) + r'\([^\n]+\): definition of ' + symbol + r'\b', trace), symbol
    final_nm = run(['nm', '-g', executable], 'linked symbols').stdout
    (OUT / 'linked-symbols.txt').write_text(final_nm)
    for symbol in ('sinf', 'cosf', 'cr_sinf', 'cr_cosf'):
        assert re.search(r' T ' + symbol + r'$', final_nm, re.MULTILINE), symbol
    dynamic = run(['readelf', '-d', executable], 'dynamic dependencies').stdout
    (OUT / 'dynamic-dependencies.txt').write_text(dynamic)
    assert 'libm.so' not in dynamic, 'probe must not obtain trig functions from native libm'
    freq_bytes = bytes.fromhex((ROOT / 'scripts/inv_freq.hex').read_text())
    assert hashlib.sha256(freq_bytes).hexdigest() == INV_FREQ_SHA
    (OUT / 'inv_freq.bin').write_bytes(freq_bytes)
    receipt = {'schema': 'rotary.execution-host-math-build.v1', 'revision': REVISION,
               'license': 'MIT', 'isa': 'x86-64-v3', 'abi': 'sysv-x86_64',
               'compiler': run([cc, '--version'], 'compiler version').stdout.splitlines()[0],
               'compiler_path': cc, 'compiler_sha256': digest(cc), 'flags': flags,
               'link_flags': link_flags, 'archive_path': str(archive), 'archive_sha256': digest(archive),
               'source_sha256': {p.name: digest(p) for p in (vendor / 'sinf.c', vendor / 'cosf.c', wrapper)},
               'source_paths': [str(vendor / 'sinf.c'), str(vendor / 'cosf.c')],
               'defined_symbols': sorted(line.split()[-1] for line in nm.splitlines() if ' T ' in line),
               'undefined_symbols': sorted(set(line.split()[-1] for line in nm.splitlines() if ' U ' in line)),
               'fma': 'Implicit contraction disabled; explicit __builtin_fmaf lowered to native fused instructions.',
               'fused_instructions': fused, 'link_symbol_trace_sha256': digest(OUT / 'link-symbol-trace.txt'),
               'link_map_sha256': digest(OUT / 'execution_host_probe.map'),
               'executable_sha256': digest(executable), 'probe_source_sha256': digest(probe),
               'production_archive_path': str(production), 'production_archive_sha256': digest(production),
               'production_build_receipt_sha256': digest(production_receipt),
               'inv_freq_sha256': INV_FREQ_SHA,
               'linkage': {'schema': 'merlin.host_linkage_contract.v1', 'supplier': 'selected_math_archive',
                           'archive_sha256': digest(archive), 'symbols': ['cosf', 'sinf']},
               'thread_environment': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
               'nice': 10}
    assert digest(production) == PRODUCTION_SHA
    (OUT / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (OUT / 'host-linkage-contract.json').write_text(json.dumps(receipt['linkage'], indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
