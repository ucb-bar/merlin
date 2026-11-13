"""Build only the two reviewed CORE-MATH source files and our symbol wrapper."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
vendor = root / 'third_party/core-math'
out = root / 'out/core-math'
out.mkdir(parents=True, exist_ok=True)
riscv = Path(os.environ['RISCV'])
flags = ['-march=rv64gc', '-mabi=lp64d', '-O2', '-std=c11', '-fno-fast-math', '-frounding-math', '-ffp-contract=off', '-fno-strict-aliasing', '-fno-math-errno']
# Explicit __builtin_fmaf calls remain fused; implicit contraction is disabled.
wrapper = out / 'wrapper.c'
wrapper.write_text('extern float cr_sinf(float);\nextern float cr_cosf(float);\nfloat sinf(float x) { return cr_sinf(x); }\nfloat cosf(float x) { return cr_cosf(x); }\n')
inputs = [vendor / 'sinf.c', vendor / 'cosf.c', wrapper]
objects=[]
for source in inputs:
    obj = out / (source.stem+'.o')
    subprocess.run(['nice','-n','10',str(riscv/'bin/riscv64-unknown-elf-gcc'),*flags,'-c',str(source),'-o',str(obj)],check=True)
    objects.append(obj)
archive = out / 'libcore-math-rotary.a'
# Deterministic archive, fresh so no stale members can survive.
archive.unlink(missing_ok=True)
subprocess.run(['nice','-n','10',str(riscv/'bin/riscv64-unknown-elf-ar'),'rcsD',str(archive),*map(str,objects)],check=True)
nm = subprocess.check_output([str(riscv/'bin/riscv64-unknown-elf-nm'),'-g',str(archive)],text=True)
(out/'symbols.txt').write_text(nm)
assembly = subprocess.check_output([str(riscv/'bin/riscv64-unknown-elf-objdump'),'-d',str(archive)],text=True)
(out/'disassembly.txt').write_text(assembly)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {'schema':'rotary.core-math-build.v1','revision':(vendor/'REVISION').read_text().strip(),
 'license':'MIT','flags':flags,'compiler':subprocess.check_output([str(riscv/'bin/riscv64-unknown-elf-gcc'),'--version'],text=True).splitlines()[0],
 'source_sha256':{p.name:digest(p) for p in inputs},'archive_path':str(archive.resolve()),'archive_sha256':digest(archive),
 'defined_symbols':sorted(line.split()[-1] for line in nm.splitlines() if ' T ' in line),
 'undefined_symbols':sorted(set(line.split()[-1] for line in nm.splitlines() if ' U ' in line)),
 'fma':'Implicit contraction off; source explicit __builtin_fmaf preserved as RV64 fnmsub.s in both translation units.',
 'fused_instructions':[line.strip() for line in assembly.splitlines() if any(op in line for op in ['fmadd.','fmsub.','fnmadd.','fnmsub.'])]}
(out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
