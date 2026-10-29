#!/usr/bin/env python3
"""Build a task-local FASED boundary reference with explicitly supplied tools.

Requires an already provisioned FireSim `sim` source tree and JVM dependencies.
Does not fetch dependencies, edit a deployed target, launch FireSim or run a
campaign. The selected Config and AXI contract are compile inputs, not proof of
what FPGA was deployed. See README.md for the remaining acceptance gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from expand_else_when import expand
from generate_boundary_driver import generate


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','selection','output','java','firtool','circt_translate','gsim','verilator','cxx'):
        parser.add_argument('--'+name.replace('_','-'),required=True)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if not 1 <= args.jobs <= 4: raise ValueError('bounded native build requires 1..4 jobs')
    project=Path(args.project).resolve(strict=True); output=Path(args.output).resolve()
    if project.is_symlink() or '/main/' in str(project) or '/hw/' in str(project):
        raise ValueError('project must be an explicitly provisioned task-local editable copy')
    output.mkdir(parents=True,exist_ok=True)
    selection=Path(args.selection).resolve(strict=True)
    assets=Path(__file__).resolve().parent
    wrapper=project/'midas/src/main/scala/FasedTimingStandalone.scala'
    shutil.copyfile(assets/'FasedTimingStandalone.scala',wrapper)
    tools={k:str(Path(getattr(args,k)).resolve(strict=True)) for k in ('java','firtool','circt_translate','gsim','verilator','cxx')}
    receipt={'schema':'merlin.fased_native_build.v1','selection':json.loads(selection.read_text()),
        'selection_sha256':sha(selection),'tools':{k:{'path':v,'sha256':sha(v)} for k,v in tools.items()},
        'assets':{p.name:sha(p) for p in assets.iterdir() if p.suffix in ('.py','.scala')},
        'commands':[],'deployment_equivalence':'NOT_ESTABLISHED','production_adapter':False}
    source_dirs=[project/'midas/src/main/scala',project/'src/main/scala',project/'fasedtests/src/main/scala']
    receipt['source_sha256s']={str(p.relative_to(project)):sha(p) for root in source_dirs if root.exists() for p in sorted(root.rglob('*.scala'))}
    def run(name, argv, cwd=output):
        started=time.monotonic()
        with (output/(name+'.log')).open('w') as log:
            result=subprocess.run(list(map(str,argv)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        receipt['commands'].append({'name':name,'argv':list(map(str,argv)),'cwd':str(cwd),
                                   'seconds':time.monotonic()-started,'returncode':result.returncode})
        (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        result.check_returncode()
    rtl=output/'rtl';rtl.mkdir(exist_ok=True)
    run('elaborate',[tools['java'],f'-XX:ActiveProcessorCount={args.jobs}','-Xmx4G','-Dsbt.supershell=false',
        '-Dsbt.log.noformat=true','-jar',project/'sbt-launch.jar',
        f'midas/runMain merlin.reference.FasedTimingStandaloneMain {selection} {rtl}'],project)
    top='FasedTimingStandalone'; parsed=rtl/(top+'.parsed.mlir'); chirrtl=rtl/(top+'.chirrtl.fir')
    run('parse',[tools['firtool'],'--format=fir','--parse-only',rtl/(top+'.fir'),'-o',parsed])
    run('export',[tools['circt_translate'],'--export-firrtl','--target-line-length=10000',parsed,'-o',chirrtl])
    normalized=rtl/(top+'.gsim.fir');normalized.write_text(expand(chirrtl.read_text()))
    gsim=output/'gsim';gsim.mkdir(exist_ok=True)
    run('gsim',[tools['gsim'],'--dynamic-clocks','--log-level=1','--dir='+str(gsim),normalized])
    verilator=output/'verilator';verilator.mkdir(exist_ok=True)
    common=['-Wno-fatal','--assert','--top-module',top,'--Mdir',verilator,rtl/(top+'.v')]
    run('verilator-xml',[tools['verilator'],'--xml-only',*common])
    for engine in ('gsim','verilator'):
        generate(verilator/('V'+top+'.xml'),top=top,engine=engine,output=output/(engine+'-driver'),gsim_header=gsim/(top+'.h'))
    run('gsim-cxx',[tools['cxx'],'-std=c++17','-O1','-I'+str(gsim),output/'gsim-driver/driver.cpp',
        *sorted(gsim.glob(top+'*.cpp')),'-lstdc++','-o',output/'gsim-driver/run'])
    run('verilator-build',[tools['verilator'],'--cc','--exe','--build','-j',str(args.jobs),*common,output/'verilator-driver/driver.cpp'])
    receipt['outputs']={str(p.relative_to(output)):sha(p) for p in [rtl/(top+'.v'),rtl/(top+'.fir'),normalized,
        rtl/'runtime-defaults.conf',output/'gsim-driver/run',verilator/('V'+top)]}
    (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__': main()
