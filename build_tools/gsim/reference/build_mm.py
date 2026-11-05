#!/usr/bin/env python3
"""Build the native mm_t adapter and optionally link the existing gSIM harness.

The optional harness operation copies/patches only the selected blackbox source
into task output and reuses byte-pinned existing model objects. It never edits
the source checkout or selects this emulator for a campaign.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from generate_native_core import generate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('unsupported blackbox source boundary: '+old[:70])
    return source.replace(old,new,1)


def patch_blackboxes(source):
    source=replace_once(source,'#include "mm.h"','#include "mm.h"\n#include "mm_fased.h"')
    source=replace_once(source,'std::unique_ptr<mm_magic_t> model','std::unique_ptr<mm_fased_t> model')
    source=replace_once(source,'std::make_unique<mm_magic_t>(base, size, data_bits / 8, line_size, backing)',
        'std::make_unique<mm_fased_t>(base, size, data_bits / 8, line_size, backing, address_bits, id_bits)')
    guard='''  if (!reset && ((aw_valid && (aw_lock || (aw_burst != 1 && !(aw_burst == 0 && aw_len == 0)))) ||
                 (ar_valid && (ar_lock || (ar_burst != 1 && !(ar_burst == 0 && ar_len == 0))))))
    throw std::runtime_error("native FASED SimDRAM requires unlocked INCR or single-beat FIXED requests");
'''
    return replace_once(source,'  model->tick(reset,',guard+'  model->tick(reset,')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-build','runtime-settings','vendor-csrc','riscv-include','verilator-include','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--harness-receipt')
    a=p.parse_args();native=Path(a.native_build).resolve();output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=True)
    receipt_path=native/'build-receipt.json';receipt=json.loads(receipt_path.read_text())
    for path,expected in receipt['outputs'].items():
        if sha(native/path)!=expected:raise ValueError('native source/output changed: '+path)
    selection=output/'selection.json';selection.write_text(json.dumps(receipt['selection'],indent=2)+'\n')
    generate(native/'verilator/VFasedTimingStandalone.xml',native/'gsim/FasedTimingStandalone.h',selection,
             Path(a.runtime_settings).resolve(),output)
    assets=Path(__file__).resolve().parent;vendor=Path(a.vendor_csrc).resolve();riscv=Path(a.riscv_include).resolve()
    cxx=receipt['tools']['cxx']['path']
    if sha(cxx)!=receipt['tools']['cxx']['sha256']:raise ValueError('native compiler changed')
    provenance={'schema':'merlin.fased_simdram_integration.v1','native_build_receipt_sha256':sha(receipt_path),'runtime_settings_sha256':sha(a.runtime_settings),
        'configuration':json.loads((output/'configuration.json').read_text()),'commands':[],
        'inputs':{str(x):sha(x) for x in [vendor/'mm.h',vendor/'mm.cc',assets/'mm_fased.cpp',assets/'mm_fased.h',assets/'mm_driver.cpp']},
        'native_source_sha256s':{str(x.relative_to(native)):sha(x) for x in [native/'gsim/FasedTimingStandalone.h',*sorted((native/'gsim').glob('*.cpp'))]},
        'deployment_equivalence':'NOT_ESTABLISHED','model_activated':False}
    def save():(output/'integration-receipt.json').write_text(json.dumps(provenance,indent=2)+'\n')
    def run(name,cmd):
        start=time.monotonic()
        with (output/(name+'.log')).open('w') as stream:
            result=subprocess.run(list(map(str,cmd)),stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        provenance['commands'].append(dict(name=name,argv=list(map(str,cmd)),seconds=time.monotonic()-start,returncode=result.returncode));save();result.check_returncode()
    flags=[cxx,'--driver-mode=g++','-O1','-std=c++17','-I'+str(assets),'-I'+str(output),'-I'+str(vendor),'-I'+str(riscv)]
    core_objects=[]
    for index,source in enumerate(sorted((native/'gsim').glob('FasedTimingStandalone*.cpp'))):
        obj=output/f'core-{index}.o';run(f'core-{index}',[*flags,'-Dgprintf=fased_gprintf','-I'+str(native/'gsim'),'-c',source,'-o',obj]);core_objects.append(obj)
    run('mm',[*flags,'-c',vendor/'mm.cc','-o',output/'mm.o'])
    run('adapter',[*flags,'-I'+str(native/'gsim'),'-c',assets/'mm_fased.cpp','-o',output/'adapter.o'])
    run('driver',[*flags,'-c',assets/'mm_driver.cpp','-o',output/'driver.o'])
    run('gsim-driver',[cxx,'--driver-mode=g++',output/'mm.o',output/'adapter.o',output/'driver.o',*core_objects,'-o',output/'gsim-mm'])
    run('verilator-adapter',[*flags,'-DMERLIN_FASED_VERILATOR','-I'+str(native/'verilator'),'-I'+str(Path(a.verilator_include).resolve()),'-c',assets/'mm_fased.cpp','-o',output/'verilator-adapter.o'])
    run('verilator-driver',[cxx,'--driver-mode=g++',output/'mm.o',output/'verilator-adapter.o',output/'driver.o',
        native/'verilator/VFasedTimingStandalone__ALL.a',native/'verilator/verilated.o',native/'verilator/verilated_threads.o',
        '-pthread','-latomic','-o',output/'verilator-mm'])
    if a.harness_receipt:
        harness_path=Path(a.harness_receipt).resolve();harness=json.loads(harness_path.read_text())
        binary=harness['artifacts']['binary'];
        if sha(binary['path'])!=binary['sha256']:raise ValueError('selected original harness binary changed')
        blackbox=next(c for c in harness['commands'] if c['stage']=='compile' and any(Path(x).name=='blackboxes.cpp' for x in c['argv']))
        source=Path(blackbox['argv'][blackbox['argv'].index('-c')+1])
        pin=next(x for x in harness['inputs'] if x['path']==str(source))
        if sha(source)!=pin['sha256']:raise ValueError('selected blackbox source changed')
        patched=output/'blackboxes.cpp';patched.write_text(patch_blackboxes(source.read_text()))
        cmd=list(blackbox['argv']);old_object=cmd[cmd.index('-o')+1]
        cmd[cmd.index('-c')+1]=str(patched);cmd[cmd.index('-o')+1]=str(output/'blackboxes.o')
        cmd.insert(1,'-I'+str(assets));cmd.insert(1,'-I'+str(source.parent))
        run('blackboxes',cmd)
        link=list(next(c['argv'] for c in harness['commands'] if c['stage']=='link'))
        objects=[Path(x) for x in link if x.endswith(('.o','.a'))]
        provenance['reused_objects']={str(x):sha(x) for x in objects if str(x)!=old_object}
        provenance['original_harness_receipt_sha256']=sha(harness_path)
        provenance['original_blackboxes_sha256']=sha(source);provenance['patched_blackboxes_sha256']=sha(patched)
        link[link.index(old_object)]=str(output/'blackboxes.o');link[link.index('-o')+1]=str(output/'emulator-fased')
        link[1:1]=list(map(str,[output/'adapter.o',*core_objects]))
        run('emulator-link',link)
    provenance['outputs']={str(x.relative_to(output)):sha(x) for x in [output/'gsim-mm',output/'verilator-mm',output/'fased_native_core.h']}
    if (output/'emulator-fased').exists():provenance['outputs']['emulator-fased']=sha(output/'emulator-fased')
    save()


if __name__=='__main__':main()
