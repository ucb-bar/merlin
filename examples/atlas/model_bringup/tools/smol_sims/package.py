#!/usr/bin/env python3
"""Prepare Smol SIMS package from the admitted generated host bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).absolute().parent
sys.path.insert(0,str(PACKAGE.parent))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from smol_full_phases import execution_profile


def bundle_profile(root):
    """Bind packaging configuration to the admitted bundle's exact contract."""
    provenance=json.loads(checked(root/'provenance.json').read_bytes())
    if (provenance.get('schema')!='atlas.full_bundle.v1' or provenance.get('workload')!='smol-full'
        or provenance.get('contract')!='host/smol-full-contract.json'):
        raise ValueError('Require a full Smol bundle contract')
    raw=checked(root/'host/smol-full-contract.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=provenance.get('contract_sha256'):
        raise ValueError('Smol package contract identity differs')
    contract=json.loads(raw)
    if contract.get('schema')!='atlas.smol_full_contract.v1':
        raise ValueError('Require a full Smol host contract')
    profile=execution_profile(contract)
    if (execution_profile(provenance)!=profile or provenance['kernels']!=contract['kernels']
        or provenance.get('attention_kernels',[])!=contract.get('attention_kernels',[])):
        raise ValueError('Smol package execution inventory differs')
    from smol_fusion_assets import profile as fusion_profile
    if (fusion_profile(provenance)!=fusion_profile(contract)
        or provenance.get('call_views')!=contract.get('call_views')
        or provenance.get('fusion_symbol')!=contract.get('fusion_symbol')):
        raise ValueError('Smol package fusion profile/typed inventory differs')
    return profile


def sims_cmake(text,profile,fusion_set='none'):
    text=text.replace('smol_full_driver.c smol_vision_attention.c atlas_host.c',
                      'smol_full_driver.c smol_attention.c smol_vision_attention.c atlas_host.c')
    marker='ATLAS_SMOL_LAUNCH_HOST_TIMING=1)'
    if text.count(marker)!=1: raise ValueError('Unexpected SIMS compile definitions')
    definitions='ATLAS_SMOL_LAUNCH_HOST_TIMING=1 SMOL_ATTENTION_RUNTIME=1'
    if fusion_set != 'none':
        from smol_fusion_assets import SETS
        if fusion_set not in SETS:raise ValueError('Unsupported SIMS fusion profile')
        text=text.replace('smol_full_driver.c smol_attention.c','smol_fusion.c smol_full_driver.c smol_attention.c',1)
        definitions+=' SMOL_FUSION_RUNTIME=1'
    if not profile.attention_count:
        definitions+=' SMOL_FULL_LAUNCH_COUNT=2715 SMOL_FULL_ATTENTION=0'
    return text.replace(marker,definitions+')')



def copy_tree(source,target):
    source,target = checked(source),checked(target)
    target.mkdir(parents=True,exist_ok=True)
    with os.scandir(source) as entries:
        for entry in entries:
            if entry.is_symlink(): raise ValueError('Symlink source is unsupported')
            destination=target/entry.name
            if entry.is_dir(follow_symlinks=False): copy_tree(Path(entry.path),destination)
            elif entry.is_file(follow_symlinks=False): shutil.copyfile(entry.path,destination)
            else: raise ValueError('Unsupported compilation input')

def sims_linker(text):
    if text.count('SECTIONS {')!=1 or text.count('}> DRAM')!=9:
        raise ValueError('Unexpected delivered SIMS linker template')
    text=text.replace('SECTIONS {', '''PHDRS {
  text PT_LOAD FLAGS(5);
  readonly PT_LOAD FLAGS(4);
  writable PT_LOAD FLAGS(6);
  assets PT_LOAD FLAGS(4);
}

SECTIONS {''',1)
    text=text.replace('128K;', '64K;',1)
    text=text.replace('}> DRAM', '}> DRAM :writable')
    text=text.replace('}> DRAM :writable', '}> DRAM :text',1)
    marker='  /* Section for initialized data */'
    before,after=text.split(marker,1)
    before=before.rsplit('}> DRAM :writable',1)
    text=before[0]+'}> DRAM :readonly'+before[1]+marker+after
    text=text.replace('> ASSETS', '> ASSETS :assets')
    text=text.replace('  .bss : ALIGN (8)', '  .bss (NOLOAD) : ALIGN (8)')
    text=text.replace('  PROVIDE(__stack_shift',
        '  __stack_bottom = __stack_start;\n  __stack_top = __stack_end;\n  PROVIDE(__stack_shift')
    return text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','build-work','out'): p.add_argument('--'+name,type=checked,required=True)
    p.add_argument('--template',type=checked,default=PACKAGE/'templates',help='SIMS integration template directory')
    p.add_argument('--platform-root',type=checked,required=True,help='Board-support tree for the target platform')
    a=p.parse_args()
    if a.out.exists(): raise ValueError('Refuse existing package')
    profile=bundle_profile(a.bundle)
    from smol_fusion_assets import profile as fusion_profile
    fusion_set=fusion_profile(json.loads(checked(a.bundle/'provenance.json').read_bytes()))
    a.out.mkdir(parents=True)
    copy_tree(a.build_work/'source',a.out/'source')
    copy_tree(a.build_work/'support',a.out/'source/support')
    shutil.copyfile(checked(a.build_work/'main.c'),a.out/'source/main.c')
    text=checked(a.build_work/'assets.S').read_text()
    text=re.sub(r'(?m)^\.incbin "([^"\n]+)"$',lambda m: '.incbin "'+Path(m[1]).name+'"',text)
    (a.out/'source/assets.S').write_text(text)
    copy_tree(a.template,a.out/'platform_root')
    cmake=a.out/'platform_root/CMakeLists.txt'
    cmake.write_text(sims_cmake(cmake.read_text(),profile,fusion_set))
    linker=a.out/'platform_root/penguin26-smolvla.ld'
    external_linker=checked(a.platform_root/'platform/penguin26/penguin26-smolvla.ld')
    linker.write_text(sims_linker(external_linker.read_text()))
    (a.out/'tools').mkdir()
    shutil.copyfile(PACKAGE/'extract_blobs.py',a.out/'tools/extract_blobs.py')
    os.link(checked(a.bundle/'host/host.elf'),a.out/'smolvla-full-f2.elf')
    (a.out/'preparation.json').write_text(json.dumps(dict(status='PREPARED_SIM_PACKAGE_NOT_EXECUTED',
        bundle=str(a.bundle),build_work=str(a.build_work),template=str(a.template),
        template_sha256={name:hashlib.sha256(checked(a.template/name).read_bytes()).hexdigest() for name in ('CMakeLists.txt','smolvla_main.c')},
        platform_root=str(a.platform_root),external_linker_sha256=hashlib.sha256(external_linker.read_bytes()).hexdigest(),
        sims_linker_sha256=hashlib.sha256(linker.read_bytes()).hexdigest(),sims_stack_bytes=65536),indent=2)+'\n')
    if fusion_set!='none':
        record=json.loads((a.out/'preparation.json').read_bytes());record['fusion_set']=fusion_set
        (a.out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__': main()
