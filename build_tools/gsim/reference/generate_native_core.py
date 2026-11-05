#!/usr/bin/env python3
"""Generate native core bindings and explicit immutable runtime configuration."""
import argparse
import hashlib
import json
from pathlib import Path
from generate_boundary_driver import ports, accessors


def generate(xml, gsim_header, selection, settings, output):
    io=ports(xml,'FasedTimingStandalone'); funcs=accessors(gsim_header)
    selection=json.loads(Path(selection).read_text()); axi=selection['axi']
    config={};requested={}
    for token in Path(settings).read_text().split():
        key,sep,value=token.partition('=')
        if not key.startswith('+mm_') or not key.endswith('_0') or not sep or not value.isdecimal() or key in requested:
            raise ValueError('runtime settings must be explicit unique model-0 nonnegative assignments')
        port='io_mmReg_'+key[4:-2]
        if port not in io['inputs']: raise ValueError('unknown runtime port '+port)
        requested[key]=int(value);config[port]=int(value)&((1<<io['inputs'][port])-1)
    if set(config)!={name for name in io['inputs'] if name.startswith('io_mmReg_')}:
        raise ValueError('runtime configuration is incomplete')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lines=['#pragma once','#include <cstdint>','#include <memory>',
      '#ifdef MERLIN_FASED_VERILATOR','#include "VFasedTimingStandalone.h"',
      'using FasedRTL = VFasedTimingStandalone;','#else','#include "FasedTimingStandalone.h"',
      'using FasedRTL = SFasedTimingStandalone;','#endif','namespace merlin_fased {']
    for direction in ('inputs','outputs'):
        lines.append('struct '+direction.title()+' {')
        lines.extend('uint64_t '+name+' = 0;' for name in io[direction])
        lines.append('};')
    lines+=['struct Core {','std::unique_ptr<FasedRTL> rtl = std::make_unique<FasedRTL>();',
            'Outputs evaluate(const Inputs& in, bool clock) {','Outputs out;','#ifdef MERLIN_FASED_VERILATOR','rtl->clock=clock;']
    lines+=['rtl->'+name+'=in.'+name+';' for name in io['inputs']]
    lines+=['rtl->eval();']+['out.'+name+'=rtl->'+name+';' for name in io['outputs']]
    lines+=['#else','rtl->'+funcs['set']['clock']+'(clock);']
    lines+=['rtl->'+funcs['set'][name]+'(in.'+name+');' for name in io['inputs']]
    lines+=['rtl->step();']+['out.'+name+'=rtl->'+funcs['get'][name]+'();' for name in io['outputs']]
    lines+=['#endif','return out;','}','};','inline Inputs configuration() { Inputs in;']
    lines += ['in.'+name+'='+str(value)+'ULL;' for name,value in sorted(config.items())]
    lines+=['return in; }']
    for name in ('data_bits','address_bits','id_bits','max_read_bytes','max_write_bytes','address_base','address_size'):
        lines += [f'constexpr uint64_t {name} = {axi[name]}ULL;']
    identity={'selection':selection,'requested_settings':requested,'effective_settings':config,
              'xml_sha256':hashlib.sha256(Path(xml).read_bytes()).hexdigest()}
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    lines += [f'constexpr const char* configuration_sha256="{digest}";','}']
    (output/'fased_native_core.h').write_text('\n'.join(lines)+'\n')
    (output/'configuration.json').write_text(json.dumps(dict(identity,configuration_sha256=digest),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('xml','gsim-header','selection','settings','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();generate(a.xml,a.gsim_header,a.selection,a.settings,a.output)


if __name__=='__main__':main()
