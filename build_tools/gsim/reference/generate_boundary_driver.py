#!/usr/bin/env python3
"""Generate a scalar-port cycle driver from Verilator XML, with no RTL parsing.

The stdin/stdout protocol has one ordered decimal value per declared port per
cycle. Values are sampled before the rising edge. All clocks/settings must be
explicit in the caller's stimulus; this driver selects no memory timing prices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def ports(xml_path, top):
    root = ET.parse(xml_path).getroot()
    module = next(x for x in root.findall('.//module') if x.get('name') == top)
    types = {x.get('id'): x for x in root.findall('.//typetable/basicdtype')}
    result = {'inputs': {}, 'outputs': {}}
    for var in module.findall('var'):
        direction = var.get('dir')
        if direction not in ('input', 'output'):
            continue
        name = var.get('name')
        dtype = types.get(var.get('dtype_id'))
        if dtype is None or dtype.get('name') not in ('logic', 'bit'):
            raise ValueError('only explicit scalar logic ports are supported')
        width = abs(int(dtype.get('left', 0)) - int(dtype.get('right', 0))) + 1
        if not 1 <= width <= 64:
            raise ValueError('scalar boundary driver supports 1..64 bit ports')
        if not name.isidentifier() or not name.isascii():
            raise ValueError('port name is not a portable C++ identifier')
        result[direction+'s'][name] = width
    if result['inputs'].pop('clock', None) != 1:
        raise ValueError('one one-bit clock port is required')
    return {key: dict(sorted(value.items())) for key, value in result.items()}


def accessors(header):
    found = {'set': {}, 'get': {}}
    for line in Path(header).read_text().splitlines():
        for mode in found:
            marker = mode+'_'
            begin = line.find(marker)
            if begin < 0:
                continue
            tail = line[begin:].partition('(')[0].strip()
            if not tail.startswith(marker) or any(c in tail for c in ' ;{}'):
                continue
            # gSIM uses dollar separators for aggregate paths. Refuse collisions.
            name = '_'.join(part for part in tail[len(marker):].split('$') if part)
            if name in found[mode] and found[mode][name] != tail:
                raise ValueError('ambiguous generated gSIM accessor')
            found[mode][name] = tail
    return found


def generate(xml_path, *, top, engine, output, gsim_header=None):
    if not top.isidentifier() or not top.isascii():
        raise ValueError('invalid top module identifier')
    io = ports(xml_path, top)
    if engine == 'gsim':
        funcs = accessors(gsim_header)
        for name in [*io['inputs'], 'clock']:
            if name not in funcs['set']:
                raise ValueError('missing gSIM input accessor: '+name)
        for name in io['outputs']:
            if name not in funcs['get']:
                raise ValueError('missing gSIM output accessor: '+name)
        header, cls = top+'.h', 'S'+top
        set_value = lambda name, value: f'dut->{funcs["set"][name]}({value});'
        get_value = lambda name: f'dut->{funcs["get"][name]}()'
        evaluate = 'dut->step();'
        diagnostic = ''  # gSIM's emitted translation unit supplies gprintf.
    elif engine == 'verilator':
        header, cls = 'V'+top+'.h', 'V'+top
        set_value = lambda name, value: f'dut->{name} = {value};'
        get_value = lambda name: f'dut->{name}'
        evaluate, diagnostic = 'dut->eval();', ''
    else:
        raise ValueError('unsupported engine')
    lines = [f'#include "{header}"', '#include <iostream>', '#include <memory>',
             '#include <sstream>', '#include <string>', '#include <cstdarg>', diagnostic,
             'int main() {', f'auto dut = std::make_unique<{cls}>();',
             'std::string line; while (std::getline(std::cin,line)) {',
             'std::istringstream row(line); uint64_t value;', set_value('clock', '0')]
    for name in io['inputs']:
        lines += ['if (!(row >> value)) return 2;']
        if io['inputs'][name] < 64:
            lines += [f'if (value >= (uint64_t(1) << {io["inputs"][name]})) return 3;']
        lines += [set_value(name, 'value')]
    lines += ['if (row >> value) return 2;', evaluate]
    for name in io['outputs']:
        lines += [f'std::cout << static_cast<uint64_t>({get_value(name)}) << " ";']
    lines += ['std::cout << std::endl;', set_value('clock', '1'), evaluate, '} return 0; }']
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'driver.cpp').write_text('\n'.join(lines)+'\n')
    info = dict(schema='merlin.scalar_boundary_driver.v1', engine=engine, top=top, ports=io,
                sampling='before_rising_edge', xml_sha256=hashlib.sha256(Path(xml_path).read_bytes()).hexdigest())
    (output/'ports.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml',required=True);parser.add_argument('--top',required=True)
    parser.add_argument('--engine',choices=['gsim','verilator'],required=True)
    parser.add_argument('--gsim-header');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    generate(args.xml,top=args.top,engine=args.engine,output=args.output,gsim_header=args.gsim_header)


if __name__ == '__main__':
    main()
