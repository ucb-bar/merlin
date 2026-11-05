"""Cheap parser/driver contract tests; actual RTL probes are separately recorded."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).parent

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_else_chains_preserve_nesting_and_comments():
    source='''when a :
  x <= a
else when b :
  x <= b
; comment outside indentation
else when c :
  x <= c
else :
  x <= d
y <= e
'''
    assert load('expand_else_when').expand(source)=='''when a :
  x <= a
else :
  when b :
    x <= b
; comment outside indentation
  else :
    when c :
      x <= c
    else :
      x <= d
y <= e
'''


def test_verification_expressions_preserved_labels_unique():
    source='  assert(clock, cond, UInt<1>(1), "hello") : check @[a 1:2]\n'*2
    result=load('expand_else_when').expand(source)
    assert result.count('assert(clock, cond, UInt<1>(1), "hello")')==2
    assert 'check_boundary_0' in result and 'check_boundary_1' in result


def test_tab_indentation_rejected():
    with pytest.raises(ValueError,match='tab'):
        load('expand_else_when').expand('\twhen a :\n')


def xml(tmp_path,width=8):
    path=tmp_path/'dut.xml'
    path.write_text(f'''<verilator_xml><netlist><module name="Dut">
      <var name="clock" dir="input" dtype_id="1"/>
      <var name="value" dir="input" dtype_id="2"/>
      <var name="result" dir="output" dtype_id="2"/>
      </module><typetable><basicdtype id="1" name="logic"/>
      <basicdtype id="2" name="logic" left="{width-1}" right="0"/>
      </typetable></netlist></verilator_xml>''')
    return path


def test_driver_rejects_out_of_width_input(tmp_path):
    load('generate_boundary_driver').generate(xml(tmp_path),top='Dut',engine='verilator',output=tmp_path/'driver')
    text=(tmp_path/'driver/driver.cpp').read_text()
    assert 'if (value >= (uint64_t(1) << 8)) return 3;' in text
    assert text.index('dut->eval();') < text.index('std::cout') < text.index('dut->clock = 1;')


def test_driver_rejects_wide_boundary(tmp_path):
    with pytest.raises(ValueError,match='1..64'):
        load('generate_boundary_driver').ports(xml(tmp_path,128),'Dut')


def test_gsim_accessors_refuse_collisions(tmp_path):
    path=tmp_path/'dut.h';path.write_text('void set_io$$data(int x);\nvoid set_io$data(int x);\n')
    with pytest.raises(ValueError,match='ambiguous'):
        load('generate_boundary_driver').accessors(path)


def test_production_patch_preserves_existing_shared_backing():
    sys.path.insert(0,str(ROOT))
    try:
        module=load('build_mm')
    finally:
        sys.path.pop(0)
    source='''#include "mm.h"
std::unique_ptr<mm_magic_t> model;
model = std::make_unique<mm_magic_t>(base, size, data_bits / 8, line_size, backing);
  model->tick(reset, remaining_args);
'''
    result=module.patch_blackboxes(source)
    assert 'line_size, backing, address_bits, id_bits)' in result
    assert 'mm_magic_t' not in result
    assert 'aw_lock' in result and 'ar_burst' in result
    assert 'model->tick(reset, remaining_args)' in result
    with pytest.raises(ValueError,match='unsupported blackbox'):
        module.patch_blackboxes(result)


def test_production_configuration_requires_all_explicit_runtime_settings(tmp_path):
    sys.path.insert(0,str(ROOT))
    try:
        module=load('generate_native_core')
    finally:
        sys.path.pop(0)
    header=tmp_path/'dut.h';header.write_text('void set_clock(int);\nvoid set_value(int);\nint get_result();\n')
    selection=tmp_path/'selection.json';selection.write_text('{"axi":{}}')
    settings=tmp_path/'settings.conf';settings.write_text('+mm_missing_0=1\n')
    with pytest.raises(ValueError,match='unknown runtime port'):
        path=xml(tmp_path);path.write_text(path.read_text().replace('name="Dut"','name="FasedTimingStandalone"'))
        module.generate(path,header,selection,settings,tmp_path/'output')
