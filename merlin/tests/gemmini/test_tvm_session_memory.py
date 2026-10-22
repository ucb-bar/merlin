"""Absolute session placement rejects overlap and executes above 4 GiB."""
import importlib
import shutil
import struct
import subprocess

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/comparisons/tvm"))
    return importlib.import_module("package_session"), importlib.import_module("session_memory")


def example(bridge, memory):
    tensor = {"dtype": "int32", "shape": [2], "bytes": 8}
    stage = {"inputs": [tensor], "outputs": [tensor], "workspace_bytes": 64,
             "constants_bytes": 0, "alignment_bytes": 64,
             "heap_required_by_orchestration": False, "runtime_workspace_allocator_required": False,
             "constants_binding": memory.CONSTANT_BINDING}
    stages = {"step": stage}
    route = {"version": 1, "calls": [
        {"stage": "step", "inputs": [{"fixture": 0}]},
        {"stage": "step", "inputs": [{"output": [0, 0]}]},
    ]}
    return bridge.plan_session(stages, route), stages


@pytest.mark.parametrize("fault", ["overlap", "outside", "unaligned", "half", "near_large", "old_package", "no_code"])
def test_refuse_invalid_absolute_memory_map(modules, fault):
    bridge, memory = modules
    plan, stages = example(bridge, memory)
    args = {"weights_base": 0x200000000, "arena_base": 0x300000000}
    if fault == "overlap":
        args["arena_base"] = args["weights_base"]
    elif fault == "outside":
        args["arena_base"] = 0x480000000
    elif fault == "unaligned":
        args["weights_base"] += 1
    elif fault == "half":
        args.pop("arena_base")
    elif fault == "near_large":
        args = {}
    elif fault == "old_package":
        stages["step"].pop("constants_binding")
    else:
        args["arena_base"] = 0x80001000
    with pytest.raises(ValueError):
        memory.plan_memory(plan, stages, 16 << 30, 256 << 10, **args)


def test_layout_accounts_for_padding_and_large_constants_without_allocation(modules):
    bridge, memory = modules
    plan, stages = example(bridge, memory)
    stages["step"]["constants_bytes"] = (3 << 30) + 3
    layout = memory.plan_memory(plan, stages, 16 << 30, 256 << 10,
                                weights_base=0x200000000, arena_base=0x300000000)
    assert layout["references"]["input_0"] == 0x2c0000040
    assert layout["weights_bytes"] == (3 << 30) + 256
    assert layout["near_limit"] == 0x100000000
    assert layout["arena_bytes"] == plan["storage_bytes"]


@pytest.mark.parametrize("negative,expected", [(0, 0), (1, 22)])
def test_absolute_recurrence_reset_and_wrong_oracle(modules, tmp_path, negative, expected):
    bridge, memory = modules
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("Requires a host C compiler")
    plan, stages = example(bridge, memory)
    layout = memory.plan_memory(plan, stages, 16 << 30, 256 << 10,
                                weights_base=0x200000000, arena_base=0x300000000)
    source = '#define _GNU_SOURCE\n#include <sys/mman.h>\n'
    source += bridge.render_session(plan, rtol=0, atol=0, negative_output=1, memory_layout=layout)
    setup = []
    for name, values in {"input_0": [-7, 42], "output_0": [-4, 45], "output_1": [-1, 48]}.items():
        data = ','.join(str(byte) for byte in struct.pack('<2i', *values))
        source += f'const uint8_t blob_{name}[]={{{data}}};\n'
        setup.append(f'memcpy((void*)session_{name},blob_{name},8);')
    source += r'''
int32_t stage_0_model_run(const void *const *in,void *const *out,uint8_t *w,uint64_t n) {
  for(int i=0;i<2;i++)((int32_t*)out[0])[i]=((const int32_t*)in[0])[i]+3;
  return 0;
}
int main(void) {
  void *weights=mmap((void*)0x200000000ULL,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
  void *arena=mmap((void*)0x300000000ULL,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED_NOREPLACE,-1,0);
  if(weights==MAP_FAILED||arena==MAP_FAILED)return 77;
'''
    source += '\n'.join(setup)
    source += f'''
  for(int repeat=0;repeat<2;repeat++) {{
    memset(session_storage,0x33,{plan['storage_bytes']});
    session_reset();int rc=session_run();if(rc)return rc;
    rc=session_validate(repeat ? {negative}:0);if(rc)return rc;
  }}
  return 0;
}}
'''
    path, executable = tmp_path / 'session.c', tmp_path / 'session'
    path.write_text(source)
    subprocess.run([cc, '-std=c11', '-O2', str(path), '-o', str(executable)], check=True, capture_output=True)
    run = subprocess.run([str(executable)], capture_output=True)
    if run.returncode == 77:
        pytest.skip('Host could not reserve the explicit high-address test mappings')
    assert run.returncode == expected, run.stdout
