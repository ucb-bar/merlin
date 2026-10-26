"""Execute the pointer-array session adapter against independent C stages."""

import copy
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def bridge():
    path = repo_root() / "examples/gemmini/comparisons/tvm/package_session.py"
    spec = importlib.util.spec_from_file_location("tvm_package_session_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor(dtype="float32", size=8):
    return {"dtype": dtype, "shape": [2], "bytes": size}


def stage(inputs, outputs):
    return {
        "inputs": inputs,
        "outputs": outputs,
        "workspace_bytes": 64,
        "constants_bytes": 0,
        "alignment_bytes": 64,
        "heap_required_by_orchestration": False,
        "runtime_workspace_allocator_required": False,
    }


def example():
    stages = {"prefill": stage([tensor()], [tensor()]), "decode": stage([tensor(), tensor()], [tensor(), tensor()])}
    route = {
        "version": 1,
        "calls": [
            {"stage": "prefill", "inputs": [{"fixture": 0}]},
            {"stage": "decode", "inputs": [{"fixture": 1}, {"output": [0, 0]}]},
            {"stage": "decode", "inputs": [{"fixture": 2}, {"output": [1, 1]}]},
        ],
    }
    return stages, route


def test_routes_keep_candidate_state_and_all_outputs(bridge):
    stages, route = example()
    plan = bridge.plan_session(stages, route)
    assert plan["calls"][2]["inputs"][1] == ["output", 2]
    assert len(plan["outputs"]) == 5
    assert len(set(plan["offsets"]["output"])) == 5
    assert all(offset % 64 == 0 for offsets in plan["offsets"].values() for offset in offsets)


@pytest.mark.parametrize("fault", ["future", "wrong_dtype", "missing", "sparse", "hidden", "extent"])
def test_refuse_invalid_session_before_codegen(bridge, fault):
    stages, route = example()
    if fault == "future":
        route["calls"][1]["inputs"][1] = {"output": [1, 0]}
    elif fault == "wrong_dtype":
        stages["decode"]["inputs"][1]["dtype"] = "int32"
    elif fault == "missing":
        route["calls"][1]["inputs"].pop()
    elif fault == "sparse":
        route["calls"][0]["inputs"] = [{"fixture": 3}]
    elif fault == "hidden":
        stages["decode"]["runtime_workspace_allocator_required"] = True
    else:
        stages["prefill"]["inputs"][0]["bytes"] = 9
    with pytest.raises(ValueError):
        bridge.plan_session(stages, route)


@pytest.mark.parametrize("fault,expected", [(0, 0), (1, 22), (2, 23), (3, 21), (4, 22)])
def test_compiled_session_recurrence_guards_negative_and_reset(bridge, tmp_path, fault, expected):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("Requires a host C compiler")
    plan = bridge.plan_session(*example())
    source = bridge.render_session(plan, rtol=1e-5, atol=2e-6, negative_output=4)
    # The byte blobs have the independent reference values. Return the cache
    # through output 1, which deliberately differs from the logits in output 0.
    import struct

    values = {
        "input_0": [2.0, 4.0],
        "input_1": [1.0, -2.0],
        "input_2": [3.0, 1.0],
        "output_0": [3.0, 5.0],
        "output_1": [5.0, 1.0],
        "output_2": [4.0, 3.0],
        "output_3": [10.0, 5.0],
        "output_4": [7.0, 4.0],
    }
    for name, value in values.items():
        data = ",".join(str(byte) for byte in struct.pack("<2f", *value))
        source += f"\n_Alignas(64) const uint8_t session_{name}[]={{{data}}};\n"
    source += r"""
static int count;
int32_t stage_0_model_run(const void *const *in,void *const *out,uint8_t *w,uint64_t n) {
  for(int i=0;i<2;i++) ((float *)out[0])[i]=((const float *)in[0])[i]+1;
  count++;return 0;
}
int32_t stage_1_model_run(const void *const *in,void *const *out,uint8_t *w,uint64_t n) {
  for(int i=0;i<2;i++) {
    float t=((const float *)in[0])[i],s=((const float *)in[1])[i];
    ((float *)out[0])[i]=s+2*t;((float *)out[1])[i]=s+t;
  }
  if(FAULT==2) ((uint8_t *)out[1])[8]=0;
  if(FAULT==4) ((float *)out[1])[0]+=1;
  count++;return FAULT==3 ? -7:0;
}
int main(void) {
  for(int repeat=0;repeat<2;repeat++) {
    session_reset();int rc=session_run();if(rc)return rc;
    rc=session_validate(FAULT==1);if(rc)return rc;
    memset(session_storage,0x33,sizeof(session_storage));
  }
  return count==6 ? 0:99;
}
"""
    path = tmp_path / "session.c"
    path.write_text(source)
    binary = tmp_path / "session"
    subprocess.run(
        [cc, "-std=c11", "-O2", f"-DFAULT={fault}", str(path), "-o", str(binary)], check=True, capture_output=True
    )
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == expected, result.stdout


def test_boolean_route_and_nan_policy(bridge):
    stages, route = example()
    with pytest.raises(ValueError):
        bridge.render_session(bridge.plan_session(stages, route), rtol=float("nan"), atol=0)
    changed = copy.deepcopy(route)
    changed["calls"][0]["inputs"][0] = {"fixture": False}
    with pytest.raises(ValueError):
        bridge.plan_session(stages, changed)


def test_binary_readback_uses_actual_signed_words(bridge, monkeypatch, tmp_path):
    from merlin.runtime.out_bin import parse_binary_console_details

    cc = shutil.which("cc")
    if not cc:
        pytest.skip("Requires a host C compiler")
    root = repo_root()
    monkeypatch.syspath_prepend(str(root / "examples/gemmini/comparisons/tvm"))
    from build_package_session import readback_source

    plan = bridge.plan_session(
        {"one": stage([tensor("int32")], [tensor("int32")])},
        {"version": 1, "calls": [{"stage": "one", "inputs": [{"fixture": 0}]}]},
    )
    source = bridge.render_session(plan, rtol=0, atol=0)
    source += r"""
void htif_putc(char c) { putchar(c); }
void htif_puts(const char *s) { fputs(s,stdout); }
void htif_line_flush(int value) { (void)value;fflush(stdout); }
const uint8_t session_input_0[]={0,0,0,0,0,0,0,0};
const uint8_t session_output_0[]={0,0,0,0,0,0,0,0};
int32_t stage_0_model_run(const void *const *i,void *const *o,uint8_t *w,uint64_t n) {
  ((int32_t *)o[0])[0]=-1234567;((int32_t *)o[0])[1]=INT32_MIN;return 0;
}
"""
    source += readback_source(plan)
    source += 'int main(void){session_reset();session_run();int rc=session_readback();if(!rc)puts("DONE");return rc;}\n'
    path, binary = tmp_path / "readback.c", tmp_path / "readback"
    path.write_text(source)
    subprocess.run(
        [cc, "-std=c11", "-O2", "-I" + str(root / "merlin/runtime/baremetal"), str(path), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True)
    outputs, _, frames = parse_binary_console_details(result.stdout)
    assert outputs["out0"] == [[-1234567, -(1 << 31)]]
    assert frames["out0"].signed


def test_stage_profile_keeps_warm_and_measured_buffers_and_prints_later(bridge, tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("Requires a host C compiler")
    plan = bridge.plan_session(
        {"one": stage([tensor("int32")], [tensor("int32")])},
        {"version": 1, "calls": [{"stage": "one", "inputs": [{"fixture": 0}]}]},
    )
    source = bridge.render_session(plan, rtol=0, atol=0, profile_stages=True)
    source += r"""
const uint8_t session_input_0[8]={0};
const uint8_t session_output_0[8]={0};
static uint64_t tick;
static uint64_t read_cycles(void) { return ++tick; }
int32_t stage_0_model_run(const void *const *in,void *const *out,uint8_t *w,uint64_t n) {
  memcpy(out[0],in[0],8);tick+=10;return 0;
}
int main(void) {
  for(int r=0;r<2;++r) {
    session_reset();if(session_run()||session_validate(0))return 1;
  }
  if(session_stage_cycles[0][0]!=11||session_stage_cycles[1][0]!=11)return 2;
  puts("validated");session_report_stage_cycles();
  return session_run()==26 ? 0:3;
}
"""
    path = tmp_path / "profile.c"
    path.write_text(source)
    binary = tmp_path / "profile"
    subprocess.run([cc, "-std=c11", "-O2", str(path), "-o", str(binary)], check=True, capture_output=True)
    process = subprocess.run([str(binary)], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout
    assert process.stdout.splitlines() == [
        "validated",
        "TVM_SESSION_STAGE_CYCLES phase=warm call=0 stage=one cycles=11",
        "TVM_SESSION_STAGE_CYCLES phase=measured call=0 stage=one cycles=11",
    ]


def test_operator_profile_keeps_each_call_and_both_phases(bridge, tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("Requires a host C compiler")
    profiled = stage([tensor("int32")], [tensor("int32")])
    profiled["calls"] = [{"symbol": "add"}]
    profiled["operator_profile"] = {
        "count": 1,
        "count_entrypoint": "model_profile_count",
        "cycles_entrypoint": "model_profile_cycles",
        "symbol_entrypoint": "model_profile_symbol",
    }
    plan = bridge.plan_session(
        {"one": profiled},
        {
            "version": 1,
            "calls": [
                {"stage": "one", "inputs": [{"fixture": 0}]},
                {"stage": "one", "inputs": [{"output": [0, 0]}]},
            ],
        },
    )
    source = bridge.render_session(plan, rtol=0, atol=0, profile_stages=True)
    source += r"""
const uint8_t session_input_0[8]={0},session_output_0[8]={0},session_output_1[8]={0};
static uint64_t tick,calls;
static uint64_t read_cycles(void) { return ++tick; }
uint32_t stage_0_model_profile_count(void) { return 1; }
uint64_t stage_0_model_profile_cycles(uint32_t i) { return 100+calls; }
const char *stage_0_model_profile_symbol(uint32_t i) { return "add"; }
int32_t stage_0_model_run(const void *const *in,void *const *out,uint8_t *w,uint64_t n) {
  memcpy(out[0],in[0],8);++calls;return 0;
}
int main(void) {
  for(int r=0;r<2;++r) {session_reset();if(session_run()||session_validate(0))return 1;}
  puts("validated");session_report_stage_cycles();return 0;
}
"""
    path = tmp_path / "operators.c"
    path.write_text(source)
    binary = tmp_path / "operators"
    subprocess.run([cc, "-std=c11", "-O2", str(path), "-o", str(binary)], check=True, capture_output=True)
    process = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    lines = process.stdout.splitlines()
    assert lines[0] == "validated"
    operators = [line for line in lines if line.startswith("TVM_SESSION_OPERATOR_CYCLES")]
    assert operators == [
        "TVM_SESSION_OPERATOR_CYCLES phase=warm call=0 stage=one operator=0 symbol=add cycles=101",
        "TVM_SESSION_OPERATOR_CYCLES phase=measured call=0 stage=one operator=0 symbol=add cycles=103",
        "TVM_SESSION_OPERATOR_CYCLES phase=warm call=1 stage=one operator=0 symbol=add cycles=102",
        "TVM_SESSION_OPERATOR_CYCLES phase=measured call=1 stage=one operator=0 symbol=add cycles=104",
    ]
    # A regular session never calls the optional profiling API.
    assert "stage_0_model_profile_cycles" not in bridge.render_session(plan, rtol=0, atol=0)
    profiled["operator_profile"]["count"] = 2
    with pytest.raises(ValueError, match="profiling ABI"):
        bridge.plan_session({"one": profiled}, {"version": 1, "calls": [{"stage": "one", "inputs": [{"fixture": 0}]}]})
