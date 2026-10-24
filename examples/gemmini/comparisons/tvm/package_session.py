"""Validate explicit TVM stage routes and emit their pointer-array ABI adapter.

This comparison edge uses Merlin's existing warm-profile and board runtime. It
does not compile operators, select model semantics or implement an executor.
Every invocation output is retained for validation outside the measured window.
"""

from __future__ import annotations

import math

WIDTHS = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float32": 4,
    "float64": 8,
    "bfloat16": 2,
}


def aligned(size):
    return (size + 63) // 64 * 64


def tensor_abi(tensor):
    dtype, shape, size = (tensor.get(key) for key in ("dtype", "shape", "bytes"))
    if dtype not in WIDTHS or not isinstance(shape, list) or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("Unsupported tensor dtype or nonpositive/static shape")
    if type(size) is not int or size != math.prod(shape) * WIDTHS[dtype] or size >= 1 << 63:
        raise ValueError("Tensor byte extent differs from its dtype/shape")
    return dtype, tuple(shape), size


def plan_session(stages, route):
    """Use the verifier's version-one routes; forbid missing or inferred edges."""
    if (
        not isinstance(stages, dict)
        or not stages
        or any(not isinstance(name, str) or not name.isascii() or not name.isidentifier() for name in stages)
    ):
        raise ValueError("Require named static stages")
    for stage in stages.values():
        if (
            stage.get("alignment_bytes") != 64
            or stage.get("heap_required_by_orchestration") is not False
            or stage.get("runtime_workspace_allocator_required") is not False
        ):
            raise ValueError("Require aligned static stages without hidden allocation")
        for group in ("inputs", "outputs"):
            if not isinstance(stage.get(group), list) or not stage[group]:
                raise ValueError("Require complete stage input/output ABI")
            for tensor in stage[group]:
                tensor_abi(tensor)
        for field in ("workspace_bytes", "constants_bytes"):
            if type(stage.get(field)) is not int or not 0 <= stage[field] < 1 << 63:
                raise ValueError("Invalid stage storage extent")
    if (
        not isinstance(route, dict)
        or set(route) != {"version", "calls"}
        or type(route["version"]) is not int
        or route["version"] != 1
        or not isinstance(route["calls"], list)
        or not route["calls"]
    ):
        raise ValueError("Require version-one explicit session calls")
    fixtures, outputs, calls, used = {}, [], [], set()
    for ordinal, call in enumerate(route["calls"]):
        if not isinstance(call, dict) or set(call) != {"stage", "inputs"} or call["stage"] not in stages:
            raise ValueError("Unknown or malformed stage call")
        stage = stages[call["stage"]]
        if not isinstance(call["inputs"], list) or len(call["inputs"]) != len(stage["inputs"]):
            raise ValueError("Every input requires an explicit route")
        used.add(call["stage"])
        inputs = []
        for edge, tensor in zip(call["inputs"], stage["inputs"]):
            if not isinstance(edge, dict) or len(edge) != 1:
                raise ValueError("Malformed input route")
            if "fixture" in edge:
                index = edge["fixture"]
                if type(index) is not int or not 0 <= index <= 65535:
                    raise ValueError("Invalid fixture index")
                if index in fixtures and tensor_abi(fixtures[index]) != tensor_abi(tensor):
                    raise ValueError("Conflicting fixture ABI")
                fixtures[index] = tensor
                inputs.append(["input", index])
            elif "output" in edge:
                source = edge["output"]
                if (
                    not isinstance(source, list)
                    or len(source) != 2
                    or any(type(i) is not int for i in source)
                    or not 0 <= source[0] < ordinal
                    or not 0 <= source[1] < len(calls[source[0]]["outputs"])
                ):
                    raise ValueError("State route must select an earlier invocation output")
                index = calls[source[0]]["outputs"][source[1]]
                if tensor_abi(outputs[index]) != tensor_abi(tensor):
                    raise ValueError("Routed state ABI mismatch")
                inputs.append(["output", index])
            else:
                raise ValueError("Routes admit only fixtures and prior outputs")
        indices = list(range(len(outputs), len(outputs) + len(stage["outputs"])))
        outputs.extend(stage["outputs"])
        calls.append({"stage": call["stage"], "inputs": inputs, "outputs": indices})
    if used != set(stages) or set(fixtures) != set(range(len(fixtures))):
        raise ValueError("All stages and a dense fixture inventory must be used")
    inputs = [fixtures[i] for i in range(len(fixtures))]
    workspace = max(stage["workspace_bytes"] for stage in stages.values())
    storage, offsets = 0, {}
    for kind, tensors in (("input", inputs), ("output", outputs), ("workspace", [{"bytes": workspace}])):
        offsets[kind] = []
        for tensor in tensors:
            # Each buffer has independent 64-byte leading and trailing guards.
            offsets[kind].append(storage + 64)
            storage += 128 + aligned(tensor["bytes"])
    return {
        "stages": list(stages),
        "calls": calls,
        "inputs": inputs,
        "outputs": outputs,
        "offsets": offsets,
        "workspace_bytes": workspace,
        "storage_bytes": storage,
        "constants_bytes": sum(stage["constants_bytes"] for stage in stages.values()),
        "reference_bytes": sum(aligned(t["bytes"]) for t in inputs + outputs),
        "memory_scope": "All invocation outputs retained; serial stage workspace; guarded inputs/outputs. Add linked code, constants, runtime and stack.",
    }


_CHECK_HELPERS = r"""
static int session_bytes_equal(const uint8_t *a,const uint8_t *b,uint64_t n) {
  for(uint64_t i=0;i<n;i++) if(a[i]!=b[i]) return 0;
  return 1;
}
static int session_guard(uint64_t at,uint64_t n) {
  for(uint64_t i=0;i<64;i++) if(session_storage[at-64+i]!=0xa5) return 0;
  uint64_t end=(n+63)/64*64+64;
  for(uint64_t i=n;i<end;i++) if(session_storage[at+i]!=0xa5) return 0;
  return 1;
}
static double session_float(const uint8_t *p,int bits) {
  if(bits==64) { double v; memcpy(&v,p,8); return v; }
  union {uint32_t u;float f;} v;
  if(bits==16) { uint16_t b;memcpy(&b,p,2);v.u=(uint32_t)b<<16; }
  else memcpy(&v.u,p,4);
  return (double)v.f;
}
static int session_float_equal(const uint8_t *a,const uint8_t *b,uint64_t n,
                               int bits,double rtol,double atol) {
  for(uint64_t i=0;i<n;i+=bits/8) {
    double x=session_float(a+i,bits),y=session_float(b+i,bits);
    if(!__builtin_isfinite(x)||!__builtin_isfinite(y)) return 0;
    double d=x-y;if(d<0)d=-d;if(y<0)y=-y;
    if(!(d<=atol+rtol*y)) return 0;
  }
  return 1;
}
"""


def render_session(plan, *, rtol, atol, negative_output=0, memory_layout=None):
    """Emit only the stage ABI adapter; reference blobs are external constants."""
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if type(negative_output) is not int or not 0 <= negative_output < len(plan["outputs"]):
        raise ValueError("Wrong-oracle output is outside the session")
    absolute = memory_layout is not None and memory_layout["kind"] == "absolute"
    lines = [
        "#include <stdint.h>",
        "#include <string.h>",
        "#include <stdio.h>",
        "#ifndef BAD_ORACLE\n#define BAD_ORACLE 0\n#endif",
    ]
    lines.append(
        f"#define session_storage ((uint8_t *)(uintptr_t)UINT64_C({memory_layout['arena_base']}))"
        if absolute
        else f"_Alignas(64) static uint8_t session_storage[{plan['storage_bytes']}];"
    )
    for index, _ in enumerate(plan["stages"]):
        lines.append(f"extern int32_t stage_{index}_model_run(const void *const *,void *const *,uint8_t *,uint64_t);")
    for kind in ("input", "output"):
        for index, _ in enumerate(plan[kind + "s"]):
            name = f"{kind}_{index}"
            lines.append(
                f"#define session_{name} ((const uint8_t *)(uintptr_t)UINT64_C({memory_layout['references'][name]}))"
                if absolute
                else f"extern const uint8_t session_{name}[];"
            )
    lines.append(_CHECK_HELPERS)
    lines += ["static void session_reset(void) {", f"  memset(session_storage,0xa5,{plan['storage_bytes']});"]
    for index, tensor in enumerate(plan["inputs"]):
        lines.append(
            f"  memcpy(session_storage+{plan['offsets']['input'][index]},session_input_{index},{tensor['bytes']});"
        )
    lines += ["}", "static int session_run(void) {"]
    workspace = plan["offsets"]["workspace"][0]
    for ordinal, call in enumerate(plan["calls"]):
        args = ",".join(f"session_storage+{plan['offsets'][kind][i]}" for kind, i in call["inputs"])
        outs = ",".join(f"session_storage+{plan['offsets']['output'][i]}" for i in call["outputs"])
        stage = plan["stages"].index(call["stage"])
        lines += [
            f"  {{ const void *inputs[]={{{args}}}; void *outputs[]={{{outs}}};",
            f"    int rc=stage_{stage}_model_run(inputs,outputs,session_storage+{workspace},{plan['workspace_bytes']});",
            f'    if(rc) {{ printf("TVM_SESSION_STAGE_FAILURE call={ordinal} rc=%d\\n",rc);return 21; }} }}',
        ]
    lines += ["  return 0;", "}", "static int session_validate(int negative) {"]
    for kind in ("input", "output"):
        for index, tensor in enumerate(plan[kind + "s"]):
            at = plan["offsets"][kind][index]
            size, dtype = tensor["bytes"], tensor["dtype"]
            lines.append(f"  if(!session_guard({at},{size})) return 23;")
            args = f"session_storage+{at},session_{kind}_{index},{size}"
            if kind == "output" and dtype in ("float32", "float64", "bfloat16"):
                bits = 16 if dtype == "bfloat16" else WIDTHS[dtype] * 8
                check = f"session_float_equal({args},{bits},{float(rtol).hex()},{float(atol).hex()})"
            else:
                check = f"session_bytes_equal({args})"
            # The negative build compares a deliberately different reference
            # value through the same comparison function, never skips execution.
            if kind == "output" and index == negative_output:
                lines += [
                    "  if(negative) { _Alignas(64) uint8_t wrong[8];",
                    f"    memcpy(wrong,session_output_{index},{WIDTHS[dtype]});",
                ]
                if dtype in ("float32", "float64", "bfloat16"):
                    # Finite, opposite-sign maximum tests numerical comparison,
                    # not just the nonfinite-value rejection path.
                    bits = 16 if dtype == "bfloat16" else WIDTHS[dtype] * 8
                    value = {16: "UINT16_C(0x7f7f)", 32: "UINT32_C(0x7f7fffff)", 64: "UINT64_C(0x7fefffffffffffff)"}[
                        bits
                    ]
                    lines += [
                        f"    uint{bits}_t bad={value};",
                        f"    if(session_float(wrong,{bits})>=0) bad|=((uint{bits}_t)1<<{bits - 1});",
                        f"    memcpy(wrong,&bad,{bits // 8});",
                        f"    if(!session_float_equal(session_storage+{at},wrong,{bits // 8},{bits},{float(rtol).hex()},{float(atol).hex()})) return 22;",
                    ]
                else:
                    lines += [
                        "    wrong[0]^=1;",
                        f"    if(!session_bytes_equal(session_storage+{at},wrong,{WIDTHS[dtype]})) return 22;",
                    ]
                lines += ["  }"]
            lines.append(f'  if(!{check}) {{ printf("TVM_SESSION_MISMATCH {kind}={index}\\n");return 22; }}')
    lines += [f"  if(!session_guard({workspace},{plan['workspace_bytes']})) return 23;", "  return 0;", "}"]
    return "\n".join(lines) + "\n"
