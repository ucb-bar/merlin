"""The whole-model device route calls a package's kernels under the logical kernel ABI.

A candidate compiler is graded capsule by capsule through the runner-owned logical harness: dense
row-major logical tensors, pointers in the interface's declaration order, then the results. The
whole-model build (the private full-model gate's route) must call the same kernels the same way, or a
kernel that passed every capsule is handed padded copies in another order and computes garbage that
still returns. These tests pin the selection (the logical ABI unless a support explicitly selects the
version-1 resident ABI), the pointer order resolved from each group's own interface, and -- compiled
and run on the host -- a device build of off-tile-edge contractions whose kernels follow the logical
ABI and whose results are exact.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower import device_shim as DS
from merlin.targetgen.contract import harness_abi as HA
from merlin.targetgen.contract.interface_emit import emit_interface_mlir

pytestmark = pytest.mark.target("gemmini")

_CC = shutil.which("cc") or shutil.which("gcc")


def _resident(order):
    tensors = {
        "W": {"shape": [8, 6], "dtype": "i8", "role": "weight"},
        "A": {"shape": [5, 8], "dtype": "i8", "role": "input"},
    }
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {name: tensors[name] for name in order},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}, "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "W_res", "dst": "acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc", "dst": "Y"},
                "attributes": {"output_dtype": "i32", "epilogue": []},
            },
        ],
    }


# ------------------------------------------------------------------ which ABI is selected


def test_the_logical_abi_is_the_default_and_the_resident_abi_is_opt_in():
    assert HA.kernel_abi_version({"harness_abi": {"entry_symbol": "t_kernel"}}, target="t") == 2
    assert HA.kernel_abi_version(None, target="t") == 2
    assert HA.kernel_abi_version({"harness_abi": {"kernel_abi_version": 1}}, target="t") == 1
    with pytest.raises(HA.HarnessAbiError):
        HA.kernel_abi_version({"harness_abi": {"kernel_abi_version": "1"}}, target="t")


def test_the_selected_gemmini_support_gets_the_logical_abi():
    abi = DS.kernel_abi_for("gemmini")
    assert abi is not None and abi.version == 2 and abi.symbol == "gemmini_kernel"
    assert "no padding" in abi.pointee_layout


def test_an_explicit_legacy_selection_gets_the_resident_abi(monkeypatch):
    monkeypatch.setattr(HA, "kernel_abi_version_for", lambda _device: HA.LEGACY_KERNEL_ABI_VERSION)
    abi = DS.kernel_abi_for("gemmini")
    assert abi is not None and abi.version == 1 and "resident_matmul" in abi.arg_order


# ------------------------------------------------------------------ pointer order per interface


@pytest.mark.parametrize(("order", "roles"), [(("W", "A"), ("rhs", "lhs", "out")), (("A", "W"), ("lhs", "rhs", "out"))])
def test_the_pointer_order_is_the_interface_declaration_order(order, roles):
    assert DS.logical_argument_roles(emit_interface_mlir(_resident(order))) == (roles, "")


def test_an_interface_with_a_fourth_pointer_is_refused_not_guessed():
    cb = _resident(("W", "A"))
    cb["tensors"]["bias"] = {"shape": [6], "dtype": "i32", "role": "bias"}
    cb["commands"][2]["attributes"]["epilogue"] = ["bias_add"]
    cb["commands"][2]["operands"]["bias"] = "bias"
    roles, why = DS.logical_argument_roles(emit_interface_mlir(cb))
    assert roles is None and "4 pointers" in why


def test_a_symbol_without_a_resolved_order_is_declined():
    unit = DS.emit_translation_unit("gemmini", {"s": (20, 24, 40)}, {"s": ("i8", "i8", "i32")})
    assert not unit.symbols and "no logical kernel argument order" in unit.skipped[0][1]


def test_off_edge_extents_are_called_dense_in_the_resolved_order():
    unit = DS.emit_translation_unit(
        "gemmini", {"s": (20, 24, 40)}, {"s": ("i8", "i8", "i32")}, argument_roles={"s": ("lhs", "rhs", "out")}
    )
    assert unit.symbols == ("s",)
    assert "s_a[" not in unit.text, "the logical ABI is dense: nothing is staged or padded"
    assert "gemmini_kernel((void *)a_addr, (void *)b_addr, (void *)c_addr);" in unit.text


# ------------------------------------------------------------------ a whole-model device build, run

#: The test's stand-in device package: emits, for each group interface, an LLVM-dialect kernel that
#: reads its pointers the way the logical ABI passes them (declaration order, then the result) and
#: hands them to a host reference contraction. No padding, no fixed weight-first order.
_TOOL = r'''#!{python}
import json, sys
from merlin.targetgen.contract.interface_emit import parse_interface_mlir

args = sys.argv[1:]
src = args[-1]
cb = parse_interface_mlir(open(src, encoding="utf-8").read())
names = [n for n, s in cb["tensors"].items() if s.get("role") in ("input", "weight")]
weight = next(n for n, s in cb["tensors"].items() if s.get("role") == "weight")
lhs = next(n for n in names if n != weight)
m, k = cb["tensors"][lhs]["shape"]
n = cb["tensors"][weight]["shape"][1]
if any(a.startswith("--emit-command-buffer=") for a in args):
    open(next(a for a in args if a.startswith("--emit-command-buffer=")).split("=", 1)[1], "w").write(json.dumps(cb))
    sys.exit(0)
if "--emit-target-artifact" in args:
    i_lhs, i_rhs = names.index(lhs), names.index(weight)
    print(f"""module {{
  llvm.func @ref_matmul_i8_i32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64)
  llvm.func @gemmini_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {{
    %m = llvm.mlir.constant({m} : i64) : i64
    %n = llvm.mlir.constant({n} : i64) : i64
    %k = llvm.mlir.constant({k} : i64) : i64
    llvm.call @ref_matmul_i8_i32(%arg{i_lhs}, %arg{i_rhs}, %arg2, %m, %n, %k)
      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
    llvm.return
  }}
}}""")
    sys.exit(0)
print(open(src, encoding="utf-8").read())
'''

_CONVERT = ["{tool}", "--convert-iface-to-gemmini"]
_MANIFEST = {
    "artifact_type": "mlir_oot_target_backend",
    "target": "gemmini",
    "package_id": "logical_abi_probe",
    "language": "python",
    "authoring": {"mode": "deterministic_generated_from_spec", "author": "test", "generated_by_agent": False},
    "integrity_exempt": False,
    "entrypoints": {"tool": "tool.py"},
    "commands": {
        "parse": {"argv": ["{tool}", "--verify-diagnostics", "{input_mlir}"]},
        "lower_interface_to_target": {"argv": [*_CONVERT, "{input_mlir}"]},
        "emit_command_buffer": {"argv": [*_CONVERT, "--emit-command-buffer={output_json}", "{input_mlir}"]},
        "lower_target_to_llvm": {"argv": [*_CONVERT, "--emit-target-artifact", "{input_mlir}"]},
    },
}

_HOST = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct { void *a; void *b; intptr_t o; intptr_t s[2]; intptr_t st[2]; } mr2;
void ref_matmul_i8_i32(const int8_t *a, const int8_t *b, int32_t *c, int64_t m, int64_t n, int64_t k) {
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      int32_t s = 0;
      for (int64_t q = 0; q < k; ++q) s += (int32_t)a[i * k + q] * (int32_t)b[q * n + j];
      c[i * n + j] = s;
    }
}
"""


def _driver(sym, m, n, k):
    return f"""
extern mr2 {sym}(void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                 void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                 void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t);
static void run_{sym}(FILE *in, FILE *out) {{
  static int8_t A[{m}*{k}], B[{k}*{n}];
  static int32_t C[{m}*{n}];
  if (fread(A, 1, sizeof A, in) != sizeof A || fread(B, 1, sizeof B, in) != sizeof B) exit(2);
  {sym}(A,A,0,{m},{k},{k},1,  B,B,0,{k},{n},{n},1,  C,C,0,{m},{n},{n},1);
  fwrite(C, 1, sizeof C, out);
}}
"""


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_logical_abi_device_build_computes_off_edge_contractions_exactly(tmp_path):
    from merlin.llvmlower.device_build import build_device_objects

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "manifest.yaml").write_text(json.dumps(_MANIFEST), encoding="utf-8")
    tool = pkg / "tool.py"
    tool.write_text(_TOOL.replace("{python}", sys.executable), encoding="utf-8")
    tool.chmod(0o755)
    signatures = {"g0": (20, 24, 40), "g1": (16, 16, 16)}  # (M, N, K): the first misses a 16 tile edge
    dtypes = {sym: ("i8", "i8", "i32") for sym in signatures}
    built = build_device_objects(
        "gemmini",
        signatures,
        dtypes,
        package_dir=pkg,
        workdir=tmp_path / "work",
        operand_dtype="int8",
        accum_dtype="i32",
        codegen_target="x86",
        cflags=["-O1", "-fPIC"],
        timeout=300,
    )
    if not built.kernels:
        pytest.skip(f"device build unavailable here: {built.skipped}")
    assert set(built.kernels) == set(signatures), built.skipped
    shim = built.shim_object.with_suffix(".c").read_text(encoding="utf-8")
    assert "_a[" not in shim and "pointer order of the logical kernel ABI for this group: rhs, lhs, out" in shim
    host = tmp_path / "host.c"
    body = "".join(_driver(sym, m, n, k) for sym, (m, n, k) in signatures.items())
    calls = "".join(f"  run_{sym}(stdin, stdout);\n" for sym in signatures)
    host.write_text(_HOST + body + f"int main(void) {{\n{calls}  return 0;\n}}\n", encoding="utf-8")
    exe = tmp_path / "model"
    subprocess.run([_CC, "-O1", "-o", str(exe), str(host), *map(str, built.objects)], check=True, capture_output=True)
    rng = np.random.default_rng(7)
    feed, want = b"", []
    for m, n, k in signatures.values():
        a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
        b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
        feed += a.tobytes() + b.tobytes()
        want.append(a.astype(np.int32) @ b.astype(np.int32))
    got = subprocess.run([str(exe)], input=feed, capture_output=True, check=True).stdout
    offset = 0
    for expected in want:
        size = expected.size * 4
        assert np.array_equal(
            np.frombuffer(got[offset : offset + size], dtype=np.int32).reshape(expected.shape), expected
        )
        offset += size
    assert offset == len(got)
    assert list(built.skipped) == []


# ------------------------------------------------------------------ whole groups: every pointer of the interface

#: A stand-in device package for whole groups. For each contraction of its interface it emits a call
#: to a host reference that applies the commit's own readout (bias, scale, relu, narrowing), reading
#: every pointer where the logical ABI puts it: the interface's inputs in declaration order, then its
#: results in result order.
_GROUP_TOOL = r"""#!{python}
import json, struct, sys
from merlin.targetgen.contract import harness_render as HR
from merlin.targetgen.contract.interface_emit import parse_interface_mlir

args = sys.argv[1:]
src = args[-1]
cb = parse_interface_mlir(open(src, encoding="utf-8").read())
if any(a.startswith("--emit-command-buffer=") for a in args):
    open(next(a for a in args if a.startswith("--emit-command-buffer=")).split("=", 1)[1], "w").write(json.dumps(cb))
    sys.exit(0)
if "--emit-target-artifact" not in args:
    print(open(src, encoding="utf-8").read())
    sys.exit(0)
order = HR.kernel_arg_order(cb)
tensors = cb["tensors"]
resident = {c["operands"]["dst"]: c["operands"]["src"] for c in cb["commands"] if c["opcode"] == "RES_PACK"}
commits = {c["operands"]["src"]: c for c in cb["commands"] if c["opcode"] == "COMMIT"}
body = ["    %null = llvm.mlir.zero : !llvm.ptr"]
for i, c in enumerate(cb["commands"]):
    if c["opcode"] == "RESIDUAL_ADD":
        attrs, ops = c.get("attributes") or {}, c["operands"]
        lhs, rhs, out = ops["lhs"], ops["rhs"], ops["dst"]
        count = 1
        for extent in tensors[lhs]["shape"]:
            count *= extent
        bits = [struct.unpack("<i", struct.pack("<f", float(attrs[key])))[0] for key in ("lhs_scale", "rhs_scale")]
        body.append(f"    %rn{i} = llvm.mlir.constant({count} : i64) : i64")
        for j, value in enumerate([*bits, int("relu" in (attrs.get("epilogue") or []))]):
            body.append(f"    %rf{i}_{j} = llvm.mlir.constant({value} : i32) : i32")
        body.append(
            f"    llvm.call @ref_group_add(%arg{order.index(lhs)}, %arg{order.index(rhs)}, %arg{order.index(out)}, "
            f"%rn{i}, %rf{i}_0, %rf{i}_1, %rf{i}_2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32) -> ()"
        )
        continue
    if c["opcode"] == "CONV2D":
        attrs, ops = c.get("attributes") or {}, c["operands"]
        ifm, w, out = ops["ifm"], resident.get(ops["weight"], ops["weight"]), ops["dst"]
        _n, h, wd, ci = tensors[ifm]["shape"]
        kh, kw, _ci, co = attrs["kernel"]
        epilogue = attrs.get("epilogue") or []
        bias = attrs.get("bias")
        scale = float(attrs.get("acc_scale", 1.0)) if "acc_scale" in epilogue else 1.0
        bits = struct.unpack("<i", struct.pack("<f", scale))[0]
        pool = [*attrs.get("pool_size", [1, 1]), *attrs.get("pool_stride", [1, 1])]
        pool += [*attrs.get("pool_padding", [0, 0, 0, 0]), int(attrs.get("pool_pad_value", 0))]
        geometry = [h, wd, ci, co, kh, kw, *attrs["stride"], *attrs["padding"], *pool]
        flags = [bits, int("relu" in epilogue), int(str(attrs.get("output_dtype")) == "i8")]
        names = []
        for j, value in enumerate(geometry):
            body.append(f"    %g{i}_{j} = llvm.mlir.constant({value} : i64) : i64")
            names.append(f"%g{i}_{j}")
        for j, value in enumerate(flags):
            body.append(f"    %f{i}_{j} = llvm.mlir.constant({value} : i32) : i32")
            names.append(f"%f{i}_{j}")
        b = f"%arg{order.index(bias)}" if bias else "%null"
        types = ", ".join(["!llvm.ptr"] * 4 + ["i64"] * len(geometry) + ["i32"] * len(flags))
        body.append(
            f"    llvm.call @ref_group_conv(%arg{order.index(ifm)}, %arg{order.index(w)}, {b}, "
            f"%arg{order.index(out)}, {', '.join(names)}) : ({types}) -> ()"
        )
        continue
    if c["opcode"] not in ("MATMUL", "MATMUL_RESIDENT"):
        continue
    lhs, rhs = c["operands"]["lhs"], resident.get(c["operands"]["rhs"], c["operands"]["rhs"])
    commit = commits[c["operands"]["dst"]]
    attrs, out = commit.get("attributes") or {}, commit["operands"]["dst"]
    m, k = tensors[lhs]["shape"]
    n = tensors[rhs]["shape"][1]
    epilogue = attrs.get("epilogue") or []
    bias = attrs.get("bias") or (commit["operands"].get("bias"))
    scale = float(attrs.get("acc_scale", 1.0)) if "acc_scale" in epilogue else 1.0
    bits = struct.unpack("<i", struct.pack("<f", scale))[0]
    narrow = 1 if str(attrs.get("output_dtype")) == "i8" else 0
    values = [("m", m, "i64"), ("n", n, "i64"), ("k", k, "i64"), ("s", bits, "i32"),
              ("r", int("relu" in epilogue), "i32"), ("q", narrow, "i32")]
    for tag, value, ty in values:
        body.append(f"    %{tag}{i} = llvm.mlir.constant({value} : {ty}) : {ty}")
    b = f"%arg{order.index(bias)}" if bias else "%null"
    body.append(
        f"    llvm.call @ref_group_mm(%arg{order.index(lhs)}, %arg{order.index(rhs)}, {b}, %arg{order.index(out)}, "
        f"%m{i}, %n{i}, %k{i}, %s{i}, %r{i}, %q{i}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, "
        "i64, i64, i64, i32, i32, i32) -> ()"
    )
params = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(len(order)))
print("module {")
print("  llvm.func @ref_group_mm(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32)")
conv_types = ", ".join(["!llvm.ptr"] * 4 + ["i64"] * 21 + ["i32"] * 3)
print(f"  llvm.func @ref_group_conv({conv_types})")
print("  llvm.func @ref_group_add(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32)")
print(f"  llvm.func @gemmini_kernel({params}) {{")
print("\n".join(body))
print("    llvm.return\n  }\n}")
"""

_GROUP_REFERENCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/* The stand-in device's arithmetic: int32 accumulation, then bias, scale (round half to even), relu,
 * and an optional int8 narrowing -- the readout the commit states. */
void ref_group_mm(const int8_t *a, const int8_t *w, const int32_t *bias, void *y, int64_t m, int64_t n,
                  int64_t k, int32_t scale_bits, int32_t relu, int32_t narrow) {
  float scale;
  memcpy(&scale, &scale_bits, sizeof scale);
  for (int64_t i = 0; i < m; ++i)
    for (int64_t j = 0; j < n; ++j) {
      int32_t acc = 0;
      for (int64_t q = 0; q < k; ++q) acc += (int32_t)a[i * k + q] * (int32_t)w[q * n + j];
      if (bias) acc += bias[j];
      float v = nearbyintf((float)acc * scale);
      if (relu && v < 0) v = 0;
      if (narrow) ((int8_t *)y)[i * n + j] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
      else ((int32_t *)y)[i * n + j] = (int32_t)v;
    }
}

/* A residual sum as the interface states it: relu(sat(roundeven(f32(a) * ls + f32(b) * rs))), int8. */
void ref_group_add(const int8_t *a, const int8_t *b, int8_t *y, int64_t n, int32_t ls_bits, int32_t rs_bits,
                   int32_t relu) {
  float ls, rs;
  memcpy(&ls, &ls_bits, sizeof ls);
  memcpy(&rs, &rs_bits, sizeof rs);
  for (int64_t i = 0; i < n; ++i) {
    float v = nearbyintf((float)a[i] * ls + (float)b[i] * rs);
    v = v > 127 ? 127 : v < -128 ? -128 : v;
    if (relu && v < 0) v = 0;
    y[i] = (int8_t)v;
  }
}

/* A convolution read the way the interface states it: IFM [1, H, W, C] (NHWC), W [kh*kw*C, Co] in
 * (tap_h, tap_w, channel) order, zero padding [top, left, bottom, right]; the readout (bias, scale,
 * relu, narrowing) then, when declared, a max pool of the readout's values whose padded cells hold
 * pool_pad_value; result [rows*cols, Co]. */
void ref_group_conv(const int8_t *x, const int8_t *w, const int32_t *bias, void *y, int64_t h, int64_t wd,
                    int64_t c, int64_t co, int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t pt, int64_t pl,
                    int64_t pb, int64_t pr, int64_t qh, int64_t qw, int64_t qsh, int64_t qsw, int64_t qt,
                    int64_t ql, int64_t qb, int64_t qr, int64_t qpad, int32_t scale_bits, int32_t relu,
                    int32_t narrow) {
  float scale;
  memcpy(&scale, &scale_bits, sizeof scale);
  int64_t ho = (h + pt + pb - kh) / sh + 1, wo = (wd + pl + pr - kw) / sw + 1;
  int32_t *v = malloc(sizeof(int32_t) * ho * wo * co);
  for (int64_t oh = 0; oh < ho; ++oh)
    for (int64_t ow = 0; ow < wo; ++ow)
      for (int64_t o = 0; o < co; ++o) {
        int32_t acc = 0;
        for (int64_t i = 0; i < kh; ++i)
          for (int64_t j = 0; j < kw; ++j) {
            int64_t ih = oh * sh + i - pt, iw = ow * sw + j - pl;
            if (ih < 0 || iw < 0 || ih >= h || iw >= wd) continue;
            for (int64_t ch = 0; ch < c; ++ch)
              acc += (int32_t)x[(ih * wd + iw) * c + ch] * (int32_t)w[((i * kw + j) * c + ch) * co + o];
          }
        if (bias) acc += bias[o];
        float r = nearbyintf((float)acc * scale);
        if (relu && r < 0) r = 0;
        if (narrow) r = r > 127 ? 127 : r < -128 ? -128 : r;
        v[(oh * wo + ow) * co + o] = (int32_t)r;
      }
  int64_t po = (ho + qt + qb - qh) / qsh + 1, pw = (wo + ql + qr - qw) / qsw + 1;
  for (int64_t a = 0; a < po; ++a)
    for (int64_t b = 0; b < pw; ++b)
      for (int64_t o = 0; o < co; ++o) {
        int32_t best = INT32_MIN;
        for (int64_t i = 0; i < qh; ++i)
          for (int64_t j = 0; j < qw; ++j) {
            int64_t ih = a * qsh + i - qt, iw = b * qsw + j - ql;
            int32_t cell = (ih < 0 || iw < 0 || ih >= ho || iw >= wo) ? (int32_t)qpad : v[(ih * wo + iw) * co + o];
            if (cell > best) best = cell;
          }
        int64_t at = (a * pw + b) * co + o;
        if (narrow) ((int8_t *)y)[at] = (int8_t)best;
        else ((int32_t *)y)[at] = best;
      }
  free(v);
}
"""


def _group_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    manifest = dict(_MANIFEST, package_id="logical_group_probe")
    (pkg / "manifest.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    tool = pkg / "tool.py"
    tool.write_text(_GROUP_TOOL.replace("{python}", sys.executable), encoding="utf-8")
    tool.chmod(0o755)
    return pkg


def _lowered_object(text: str, work):
    """The module through Merlin's own host lowering, as an x86 object."""
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.toolchain import clang

    lowered = lower_model(text, work, targets=())
    obj = work / "model.o"
    subprocess.run(
        [clang(), "-O1", "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)], check=True, capture_output=True
    )
    return obj


def _run(tmp_path, objects, main_c: str, feed: bytes) -> bytes:
    from merlin.llvmlower.toolchain import clang

    host = tmp_path / "main.c"
    host.write_text(_GROUP_REFERENCE + main_c, encoding="utf-8")
    exe = tmp_path / "model"
    subprocess.run(
        [clang(), "-O1", "-o", str(exe), str(host), *map(str, objects), "-lm"], check=True, capture_output=True
    )
    return subprocess.run([str(exe)], input=feed, capture_output=True, check=True).stdout


_MEMREF_C = """
typedef struct { void *al; void *ali; intptr_t off; intptr_t s[1]; intptr_t st[1]; } m1;
typedef struct { void *al; void *ali; intptr_t off; intptr_t s[2]; intptr_t st[2]; } m2;
static m1 d1(void *p, intptr_t n) { m1 d = {p, p, 0, {n}, {1}}; return d; }
static m2 d2(void *p, intptr_t r, intptr_t c) { m2 d = {p, p, 0, {r, c}, {c, 1}}; return d; }
"""


def _rne(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


def _fused_route(capture: str, work: str, package: str, model: str = "two") -> dict:
    """The group route and device build of the capture at ``capture``, as plain data.

    Run in a CHILD PROCESS with the gemmini support selected (the route reads which readout stages
    the target closes from that contract): selecting a support loads its plugin for the life of the
    process, and the rest of the suite runs with none selected."""
    from pathlib import Path

    from merlin.common import mlir_query as mq
    from merlin.llvmlower.device_build import build_device_objects
    from merlin.llvmlower.device_offload import rewrite_groups_to_device
    from merlin.xdsl_dialects._common import text as to_text

    module = mq.parse((Path(capture) / "model.mlir").read_text(encoding="utf-8"))
    rewrite = rewrite_groups_to_device(
        module, "gemmini", select=lambda _shape: True, model=model, capture=Path(capture) / "model.mlir"
    )
    built = build_device_objects(
        "gemmini",
        rewrite.signatures,
        {r.symbol: r.dtypes for r in rewrite.routed},
        package_dir=package,
        workdir=Path(work) / "device",
        operand_dtype="int8",
        accum_dtype="i32",
        codegen_target="x86",
        cflags=["-O1", "-fPIC"],
        entries=rewrite.entries,
        call_buffers=rewrite.call_buffers,
        timeout=300,
    )
    (Path(work) / "routed.mlir").write_text(to_text(module), encoding="utf-8")
    return {
        "skipped": [list(s) for s in rewrite.skipped],
        "signatures": sorted(rewrite.signatures),
        "call_buffers": rewrite.call_buffers,
        "kernels": sorted(built.kernels),
        "build_skipped": [list(s) for s in built.skipped],
        "objects": [str(o) for o in built.objects],
        "shim": str(built.shim_object.with_suffix(".c")) if built.shim_object else None,
        "ops": {sym: entry.get("op") for sym, entry in rewrite.entries.items()},
        "epilogues": {sym: list(entry.get("epilogue") or ()) for sym, entry in rewrite.entries.items()},
    }


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_fused_bias_scale_relu_whole_model_runs_exactly_under_the_logical_abi(tmp_path):
    """Two chained quantized linear layers, each bias_add + acc_scale + relu + quantize: the route
    rewrites each into ONE call passing the activation, the stored weight, the folded bias and the
    destination; the device build matches each to the kernel's logical interface (W, A0, B, Y0) and
    the program, lowered by Merlin's host pipeline and run, equals the capture's quantized semantics."""
    import os
    import struct as _struct

    from merlin.common.paths import repo_root

    sys.path.insert(0, str(repo_root() / "merlin" / "tests" / "runtime"))
    from test_whole_model_group_route import _two_layers  # noqa: PLC0415

    capture = tmp_path / "capture"
    capture.mkdir()
    text = _two_layers()
    (capture / "model.mlir").write_text(text, encoding="utf-8")
    rng = np.random.default_rng(3)
    stored = {
        "wa": rng.integers(-8, 8, (8, 16)).astype(np.int8),
        "biasa": (rng.integers(-12, 12, 16) * 0.25).astype(np.float32),
        "wb": rng.integers(-8, 8, (16, 32)).astype(np.int8),
        "biasb": (rng.integers(-12, 12, 32) * 0.25).astype(np.float32),
        "biasr": np.zeros(4, dtype=np.float32),
    }
    header, payload = {}, b""
    for name, array in stored.items():
        raw = array.tobytes()
        spelled = "I8" if array.dtype == np.int8 else "F32"
        header[name] = {
            "dtype": spelled,
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    blob = json.dumps(header).encode()
    (capture / "model.safetensors").write_bytes(_struct.pack("<Q", len(blob)) + blob + payload)
    manifest = {"0": {"kind": "input"}, **{str(i + 1): {"kind": "weight", "weight": n} for i, n in enumerate(stored)}}
    (capture / "model.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, MERLIN_TARGET_PATH=str(repo_root() / "examples" / "gemmini" / "support"))
    script = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); from test_device_shim_logical_abi import _fused_route; "
        "print(json.dumps(_fused_route(*sys.argv[2:])))"
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parent),
            str(capture),
            str(work),
            str(_group_package(tmp_path)),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert child.returncode == 0, child.stderr[-3000:]
    route = json.loads(child.stdout.strip().splitlines()[-1])
    assert len(route["signatures"]) == 2, route["skipped"]
    for call in route["call_buffers"].values():
        assert [op["role"] for op in call] == ["input_0", "weight_0", "bias_0", "out_0"]
    assert route["kernels"] == route["signatures"], route["build_skipped"]
    assert "pointer order W, A0, B, Y0" in Path(route["shim"]).read_text(encoding="utf-8")
    model = _lowered_object((work / "routed.mlir").read_text(encoding="utf-8"), tmp_path / "lower")
    built_objects = [Path(o) for o in route["objects"]]
    main_c = (
        _MEMREF_C
        + """
#include <stdio.h>
/* Merlin's host lowering passes the model's result as a caller-allocated out-parameter, last. */
extern void _mlir_ciface_forward(m2 *, m2 *, m1 *, m2 *, m1 *, m1 *, m2 *);
int main(void) {
  static int8_t x[4 * 8], wa[8 * 16], wb[16 * 32], y[4 * 32];
  static float ba[16], bb[32], br[4];
  if (fread(x, 1, sizeof x, stdin) != sizeof x || fread(wa, 1, sizeof wa, stdin) != sizeof wa ||
      fread(wb, 1, sizeof wb, stdin) != sizeof wb) return 2;
  m2 dy = d2(y, 4, 32), dx = d2(x, 4, 8), dwa = d2(wa, 8, 16), dwb = d2(wb, 16, 32);
  m1 dba = d1(ba, 16), dbb = d1(bb, 32), dbr = d1(br, 4);
  _mlir_ciface_forward(&dx, &dwa, &dba, &dwb, &dbb, &dbr, &dy);
  fwrite(y, 1, sizeof y, stdout);
  return 0;
}
"""
    )
    x = rng.integers(-128, 128, (4, 8)).astype(np.int8)
    got = _run(tmp_path, [model, *built_objects], main_c, x.tobytes() + stored["wa"].tobytes() + stored["wb"].tobytes())

    def layer(act, weight, bias):  # the capture's own float program: dq, matmul, bias, relu, quantize
        value = (act.astype(np.float64) * 0.5) @ (weight.astype(np.float64) * 0.5) + bias
        return _rne(np.maximum(value, 0.0) / 0.5)

    want = layer(layer(x, stored["wa"], stored["biasa"]), stored["wb"], stored["biasb"])
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(4, 32), want)


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_two_result_group_passes_both_results_and_runs_exactly(tmp_path):
    """One resident weight read by two contractions, each committing its own result: the call passes
    both activations, the weight and BOTH destinations (in an order unlike the kernel's), and both
    results come back through Merlin's host lowering, exact."""
    from merlin.llvmlower.device_build import build_device_objects

    entry = {
        "name": "two_results",
        "op": "resident_reuse",
        "kind": "op",
        "source_role": "model_derived",
        "source_reference": "test",
        "K": 12,
        "N": 7,
        "matmuls": [{"lhs": "A0", "out": "Y0", "M": 3}, {"lhs": "A1", "out": "Y1", "M": 5, "epilogue": ["relu"]}],
    }
    sym = "merlin_dev_gemmini_0"
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.targetgen import corpus_spec as CS

    buffers, why = DS.logical_buffers(CS.build(dict(entry), _mesh_tile_binding("gemmini", "int8", "i32"))[1])
    assert buffers is not None, why
    by_role = {b.role: b for b in buffers}
    assert [b.role for b in buffers] == ["weight_0", "input_0", "input_1", "out_0", "out_1"]
    call = [
        {"role": role, "shape": list(by_role[role].shape), "dtype": by_role[role].dtype}
        for role in ("input_0", "input_1", "weight_0", "out_0", "out_1")
    ]
    built = build_device_objects(
        "gemmini",
        {sym: (3, 7, 12)},
        {sym: ("i8", "i8", "i32")},
        package_dir=_group_package(tmp_path),
        workdir=tmp_path / "device",
        operand_dtype="int8",
        accum_dtype="i32",
        codegen_target="x86",
        cflags=["-O1", "-fPIC"],
        entries={sym: entry},
        call_buffers={sym: call},
        timeout=300,
    )
    assert set(built.kernels) == {sym}, built.skipped
    t0, t1 = (f"tensor<{'x'.join(map(str, by_role[r].shape))}x{by_role[r].dtype}>" for r in ("out_0", "out_1"))
    args = "tensor<3x12xi8>, tensor<5x12xi8>, tensor<12x7xi8>, " + f"{t0}, {t1}"
    access = ", ".join(f'{{bufferization.access = "{a}"}}' for a in ("read", "read", "read", "write", "write"))
    text = f"""builtin.module {{
  func.func @forward(%a0: tensor<3x12xi8>, %a1: tensor<5x12xi8>, %w: tensor<12x7xi8>) -> ({t0}, {t1}) {{
    %e0 = tensor.empty() : {t0}
    %e1 = tensor.empty() : {t1}
    %r:2 = func.call @{sym}(%a0, %a1, %w, %e0, %e1) : ({args}) -> ({t0}, {t1})
    func.return %r#0, %r#1 : {t0}, {t1}
  }}
  "func.func"() <{{sym_name = "{sym}", function_type = ({args}) -> ({t0}, {t1}),
      sym_visibility = "private", arg_attrs = [{access}]}}> ({{
  }}) : () -> ()
}}
"""
    model = _lowered_object(text, tmp_path / "lower")
    n0, n1 = (int(np.prod(by_role[r].shape)) for r in ("out_0", "out_1"))
    e0, e1 = (4 if by_role[r].dtype == "i32" else 1 for r in ("out_0", "out_1"))
    main_c = (
        _MEMREF_C
        + f"""
#include <stdio.h>
/* Merlin's host lowering passes each model result as a caller-allocated out-parameter, last. */
extern void _mlir_ciface_forward(m2 *, m2 *, m2 *, m2 *, m2 *);
int main(void) {{
  static int8_t a0[3 * 12], a1[5 * 12], w[12 * 7];
  static char y0[{n0 * e0}], y1[{n1 * e1}];
  if (fread(a0, 1, sizeof a0, stdin) != sizeof a0 || fread(a1, 1, sizeof a1, stdin) != sizeof a1 ||
      fread(w, 1, sizeof w, stdin) != sizeof w) return 2;
  m2 da0 = d2(a0, 3, 12), da1 = d2(a1, 5, 12), dw = d2(w, 12, 7), dy0 = d2(y0, 3, 7), dy1 = d2(y1, 5, 7);
  _mlir_ciface_forward(&da0, &da1, &dw, &dy0, &dy1);
  fwrite(y0, 1, sizeof y0, stdout);
  fwrite(y1, 1, sizeof y1, stdout);
  return 0;
}}
"""
    )
    rng = np.random.default_rng(5)
    a0, a1 = rng.integers(-128, 128, (3, 12)).astype(np.int8), rng.integers(-128, 128, (5, 12)).astype(np.int8)
    w = rng.integers(-128, 128, (12, 7)).astype(np.int8)
    got = _run(tmp_path, [model, *built.objects], main_c, a0.tobytes() + a1.tobytes() + w.tobytes())
    acc0, acc1 = a0.astype(np.int32) @ w.astype(np.int32), a1.astype(np.int32) @ w.astype(np.int32)
    want0 = acc0 if e0 == 4 else _rne(acc0)
    want1 = np.maximum(acc1, 0) if e1 == 4 else _rne(np.maximum(acc1, 0))
    dt0, dt1 = (np.int32 if e == 4 else np.int8 for e in (e0, e1))
    assert np.array_equal(np.frombuffer(got[: n0 * e0], dtype=dt0).reshape(3, 7), want0)
    assert np.array_equal(np.frombuffer(got[n0 * e0 :], dtype=dt1).reshape(5, 7), want1)


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_batched_call_runs_the_slice_kernel_once_per_disjoint_slice(tmp_path):
    """A batch is a loop: the host passes [B, ...] buffers, the kernel takes one slice's logical tensors."""
    buffers = (
        DS.LogicalBuffer("A", "input", "input_0", (2, 4), "i8"),
        DS.LogicalBuffer("B", "input", "input_1", (4, 5), "i8"),
        DS.LogicalBuffer("Y", "output", "out_0", (2, 5), "i32"),
    )
    call = [{"role": b.role, "shape": [3, *b.shape], "dtype": b.dtype} for b in buffers]
    unit = DS.emit_logical_translation_unit("gemmini", {"s": (buffers, call)}, kernel_symbol_for=lambda _s: "k")
    assert unit.symbols == ("s",), unit.skipped
    (tmp_path / "shim.c").write_text(unit.text, encoding="utf-8")
    (tmp_path / "main.c").write_text(
        _GROUP_REFERENCE
        + """
#include <stdio.h>
typedef struct { void *al; void *ali; intptr_t off; intptr_t s[3]; intptr_t st[3]; } m3;
void _mlir_ciface_s(m3 *, const m3 *, const m3 *, const m3 *);
void k(void *a, void *b, void *y) { ref_group_mm(a, b, 0, y, 2, 5, 4, 0x3f800000, 0, 0); }
int main(void) {
  static int8_t a[3 * 2 * 4], b[3 * 4 * 5]; static int32_t y[3 * 2 * 5]; m3 r;
  if (fread(a, 1, sizeof a, stdin) != sizeof a || fread(b, 1, sizeof b, stdin) != sizeof b) return 2;
  m3 da = {a, a, 0, {3, 2, 4}, {8, 4, 1}}, db = {b, b, 0, {3, 4, 5}, {20, 5, 1}}, dy = {y, y, 0, {3, 2, 5}, {10, 5, 1}};
  _mlir_ciface_s(&r, &da, &db, &dy);
  fwrite(y, 1, sizeof y, stdout);
  return r.ali == (void *)y ? 0 : 3;
}
""",
        encoding="utf-8",
    )
    exe = tmp_path / "batched"
    subprocess.run(
        [_CC, "-Wall", "-Werror", "-O1", "-o", str(exe), str(tmp_path / "main.c"), str(tmp_path / "shim.c"), "-lm"],
        check=True,
        capture_output=True,
    )
    rng = np.random.default_rng(11)
    a = rng.integers(-128, 128, (3, 2, 4)).astype(np.int8)
    b = rng.integers(-128, 128, (3, 4, 5)).astype(np.int8)
    got = subprocess.run([str(exe)], input=a.tobytes() + b.tobytes(), capture_output=True, check=True).stdout
    want = np.einsum("bmk,bkn->bmn", a.astype(np.int32), b.astype(np.int32))
    assert np.array_equal(np.frombuffer(got, dtype=np.int32).reshape(3, 2, 5), want)


# ------------------------------------------------------------------ captured convolutions, routed from their image


class _Capture:
    """A quantized model spelled the way a capture lowers it on the host (generic MLIR form).

    A convolution is strided ``extract_slice``s of the zero-padded NCHW image, one per tap, each
    transposed to NHWC and flattened to ``[positions, channels]``, concatenated into the patch matrix
    an integer matmul reads against the weight's ``[Co, C, kh, kw] -> [kh*kw*C, Co]`` view, then the
    NCHW readout: two scales, a per-channel bias, relu and a quantize. A residual connection is two
    dequantizes, a sum, relu and a quantize. Every tensor between layers is NCHW int8."""

    def __init__(self):
        self.lines: list[str] = []
        self.count = 0
        self.args: list[tuple[str, str]] = []

    def fresh(self):
        self.count += 1
        return f"%v{self.count}"

    @staticmethod
    def ty(shape, dt):
        return f"tensor<{''.join(f'{d}x' for d in shape)}{dt}>"

    @staticmethod
    def arr(values):
        return f"array<i64: {', '.join(map(str, values))}>"

    @staticmethod
    def reassoc(groups):
        return "[" + ", ".join("[" + ", ".join(f"{i} : i64" for i in g) + "]" for g in groups) + "]"

    def arg(self, name, shape, dt):
        self.args.append((f"%{name}", self.ty(shape, dt)))
        return f"%{name}"

    def op(self, text):
        name = self.fresh()
        self.lines.append(f"    {name} = {text}")
        return name

    def empty(self, shape, dt):
        return self.op(f'"tensor.empty"() : () -> {self.ty(shape, dt)}')

    def transpose(self, src, shape, perm, dt):
        out = [shape[i] for i in perm]
        init = self.empty(out, dt)
        region = f'({{\n    ^bb0(%a: {dt}, %b: {dt}):\n      "linalg.yield"(%a) : ({dt}) -> ()\n    }})'
        return self.op(
            f'"linalg.transpose"({src}, {init}) <{{permutation = {self.arr(perm)}}}> {region} '
            f": ({self.ty(shape, dt)}, {self.ty(out, dt)}) -> {self.ty(out, dt)}"
        ), out

    def collapse(self, src, shape, groups, dt):
        out = [int(np.prod([shape[i] for i in g])) for g in groups]
        return self.op(
            f'"tensor.collapse_shape"({src}) <{{reassociation = {self.reassoc(groups)}}}> '
            f": ({self.ty(shape, dt)}) -> {self.ty(out, dt)}"
        ), out

    def expand(self, src, shape, groups, out, dt):
        attrs = f"reassociation = {self.reassoc(groups)}, static_output_shape = {self.arr(out)}"
        return self.op(
            f'"tensor.expand_shape"({src}) <{{{attrs}}}> : ({self.ty(shape, dt)}) -> {self.ty(out, dt)}'
        ), out

    def extract(self, src, shape, offsets, sizes, strides, dt):
        attrs = (
            f"static_offsets = {self.arr(offsets)}, static_sizes = {self.arr(sizes)}, "
            f"static_strides = {self.arr(strides)}, operandSegmentSizes = array<i32: 1, 0, 0, 0>"
        )
        return self.op(f'"tensor.extract_slice"({src}) <{{{attrs}}}> : ({self.ty(shape, dt)}) -> {self.ty(sizes, dt)}')

    def splat(self, value, dt, shape):
        constant = self.op(f'"arith.constant"() <{{value = {value} : {dt}}}> : () -> {dt}')
        return self.op(f'"tensor.splat"({constant}) : ({dt}) -> {self.ty(shape, dt)}')

    def elementwise(self, inputs, types, shape, body, maps=None):
        init = self.empty(shape, "f32")
        dims = ", ".join(f"d{i}" for i in range(len(shape)))
        identity = f"affine_map<({dims}) -> ({dims})>"
        maps = [*(maps or [identity] * len(inputs)), identity]
        kinds = ", ".join(["#linalg.iterator_type<parallel>"] * len(shape))
        args = ", ".join(f"%a{i}: {t.split('x')[-1][:-1]}" for i, t in enumerate(types)) + ", %o: f32"
        attrs = (
            f"indexing_maps = [{', '.join(maps)}], iterator_types = [{kinds}], "
            f"operandSegmentSizes = array<i32: {len(inputs)}, 1>"
        )
        signature = f"({', '.join([*types, self.ty(shape, 'f32')])}) -> {self.ty(shape, 'f32')}"
        return self.op(
            f'"linalg.generic"({", ".join(inputs)}, {init}) <{{{attrs}}}> ({{\n    ^bb0({args}):\n{body}\n'
            f'      "linalg.yield"(%r) : (f32) -> ()\n    }}) : {signature}'
        )

    def quantize(self, value, shape, scale):
        q_scale, q_zero = self.splat(f"{scale:e}", "f32", []), self.splat(0, "i64", [])
        return self.op(
            f'"quant_ext.quantize_per_tensor"({value}, {q_scale}, {q_zero}) <{{quant_min = -128 : i64, '
            f'quant_max = 127 : i64, output_dtype = "int8"}}> : ({self.ty(shape, "f32")}, tensor<f32>, '
            f"tensor<i64>) -> {self.ty(shape, 'i8')}"
        )

    def dequantize(self, value, shape, scale):
        q_scale, q_zero = self.splat(f"{scale:e}", "f32", []), self.splat(0, "i64", [])
        return self.op(
            f'"quant_ext.dequantize_per_tensor"({value}, {q_scale}, {q_zero}) <{{quant_min = -128 : i64, '
            f"quant_max = 127 : i64}}> : ({self.ty(shape, 'i8')}, tensor<f32>, tensor<i64>) -> {self.ty(shape, 'f32')}"
        )

    def relu(self, value, shape):
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        return self.elementwise(
            [value],
            [self.ty(shape, "f32")],
            shape,
            '      %z = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32\n'
            f'      %r = "arith.maximumf"(%a0, %z) {fastmath} : (f32, f32) -> f32',
        )

    def maxpool(self, value, shape, size, stride, pad):
        """A framework max-pool of an NCHW float tensor: a ``-inf`` pad, then a NaN-propagating window max."""
        n, c, h, w = shape
        hp, wp = h + 2 * pad, w + 2 * pad
        if pad:
            fill = self.splat("0xff800000", "f32", [n, c, hp, wp])
            attrs = (
                f"static_offsets = {self.arr([0, 0, pad, pad])}, static_sizes = {self.arr(shape)}, "
                f"static_strides = {self.arr([1, 1, 1, 1])}, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>"
            )
            value = self.op(
                f'"tensor.insert_slice"({value}, {fill}) <{{{attrs}}}> : ({self.ty(shape, "f32")}, '
                f"{self.ty([n, c, hp, wp], 'f32')}) -> {self.ty([n, c, hp, wp], 'f32')}"
            )
        ho, wo = (hp - size) // stride + 1, (wp - size) // stride + 1
        out = [n, c, ho, wo]
        init = self.splat("0xff800000", "f32", out)
        window = self.empty([size, size], "f32")
        maps = (
            f"affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * {stride} + d4, d3 * {stride} + d5)>, "
            "affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>"
        )
        kinds = ", ".join(["#linalg.iterator_type<parallel>"] * 4 + ["#linalg.iterator_type<reduction>"] * 2)
        fm = "#arith.fastmath<none>"
        body = (
            "    ^bb0(%a: f32, %b: f32, %o: f32):\n"
            f'      %u = "arith.cmpf"(%a, %a) <{{predicate = 14 : i64, fastmath = {fm}}}> : (f32, f32) -> i1\n'
            f'      %g = "arith.cmpf"(%a, %o) <{{predicate = 2 : i64, fastmath = {fm}}}> : (f32, f32) -> i1\n'
            '      %e = "arith.ori"(%u, %g) : (i1, i1) -> i1\n'
            '      %r = "arith.select"(%e, %a, %o) : (i1, f32, f32) -> f32\n'
            '      "linalg.yield"(%r) : (f32) -> ()\n'
        )
        padded = [n, c, hp, wp]
        return self.op(
            f'"linalg.generic"({value}, {window}, {init}) <{{indexing_maps = [{maps}], iterator_types = [{kinds}], '
            f"operandSegmentSizes = array<i32: 2, 1>}}> ({{\n{body}    }}) : ({self.ty(padded, 'f32')}, "
            f"{self.ty([size, size], 'f32')}, {self.ty(out, 'f32')}) -> {self.ty(out, 'f32')}"
        ), out

    def linear(self, x, m, k, n, weight, bias, s1, s2, so):
        """``x[m, k] @ weight[n, k]^T`` + bias, relu, quantize -- the weight transposed on the host."""
        wt, _ = self.transpose(weight, [n, k], [1, 0], "i8")
        init = self.splat(0, "i32", [m, n])
        maps = (
            "affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, "
            "affine_map<(d0, d1, d2) -> (d0, d1)>"
        )
        kinds = "#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>"
        overflow = "<{overflowFlags = #arith.overflow<none>}>"
        acc = self.op(
            f'"linalg.generic"({x}, {wt}, {init}) <{{indexing_maps = [{maps}], '
            f"iterator_types = [{kinds}], operandSegmentSizes = array<i32: 2, 1>}}> ({{\n"
            "    ^bb0(%a: i8, %b: i8, %o: i32):\n"
            '      %ea = "arith.extsi"(%a) : (i8) -> i32\n      %eb = "arith.extsi"(%b) : (i8) -> i32\n'
            f'      %m = "arith.muli"(%ea, %eb) {overflow} : (i32, i32) -> i32\n'
            f'      %s = "arith.addi"(%o, %m) {overflow} : (i32, i32) -> i32\n      "linalg.yield"(%s) : (i32) -> ()\n'
            f"    }}) : ({self.ty([m, k], 'i8')}, {self.ty([k, n], 'i8')}, {self.ty([m, n], 'i32')}) "
            f"-> {self.ty([m, n], 'i32')}"
        )
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        value = self.elementwise(
            [acc], [self.ty([m, n], "i32")], [m, n], '      %r = "arith.sitofp"(%a0) : (i32) -> f32'
        )
        for scale in (s1, s2):
            factor = self.splat(f"{scale:e}", "f32", [m, n])
            value = self.elementwise(
                [value, factor],
                [self.ty([m, n], "f32")] * 2,
                [m, n],
                f'      %r = "arith.mulf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            )
        value = self.elementwise(
            [value, bias],
            [self.ty([m, n], "f32"), self.ty([n], "f32")],
            [m, n],
            f'      %r = "arith.addf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            maps=["affine_map<(d0, d1) -> (d0, d1)>", "affine_map<(d0, d1) -> (d1)>"],
        )
        return self.quantize(self.relu(value, [m, n]), [m, n], so)

    def conv(self, image, c, h, w, co, k, stride, pad, weight, bias, s1, s2, so, pool=None):
        """A conv + bias + relu (+ max-pool) + quantize layer of the NCHW int8 ``image``;
        returns ``(value, rows, cols)``. ``pool`` is ``(size, stride, pad)``."""
        hp, wp = h + 2 * pad, w + 2 * pad
        ho, wo = (hp - k) // stride + 1, (wp - k) // stride + 1
        p, kk = ho * wo, k * k * c
        if pad:
            zero = self.splat(0, "i8", [1, c, hp, wp])
            attrs = (
                f"static_offsets = {self.arr([0, 0, pad, pad])}, static_sizes = {self.arr([1, c, h, w])}, "
                f"static_strides = {self.arr([1, 1, 1, 1])}, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>"
            )
            image = self.op(
                f'"tensor.insert_slice"({image}, {zero}) <{{{attrs}}}> : ({self.ty([1, c, h, w], "i8")}, '
                f"{self.ty([1, c, hp, wp], 'i8')}) -> {self.ty([1, c, hp, wp], 'i8')}"
            )
        pieces = []
        for dh in range(k):
            for dw in range(k):
                rows = self.extract(image, [1, c, hp, wp], [0, 0, dh, 0], [1, c, ho, wp], [1, 1, stride, 1], "i8")
                tap = self.extract(rows, [1, c, ho, wp], [0, 0, 0, dw], [1, c, ho, wo], [1, 1, 1, stride], "i8")
                nhwc, shape = self.transpose(tap, [1, c, ho, wo], [0, 2, 3, 1], "i8")
                flat, shape = self.collapse(nhwc, shape, [[0, 1, 2, 3]], "i8")
                pieces.append(self.expand(flat, shape, [[0, 1]], [p, c], "i8")[0])
        patches = pieces[0]
        if len(pieces) > 1:
            inputs = ", ".join([self.ty([p, c], "i8")] * len(pieces))
            patches = self.op(
                f'"tensor.concat"({", ".join(pieces)}) <{{dim = 1 : i64}}> : ({inputs}) -> {self.ty([p, kk], "i8")}'
            )
        wv, shape = self.transpose(weight, [co, c, k, k], [0, 2, 3, 1], "i8")
        wv, shape = self.collapse(wv, shape, [[0, 1, 2, 3]], "i8")
        wv, shape = self.expand(wv, shape, [[0, 1]], [co, kk], "i8")
        wv, _ = self.transpose(wv, [co, kk], [1, 0], "i8")
        init = self.splat(0, "i32", [p, co])
        maps = (
            "affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, "
            "affine_map<(d0, d1, d2) -> (d0, d1)>"
        )
        kinds = "#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>"
        overflow = "<{overflowFlags = #arith.overflow<none>}>"
        acc = self.op(
            f'"linalg.generic"({patches}, {wv}, {init}) <{{indexing_maps = [{maps}], '
            f"iterator_types = [{kinds}], operandSegmentSizes = array<i32: 2, 1>}}> ({{\n"
            "    ^bb0(%a: i8, %b: i8, %o: i32):\n"
            '      %ea = "arith.extsi"(%a) : (i8) -> i32\n      %eb = "arith.extsi"(%b) : (i8) -> i32\n'
            f'      %m = "arith.muli"(%ea, %eb) {overflow} : (i32, i32) -> i32\n'
            f'      %s = "arith.addi"(%o, %m) {overflow} : (i32, i32) -> i32\n      "linalg.yield"(%s) : (i32) -> ()\n'
            f"    }}) : ({self.ty([p, kk], 'i8')}, {self.ty([kk, co], 'i8')}, {self.ty([p, co], 'i32')}) "
            f"-> {self.ty([p, co], 'i32')}"
        )
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        value = self.elementwise(
            [acc], [self.ty([p, co], "i32")], [p, co], '      %r = "arith.sitofp"(%a0) : (i32) -> f32'
        )
        for scale in (s1, s2):
            factor = self.splat(f"{scale:e}", "f32", [p, co])
            value = self.elementwise(
                [value, factor],
                [self.ty([p, co], "f32")] * 2,
                [p, co],
                f'      %r = "arith.mulf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            )
        value, shape = self.collapse(value, [p, co], [[0, 1]], "f32")
        value, shape = self.expand(value, shape, [[0, 1, 2, 3]], [1, ho, wo, co], "f32")
        value, out = self.transpose(value, shape, [0, 3, 1, 2], "f32")
        bias_view, _ = self.expand(bias, [co], [[0, 1, 2, 3]], [1, co, 1, 1], "f32")
        value = self.elementwise(
            [value, bias_view],
            [self.ty(out, "f32"), self.ty([1, co, 1, 1], "f32")],
            out,
            f'      %r = "arith.addf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
            maps=["affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>", "affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>"],
        )
        value = self.relu(value, out)
        if pool is not None:
            value, out = self.maxpool(value, out, *pool)
        return self.quantize(value, out, so), out[2], out[3]

    def residual(self, a, b, shape, sa, sb, so):
        """``quantize(relu(dequantize(a) + dequantize(b)))`` over two NCHW int8 tensors."""
        fastmath = "<{fastmath = #arith.fastmath<none>}>"
        total = self.elementwise(
            [self.dequantize(a, shape, sa), self.dequantize(b, shape, sb)],
            [self.ty(shape, "f32")] * 2,
            shape,
            f'      %r = "arith.addf"(%a0, %a1) {fastmath} : (f32, f32) -> f32',
        )
        return self.quantize(self.relu(total, shape), shape, so)

    def module(self, result, shape) -> str:
        types = ", ".join(t for _n, t in self.args)
        params = ", ".join(f"{n}: {t}" for n, t in self.args)
        return (
            '"builtin.module"() ({\n'
            f'  "func.func"() <{{sym_name = "forward", function_type = ({types}) -> {self.ty(shape, "i8")}}}> ({{\n'
            f"  ^bb0({params}):\n" + "\n".join(self.lines) + "\n"
            f'    "func.return"({result}) : ({self.ty(shape, "i8")}) -> ()\n  }}) : () -> ()\n}}) : () -> ()\n'
        )


def _conv_layer_module(c, h, w, co, k, stride, pad, s1=0.5, s2=0.25, so=0.5):
    """One captured convolution layer: ``forward(x, wt, bias)``; returns the text and the output extent."""
    capture = _Capture()
    x = capture.arg("x", [1, c, h, w], "i8")
    weight = capture.arg("wt", [co, c, k, k], "i8")
    bias = capture.arg("bias", [co], "f32")
    out, ho, wo = capture.conv(x, c, h, w, co, k, stride, pad, weight, bias, s1, s2, so)
    return capture.module(out, [1, co, ho, wo]), (ho, wo)


def _write_capture(directory: Path, text: str, stored: dict) -> None:
    """``model.mlir`` plus the weights manifest and safetensors the route's prepack reads."""
    import struct as _struct

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.mlir").write_text(text, encoding="utf-8")
    header, payload = {}, b""
    for name, array in stored.items():
        raw = array.tobytes()
        offsets = [len(payload), len(payload) + len(raw)]
        spelled = "I8" if array.dtype == np.int8 else "F32"
        header[name] = {"dtype": spelled, "shape": list(array.shape), "data_offsets": offsets}
        payload += raw
    blob = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(_struct.pack("<Q", len(blob)) + blob + payload)
    manifest = {"0": {"kind": "input"}, **{str(i + 1): {"kind": "weight", "weight": n} for i, n in enumerate(stored)}}
    (directory / "model.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _routed_in_child(tmp_path, capture: Path, model: str) -> tuple[dict, Path]:
    import os

    from merlin.common.paths import repo_root

    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, MERLIN_TARGET_PATH=str(repo_root() / "examples" / "gemmini" / "support"))
    script = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); from test_device_shim_logical_abi import _fused_route; "
        "print(json.dumps(_fused_route(*sys.argv[2:])))"
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).parent), str(capture), str(work)]
        + [str(_group_package(tmp_path)), model],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert child.returncode == 0, child.stderr[-3000:]
    return json.loads(child.stdout.strip().splitlines()[-1]), work


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
@pytest.mark.parametrize(
    ("channels", "side", "features", "kernel", "stride", "pad"),
    [(3, 9, 8, 3, 1, 1), (4, 9, 8, 3, 2, 1), (5, 6, 7, 1, 1, 0), (5, 7, 6, 1, 2, 0)],
    ids=["3x3_s1_pad1", "3x3_s2_pad1", "1x1_s1", "1x1_s2"],
)
def test_a_captured_convolution_is_routed_from_its_image_and_runs_exactly(
    tmp_path, channels, side, features, kernel, stride, pad
):
    """A convolution the capture lowered on the HOST (strided slices of the padded NCHW image, an
    NCHW->NHWC transpose per tap, concatenated into a patch matrix for an integer matmul, then an NCHW
    readout of scale, bias, relu and quantize) is routed as the convolution it is: the call hands the
    device the image before the gather (NHWC, unpadded), the weight in the interface's [tap_h, tap_w,
    channel] rows, the folded bias and a [positions, features] destination; the gather is gone from
    the host program; and the result, read back to NCHW, equals the capture's quantized semantics."""
    text, (ho, wo) = _conv_layer_module(channels, side, side, features, kernel, stride, pad)
    rng = np.random.default_rng(kernel * 10 + stride)
    weight = rng.integers(-8, 8, (features, channels, kernel, kernel)).astype(np.int8)
    bias = (rng.integers(-40, 40, features) * 0.125).astype(np.float32)
    capture = tmp_path / "capture"
    _write_capture(capture, text, {"wt": weight, "bias": bias})
    route, work = _routed_in_child(tmp_path, capture, "conv")
    assert list(route["ops"].values()) == ["conv2d"], route["skipped"]
    (call,) = route["call_buffers"].values()
    assert [(op["role"], op["shape"]) for op in call] == [
        ("input_0", [1, side, side, channels]),
        ("weight_0", [kernel * kernel * channels, features]),
        ("bias_0", [features]),
        ("out_0", [ho * wo, features]),
    ]
    assert route["kernels"] == route["signatures"], route["build_skipped"]
    routed = (work / "routed.mlir").read_text(encoding="utf-8")
    assert "tensor.concat" not in routed and "tensor.extract_slice" not in routed, "the host gather survived"
    model = _lowered_object(routed, tmp_path / "lower")
    main_c = f"""
#include <stdio.h>
typedef struct {{ void *al; void *ali; intptr_t off; intptr_t s[4]; intptr_t st[4]; }} m4;
typedef struct {{ void *al; void *ali; intptr_t off; intptr_t s[1]; intptr_t st[1]; }} m1;
static m4 d4(void *p, intptr_t a, intptr_t b, intptr_t c, intptr_t d) {{
  m4 r = {{p, p, 0, {{a, b, c, d}}, {{b * c * d, c * d, d, 1}}}}; return r; }}
extern void _mlir_ciface_forward(m4 *, m4 *, m1 *, m4 *);
int main(void) {{
  static int8_t x[{channels * side * side}], w[{weight.size}], y[{features * ho * wo}];
  static float b[{features}];
  if (fread(x, 1, sizeof x, stdin) != sizeof x) return 2;
  m4 dx = d4(x, 1, {channels}, {side}, {side}), dw = d4(w, {features}, {channels}, {kernel}, {kernel});
  m4 dy = d4(y, 1, {features}, {ho}, {wo});
  m1 db = {{b, b, 0, {{{features}}}, {{1}}}};
  _mlir_ciface_forward(&dx, &dw, &db, &dy);
  fwrite(y, 1, sizeof y, stdout);
  return 0;
}}
"""
    x = rng.integers(-128, 128, (1, channels, side, side)).astype(np.int8)
    got = _run(tmp_path, [model, *map(Path, route["objects"])], main_c, x.tobytes())
    padded = np.pad(x[0].astype(np.int64), ((0, 0), (pad, pad), (pad, pad)))
    acc = np.zeros((features, ho, wo), dtype=np.int64)
    for i in range(kernel):
        for j in range(kernel):
            window = padded[:, i : i + stride * ho : stride, j : j + stride * wo : stride]
            acc += np.einsum("chw,oc->ohw", window, weight[:, :, i, j].astype(np.int64))
    value = (acc.astype(np.float32) * np.float32(0.5)) * np.float32(0.25) + bias[:, None, None]
    want = _rne(np.maximum(value, 0) / np.float32(0.5))
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(features, ho, wo), want)


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_chain_of_device_groups_carries_the_device_layout_between_them(tmp_path):
    """conv 3x3 -> conv 1x1 -> residual add of the two -> conv 3x3 stride 2, every layer on the device.

    The capture's tensors between layers are NCHW; the device's are NHWC / [positions, features]. A
    route that read each result back to NCHW and handed it straight over in NHWC would put two
    transposes on the host between every pair of layers. The host program keeps exactly one relayout
    at each true host boundary -- the model input and the model output -- and the result equals the
    capture's quantized semantics."""
    c, side, f, g = 3, 8, 8, 6
    capture = _Capture()
    x = capture.arg("x", [1, c, side, side], "i8")
    names = [("wa", [f, c, 3, 3]), ("ba", [f]), ("wb", [f, f, 1, 1]), ("bb", [f]), ("wc", [g, f, 3, 3]), ("bc", [g])]
    args = {name: capture.arg(name, shape, "i8" if len(shape) == 4 else "f32") for name, shape in names}
    qa, _h, _w = capture.conv(x, c, side, side, f, 3, 1, 1, args["wa"], args["ba"], 0.5, 0.25, 0.5)
    qb, _h, _w = capture.conv(qa, f, side, side, f, 1, 1, 0, args["wb"], args["bb"], 0.5, 0.25, 0.5)
    qr = capture.residual(qa, qb, [1, f, side, side], 0.5, 1.0, 0.5)
    qc, ho, wo = capture.conv(qr, f, side, side, g, 3, 2, 1, args["wc"], args["bc"], 0.5, 0.25, 0.5)
    text = capture.module(qc, [1, g, ho, wo])
    rng = np.random.default_rng(17)
    stored = {}
    for name, shape in names:
        stored[name] = (
            rng.integers(-4, 4, shape).astype(np.int8)
            if len(shape) == 4
            else (rng.integers(-24, 24, shape) * 0.125).astype(np.float32)
        )
    root = tmp_path / "capture"
    _write_capture(root, text, stored)
    route, work = _routed_in_child(tmp_path, root, "chain")
    assert sorted(route["ops"].values()) == ["conv2d", "conv2d", "conv2d", "residual_add"], route["skipped"]
    assert route["kernels"] == route["signatures"], route["build_skipped"]
    routed = (work / "routed.mlir").read_text(encoding="utf-8")
    assert routed.count("linalg.transpose") == 2, "only the model input and output may be relaid out on the host"
    assert "tensor.concat" not in routed and "tensor.extract_slice" not in routed
    model = _lowered_object(routed, tmp_path / "lower")
    sizes = {name: int(np.prod(shape)) for name, shape in names}
    decls = "".join(
        f"  static {'int8_t' if len(shape) == 4 else 'float'} {name}[{sizes[name]}];\n" for name, shape in names
    )
    views = "".join(
        f"  m4 d_{name} = d4({name}, {', '.join(map(str, shape))});\n"
        if len(shape) == 4
        else f"  m1 d_{name} = {{{name}, {name}, 0, {{{shape[0]}}}, {{1}}}};\n"
        for name, shape in names
    )
    call = ", ".join(f"&d_{name}" for name, _shape in names)
    main_c = f"""
#include <stdio.h>
typedef struct {{ void *al; void *ali; intptr_t off; intptr_t s[4]; intptr_t st[4]; }} m4;
typedef struct {{ void *al; void *ali; intptr_t off; intptr_t s[1]; intptr_t st[1]; }} m1;
static m4 d4(void *p, intptr_t a, intptr_t b, intptr_t c, intptr_t d) {{
  m4 r = {{p, p, 0, {{a, b, c, d}}, {{b * c * d, c * d, d, 1}}}}; return r; }}
extern void _mlir_ciface_forward(m4 *, {", ".join("m4 *" if len(s) == 4 else "m1 *" for _n, s in names)}, m4 *);
int main(void) {{
  static int8_t x[{c * side * side}], y[{g * ho * wo}];
{decls}  if (fread(x, 1, sizeof x, stdin) != sizeof x) return 2;
  m4 dx = d4(x, 1, {c}, {side}, {side}), dy = d4(y, 1, {g}, {ho}, {wo});
{views}  _mlir_ciface_forward(&dx, {call}, &dy);
  fwrite(y, 1, sizeof y, stdout);
  return 0;
}}
"""
    image = rng.integers(-128, 128, (1, c, side, side)).astype(np.int8)
    got = _run(tmp_path, [model, *map(Path, route["objects"])], main_c, image.tobytes())

    def conv(act, weight, bias, stride, pad):
        _n, _c, h, w = act.shape
        co, _ci, k, _k = weight.shape
        oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
        padded = np.pad(act[0].astype(np.int64), ((0, 0), (pad, pad), (pad, pad)))
        acc = np.zeros((co, oh, ow), dtype=np.int64)
        for i in range(k):
            for j in range(k):
                window = padded[:, i : i + stride * oh : stride, j : j + stride * ow : stride]
                acc += np.einsum("chw,oc->ohw", window, weight[:, :, i, j].astype(np.int64))
        value = (acc.astype(np.float32) * np.float32(0.5)) * np.float32(0.25) + bias[:, None, None]
        return _rne(np.maximum(value, 0) / np.float32(0.5))[None]

    a = conv(image, stored["wa"], stored["ba"], 1, 1)
    b = conv(a, stored["wb"], stored["bb"], 1, 0)
    r = _rne(np.maximum(a.astype(np.float32) * 0.5 + b.astype(np.float32) * 1.0, 0) / np.float32(0.5))
    want = conv(r, stored["wc"], stored["bc"], 2, 1)
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(want.shape), want)


def _np_conv(act, weight, bias, stride, pad, pool=None):
    """The capture's own float program for one layer: conv, two scales, bias, relu, max-pool, quantize."""
    _n, _c, h, w = act.shape
    co, _ci, k, _k = weight.shape
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    padded = np.pad(act[0].astype(np.int64), ((0, 0), (pad, pad), (pad, pad)))
    acc = np.zeros((co, oh, ow), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            window = padded[:, i : i + stride * oh : stride, j : j + stride * ow : stride]
            acc += np.einsum("chw,oc->ohw", window, weight[:, :, i, j].astype(np.int64))
    value = np.maximum((acc.astype(np.float32) * np.float32(0.5)) * np.float32(0.25) + bias[:, None, None], 0)
    if pool is not None:
        size, step, margin = pool
        value = np.pad(value, ((0, 0), (margin, margin), (margin, margin)), constant_values=-np.inf)
        ph, pw = (value.shape[1] - size) // step + 1, (value.shape[2] - size) // step + 1
        value = np.max(
            [value[:, i : i + step * ph : step, j : j + step * pw : step] for i in range(size) for j in range(size)],
            axis=0,
        )
    return _rne(value / np.float32(0.5))[None]


_M4_C = """
#include <stdio.h>
typedef struct { void *al; void *ali; intptr_t off; intptr_t s[4]; intptr_t st[4]; } m4;
typedef struct { void *al; void *ali; intptr_t off; intptr_t s[2]; intptr_t st[2]; } m2;
typedef struct { void *al; void *ali; intptr_t off; intptr_t s[1]; intptr_t st[1]; } m1;
static m4 d4(void *p, intptr_t a, intptr_t b, intptr_t c, intptr_t d) {
  m4 r = {p, p, 0, {a, b, c, d}, {b * c * d, c * d, d, 1}}; return r; }
static m2 d2(void *p, intptr_t a, intptr_t b) { m2 r = {p, p, 0, {a, b}, {b, 1}}; return r; }
static m1 d1(void *p, intptr_t a) { m1 r = {p, p, 0, {a}, {1}}; return r; }
"""


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_pooled_convolution_routes_its_whole_readout_including_the_max_pool(tmp_path):
    """A stem-shaped layer -- conv 5x5 stride 2, bias, relu, a framework max-pool (-inf pad, NaN
    propagating window max) and a quantize. The target's narrowing readout declares maxpool, so the
    whole readout is the device group's: no host readout and no relayout around the pool remain."""
    c, side, f = 3, 12, 8
    capture = _Capture()
    x = capture.arg("x", [1, c, side, side], "i8")
    weight, bias = capture.arg("wt", [f, c, 5, 5], "i8"), capture.arg("bias", [f], "f32")
    out, ho, wo = capture.conv(x, c, side, side, f, 5, 2, 2, weight, bias, 0.5, 0.25, 0.5, pool=(3, 2, 1))
    rng = np.random.default_rng(23)
    stored = {
        "wt": rng.integers(-6, 6, (f, c, 5, 5)).astype(np.int8),
        "bias": (rng.integers(-60, 60, f) * 0.125).astype(np.float32),
    }
    root = tmp_path / "capture"
    _write_capture(root, capture.module(out, [1, f, ho, wo]), stored)
    route, work = _routed_in_child(tmp_path, root, "stem")
    assert list(route["epilogues"].values()) == [["bias_add", "acc_scale", "relu", "maxpool"]], route["skipped"]
    assert route["kernels"] == route["signatures"], route["build_skipped"]
    routed = (work / "routed.mlir").read_text(encoding="utf-8")
    assert routed.count("linalg.transpose") == 2 and "linalg.generic" not in routed, "the readout stayed on the host"
    model = _lowered_object(routed, tmp_path / "lower")
    main_c = (
        _M4_C
        + f"""
extern void _mlir_ciface_forward(m4 *, m4 *, m1 *, m4 *);
int main(void) {{
  static int8_t x[{c * side * side}], w[{stored["wt"].size}], y[{f * ho * wo}];
  static float b[{f}];
  if (fread(x, 1, sizeof x, stdin) != sizeof x) return 2;
  m4 dx = d4(x, 1, {c}, {side}, {side}), dw = d4(w, {f}, {c}, 5, 5), dy = d4(y, 1, {f}, {ho}, {wo});
  m1 db = d1(b, {f});
  _mlir_ciface_forward(&dx, &dw, &db, &dy);
  fwrite(y, 1, sizeof y, stdout);
  return 0;
}}
"""
    )
    image = rng.integers(-128, 128, (1, c, side, side)).astype(np.int8)
    got = _run(tmp_path, [model, *map(Path, route["objects"])], main_c, image.tobytes())
    want = _np_conv(image, stored["wt"], stored["bias"], 2, 2, pool=(3, 2, 1))
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(want.shape), want)


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_weight_the_host_would_transpose_every_inference_is_read_from_the_prepack(tmp_path):
    """A linear layer stores its weight [out, in] and the capture transposes it on the host before the
    contraction. The call reads the prepack's copy of the transposed weight instead -- the same bytes,
    laid out once -- so the host program keeps no transpose at all."""
    m, k, n = 3, 10, 7
    capture = _Capture()
    x = capture.arg("x", [m, k], "i8")
    weight, bias = capture.arg("w", [n, k], "i8"), capture.arg("bias", [n], "f32")
    out = capture.linear(x, m, k, n, weight, bias, 0.5, 0.25, 0.5)
    rng = np.random.default_rng(29)
    stored = {
        "w": rng.integers(-8, 8, (n, k)).astype(np.int8),
        "bias": (rng.integers(-40, 40, n) * 0.125).astype(np.float32),
    }
    root = tmp_path / "capture"
    _write_capture(root, capture.module(out, [m, n]), stored)
    route, work = _routed_in_child(tmp_path, root, "linear")
    assert list(route["ops"].values()) == ["matmul"], route["skipped"]
    assert route["kernels"] == route["signatures"], route["build_skipped"]
    routed = (work / "routed.mlir").read_text(encoding="utf-8")
    assert "linalg.transpose" not in routed, "the weight is still transposed on the host every inference"
    model = _lowered_object(routed, tmp_path / "lower")
    main_c = (
        _M4_C
        + f"""
extern void _mlir_ciface_forward(m2 *, m2 *, m1 *, m2 *);
int main(void) {{
  static int8_t x[{m * k}], w[{n * k}], y[{m * n}];
  static float b[{n}];
  if (fread(x, 1, sizeof x, stdin) != sizeof x) return 2;
  m2 dx = d2(x, {m}, {k}), dw = d2(w, {n}, {k}), dy = d2(y, {m}, {n});
  m1 db = d1(b, {n});
  _mlir_ciface_forward(&dx, &dw, &db, &dy);
  fwrite(y, 1, sizeof y, stdout);
  return 0;
}}
"""
    )
    act = rng.integers(-128, 128, (m, k)).astype(np.int8)
    got = _run(tmp_path, [model, *map(Path, route["objects"])], main_c, act.tobytes())
    acc = act.astype(np.int64) @ stored["w"].astype(np.int64).T
    value = (acc.astype(np.float32) * np.float32(0.5)) * np.float32(0.25) + stored["bias"]
    want = _rne(np.maximum(value, 0) / np.float32(0.5))
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(m, n), want)
