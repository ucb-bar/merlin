from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from corefixtures import device_route
from corefixtures.device_programs import _Capture, _write_capture

from merlin.compiler.lower.model import device_shim as DS

pytestmark = pytest.mark.target("gemmini")
_CC = shutil.which("cc") or shutil.which("gcc")


@pytest.fixture(autouse=True)
def _synthetic_capabilities(monkeypatch, tmp_path):
    from corefixtures.logical_device_target import install

    install(monkeypatch)
    device_route._install_logical_backend(monkeypatch, tmp_path)


@pytest.fixture
def native_device_tools():
    from corefixtures.logical_device_target import require_native_tools

    require_native_tools(device=True)


@pytest.fixture
def native_host_tools():
    from corefixtures.logical_device_target import require_native_tools

    require_native_tools(host=True)


_TOOL = '#!{python}\nimport json, sys\nfrom merlin.development.targetgen.contract.interface_emit import parse_interface_mlir\n\nargs = sys.argv[1:]\nsrc = args[-1]\ncb = parse_interface_mlir(open(src, encoding="utf-8").read())\nnames = [n for n, s in cb["tensors"].items() if s.get("role") in ("input", "weight")]\nweight = next(n for n, s in cb["tensors"].items() if s.get("role") == "weight")\nlhs = next(n for n in names if n != weight)\nm, k = cb["tensors"][lhs]["shape"]\nn = cb["tensors"][weight]["shape"][1]\nif any(a.startswith("--emit-command-buffer=") for a in args):\n    open(next(a for a in args if a.startswith("--emit-command-buffer=")).split("=", 1)[1], "w").write(json.dumps(cb))\n    sys.exit(0)\nif "--emit-target-artifact" in args:\n    i_lhs, i_rhs = names.index(lhs), names.index(weight)\n    print(f"""module {{\n  llvm.func @ref_matmul_i8_i32(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64)\n  llvm.func @gemmini_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {{\n    %m = llvm.mlir.constant({m} : i64) : i64\n    %n = llvm.mlir.constant({n} : i64) : i64\n    %k = llvm.mlir.constant({k} : i64) : i64\n    llvm.call @ref_matmul_i8_i32(%arg{i_lhs}, %arg{i_rhs}, %arg2, %m, %n, %k)\n      : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()\n    llvm.return\n  }}\n}}""")\n    sys.exit(0)\nprint(open(src, encoding="utf-8").read())\n'
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
_HOST = "\n#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\ntypedef struct { void *a; void *b; intptr_t o; intptr_t s[2]; intptr_t st[2]; } mr2;\nvoid ref_matmul_i8_i32(const int8_t *a, const int8_t *b, int32_t *c, int64_t m, int64_t n, int64_t k) {\n  for (int64_t i = 0; i < m; ++i)\n    for (int64_t j = 0; j < n; ++j) {\n      int32_t s = 0;\n      for (int64_t q = 0; q < k; ++q) s += (int32_t)a[i * k + q] * (int32_t)b[q * n + j];\n      c[i * n + j] = s;\n    }\n}\n"


def _driver(sym, m, n, k):
    return f"\nextern mr2 {sym}(void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,\n                 void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,\n                 void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t);\nstatic void run_{sym}(FILE *in, FILE *out) {{\n  static int8_t A[{m}*{k}], B[{k}*{n}];\n  static int32_t C[{m}*{n}];\n  if (fread(A, 1, sizeof A, in) != sizeof A || fread(B, 1, sizeof B, in) != sizeof B) exit(2);\n  {sym}(A,A,0,{m},{k},{k},1,  B,B,0,{k},{n},{n},1,  C,C,0,{m},{n},{n},1);\n  fwrite(C, 1, sizeof C, out);\n}}\n"


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_logical_abi_device_build_computes_off_edge_contractions_exactly(tmp_path, native_device_tools):
    from merlin.compiler.lower.model.device_build import build_device_objects

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "manifest.yaml").write_text(json.dumps(_MANIFEST), encoding="utf-8")
    tool = pkg / "tool.py"
    tool.write_text(_TOOL.replace("{python}", sys.executable), encoding="utf-8")
    tool.chmod(493)
    signatures = {"g0": (20, 24, 40), "g1": (16, 16, 16)}
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
    assert set(built.kernels) == set(signatures), built.skipped
    shim = built.shim_object.with_suffix(".c").read_text(encoding="utf-8")
    assert "_a[" not in shim and "pointer order of the logical kernel ABI for this group: rhs, lhs, out" in shim
    host = tmp_path / "host.c"
    body = "".join((_driver(sym, m, n, k) for sym, (m, n, k) in signatures.items()))
    calls = "".join((f"  run_{sym}(stdin, stdout);\n" for sym in signatures))
    host.write_text(_HOST + body + f"int main(void) {{\n{calls}  return 0;\n}}\n", encoding="utf-8")
    exe = tmp_path / "model"
    subprocess.run([_CC, "-O1", "-o", str(exe), str(host), *map(str, built.objects)], check=True, capture_output=True)
    rng = np.random.default_rng(7)
    feed, want = (b"", [])
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


_GROUP_TOOL = '#!{python}\nimport json, struct, sys\nfrom merlin.development.targetgen.contract import harness_render as HR\nfrom merlin.development.targetgen.contract.interface_emit import parse_interface_mlir\n\nargs = sys.argv[1:]\nsrc = args[-1]\ncb = parse_interface_mlir(open(src, encoding="utf-8").read())\nif any(a.startswith("--emit-command-buffer=") for a in args):\n    open(next(a for a in args if a.startswith("--emit-command-buffer=")).split("=", 1)[1], "w").write(json.dumps(cb))\n    sys.exit(0)\nif "--emit-target-artifact" not in args:\n    print(open(src, encoding="utf-8").read())\n    sys.exit(0)\norder = HR.kernel_arg_order(cb)\ntensors = cb["tensors"]\nresident = {c["operands"]["dst"]: c["operands"]["src"] for c in cb["commands"] if c["opcode"] == "RES_PACK"}\ncommits = {c["operands"]["src"]: c for c in cb["commands"] if c["opcode"] == "COMMIT"}\nbody = ["    %null = llvm.mlir.zero : !llvm.ptr"]\nfor i, c in enumerate(cb["commands"]):\n    if c["opcode"] == "RESIDUAL_ADD":\n        attrs, ops = c.get("attributes") or {}, c["operands"]\n        lhs, rhs, out = ops["lhs"], ops["rhs"], ops["dst"]\n        count = 1\n        for extent in tensors[lhs]["shape"]:\n            count *= extent\n        bits = [struct.unpack("<i", struct.pack("<f", float(attrs[key])))[0] for key in ("lhs_scale", "rhs_scale")]\n        body.append(f"    %rn{i} = llvm.mlir.constant({count} : i64) : i64")\n        for j, value in enumerate([*bits, int("relu" in (attrs.get("epilogue") or []))]):\n            body.append(f"    %rf{i}_{j} = llvm.mlir.constant({value} : i32) : i32")\n        body.append(\n            f"    llvm.call @ref_group_add(%arg{order.index(lhs)}, %arg{order.index(rhs)}, %arg{order.index(out)}, "\n            f"%rn{i}, %rf{i}_0, %rf{i}_1, %rf{i}_2) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32) -> ()"\n        )\n        continue\n    if c["opcode"] == "CONV2D":\n        attrs, ops = c.get("attributes") or {}, c["operands"]\n        ifm, w, out = ops["ifm"], resident.get(ops["weight"], ops["weight"]), ops["dst"]\n        _n, h, wd, ci = tensors[ifm]["shape"]\n        kh, kw, _ci, co = attrs["kernel"]\n        epilogue = attrs.get("epilogue") or []\n        bias = attrs.get("bias")\n        scale = float(attrs.get("acc_scale", 1.0)) if "acc_scale" in epilogue else 1.0\n        bits = struct.unpack("<i", struct.pack("<f", scale))[0]\n        pool = [*attrs.get("pool_size", [1, 1]), *attrs.get("pool_stride", [1, 1])]\n        pool += [*attrs.get("pool_padding", [0, 0, 0, 0]), int(attrs.get("pool_pad_value", 0))]\n        geometry = [h, wd, ci, co, kh, kw, *attrs["stride"], *attrs["padding"], *pool]\n        flags = [bits, int("relu" in epilogue), int(str(attrs.get("output_dtype")) == "i8")]\n        names = []\n        for j, value in enumerate(geometry):\n            body.append(f"    %g{i}_{j} = llvm.mlir.constant({value} : i64) : i64")\n            names.append(f"%g{i}_{j}")\n        for j, value in enumerate(flags):\n            body.append(f"    %f{i}_{j} = llvm.mlir.constant({value} : i32) : i32")\n            names.append(f"%f{i}_{j}")\n        b = f"%arg{order.index(bias)}" if bias else "%null"\n        types = ", ".join(["!llvm.ptr"] * 4 + ["i64"] * len(geometry) + ["i32"] * len(flags))\n        body.append(\n            f"    llvm.call @ref_group_conv(%arg{order.index(ifm)}, %arg{order.index(w)}, {b}, "\n            f"%arg{order.index(out)}, {\', \'.join(names)}) : ({types}) -> ()"\n        )\n        continue\n    if c["opcode"] not in ("MATMUL", "MATMUL_RESIDENT"):\n        continue\n    lhs, rhs = c["operands"]["lhs"], resident.get(c["operands"]["rhs"], c["operands"]["rhs"])\n    commit = commits[c["operands"]["dst"]]\n    attrs, out = commit.get("attributes") or {}, commit["operands"]["dst"]\n    m, k = tensors[lhs]["shape"]\n    n = tensors[rhs]["shape"][1]\n    epilogue = attrs.get("epilogue") or []\n    bias = attrs.get("bias") or (commit["operands"].get("bias"))\n    scale = float(attrs.get("acc_scale", 1.0)) if "acc_scale" in epilogue else 1.0\n    bits = struct.unpack("<i", struct.pack("<f", scale))[0]\n    narrow = 1 if str(attrs.get("output_dtype")) == "i8" else 0\n    values = [("m", m, "i64"), ("n", n, "i64"), ("k", k, "i64"), ("s", bits, "i32"),\n              ("r", int("relu" in epilogue), "i32"), ("q", narrow, "i32")]\n    for tag, value, ty in values:\n        body.append(f"    %{tag}{i} = llvm.mlir.constant({value} : {ty}) : {ty}")\n    b = f"%arg{order.index(bias)}" if bias else "%null"\n    body.append(\n        f"    llvm.call @ref_group_mm(%arg{order.index(lhs)}, %arg{order.index(rhs)}, {b}, %arg{order.index(out)}, "\n        f"%m{i}, %n{i}, %k{i}, %s{i}, %r{i}, %q{i}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, "\n        "i64, i64, i64, i32, i32, i32) -> ()"\n    )\nparams = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(len(order)))\nprint("module {")\nprint("  llvm.func @ref_group_mm(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i32, i32, i32)")\nconv_types = ", ".join(["!llvm.ptr"] * 4 + ["i64"] * 21 + ["i32"] * 3)\nprint(f"  llvm.func @ref_group_conv({conv_types})")\nprint("  llvm.func @ref_group_add(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32)")\nprint(f"  llvm.func @gemmini_kernel({params}) {{")\nprint("\\n".join(body))\nprint("    llvm.return\\n  }\\n}")\n'
_GROUP_REFERENCE = "\n#include <math.h>\n#include <stdint.h>\n#include <stdlib.h>\n#include <string.h>\n/* The stand-in device's arithmetic: int32 accumulation, then bias, scale (round half to even), relu,\n * and an optional int8 narrowing -- the readout the commit states. */\nvoid ref_group_mm(const int8_t *a, const int8_t *w, const int32_t *bias, void *y, int64_t m, int64_t n,\n                  int64_t k, int32_t scale_bits, int32_t relu, int32_t narrow) {\n  float scale;\n  memcpy(&scale, &scale_bits, sizeof scale);\n  for (int64_t i = 0; i < m; ++i)\n    for (int64_t j = 0; j < n; ++j) {\n      int32_t acc = 0;\n      for (int64_t q = 0; q < k; ++q) acc += (int32_t)a[i * k + q] * (int32_t)w[q * n + j];\n      if (bias) acc += bias[j];\n      float v = nearbyintf((float)acc * scale);\n      if (relu && v < 0) v = 0;\n      if (narrow) ((int8_t *)y)[i * n + j] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);\n      else ((int32_t *)y)[i * n + j] = (int32_t)v;\n    }\n}\n\n/* A residual sum as the interface states it: relu(sat(roundeven(f32(a) * ls + f32(b) * rs))), int8. */\nvoid ref_group_add(const int8_t *a, const int8_t *b, int8_t *y, int64_t n, int32_t ls_bits, int32_t rs_bits,\n                   int32_t relu) {\n  float ls, rs;\n  memcpy(&ls, &ls_bits, sizeof ls);\n  memcpy(&rs, &rs_bits, sizeof rs);\n  for (int64_t i = 0; i < n; ++i) {\n    float v = nearbyintf((float)a[i] * ls + (float)b[i] * rs);\n    v = v > 127 ? 127 : v < -128 ? -128 : v;\n    if (relu && v < 0) v = 0;\n    y[i] = (int8_t)v;\n  }\n}\n\n/* A convolution read the way the interface states it: IFM [1, H, W, C] (NHWC), W [kh*kw*C, Co] in\n * (tap_h, tap_w, channel) order, zero padding [top, left, bottom, right]; the readout (bias, scale,\n * relu, narrowing) then, when declared, a max pool of the readout's values whose padded cells hold\n * pool_pad_value; result [rows*cols, Co]. */\nvoid ref_group_conv(const int8_t *x, const int8_t *w, const int32_t *bias, void *y, int64_t h, int64_t wd,\n                    int64_t c, int64_t co, int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t pt, int64_t pl,\n                    int64_t pb, int64_t pr, int64_t qh, int64_t qw, int64_t qsh, int64_t qsw, int64_t qt,\n                    int64_t ql, int64_t qb, int64_t qr, int64_t qpad, int32_t scale_bits, int32_t relu,\n                    int32_t narrow) {\n  float scale;\n  memcpy(&scale, &scale_bits, sizeof scale);\n  int64_t ho = (h + pt + pb - kh) / sh + 1, wo = (wd + pl + pr - kw) / sw + 1;\n  int32_t *v = malloc(sizeof(int32_t) * ho * wo * co);\n  for (int64_t oh = 0; oh < ho; ++oh)\n    for (int64_t ow = 0; ow < wo; ++ow)\n      for (int64_t o = 0; o < co; ++o) {\n        int32_t acc = 0;\n        for (int64_t i = 0; i < kh; ++i)\n          for (int64_t j = 0; j < kw; ++j) {\n            int64_t ih = oh * sh + i - pt, iw = ow * sw + j - pl;\n            if (ih < 0 || iw < 0 || ih >= h || iw >= wd) continue;\n            for (int64_t ch = 0; ch < c; ++ch)\n              acc += (int32_t)x[(ih * wd + iw) * c + ch] * (int32_t)w[((i * kw + j) * c + ch) * co + o];\n          }\n        if (bias) acc += bias[o];\n        float r = nearbyintf((float)acc * scale);\n        if (relu && r < 0) r = 0;\n        if (narrow) r = r > 127 ? 127 : r < -128 ? -128 : r;\n        v[(oh * wo + ow) * co + o] = (int32_t)r;\n      }\n  int64_t po = (ho + qt + qb - qh) / qsh + 1, pw = (wo + ql + qr - qw) / qsw + 1;\n  for (int64_t a = 0; a < po; ++a)\n    for (int64_t b = 0; b < pw; ++b)\n      for (int64_t o = 0; o < co; ++o) {\n        int32_t best = INT32_MIN;\n        for (int64_t i = 0; i < qh; ++i)\n          for (int64_t j = 0; j < qw; ++j) {\n            int64_t ih = a * qsh + i - qt, iw = b * qsw + j - ql;\n            int32_t cell = (ih < 0 || iw < 0 || ih >= ho || iw >= wo) ? (int32_t)qpad : v[(ih * wo + iw) * co + o];\n            if (cell > best) best = cell;\n          }\n        int64_t at = (a * pw + b) * co + o;\n        if (narrow) ((int8_t *)y)[at] = (int8_t)best;\n        else ((int32_t *)y)[at] = best;\n      }\n  free(v);\n}\n"


def _group_package(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    manifest = dict(_MANIFEST, package_id="logical_group_probe")
    (pkg / "manifest.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    tool = pkg / "tool.py"
    tool.write_text(_GROUP_TOOL.replace("{python}", sys.executable), encoding="utf-8")
    tool.chmod(493)
    return pkg


def _lowered_object(text: str, work):
    """The module through Merlin's own host lowering, as an x86 object."""
    from merlin.compiler.lower.model.lower import lower_model
    from merlin.compiler.lower.model.toolchain import clang

    lowered = lower_model(text, work, targets=())
    obj = work / "model.o"
    subprocess.run(
        [clang(), "-O1", "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)], check=True, capture_output=True
    )
    return obj


def _run(tmp_path, objects, main_c: str, feed: bytes) -> bytes:
    from merlin.compiler.lower.model.toolchain import clang

    host = tmp_path / "main.c"
    host.write_text(_GROUP_REFERENCE + main_c, encoding="utf-8")
    exe = tmp_path / "model"
    subprocess.run(
        [clang(), "-O1", "-o", str(exe), str(host), *map(str, objects), "-lm"], check=True, capture_output=True
    )
    return subprocess.run([str(exe)], input=feed, capture_output=True, check=True).stdout


_MEMREF_C = "\ntypedef struct { void *al; void *ali; intptr_t off; intptr_t s[1]; intptr_t st[1]; } m1;\ntypedef struct { void *al; void *ali; intptr_t off; intptr_t s[2]; intptr_t st[2]; } m2;\nstatic m1 d1(void *p, intptr_t n) { m1 d = {p, p, 0, {n}, {1}}; return d; }\nstatic m2 d2(void *p, intptr_t r, intptr_t c) { m2 d = {p, p, 0, {r, c}, {c, 1}}; return d; }\n"


def _rne(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_fused_bias_scale_relu_whole_model_runs_exactly_under_the_logical_abi(tmp_path, native_host_tools):
    """Two chained quantized linear layers, each bias_add + acc_scale + relu + quantize: the route
    rewrites each into ONE call passing the activation, the stored weight, the folded bias and the
    destination; the device build matches each to the kernel's logical interface (W, A0, B, Y0) and
    the program, lowered by Merlin's host pipeline and run, equals the capture's quantized semantics."""
    import os
    import struct as _struct

    from corefixtures.whole_model_group_route import _two_layers

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
    header, payload = ({}, b"")
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
    env = dict(os.environ, MERLIN_TARGET_PATH="")
    script = "import json, sys; sys.path.insert(0, sys.argv[1]); from corefixtures.device_route import _fused_route; print(json.dumps(_fused_route(*sys.argv[2:])))"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(device_route.__file__).resolve().parents[1]),
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
        + "\n#include <stdio.h>\n/* Merlin's host lowering passes the model's result as a caller-allocated out-parameter, last. */\nextern void _mlir_ciface_forward(m2 *, m2 *, m1 *, m2 *, m1 *, m1 *, m2 *);\nint main(void) {\n  static int8_t x[4 * 8], wa[8 * 16], wb[16 * 32], y[4 * 32];\n  static float ba[16], bb[32], br[4];\n  if (fread(x, 1, sizeof x, stdin) != sizeof x || fread(wa, 1, sizeof wa, stdin) != sizeof wa ||\n      fread(wb, 1, sizeof wb, stdin) != sizeof wb) return 2;\n  m2 dy = d2(y, 4, 32), dx = d2(x, 4, 8), dwa = d2(wa, 8, 16), dwb = d2(wb, 16, 32);\n  m1 dba = d1(ba, 16), dbb = d1(bb, 32), dbr = d1(br, 4);\n  _mlir_ciface_forward(&dx, &dwa, &dba, &dwb, &dbb, &dbr, &dy);\n  fwrite(y, 1, sizeof y, stdout);\n  return 0;\n}\n"
    )
    x = rng.integers(-128, 128, (4, 8)).astype(np.int8)
    got = _run(tmp_path, [model, *built_objects], main_c, x.tobytes() + stored["wa"].tobytes() + stored["wb"].tobytes())

    def layer(act, weight, bias):
        value = act.astype(np.float64) * 0.5 @ (weight.astype(np.float64) * 0.5) + bias
        return _rne(np.maximum(value, 0.0) / 0.5)

    want = layer(layer(x, stored["wa"], stored["biasa"]), stored["wb"], stored["biasb"])
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(4, 32), want)


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_two_result_group_passes_both_results_and_runs_exactly(tmp_path, native_host_tools):
    """One resident weight read by two contractions, each committing its own result: the call passes
    both activations, the weight and BOTH destinations (in an order unlike the kernel's), and both
    results come back through Merlin's host lowering, exact."""
    from merlin.compiler.lower.model.device_build import build_device_objects

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
    from merlin.development.targetgen import corpus_spec as CS
    from merlin.runtime.mesh import _mesh_tile_binding

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
    access = ", ".join((f'{{bufferization.access = "{a}"}}' for a in ("read", "read", "read", "write", "write")))
    text = f'builtin.module {{\n  func.func @forward(%a0: tensor<3x12xi8>, %a1: tensor<5x12xi8>, %w: tensor<12x7xi8>) -> ({t0}, {t1}) {{\n    %e0 = tensor.empty() : {t0}\n    %e1 = tensor.empty() : {t1}\n    %r:2 = func.call @{sym}(%a0, %a1, %w, %e0, %e1) : ({args}) -> ({t0}, {t1})\n    func.return %r#0, %r#1 : {t0}, {t1}\n  }}\n  "func.func"() <{{sym_name = "{sym}", function_type = ({args}) -> ({t0}, {t1}),\n      sym_visibility = "private", arg_attrs = [{access}]}}> ({{\n  }}) : () -> ()\n}}\n'
    model = _lowered_object(text, tmp_path / "lower")
    n0, n1 = (int(np.prod(by_role[r].shape)) for r in ("out_0", "out_1"))
    e0, e1 = (4 if by_role[r].dtype == "i32" else 1 for r in ("out_0", "out_1"))
    main_c = (
        _MEMREF_C
        + f"\n#include <stdio.h>\n/* Merlin's host lowering passes each model result as a caller-allocated out-parameter, last. */\nextern void _mlir_ciface_forward(m2 *, m2 *, m2 *, m2 *, m2 *);\nint main(void) {{\n  static int8_t a0[3 * 12], a1[5 * 12], w[12 * 7];\n  static char y0[{n0 * e0}], y1[{n1 * e1}];\n  if (fread(a0, 1, sizeof a0, stdin) != sizeof a0 || fread(a1, 1, sizeof a1, stdin) != sizeof a1 ||\n      fread(w, 1, sizeof w, stdin) != sizeof w) return 2;\n  m2 da0 = d2(a0, 3, 12), da1 = d2(a1, 5, 12), dw = d2(w, 12, 7), dy0 = d2(y0, 3, 7), dy1 = d2(y1, 5, 7);\n  _mlir_ciface_forward(&da0, &da1, &dw, &dy0, &dy1);\n  fwrite(y0, 1, sizeof y0, stdout);\n  fwrite(y1, 1, sizeof y1, stdout);\n  return 0;\n}}\n"
    )
    rng = np.random.default_rng(5)
    a0, a1 = (rng.integers(-128, 128, (3, 12)).astype(np.int8), rng.integers(-128, 128, (5, 12)).astype(np.int8))
    w = rng.integers(-128, 128, (12, 7)).astype(np.int8)
    got = _run(tmp_path, [model, *built.objects], main_c, a0.tobytes() + a1.tobytes() + w.tobytes())
    acc0, acc1 = (a0.astype(np.int32) @ w.astype(np.int32), a1.astype(np.int32) @ w.astype(np.int32))
    want0 = acc0 if e0 == 4 else _rne(acc0)
    want1 = np.maximum(acc1, 0) if e1 == 4 else _rne(np.maximum(acc1, 0))
    dt0, dt1 = (np.int32 if e == 4 else np.int8 for e in (e0, e1))
    assert np.array_equal(np.frombuffer(got[: n0 * e0], dtype=dt0).reshape(3, 7), want0)
    assert np.array_equal(np.frombuffer(got[n0 * e0 :], dtype=dt1).reshape(5, 7), want1)


def _stored_bytes(capture, name):
    weights = (capture / "model.safetensors").read_bytes()
    header_bytes = int.from_bytes(weights[:8], "little")
    header = json.loads(weights[8 : 8 + header_bytes])
    begin, end = header[name]["data_offsets"]
    return weights[8 + header_bytes + begin : 8 + header_bytes + end]


def _routed_in_child(tmp_path, capture: Path, model: str) -> tuple[dict, Path]:
    import os

    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, MERLIN_TARGET_PATH="")
    script = "import json, sys; sys.path.insert(0, sys.argv[1]); from corefixtures.device_route import _fused_route; print(json.dumps(_fused_route(*sys.argv[2:])))"
    child = subprocess.run(
        [sys.executable, "-c", script, str(Path(device_route.__file__).resolve().parents[1]), str(capture), str(work)]
        + [str(_group_package(tmp_path)), model],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert child.returncode == 0, child.stderr[-3000:]
    return (json.loads(child.stdout.strip().splitlines()[-1]), work)


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_chain_of_device_groups_preserves_numerics_with_explicit_layout_transitions(
    tmp_path, native_host_tools, record_property
):
    """Four device groups preserve the capture while host layout transitions remain explicit."""
    from xdsl.ir import BlockArgument

    from merlin.common import mlir_query as mq

    c, side, f, g = (3, 8, 8, 6)
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
    transposes = [op for op in mq.parse(routed).walk() if op.name == "linalg.transpose"]
    assert any((isinstance(op.operands[0], BlockArgument) for op in transposes))
    assert any((use.operation.name == "func.return" for op in transposes for use in op.results[0].uses))
    record_property(
        "retained_layout_transitions",
        json.dumps(
            [
                dict(
                    source=str(op.operands[0].type),
                    result=str(op.results[0].type),
                    permutation=str(op.permutation),
                    users=[use.operation.name for use in op.results[0].uses],
                )
                for op in transposes
            ]
        ),
    )
    assert not {"tensor.concat", "tensor.extract_slice"} & {mq.op_name(op) for op in mq.parse(routed).walk()}
    model = _lowered_object(routed, tmp_path / "lower")
    sizes = {name: int(np.prod(shape)) for name, shape in names}
    decls = "".join(
        (f"  static {('int8_t' if len(shape) == 4 else 'float')} {name}[{sizes[name]}];\n" for name, shape in names)
    )
    views = "".join(
        (
            f"  m4 d_{name} = d4({name}, {', '.join(map(str, shape))});\n"
            if len(shape) == 4
            else f"  m1 d_{name} = {{{name}, {name}, 0, {{{shape[0]}}}, {{1}}}};\n"
            for name, shape in names
        )
    )
    call = ", ".join((f"&d_{name}" for name, _shape in names))
    main_c = f"\n#include <stdio.h>\ntypedef struct {{ void *al; void *ali; intptr_t off; intptr_t s[4]; intptr_t st[4]; }} m4;\ntypedef struct {{ void *al; void *ali; intptr_t off; intptr_t s[1]; intptr_t st[1]; }} m1;\nstatic m4 d4(void *p, intptr_t a, intptr_t b, intptr_t c, intptr_t d) {{\n  m4 r = {{p, p, 0, {{a, b, c, d}}, {{b * c * d, c * d, d, 1}}}}; return r; }}\nextern void _mlir_ciface_forward(m4 *, {', '.join(('m4 *' if len(s) == 4 else 'm1 *' for _n, s in names))}, m4 *);\nint main(void) {{\n  static int8_t x[{c * side * side}], y[{g * ho * wo}];\n{decls}  if (fread(x, 1, sizeof x, stdin) != sizeof x) return 2;\n  m4 dx = d4(x, 1, {c}, {side}, {side}), dy = d4(y, 1, {g}, {ho}, {wo});\n{views}  _mlir_ciface_forward(&dx, {call}, &dy);\n  fwrite(y, 1, sizeof y, stdout);\n  return 0;\n}}\n"
    image = rng.integers(-128, 128, (1, c, side, side)).astype(np.int8)
    got = _run(tmp_path, [model, *map(Path, route["objects"])], main_c, image.tobytes())

    def conv(act, weight, bias, stride, pad):
        _n, _c, h, w = act.shape
        co, _ci, k, _k = weight.shape
        oh, ow = ((h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1)
        padded = np.pad(act[0].astype(np.int64), ((0, 0), (pad, pad), (pad, pad)))
        acc = np.zeros((co, oh, ow), dtype=np.int64)
        for i in range(k):
            for j in range(k):
                window = padded[:, i : i + stride * oh : stride, j : j + stride * ow : stride]
                acc += np.einsum("chw,oc->ohw", window, weight[:, :, i, j].astype(np.int64))
        value = acc.astype(np.float32) * np.float32(0.5) * np.float32(0.25) + bias[:, None, None]
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
    oh, ow = ((h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1)
    padded = np.pad(act[0].astype(np.int64), ((0, 0), (pad, pad), (pad, pad)))
    acc = np.zeros((co, oh, ow), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            window = padded[:, i : i + stride * oh : stride, j : j + stride * ow : stride]
            acc += np.einsum("chw,oc->ohw", window, weight[:, :, i, j].astype(np.int64))
    value = np.maximum(acc.astype(np.float32) * np.float32(0.5) * np.float32(0.25) + bias[:, None, None], 0)
    if pool is not None:
        size, step, margin = pool
        value = np.pad(value, ((0, 0), (margin, margin), (margin, margin)), constant_values=-np.inf)
        ph, pw = ((value.shape[1] - size) // step + 1, (value.shape[2] - size) // step + 1)
        value = np.max(
            [value[:, i : i + step * ph : step, j : j + step * pw : step] for i in range(size) for j in range(size)],
            axis=0,
        )
    return _rne(value / np.float32(0.5))[None]


_M4_C = "\n#include <stdio.h>\ntypedef struct { void *al; void *ali; intptr_t off; intptr_t s[4]; intptr_t st[4]; } m4;\ntypedef struct { void *al; void *ali; intptr_t off; intptr_t s[2]; intptr_t st[2]; } m2;\ntypedef struct { void *al; void *ali; intptr_t off; intptr_t s[1]; intptr_t st[1]; } m1;\nstatic m4 d4(void *p, intptr_t a, intptr_t b, intptr_t c, intptr_t d) {\n  m4 r = {p, p, 0, {a, b, c, d}, {b * c * d, c * d, d, 1}}; return r; }\nstatic m2 d2(void *p, intptr_t a, intptr_t b) { m2 r = {p, p, 0, {a, b}, {b, 1}}; return r; }\nstatic m1 d1(void *p, intptr_t a) { m1 r = {p, p, 0, {a}, {1}}; return r; }\n"


@pytest.mark.skipif(_CC is None, reason="no host C compiler")
def test_a_pooled_convolution_preserves_unproved_float_readout_on_host(tmp_path, native_host_tools):
    """An unproved float pool stays on the host after an exact raw device contraction."""
    from merlin.common import mlir_query as mq
    from merlin.compiler.dialects.lowering import compute_groups as CG
    from merlin.compiler.dialects.lowering import contraction_coverage as CC
    from merlin.compiler.dialects.lowering import group_exact_float as GEF

    c, side, f = (3, 12, 8)
    capture = _Capture()
    x = capture.arg("x", [1, c, side, side], "i8")
    weight, bias = (capture.arg("wt", [f, c, 5, 5], "i8"), capture.arg("bias", [f], "f32"))
    out, ho, wo = capture.conv(x, c, side, side, f, 5, 2, 2, weight, bias, 0.5, 0.25, 0.5, pool=(3, 2, 1))
    rng = np.random.default_rng(23)
    stored = {
        "wt": rng.integers(-6, 6, (f, c, 5, 5)).astype(np.int8),
        "bias": (rng.integers(-60, 60, f) * 0.125).astype(np.float32),
    }
    root = tmp_path / "capture"
    _write_capture(root, capture.module(out, [1, f, ho, wo]), stored)
    route, work = _routed_in_child(tmp_path, root, "stem")
    with CC.select_max_is_max():
        groups = CG.form_groups(mq.parse((root / "model.mlir").read_text()), "gemmini")
        refusals = GEF.refusals(groups, root)
    assert len(refusals) == 1 and "stage the proof does not model" in next(iter(refusals.values()))
    assert list(route["epilogues"].values()) == [[]], route["skipped"]
    assert route["kernels"] == route["signatures"], route["build_skipped"]
    routed = (work / "routed.mlir").read_text(encoding="utf-8")
    calls = [op for op in mq.parse(routed).walk() if op.name == "func.call"]
    assert len(calls) == 1 and mq.type_shape_dtype(calls[0].results[0].type)[1] == "i32"
    assert {"linalg.generic", "quant_ext.quantize_per_tensor"} <= {mq.op_name(op) for op in mq.parse(routed).walk()}
    model = _lowered_object(routed, tmp_path / "lower")
    main_c = (
        _M4_C
        + f"\nextern void _mlir_ciface_forward(m4 *, m4 *, m1 *, m4 *);\nint main(void) {{\n  static int8_t x[{c * side * side}], w[{stored['wt'].size}], y[{f * ho * wo}];\n  static float b[{f}];\n  if (fread(x, 1, sizeof x, stdin) != sizeof x) return 2;\n  if (fread(b, 1, sizeof b, stdin) != sizeof b) return 2;\n  m4 dx = d4(x, 1, {c}, {side}, {side}), dw = d4(w, {f}, {c}, 5, 5), dy = d4(y, 1, {f}, {ho}, {wo});\n  m1 db = d1(b, {f});\n  _mlir_ciface_forward(&dx, &dw, &db, &dy);\n  fwrite(y, 1, sizeof y, stdout);\n  return 0;\n}}\n"
    )
    image = rng.integers(-128, 128, (1, c, side, side)).astype(np.int8)
    got = _run(tmp_path, [model, *map(Path, route["objects"])], main_c, image.tobytes() + _stored_bytes(root, "bias"))
    want = _np_conv(image, stored["wt"], stored["bias"], 2, 2, pool=(3, 2, 1))
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(want.shape), want)
