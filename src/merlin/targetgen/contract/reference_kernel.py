"""A naive C reference kernel for a command buffer, for qualifying the runner-owned harness.

The grader's harness (:mod:`.harness_render`) is only trustworthy if a kernel that computes the
capsule's semantics, called through it, reproduces :func:`merlin.runtime.reference.reference_outputs`
exactly -- and a kernel that does NOT (a shifted index, a wrong scale, a transpose, an output it never
writes) fails. This module writes such a kernel straight from the command buffer's semantics: plain C
loops over the logical tensors the harness passes, no accelerator instruction, no tiling, no target
library. It is a test instrument, never a compiler and never graded.

Arithmetic mirrors :mod:`merlin.runtime.tensor` exactly: integer accumulation in 64 bits, the rounding
shift, the float32 ``acc_scale`` with ties to even, the saturating readout to the declared container.
Commands or stages it does not model raise :class:`ReferenceKernelUnsupported` (a buffer outside the
instrument's reach is skipped by name, never approximated).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

#: Deliberate defects the qualification gate requires the harness + comparison to catch.
MUTATIONS = ("off_by_one", "wrong_scale", "transposed", "uninitialized_output", "swapped_inputs")


class ReferenceKernelUnsupported(ValueError):
    """The buffer uses a command, stage or dtype this instrument does not model."""


@dataclass
class _Value:
    var: str
    shape: tuple[int, ...]
    dtype: str


def _int_dtype(dtype: str) -> tuple[int, bool]:
    kind, digits = dtype[:1], dtype[1:]
    if kind not in ("i", "u") or not digits.isdigit():
        raise ReferenceKernelUnsupported(f"dtype {dtype!r} is not an integer container")
    return int(digits), kind == "i"


class _Emitter:
    def __init__(self, cb: dict):
        self.cb = cb
        self.lines: list[str] = []
        self.decls: list[str] = []
        self.env: dict[str, _Value] = {}
        self.counter = 0
        self.default_shift = int((cb.get("params") or {}).get("requant_shift", 4))

    def fresh(self, shape, dtype: str, hint: str = "t") -> _Value:
        self.counter += 1
        var = f"v{self.counter}_{''.join(c if c.isalnum() else '_' for c in hint)}"
        self.decls.append(f"static int64_t {var}[{max(prod(shape), 1)}];")
        return _Value(var, tuple(shape), dtype)

    def emit(self, line: str) -> None:
        self.lines.append("  " + line)

    # -- primitives ------------------------------------------------------------------------------
    def matmul(self, a: _Value, b: _Value, hint: str) -> _Value:
        (m, k), (k2, n) = a.shape, b.shape
        if k != k2:
            raise ReferenceKernelUnsupported(f"matmul shape mismatch {a.shape} x {b.shape}")
        out = self.fresh((m, n), "i32", hint)
        self.emit(f"for (long i = 0; i < {m}; i++) for (long j = 0; j < {n}; j++) {{")
        self.emit(f"  int64_t s = 0; for (long p = 0; p < {k}; p++) s += {a.var}[i*{k}+p] * {b.var}[p*{n}+j];")
        self.emit(f"  {out.var}[i*{n}+j] = s; }}")
        return out

    def map(self, t: _Value, expr: str, dtype: str | None = None) -> _Value:
        out = self.fresh(t.shape, dtype or t.dtype, "m")
        self.emit(
            f"for (long e = 0; e < {max(prod(t.shape), 1)}; e++) {{ int64_t x = {t.var}[e]; {out.var}[e] = {expr}; }}"
        )
        return out

    def bias(self, t: _Value, b: _Value) -> _Value:
        m, n = t.shape
        if prod(b.shape) != n:
            raise ReferenceKernelUnsupported(f"bias {b.shape} is not one value per column of {t.shape}")
        out = self.fresh(t.shape, t.dtype, "bias")
        self.emit(
            f"for (long i = 0; i < {m}; i++) for (long j = 0; j < {n}; j++) "
            f"{out.var}[i*{n}+j] = {t.var}[i*{n}+j] + {b.var}[j];"
        )
        return out

    def requant(self, t: _Value, shift: int) -> _Value:
        if shift <= 0:
            return t
        return self.map(t, f"(x + {1 << (shift - 1)}LL) >> {shift}")

    def acc_scale(self, t: _Value, scale: float) -> _Value:
        import struct

        bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
        return self.map(t, f"merlin_ref_acc_scale(x, {bits}u)")

    def relu(self, t: _Value) -> _Value:
        return self.map(t, "x > 0 ? x : 0")

    def narrow(self, t: _Value, dtype: str) -> _Value:
        kind, digits = (dtype[:1], dtype[1:]) if dtype else ("", "")
        if kind not in ("i", "u") or not digits.isdigit():
            return _Value(t.var, t.shape, dtype)
        bits, signed = int(digits), kind == "i"
        if bits >= 32:
            return _Value(t.var, t.shape, dtype)
        lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        return self.map(t, f"x < {lo}LL ? {lo}LL : (x > {hi}LL ? {hi}LL : x)", dtype)

    def maxpool(self, t: _Value, attrs: dict, op: str) -> _Value:
        from merlin.runtime.commandbuffer import pool_params
        from merlin.runtime.tensor import pool_out_dims

        p = pool_params(attrs, op=op)
        rows, channels = t.shape
        h, w = p["pool_in_dims"]
        ph, pw = p["pool_size"]
        sh, sw = p["pool_stride"]
        pt, pl, pb, pr = p["pool_padding"]
        pad = p["pad_value"]
        if (pt or pl or pb or pr) and pad is None:
            raise ReferenceKernelUnsupported(f"{op}: padded pooling without a declared pad value")
        batch = rows // (h * w)
        ho, wo = pool_out_dims(h, w, (ph, pw), (sh, sw), (pt, pl, pb, pr))
        out = self.fresh((batch * ho * wo, channels), t.dtype, "pool")
        self.emit(
            f"for (long n = 0; n < {batch}; n++) for (long oy = 0; oy < {ho}; oy++) "
            f"for (long ox = 0; ox < {wo}; ox++) for (long c = 0; c < {channels}; c++) {{"
        )
        self.emit("  int have = 0; int64_t best = 0;")
        self.emit(f"  for (long ky = 0; ky < {ph}; ky++) for (long kx = 0; kx < {pw}; kx++) {{")
        self.emit(f"    long y = oy*{sh} - {pt} + ky, x = ox*{sw} - {pl} + kx; int64_t v;")
        self.emit(
            f"    if (y >= 0 && y < {h} && x >= 0 && x < {w}) v = {t.var}[(n*{h * w} + y*{w} + x)*{channels} + c];"
        )
        self.emit(f"    else v = {int(pad) if pad is not None else 0}LL;")
        self.emit("    if (!have || v > best) { best = v; have = 1; } }")
        self.emit(f"  {out.var}[((n*{ho} + oy)*{wo} + ox)*{channels} + c] = best; }}")
        return out

    # -- commands --------------------------------------------------------------------------------
    def epilogue(self, t: _Value, ops: dict, attrs: dict, op: str, stages_allowed) -> _Value:
        from merlin.runtime.commandbuffer import BIAS_STAGES, bias_tensor_name

        for stage in attrs.get("epilogue") or []:
            if stage not in stages_allowed:
                raise ReferenceKernelUnsupported(f"{op}: epilogue stage {stage!r}")
            if stage in BIAS_STAGES:
                t = self.bias(t, self.env[bias_tensor_name(ops, attrs, op=op)])
            elif stage == "requant":
                t = self.requant(t, int(attrs.get("requant_shift", self.default_shift)))
            elif stage == "acc_scale":
                t = self.acc_scale(t, float(attrs.get("acc_scale", 1.0)))
            elif stage == "relu":
                t = self.relu(t)
            elif stage == "maxpool":
                t = self.maxpool(t, attrs, op)
        return self.narrow(t, str(attrs.get("output_dtype", "i32")))

    def run(self) -> None:
        from merlin.runtime.commandbuffer import batched_matmul_geometry, conv_out_dims

        resident: dict[str, str] = {}
        accumulators: dict[str, _Value] = {}
        tensors = self.cb.get("tensors") or {}
        all_stages = ("bias_add", "bias", "requant", "acc_scale", "relu", "maxpool")
        for cmd in self.cb.get("commands") or []:
            op, ops, attrs = cmd["opcode"], cmd.get("operands") or {}, cmd.get("attributes") or {}
            if op == "RES_PACK":
                if "scale" in ops:
                    raise ReferenceKernelUnsupported("dequantizing resident pack")
                resident[ops["dst"]] = ops["src"]
            elif op == "EVICT":
                continue
            elif op in ("MATMUL", "MATMUL_RESIDENT"):
                rhs = self.env[resident.get(ops["rhs"], ops["rhs"])]
                accumulators[ops["dst"]] = self.matmul(self.env[ops["lhs"]], rhs, ops["dst"])
            elif op == "COMMIT":
                self.env[ops["dst"]] = self.epilogue(
                    accumulators[ops["src"]], ops, attrs, f"COMMIT {ops['dst']}", all_stages
                )
            elif op == "BIAS_ADD":
                from merlin.runtime.commandbuffer import bias_tensor_name

                t = self.bias(self.env[ops["src"]], self.env[bias_tensor_name(ops, attrs, op="BIAS_ADD")])
                self.env[ops["dst"]] = self.narrow(t, str(attrs.get("output_dtype", "i32")))
            elif op in ("ATTENTION_QK", "ATTENTION_PV"):
                if op == "ATTENTION_QK":
                    q, k = self.env[ops["q"]], self.env[ops["k"]]
                    n, d = k.shape
                    kt = self.fresh((d, n), k.dtype, "kT")
                    self.emit(
                        f"for (long i = 0; i < {d}; i++) for (long j = 0; j < {n}; j++) "
                        f"{kt.var}[i*{n}+j] = {k.var}[j*{d}+i];"
                    )
                    t = self.matmul(q, kt, ops["dst"])
                else:
                    t = self.matmul(self.env[ops["p"]], self.env[ops["v"]], ops["dst"])
                self.env[ops["dst"]] = self.epilogue(t, ops, attrs, op, ("acc_scale", "requant", "relu"))
            elif op == "CONV2D":
                ifm = self.env[ops["ifm"]]
                weight = self.env[resident.get(ops["weight"], ops["weight"])]
                if attrs.get("layout", "nhwc") != "nhwc":
                    raise ReferenceKernelUnsupported("CONV2D layout other than nhwc")
                kh, kw, ci, co = (int(v) for v in attrs["kernel"])
                shape = ifm.shape if len(ifm.shape) == 4 else tuple((tensors.get(ops["ifm"]) or {}).get("shape") or ())
                if len(shape) != 4:
                    raise ReferenceKernelUnsupported("CONV2D activation is not rank-4 NHWC")
                nb, h, w, c = shape
                sh, sw = (int(v) for v in attrs.get("stride", [1, 1]))
                pt, pl, _pb, _pr = (int(v) for v in attrs.get("padding", [0, 0, 0, 0]))
                dh, dw = (int(v) for v in attrs.get("dilation", [1, 1]))
                ho, wo = conv_out_dims(h, w, kh, kw, (sh, sw), attrs.get("padding", [0, 0, 0, 0]), (dh, dw))
                out = self.fresh((nb * ho * wo, co), "i32", ops["dst"])
                self.emit(
                    f"for (long b = 0; b < {nb}; b++) for (long oy = 0; oy < {ho}; oy++) "
                    f"for (long ox = 0; ox < {wo}; ox++) for (long o = 0; o < {co}; o++) {{"
                )
                self.emit("  int64_t s = 0;")
                self.emit(
                    f"  for (long ky = 0; ky < {kh}; ky++) for (long kx = 0; kx < {kw}; kx++) "
                    f"for (long cc = 0; cc < {ci}; cc++) {{"
                )
                self.emit(f"    long y = oy*{sh} - {pt} + ky*{dh}, x = ox*{sw} - {pl} + kx*{dw};")
                self.emit(f"    if (y < 0 || y >= {h} || x < 0 || x >= {w}) continue;")
                self.emit(
                    f"    s += {ifm.var}[((b*{h} + y)*{w} + x)*{c} + cc] * "
                    f"{weight.var}[((ky*{kw} + kx)*{ci} + cc)*{co} + o]; }}"
                )
                self.emit(f"  {out.var}[((b*{ho} + oy)*{wo} + ox)*{co} + o] = s; }}")
                self.env[ops["dst"]] = self.epilogue(out, ops, attrs, "CONV2D", all_stages)
            elif op == "BATCHED_MATMUL":
                a, wt = self.env[ops["a"]], self.env[ops["w"]]
                dst_shape = (tensors.get(ops["dst"]) or {}).get("shape")
                g = batched_matmul_geometry(a.shape, wt.shape, dst_shape, op="BATCHED_MATMUL")
                out = self.fresh(
                    tuple(g.output_shape), str((tensors.get(ops["dst"]) or {}).get("dtype") or "i32"), ops["dst"]
                )
                self.emit(
                    f"for (long bb = 0; bb < {g.batch_count}; bb++) for (long i = 0; i < {g.m}; i++) "
                    f"for (long j = 0; j < {g.n}; j++) {{ int64_t s = 0;"
                )
                self.emit(
                    f"  for (long p = 0; p < {g.k}; p++) s += {a.var}[bb*{g.m * g.k} + i*{g.k} + p] * "
                    f"{wt.var}[bb*{g.k * g.n} + p*{g.n} + j];"
                )
                self.emit(f"  {out.var}[bb*{g.m * g.n} + i*{g.n} + j] = s; }}")
                self.env[ops["dst"]] = out
            elif op == "MOVEMENT":
                src = self.env[ops["src"]]
                self.env[ops["dst"]] = _Value(src.var, src.shape, str(attrs.get("output_dtype", src.dtype)))
            elif op == "VECTOR_MAP":
                combine = attrs.get("combine", "add")
                a = self.env[ops["lhs"]]
                if combine == "identity":
                    t = a
                else:
                    b = self.env[ops["rhs"]]
                    out = self.fresh(a.shape, a.dtype, "vmap")
                    sym = "+" if combine == "add" else "*"
                    self.emit(f"for (long e = 0; e < {prod(a.shape)}; e++) {out.var}[e] = {a.var}[e] {sym} {b.var}[e];")
                    t = out
                for stage in attrs.get("activation") or []:
                    if stage == "relu":
                        t = self.relu(t)
                self.env[ops["dst"]] = t
            else:
                raise ReferenceKernelUnsupported(f"opcode {op!r}")


_HELPERS = r"""
static int64_t merlin_ref_acc_scale(int64_t x, uint32_t scale_bits) {
  union { uint32_t u; float f; } s; s.u = scale_bits;
  volatile float product = (float)x * s.f;   /* f32(f32(x) * f32(scale)), one rounding */
  int64_t i = (int64_t)product;               /* truncate toward zero */
  int64_t next = product < 0 ? i - 1 : i + 1;
  double rem = (double)product - (double)i; if (rem < 0) rem = -rem;
  if (rem < 0.5) return i;
  if (rem > 0.5) return next;
  return (i % 2 == 0) ? i : next;
}
"""


def render_reference_kernel(cb: dict, *, symbol: str, mutation: str | None = None) -> str:
    """C source defining ``void <symbol>(void*, ...)`` that computes ``cb``'s results naively."""
    from merlin.targetgen.contract.harness_render import container_for, logical_abi, logical_interface

    if mutation is not None and mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}")
    buffers = logical_interface(cb, logical_abi())
    em = _Emitter(cb)
    inputs = [b for b in buffers if b.kind == "input"]
    pointer_of = {b.name: f"arg{i}" for i, b in enumerate(buffers)}
    if mutation == "swapped_inputs":
        pair = next(
            (
                (x, y)
                for i, x in enumerate(inputs)
                for y in inputs[i + 1 :]
                if x.elements == y.elements and x.dtype == y.dtype
            ),
            None,
        )
        if pair is None:
            raise ReferenceKernelUnsupported("no two same-size inputs to swap")
        pointer_of[pair[0].name], pointer_of[pair[1].name] = pointer_of[pair[1].name], pointer_of[pair[0].name]
    for buf in inputs:
        _int_dtype(buf.dtype)
        container = container_for(buf.dtype)
        value = em.fresh(buf.shape, buf.dtype, buf.name)
        em.emit(
            f"for (long e = 0; e < {buf.elements}; e++) "
            f"{value.var}[e] = (({container.ctype} *){pointer_of[buf.name]})[e];"
        )
        em.env[buf.name] = value
    em.run()
    for buf in buffers:
        if buf.kind != "output":
            continue
        _int_dtype(buf.dtype)
        if buf.name not in em.env:
            raise ReferenceKernelUnsupported(f"result {buf.name!r} is never computed")
        value = em.env[buf.name]
        if prod(value.shape or (1,)) != buf.elements:
            raise ReferenceKernelUnsupported(f"result {buf.name!r} has {value.shape}, the harness passes {buf.shape}")
        if mutation == "uninitialized_output":
            continue
        container = container_for(buf.dtype)
        n = buf.elements
        rows, cols = buf.matrix
        source = {
            None: "e",
            "off_by_one": f"(e + 1) % {n}",
            "wrong_scale": "e",
            "transposed": f"((e % {cols}) * {rows} + e / {cols})",
            "swapped_inputs": "e",
        }[mutation]
        scale = " * 2" if mutation == "wrong_scale" else ""
        em.emit(
            f"for (long e = 0; e < {n}; e++) (({container.ctype} *){pointer_of[buf.name]})[e] = "
            f"({container.ctype})({value.var}[{source}]{scale});"
        )
    params = ", ".join(f"void *arg{i}" for i in range(len(buffers))) or "void"
    return (
        "#include <stdint.h>\n"
        + _HELPERS
        + "\n".join(em.decls)
        + f"\nvoid {symbol}({params}) {{\n"
        + "\n".join(em.lines)
        + "\n}\n"
    )
