"""Runner-owned self-contained-C harness for the fork-free SIMT path.

The generic ``kernel_abi`` contract has the submission package emit only a kernel FUNCTION and the runner
own the harness that embeds the leaf tensors, calls the kernel, and prints the OUT/METRIC/DONE protocol.
This module builds that harness as a **self-contained, relocation-free C program** so the fork-free driver
(:func:`muon.compile_kernel_forkfree`) can build it with a stock toolchain — no vendor fork, no libc:

  * operands are materialized ELEMENT-WISE from their IEEE bit patterns (integer immediates -> no
    constant pool -> no relocations; an aggregate initializer would emit a PC-relative memcpy);
  * floats are reconstructed by bit-cast and printed with a manual fixed-decimal routine (no soft-float
    libcalls, no printf), integers by a manual base-10 routine;
  * the whole thing runs on one thread (a hart-0 guard) so the console byte stream is clean.

The harness is TRUSTED infra: it reads the canonical operands (runner-side) and never exposes them to the
agent, whose artifact is only the kernel function. Target-agnostic — the caller supplies the operands and
the output spec; nothing here names a target.

The kernel is compiled into ``main`` with ``always_inline`` so its single call site leaves no
``R_RISCV_CALL`` relocation (the reassemble-after-transcode kernel path cannot carry a kernel-internal
relocation — a relocation-preserving transcode is the general fix, tracked with the multi-thread lane).
This covers the whole-computation kernel functions the functional ladder needs.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

@dataclass
class TensorArg:
    """One kernel argument: a named tensor with a shape and its flat row-major values."""
    name: str
    rows: int
    cols: int
    values: list[float]      # flat, row-major, length rows*cols
    dtype: str               # "f32" | "i32" (the C element type the kernel sees)


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _emit_fill(arr: str, arg: TensorArg) -> list[str]:
    """Materialize ``arg`` into a STACK array ``arr`` element-wise. Float elements are written as their
    u32 IEEE bit pattern into a uint32_t array (bit-cast to float* at the call); integer elements directly.
    The array is stack-local (SP-relative, not PC-relative like a ``static`` in .bss) and ``volatile`` so
    the compiler cannot coalesce the element-wise stores into a .rodata memcpy — both would introduce a
    relocation the fork-free transcoder fails closed on. So the object stays relocation-free."""
    n = arg.rows * arg.cols
    lines = [f"  volatile uint32_t {arr}[{n}];"]
    for i, v in enumerate(arg.values):
        bits = _f32_bits(v) if arg.dtype == "f32" else (int(v) & 0xFFFFFFFF)
        lines.append(f"  {arr}[{i}]=0x{bits:08x}u;")
    return lines


def _render_helpers(model) -> str:
    """The harness's print/identity helpers, with the hart-id CSR and the console-MMIO putchar aperture DERIVED
    from the target's ``runtime_abi`` — no hardcoded CSR number or address. The hart-id CSR is the RISC-V machine
    ``mhartid``; the console aperture is the target's own (sim/BSP-defined) putchar region. Fail closed if the
    runtime ABI does not carry them."""
    hid_csr = model.special_csr("mhartid")
    console = model.aperture("console_mmio")
    return f"""
#include <stdint.h>
static inline uint32_t _hid(void){{uint32_t r;__asm__ volatile("csrr %0,{hid_csr:#x}":"=r"(r));return r;}}
static inline void _pc(char c){{*(volatile char*)0x{console:08x}u=c;}}
static void _pu(uint32_t v){{char b[12];int n=0;if(!v){{_pc('0');return;}}while(v){{b[n++]=(char)('0'+v%10);v/=10;}}while(n)_pc(b[--n]);}}
static void _ps(const char*s){{while(*s)_pc(*s++);}}
static float _u2f(uint32_t b){{union{{uint32_t u;float f;}}x;x.u=b;return x.f;}}
/* fixed 4-decimal float print with carry (no soft-float libcalls, no printf) */
static void _pf(float x){{if(x<0.0f){{_pc('-');x=-x;}}uint32_t ip=(uint32_t)x;float fr=x-(float)ip;
  float k=_u2f(0x461c4000u),h=_u2f(0x3f000000u);/*10000.0, 0.5*/
  uint32_t fp=(uint32_t)(fr*k+h);if(fp>=10000u){{ip++;fp-=10000u;}}
  _pu(ip);_pc('.');if(fp<1000u)_pc('0');if(fp<100u)_pc('0');if(fp<10u)_pc('0');_pu(fp);}}
"""


def build_program(kernel_fn_src: str, args: list[TensorArg], outputs: list[TensorArg],
                  *, kernel_symbol: str, model) -> str:
    """Assemble the self-contained C program: helpers + the agent's kernel function + a ``main`` that
    embeds every input, calls ``kernel_symbol(<inputs>, <outputs>)``, and prints ``OUT <name> <r> <c> ...``
    for each output followed by ``DONE``. ``args`` is the kernel's input arguments in ABI order (weight,
    then lhs...); ``outputs`` the output tensors (also passed to the kernel, then printed). Float outputs
    print as fixed decimals (matched by the runner's float tolerance); integer outputs as base-10."""
    # Force the kernel to inline into main (single call site) so no R_RISCV_CALL relocation survives the
    # reassemble-after-transcode path. Prepended to the agent's definition (which starts with its return
    # type), yielding e.g. `static inline __attribute__((always_inline)) void radiance_kernel(...)`.
    kernel_inlined = "static inline __attribute__((always_inline)) " + kernel_fn_src.strip()
    body: list[str] = [_render_helpers(model).strip(), "", kernel_inlined, "", "int main(void){",
                       "  if(_hid()!=0)return 0;"]
    call_ptrs: list[str] = []
    for a in args:
        arr = f"_in_{a.name}"
        body += _emit_fill(arr, a)
        call_ptrs.append(f"(float*){arr}" if a.dtype == "f32" else f"(int32_t*){arr}")
    for o in outputs:
        arr = f"_out_{o.name}"
        body.append(f"  volatile uint32_t {arr}[{o.rows * o.cols}];")   # stack (SP-relative -> no reloc)
        call_ptrs.append(f"(float*){arr}" if o.dtype == "f32" else f"(int32_t*){arr}")
    body.append(f"  {kernel_symbol}({', '.join(call_ptrs)});")
    for o in outputs:
        arr = f"_out_{o.name}"
        body.append(f'  _ps("OUT {o.name} {o.rows} {o.cols}");')
        if o.dtype == "f32":
            body.append(f"  for(int i=0;i<{o.rows * o.cols};i++){{_pc(' ');_pf(_u2f({arr}[i]));}}")
        else:
            body.append(f"  for(int i=0;i<{o.rows * o.cols};i++){{_pc(' ');_pu({arr}[i]);}}")
        body.append("  _pc('\\n');")
    body.append('  _ps("DONE\\n");')
    body.append("  return 0;")
    body.append("}")
    return "\n".join(body) + "\n"


def _kernel_symbol(kernel_fn_src: str) -> str:
    """The kernel's function name, extracted structurally (no regex): the identifier just before the first
    ``(`` — i.e. the last token of the text preceding the parameter list. Robust to the return type/qualifiers."""
    head = kernel_fn_src.split("(", 1)[0]
    toks = head.replace("*", " ").split()
    if not toks:
        raise ValueError("cannot find a kernel function name in the emitted artifact")
    return toks[-1]


def _shape2d(shape: list) -> tuple[int, int]:
    """A tensor shape as (rows, cols): a 2-D shape verbatim, a 1-D shape as a row vector, higher rank
    flattened to (prod/last, last) so the row-major byte order the kernel/harness use is preserved."""
    dims = [int(d) for d in (shape or []) if int(d) > 0] or [1]
    if len(dims) == 1:
        return 1, dims[0]
    rows = 1
    for d in dims[:-1]:
        rows *= d
    return rows, dims[-1]


def _decode_preload(tspec: dict | None) -> list[float] | None:
    """Decode a command-buffer input tensor's INJECTED operand bytes into a flat, row-major float list —
    the seam that lets a caller INJECT real operands (a whole-model mesh LAYER's activations/weights) onto
    the command buffer instead of the deterministic, name-materialized ones the golden uses. The bytes live
    in ``tspec['preload_b64']`` (base64), encoded for the tensor's DECLARED ``dtype`` by the injector (the
    inverse of the shared operand encoder). Returns None when the tensor carries no injected operand (the
    caller keeps materializing); RAISES on a present-but-undecodable dtype so the run fails closed rather
    than silently grading the wrong (materialized) operands. Target-agnostic — dispatches on the dtype
    token, never on a target."""
    if not isinstance(tspec, dict) or not tspec.get("preload_b64"):
        return None
    import base64

    import numpy as np
    raw = base64.b64decode(tspec["preload_b64"])
    dt = str(tspec.get("dtype", "f32"))
    codec = {"i8": "<i1", "int8": "<i1", "u8": "<u1", "uint8": "<u1",
             "i32": "<i4", "int32": "<i4", "f32": "<f4", "float32": "<f4"}.get(dt)
    if codec is not None:
        arr = np.frombuffer(raw, dtype=codec)
    else:                                            # fp8 / bf16 / fp16 via the derived float codec
        try:
            from merlin.targetgen.rtl.fp8_codec import decode_bytes as _fp_decode
            arr = np.asarray(_fp_decode(raw, dt))
        except Exception as e:  # noqa: BLE001 — present operand we cannot decode: fail closed
            raise ValueError(f"injected operand of dtype {dt!r} has no decoder") from e
    return [float(x) for x in np.asarray(arr).reshape(-1)]


def args_from_cb(cb: dict) -> tuple[list[TensorArg], list[TensorArg]] | None:
    """Derive the kernel's ``(in_args, out_args)`` from a capsule's COMMAND BUFFER, in the generic
    ``kernel_abi`` order ``[weight] ++ [lhs] ++ [output]``. Input VALUES come from the SAME deterministic
    materialization the reference backend and the golden use (:func:`commandbuffer.materialize_inputs`) — NOT
    a ``canonical_inputs`` side table (which the emitted cb does not carry) — so the harnessed operands match
    the golden bit-for-bit; the output NAME is the COMMIT's dst (e.g. ``Y0``) so the printed ``OUT <name>``
    matches the graded tensor; the output SHAPE is the matmul's ``(M, N)`` (outputs are produced, not leaf
    tensors). Uses the same plan as :func:`muon_codegen_mlir.emit_kernel_mlir`, so harness and kernel agree
    on operand order + shapes. Returns None (fail-safe) on an unsupported shape (chained matmuls / no matmul).
    Shared by the inline-source path (:func:`program_from_cb`) and the object-kernel path
    (:func:`external_main_from_cb`)."""
    from merlin.runtime.commandbuffer import materialize_inputs

    all_tensors = cb.get("tensors") or {}

    def _vals(canon: dict, name: str, t) -> list[float]:
        """Operand VALUES, in precedence order: an INJECTED operand the caller preloaded onto the cb tensor
        (``preload_b64`` — a real whole-model layer's activations/weights), then the golden's decoded
        operands the runner attached (``cb['canonical_inputs']``, so the kernel runs on the SAME operands
        the independent golden used), else the deterministic materialization. Flat, row-major, as f32."""
        inj = _decode_preload(all_tensors.get(name))
        if inj is not None:
            return inj
        c = canon.get(name)
        if isinstance(c, dict):
            c = c.get("values")
        if c is not None:
            def _fl(v):                                  # deep-flatten (batched operands are rank-3)
                if isinstance(v, (list, tuple)):
                    out: list[float] = []
                    for e in v:
                        out.extend(_fl(e))
                    return out
                return [float(v)]
            return _fl(c)
        return [float(v) for row in t.to_list() for v in row]

    # --- linalg-interface fused op (softmax/layernorm/geglu/rope/attention_full) ----------------------
    # A capsule whose interface is the m2m linalg module directly (the agent compiles standard-dialect
    # linalg) is fed POSITIONALLY: ``cb['arg_order']`` is ``[in0, in1, ..., out]`` (the func-arg order), each
    # input's values come from the golden's canonical operands, and the output is produced. Operands may be
    # rank-1 (e.g. layernorm weight/bias) — flattened to (1, n).
    if cb.get("interface") == "linalg_positional":
        order = list(cb.get("arg_order") or [])
        tensors = cb.get("tensors") or {}
        canon = cb.get("canonical_inputs") or {}
        if len(order) < 2:
            return None
        ins, out = order[:-1], order[-1]

        def _rc(shp):
            if not shp:
                return None
            return (1, int(shp[0])) if len(shp) == 1 else (int(shp[0]), int(shp[1]))

        in_args = []
        for nm in ins:
            c = _decode_preload(tensors.get(nm))         # INJECTED operand takes precedence
            if c is None:
                c = canon.get(nm)
                if isinstance(c, dict):
                    c = c.get("values")
            rc = _rc((tensors.get(nm) or {}).get("shape"))
            if c is None or rc is None:
                return None
            in_args.append(TensorArg(nm, rc[0], rc[1], [float(x) for x in c], "f32"))
        orc = _rc((tensors.get(out) or {}).get("shape"))
        if orc is None:
            return None
        return in_args, [TensorArg(out, orc[0], orc[1], [0.0] * (orc[0] * orc[1]), "f32")]

    # --- non-matmul SIMT ops (attention scores Q@K^T, row rmsnorm) --------------------------------------
    # These have no matmul/commit; derive operands directly from the op command + leaf shapes, in the generic
    # kernel_abi order ([weight] ++ [inputs in command order] ++ [output]) that emit_kernel_mlir mirrors. The
    # output is produced (not a leaf) so its shape comes from the inputs (attention: (Qrows, Krows); rmsnorm:
    # X's shape). Values come from the golden's canonical decoded operands (matched bit-for-bit).
    canon0 = cb.get("canonical_inputs") or {}
    env0 = materialize_inputs(cb)
    for cmd in cb.get("commands", []):
        op = (cmd.get("opcode") or "").upper()
        o = cmd.get("operands", {})
        if op == "ATTENTION_QK":
            q, k, out = o.get("q"), o.get("k"), o.get("dst")
            if not (q and k and out) or q not in env0 or k not in env0:
                return None
            qt, kt = env0[q], env0[k]
            if len(qt.shape) != 2 or len(kt.shape) != 2 or qt.shape[1] != kt.shape[1]:
                return None
            m, d = qt.shape[0], qt.shape[1]
            n = kt.shape[0]
            in_args = [TensorArg(q, m, d, _vals(canon0, q, qt), "f32"),
                       TensorArg(k, n, d, _vals(canon0, k, kt), "f32")]
            out_args = [TensorArg(out, m, n, [0.0] * (m * n), "f32")]
            return in_args, out_args
        if op == "RMSNORM":
            _cmds = cb.get("commands", [])
            rms = [cc for cc in _cmds if (cc.get("opcode") or "").upper() == "RMSNORM"]
            _mms = [cc for cc in _cmds if (cc.get("opcode") or "").upper() in ("MATMUL", "MATMUL_RESIDENT")]
            if len(rms) == 1 and _mms:                           # fused rmsnorm -> matmul (Y = rmsnorm(X,G) @ W)
                ro = rms[0].get("operands", {})
                g, x, h = ro.get("gamma"), ro.get("src"), ro.get("dst")
                resident = {cc["operands"]["dst"]: cc["operands"]["src"] for cc in _cmds
                            if (cc.get("opcode") or "").upper() == "RES_PACK"}
                mo = _mms[0].get("operands", {})
                if mo.get("lhs") == h:
                    w = resident.get(mo.get("rhs"), mo.get("rhs"))
                    commit = next((cc for cc in _cmds if (cc.get("opcode") or "").upper() == "COMMIT"
                                   and cc["operands"].get("src") == mo.get("dst")), None)
                    y = commit["operands"]["dst"] if commit else mo.get("dst")
                    if g and x and w and y and all(nm in env0 for nm in (g, x, w)) and len(env0[x].shape) == 2:
                        r, c = env0[x].shape
                        _, nn = env0[w].shape
                        gr, gc = _shape2d(list(env0[g].shape))
                        in_args = [TensorArg(g, gr, gc, _vals(canon0, g, env0[g]), "f32"),
                                   TensorArg(w, c, nn, _vals(canon0, w, env0[w]), "f32"),
                                   TensorArg(x, r, c, _vals(canon0, x, env0[x]), "f32")]
                        return in_args, [TensorArg(y, r, nn, [0.0] * (r * nn), "f32")]
            if len(rms) == 2:                                    # gemma double rmsnorm (chained via alloca)
                a0, a1 = rms[0].get("operands", {}), rms[1].get("operands", {})
                g1, g2, xx, yy = a0.get("gamma"), a1.get("gamma"), a0.get("src"), a1.get("dst")
                if not (g1 and g2 and xx and yy) or xx not in env0 or g1 not in env0 or g2 not in env0:
                    return None
                xt = env0[xx]
                if len(xt.shape) != 2:
                    return None
                r, c = xt.shape[0], xt.shape[1]
                g1r, g1c = _shape2d(list(env0[g1].shape))
                g2r, g2c = _shape2d(list(env0[g2].shape))
                in_args = [TensorArg(g1, g1r, g1c, _vals(canon0, g1, env0[g1]), "f32"),
                           TensorArg(g2, g2r, g2c, _vals(canon0, g2, env0[g2]), "f32"),
                           TensorArg(xx, r, c, _vals(canon0, xx, xt), "f32")]
                return in_args, [TensorArg(yy, r, c, [0.0] * (r * c), "f32")]
            x, g, out = o.get("src"), o.get("gamma"), o.get("dst")
            if not (x and g and out) or x not in env0 or g not in env0:
                return None
            xt, gt = env0[x], env0[g]
            if len(xt.shape) != 2:
                return None
            r, c = xt.shape[0], xt.shape[1]
            gr, gc = _shape2d(list(gt.shape))
            # weight-first ABI: [gamma] ++ [src] ++ [out]
            in_args = [TensorArg(g, gr, gc, _vals(canon0, g, gt), "f32"),
                       TensorArg(x, r, c, _vals(canon0, x, xt), "f32")]
            out_args = [TensorArg(out, r, c, [0.0] * (r * c), "f32")]
            return in_args, out_args
        if op == "LAYERNORM":
            x, g, b, out = o.get("src"), o.get("gamma"), o.get("beta"), o.get("dst")
            if not (x and g and b and out) or x not in env0 or g not in env0 or b not in env0:
                return None
            xt, gt, bt = env0[x], env0[g], env0[b]
            if len(xt.shape) != 2:
                return None
            r, c = xt.shape[0], xt.shape[1]
            gr, gc = _shape2d(list(gt.shape))
            br, bc = _shape2d(list(bt.shape))
            # weight-first ABI: [gamma, beta] ++ [src] ++ [out]
            in_args = [TensorArg(g, gr, gc, _vals(canon0, g, gt), "f32"),
                       TensorArg(b, br, bc, _vals(canon0, b, bt), "f32"),
                       TensorArg(x, r, c, _vals(canon0, x, xt), "f32")]
            out_args = [TensorArg(out, r, c, [0.0] * (r * c), "f32")]
            return in_args, out_args
        if op == "ATTENTION_FULL":
            q, k, v, out = o.get("q"), o.get("k"), o.get("v"), o.get("dst")
            if not (q and k and v and out) or any(nm not in env0 for nm in (q, k, v)):
                return None
            qt, kt, vt = env0[q], env0[k], env0[v]
            if len(qt.shape) != 2 or len(vt.shape) != 2:
                return None
            mq, dq = qt.shape
            _, dvv = vt.shape
            in_args = [TensorArg(q, qt.shape[0], qt.shape[1], _vals(canon0, q, qt), "f32"),
                       TensorArg(k, kt.shape[0], kt.shape[1], _vals(canon0, k, kt), "f32"),
                       TensorArg(v, vt.shape[0], vt.shape[1], _vals(canon0, v, vt), "f32")]
            return in_args, [TensorArg(out, mq, dvv, [0.0] * (mq * dvv), "f32")]
        if op == "GEGLU":
            x, wg, wu, out = o.get("src"), o.get("w_gate"), o.get("w_up"), o.get("dst")
            if not (x and wg and wu and out) or any(nm not in env0 for nm in (x, wg, wu)):
                return None
            xt, wgt, wut = env0[x], env0[wg], env0[wu]
            if len(xt.shape) != 2 or len(wgt.shape) != 2:
                return None
            m, k = xt.shape
            _, n = wgt.shape
            in_args = [TensorArg(wg, k, n, _vals(canon0, wg, wgt), "f32"),
                       TensorArg(wu, k, n, _vals(canon0, wu, wut), "f32"),
                       TensorArg(x, m, k, _vals(canon0, x, xt), "f32")]
            return in_args, [TensorArg(out, m, n, [0.0] * (m * n), "f32")]
        if op in ("SOFTMAX", "GELU", "SOFTCAP"):
            x, out = o.get("src"), o.get("dst")
            if not (x and out) or x not in env0:
                return None
            xt = env0[x]
            if len(xt.shape) != 2:
                return None
            r, c = xt.shape[0], xt.shape[1]
            in_args = [TensorArg(x, r, c, _vals(canon0, x, xt), "f32")]
            out_args = [TensorArg(out, r, c, [0.0] * (r * c), "f32")]
            return in_args, out_args
        if op == "ROPE":
            _cmds = cb.get("commands", [])
            _mms = [cc for cc in _cmds if (cc.get("opcode") or "").upper() in ("MATMUL", "MATMUL_RESIDENT")]
            x, out = o.get("src"), o.get("dst")
            if _mms:                                                 # fused matmul -> rope (Y = rope(X @ W))
                resident = {cc["operands"]["dst"]: cc["operands"]["src"] for cc in _cmds
                            if (cc.get("opcode") or "").upper() == "RES_PACK"}
                mo = _mms[0].get("operands", {})
                commit = next((cc for cc in _cmds if (cc.get("opcode") or "").upper() == "COMMIT"
                               and cc["operands"].get("src") == mo.get("dst")), None)
                h = commit["operands"]["dst"] if commit else mo.get("dst")
                if x == h and out:
                    lhs = mo.get("lhs")
                    w = resident.get(mo.get("rhs"), mo.get("rhs"))
                    if lhs in env0 and w in env0 and len(env0[lhs].shape) == 2 and len(env0[w].shape) == 2:
                        m, k = env0[lhs].shape
                        _, n = env0[w].shape
                        in_args = [TensorArg(w, k, n, _vals(canon0, w, env0[w]), "f32"),
                                   TensorArg(lhs, m, k, _vals(canon0, lhs, env0[lhs]), "f32")]
                        return in_args, [TensorArg(out, m, n, [0.0] * (m * n), "f32")]
                return None
            if not (x and out) or x not in env0:                     # standalone rope over a leaf
                return None
            xt = env0[x]
            if len(xt.shape) != 2:
                return None
            r, c = xt.shape[0], xt.shape[1]
            in_args = [TensorArg(x, r, c, _vals(canon0, x, xt), "f32")]
            out_args = [TensorArg(out, r, c, [0.0] * (r * c), "f32")]
            return in_args, out_args
        if op == "BATCHED_MATMUL":
            a, w, out = o.get("a"), o.get("w"), o.get("dst")
            if not (a and w and out) or a not in env0 or w not in env0:
                return None
            at, wt = env0[a], env0[w]
            if len(at.shape) != 3 or len(wt.shape) != 3:
                return None
            batch, m, k = at.shape
            _, k2, n = wt.shape
            # flatten the batch into rows for the flat preload (matches the kernel's flat indexing);
            # weight-first ABI: [w] ++ [a] ++ [out]
            in_args = [TensorArg(w, batch * k2, n, _vals(canon0, w, wt), "f32"),
                       TensorArg(a, batch * m, k, _vals(canon0, a, at), "f32")]
            out_args = [TensorArg(out, batch * m, n, [0.0] * (batch * m * n), "f32")]
            return in_args, out_args

    # Tolerant single-matmul plan (do NOT reuse the strict muon_codegen._plan, which assumes ``dst`` on every
    # matmul): resolve RES_PACK residents, the one matmul (accepting ``dst``/``out`` and ``rhs``/``weight``),
    # and the output NAME — the COMMIT dst that sources the matmul when present, else the matmul's own dst.
    resident_source: dict[str, str] = {}
    matmuls: list[tuple[str, str, str]] = []      # (dst, lhs, rhs)
    commits: list[tuple[str, str]] = []           # (out, src)
    commit_bias: dict[str, str] = {}              # out -> bias operand (a bias_add epilogue)
    for cmd in cb.get("commands", []):
        op = (cmd.get("opcode") or "").upper()
        o = cmd.get("operands", {})
        if op == "RES_PACK":
            if o.get("dst") and o.get("src"):
                resident_source[o["dst"]] = o["src"]
        elif "MATMUL" in op or "GEMM" in op:
            dst, lhs, rhs = o.get("dst") or o.get("out"), o.get("lhs"), o.get("rhs") or o.get("weight")
            if dst and lhs and rhs:
                matmuls.append((dst, lhs, rhs))
        elif op == "COMMIT":
            if o.get("dst") and o.get("src"):
                commits.append((o["dst"], o["src"]))
                if o.get("bias") and "bias_add" in (cmd.get("attributes", {}) or {}).get("epilogue", []):
                    commit_bias[o["dst"]] = o["bias"]
    # Chained matmul: two matmuls where the second's lhs is the first's output (A@W1 then @W2). The
    # intermediate is internal to the kernel; the graded operands are A, W1, W2 (weight-first: W1,W2,A) + Y.
    if len(matmuls) == 2:
        (d0, l0, r0), (d1, l1, r1) = matmuls
        env = materialize_inputs(cb)
        out0 = next((cout for cout, csrc in commits if csrc == d0), d0)   # matmul0's committed output
        if l1 in (d0, out0):                                              # matmul1 consumes it (via commit)
            a_nm, w1, w2 = l0, resident_source.get(r0, r0), resident_source.get(r1, r1)
            y = next((cout for cout, csrc in commits if csrc == d1), d1)
            if all(nm in env and len(env[nm].shape) == 2 for nm in (a_nm, w1, w2)):
                m, k = env[a_nm].shape
                _, k2 = env[w1].shape
                _, n = env[w2].shape
                in_args = [TensorArg(w1, k, k2, _vals(canon0, w1, env[w1]), "f32"),
                           TensorArg(w2, k2, n, _vals(canon0, w2, env[w2]), "f32"),
                           TensorArg(a_nm, m, k, _vals(canon0, a_nm, env[a_nm]), "f32")]
                out_args = [TensorArg(y, m, n, [0.0] * (m * n), "f32")]
                return in_args, out_args
        return None
    if len(matmuls) != 1:
        return None
    mdst, lhs, rhs = matmuls[0]
    rhs = resident_source.get(rhs, rhs)
    out = next((cout for cout, csrc in commits if csrc == mdst), mdst)
    if not (lhs and rhs and out):
        return None
    env = materialize_inputs(cb)
    if lhs not in env or rhs not in env:
        return None
    lt, rt = env[lhs], env[rhs]
    if len(lt.shape) != 2 or len(rt.shape) != 2:
        return None
    m, k = lt.shape[0], lt.shape[1]
    k2, n = rt.shape[0], rt.shape[1]
    if k != k2:
        return None

    # Operand VALUES: the golden's decoded operands the runner attaches at grade time
    # (``cb['canonical_inputs']`` — so the kernel runs on the SAME operands the independent golden used),
    # else the deterministic materialization (which the golden also uses when no canonical raws exist). SHAPES
    # come from the leaf tensors; the output is produced (not a leaf), so its shape is the matmul (M, N).
    canon = cb.get("canonical_inputs") or {}

    def _flat(name: str, t) -> list[float] | None:
        inj = _decode_preload(all_tensors.get(name))    # INJECTED operand takes precedence
        if inj is not None:
            return inj
        c = canon.get(name)
        if isinstance(c, dict):
            c = c.get("values")
        if c is not None:
            def _fl(v):                                  # deep-flatten (batched operands are rank-3)
                if isinstance(v, (list, tuple)):
                    out: list[float] = []
                    for e in v:
                        out.extend(_fl(e))
                    return out
                return [float(v)]
            return _fl(c)
        return [float(v) for row in t.to_list() for v in row]

    lv, rv = _flat(lhs, lt), _flat(rhs, rt)
    if lv is None or rv is None or len(lv) != m * k or len(rv) != k2 * n:
        return None
    # ABI order: [weight (+bias, before lhs)] ++ [lhs] ++ [output]. weight = the resident-resolved rhs.
    in_args = [TensorArg(rhs, k2, n, rv, "f32")]
    bias = commit_bias.get(out)
    if bias is not None:
        bt = env.get(bias)
        bv = _flat(bias, bt) if bt is not None else None
        if bt is None or bv is None:
            return None
        br, bc = _shape2d(list(bt.shape))
        in_args.append(TensorArg(bias, br, bc, bv, "f32"))
    in_args.append(TensorArg(lhs, m, k, lv, "f32"))
    out_args = [TensorArg(out, m, n, [0.0] * (m * n), "f32")]
    return in_args, out_args


def build_external_kernel_main(in_args: list[TensorArg], out_args: list[TensorArg],
                               *, kernel_symbol: str, model) -> str:
    """Harness ``main`` for an OBJECT kernel (an MLIR-lowered ``kernel.o``): declares ``kernel_symbol``
    EXTERN (not inlined), embeds every input, calls it, prints ``OUT <name> <r> <c> ...`` + ``DONE``. Unlike
    :func:`build_program` (which inlines a *source* kernel to stay relocation-free), the extern call leaves a
    cross-object ``R_RISCV_CALL`` that the fork-free reloc-preserving transcode + linker resolve. Inputs are
    stack-embedded (``_emit_fill``) so ``main`` itself carries ONLY the kernel-call relocation."""
    ptrs = ", ".join(["const void*"] * len(in_args) + ["void*"] * len(out_args)) or "void"
    body: list[str] = [_render_helpers(model).strip(), "",
                       f"extern void {kernel_symbol}({ptrs});", "",
                       "int main(void){", "  if(_hid()!=0)return 0;"]
    call_ptrs: list[str] = []
    for a in in_args:
        arr = f"_in_{a.name}"
        body += _emit_fill(arr, a)
        call_ptrs.append(f"(const void*){arr}")
    for o in out_args:
        arr = f"_out_{o.name}"
        body.append(f"  volatile uint32_t {arr}[{o.rows * o.cols}];")
        call_ptrs.append(f"(void*){arr}")
    body.append(f"  {kernel_symbol}({', '.join(call_ptrs)});")
    for o in out_args:
        arr = f"_out_{o.name}"
        body.append(f'  _ps("OUT {o.name} {o.rows} {o.cols}");')
        if o.dtype == "f32":
            body.append(f"  for(int i=0;i<{o.rows * o.cols};i++){{_pc(' ');_pf(_u2f({arr}[i]));}}")
        else:
            body.append(f"  for(int i=0;i<{o.rows * o.cols};i++){{_pc(' ');_pu({arr}[i]);}}")
        body.append("  _pc('\\n');")
    body += ['  _ps("DONE\\n");', "  return 0;", "}"]
    return "\n".join(body) + "\n"


def external_main_from_cb(cb: dict, *, kernel_symbol: str, model) -> str | None:
    """The object-kernel analogue of :func:`program_from_cb`: derive the operands from the cb and render the
    EXTERN-kernel harness ``main`` (to be compiled to ``main.o`` and fork-free-linked against the MLIR
    ``kernel.o``). None when the operands aren't available (fail-safe)."""
    derived = args_from_cb(cb)
    if derived is None:
        return None
    in_args, out_args = derived
    return build_external_kernel_main(in_args, out_args, kernel_symbol=kernel_symbol, model=model)


def program_from_cb(cb: dict, kernel_fn_src: str, model) -> str | None:
    """Build the self-contained harness program for a capsule directly from its COMMAND BUFFER, or return
    None when the artifact is already a full program (has ``main``) — the caller then compiles it directly.
    Inlines the *source* kernel (:func:`build_program`); operand order from :func:`args_from_cb`."""
    if "int main" in kernel_fn_src:
        return None
    derived = args_from_cb(cb)
    if derived is None:
        return None
    in_args, out_args = derived
    return build_program(kernel_fn_src, in_args, out_args, kernel_symbol=_kernel_symbol(kernel_fn_src),
                         model=model)
