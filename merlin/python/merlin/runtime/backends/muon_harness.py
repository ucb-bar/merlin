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
    from ..commandbuffer import materialize_inputs

    def _vals(canon: dict, name: str, t) -> list[float]:
        """Operand VALUES: the golden's decoded operands the runner attached (``cb['canonical_inputs']``,
        so the kernel runs on the SAME operands the independent golden used), else the deterministic
        materialization. Flat, row-major, as f32."""
        c = canon.get(name)
        if isinstance(c, dict):
            c = c.get("values")
        if c is not None:
            return [float(x) for x in c]
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

    # Tolerant single-matmul plan (do NOT reuse the strict muon_codegen._plan, which assumes ``dst`` on every
    # matmul): resolve RES_PACK residents, the one matmul (accepting ``dst``/``out`` and ``rhs``/``weight``),
    # and the output NAME — the COMMIT dst that sources the matmul when present, else the matmul's own dst.
    resident_source: dict[str, str] = {}
    matmuls: list[tuple[str, str, str]] = []      # (dst, lhs, rhs)
    commits: list[tuple[str, str]] = []           # (out, src)
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
        c = canon.get(name)
        if isinstance(c, dict):
            c = c.get("values")
        if c is not None:
            return [float(x) for x in c]
        return [float(v) for row in t.to_list() for v in row]

    lv, rv = _flat(lhs, lt), _flat(rhs, rt)
    if lv is None or rv is None or len(lv) != m * k or len(rv) != k2 * n:
        return None
    # ABI order: [weight] ++ [lhs] ++ [output]. weight = the (resident-resolved) rhs operand.
    in_args = [TensorArg(rhs, k2, n, rv, "f32"), TensorArg(lhs, m, k, lv, "f32")]
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
