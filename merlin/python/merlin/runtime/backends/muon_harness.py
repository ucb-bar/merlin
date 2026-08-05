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

# The console MMIO putchar aperture the cyclotron/print-runtime consumes. A per-target BSP fact (the same
# address the vendor print runtime writes); kept in this per-target edge module, not a shared code path.
CONSOLE_MMIO_ADDR = 0xFF080000


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


_HELPERS = f"""
#include <stdint.h>
static inline uint32_t _hid(void){{uint32_t r;__asm__ volatile("csrr %0,0xF14":"=r"(r));return r;}}
static inline void _pc(char c){{*(volatile char*)0x{CONSOLE_MMIO_ADDR:08x}u=c;}}
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
                  *, kernel_symbol: str) -> str:
    """Assemble the self-contained C program: helpers + the agent's kernel function + a ``main`` that
    embeds every input, calls ``kernel_symbol(<inputs>, <outputs>)``, and prints ``OUT <name> <r> <c> ...``
    for each output followed by ``DONE``. ``args`` is the kernel's input arguments in ABI order (weight,
    then lhs...); ``outputs`` the output tensors (also passed to the kernel, then printed). Float outputs
    print as fixed decimals (matched by the runner's float tolerance); integer outputs as base-10."""
    # Force the kernel to inline into main (single call site) so no R_RISCV_CALL relocation survives the
    # reassemble-after-transcode path. Prepended to the agent's definition (which starts with its return
    # type), yielding e.g. `static inline __attribute__((always_inline)) void radiance_kernel(...)`.
    kernel_inlined = "static inline __attribute__((always_inline)) " + kernel_fn_src.strip()
    body: list[str] = [_HELPERS.strip(), "", kernel_inlined, "", "int main(void){", "  if(_hid()!=0)return 0;"]
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


def program_from_cb(cb: dict, kernel_fn_src: str) -> str | None:
    """Build the self-contained harness program for a capsule directly from its COMMAND BUFFER, or return
    None when the artifact is already a full program (has ``main``) — the caller then compiles it directly.

    Derives the kernel ABI order from the cb the runner already validated: each matmul/gemm command names
    its ``weight``/``rhs``, ``lhs`` and ``out``/``dst`` operands, so the argument order is
    ``[weight] ++ [lhs in command order] ++ [outputs in command order]`` (the generic kernel_abi). Operand
    VALUES come from ``cb['canonical_inputs']`` (the runner attaches the golden's decoded operands there);
    shapes from ``cb['tensors']``. Returns None (fail-safe to direct compile) if the operands are not
    available, so this never silently grades an un-fed kernel."""
    if "int main" in kernel_fn_src:
        return None
    tensors = cb.get("tensors") or {}
    values = cb.get("canonical_inputs") or {}
    if not values:
        return None                                   # no operands attached -> let the caller compile as-is

    def _pick(operands: dict, *keys: str) -> str | None:
        for k in keys:
            if operands.get(k):
                return operands[k]
        return None

    weights, lhses, outs, seen = [], [], [], set()
    for cmd in cb.get("commands", []):
        op = (cmd.get("opcode") or "").upper()
        if "MATMUL" not in op and "GEMM" not in op:
            continue
        o = cmd.get("operands", {})
        for role, bucket in ((("weight", "rhs"), weights), (("lhs",), lhses), (("out", "dst"), outs)):
            nm = _pick(o, *role)
            if nm and nm not in seen:
                seen.add(nm)
                bucket.append(nm)
    if not (lhses and outs):
        return None                                   # not a shape we can harness -> compile as-is

    def _arg(name: str, with_values: bool) -> TensorArg | None:
        shp = (tensors.get(name) or {}).get("shape")
        r, c = _shape2d(shp)
        if with_values:
            v = (values.get(name) or {}).get("values")
            if v is None or len(v) != r * c:
                return None
            return TensorArg(name, r, c, [float(x) for x in v], "f32")
        return TensorArg(name, r, c, [0.0] * (r * c), "f32")

    in_args = [_arg(n, True) for n in (weights + lhses)]
    out_args = [_arg(n, False) for n in outs]
    if any(a is None for a in in_args + out_args):
        return None                                   # a missing operand/shape -> fail safe, do not guess
    return build_program(kernel_fn_src, in_args, out_args, kernel_symbol=_kernel_symbol(kernel_fn_src))
