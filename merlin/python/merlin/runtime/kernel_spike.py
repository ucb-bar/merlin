"""Run ONE lowered kernel on spike, against data the caller chose.

The existing spike paths take either a command buffer (`backends.spike`) or a whole captured model
(`backends.spike_model`). Neither fits a single standalone kernel that took the generic LLVM route:
there is no command buffer, and there is no model directory with weights and an input bundle. This
module is that missing rung — MLIR in, numbers out — and it is deliberately kernel-generic, so it
serves any frontend that produces linalg-on-tensors rather than the one that motivated it.

Inputs are baked into the image as bytes and outputs come back over the console as **raw bit
patterns**, never as formatted decimals. Two reasons: a freestanding harness has no float printer
worth trusting, and a bit pattern makes "the same answer" an exact question instead of a question
about how many digits were printed.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.paths import runtime_dir

from .backends import spike as _spike
from .backends.spike_model import RVV_CFLAGS

# Inputs are emitted as C initializers, so a large tensor becomes a pathological source file long
# before it becomes a memory problem. Refuse early with a clear cause.
MAX_EMBEDDED_BYTES = 1 << 20

# How each element type crosses the console. Floats travel as their bit pattern because the harness
# has no float formatter; the receiving side views the integers back, so the comparison is exact.
_ELEMENT = {
    "i8": ("signed char", np.int8, np.int8),
    "i16": ("short", np.int16, np.int16),
    "i32": ("int", np.int32, np.int32),
    "i64": ("long", np.int64, np.int64),
    "f16": ("unsigned short", np.uint16, np.float16),
    "bf16": ("unsigned short", np.uint16, None),      # no numpy bfloat16; caller reinterprets
    "f32": ("unsigned int", np.uint32, np.float32),
    "f64": ("unsigned long", np.uint64, np.float64),
}


class KernelSpikeError(RuntimeError):
    pass


@dataclass
class KernelRun:
    """One kernel executed on spike, plus the evidence for how it was compiled."""

    elf: Path
    outputs: list[np.ndarray]
    console: str
    vector_ops: list[str] = field(default_factory=list)
    workdir: Path | None = None


def available() -> bool:
    return _spike.available()


def _entry_name(module) -> str:
    fns = [op for op in module.walk() if op.name == "func.func"]
    if len(fns) != 1:
        raise KernelSpikeError(f"expected exactly one func.func, found {len(fns)}")
    return fns[0].sym_name.data


def _byte_literals(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array).tobytes()
    if len(raw) > MAX_EMBEDDED_BYTES:
        raise KernelSpikeError(
            f"{len(raw)} bytes of input exceeds the {MAX_EMBEDDED_BYTES} this harness embeds in the "
            "image; a kernel that large needs the whole-model path, which loads a weights blob")
    return ", ".join(str(b) for b in raw)


def _descriptor_struct(rank: int) -> str:
    dims = max(rank, 1)
    return (f"typedef struct {{ void *a; void *b; long off; long sizes[{dims}]; "
            f"long strides[{dims}]; }} mr{rank};\n")


def _descriptor_init(name: str, buffer: str, shape: tuple[int, ...]) -> str:
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    sizes = ", ".join(str(s) for s in shape) or "1"
    stride_text = ", ".join(str(s) for s in strides) or "1"
    return (f"  static mr{len(shape)} {name} = {{0, 0, 0, {{{sizes}}}, {{{stride_text}}}}};\n"
            f"  {name}.a = (void*){buffer}; {name}.b = (void*){buffer};\n")


def generate_main(entry: str, inputs: list[np.ndarray], out_shapes: list[tuple[int, ...]],
                  out_dtypes: list[str]) -> str:
    """The C driver: embed the inputs, call the kernel, print each output's bit patterns."""
    ranks = sorted({len(a.shape) for a in inputs} | {len(s) for s in out_shapes})
    parts = ['#include "htif.h"\n\n']
    parts += [_descriptor_struct(r) for r in ranks]

    for i, array in enumerate(inputs):
        parts.append(f"static unsigned char in{i}[] __attribute__((aligned(64))) = "
                     f"{{{_byte_literals(array)}}};\n")
    for i, (shape, dtype) in enumerate(zip(out_shapes, out_dtypes)):
        element, _, _ = _element_of(dtype)
        count = int(np.prod(shape)) if shape else 1
        parts.append(f"static {element} out{i}[{count}] __attribute__((aligned(64)));\n")

    args = ", ".join(["void*"] * (len(inputs) + len(out_shapes)))
    parts.append(f"\nextern void _mlir_ciface_{entry}({args});\n\n")
    parts.append("int main(void) {\n  console_init();\n")
    call = []
    for i, array in enumerate(inputs):
        parts.append(_descriptor_init(f"d_in{i}", f"in{i}", tuple(array.shape)))
        call.append(f"&d_in{i}")
    for i, shape in enumerate(out_shapes):
        parts.append(_descriptor_init(f"d_out{i}", f"out{i}", tuple(shape)))
        call.append(f"&d_out{i}")
    parts.append(f"  _mlir_ciface_{entry}({', '.join(call)});\n")

    for i, (shape, dtype) in enumerate(zip(out_shapes, out_dtypes)):
        count = int(np.prod(shape)) if shape else 1
        parts.append(
            f'  htif_puts("OUT out{i} 1 {count}");\n'
            f"  for (int k = 0; k < {count}; k++) {{ htif_putc(' '); htif_putd((long)out{i}[k]); }}\n"
            '  htif_putc(\'\\n\');\n')
    parts.append('  htif_puts("DONE\\n");\n  htif_exit(0);\n  return 0;\n}\n')
    return "".join(parts)


def _element_of(dtype: str):
    try:
        return _ELEMENT[dtype]
    except KeyError:
        raise KernelSpikeError(f"unsupported element type {dtype!r}") from None


def _run(cmd: list, timeout: int = 900) -> None:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise KernelSpikeError(f"{cmd[0]} failed:\n{proc.stdout}\n{proc.stderr}")


def build(module, inputs: list[np.ndarray], workdir: str | Path, *, vectorize: bool = True,
          transform_schedule: str | None = None) -> dict[str, Any]:
    """Lower ``module`` to an rv64gcv ELF that runs it on ``inputs``. Returns the build record."""
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.kernel_backend import signature_of
    from merlin.llvmlower.lower import lower_model
    from merlin.xdsl_dialects._common import text

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    entry = _entry_name(module)
    fn = next(op for op in module.walk() if op.name == "func.func")
    sig = signature_of(fn)
    if len(inputs) != len(sig.in_shapes):
        raise KernelSpikeError(f"kernel takes {len(sig.in_shapes)} input(s), got {len(inputs)}")
    for i, (array, shape) in enumerate(zip(inputs, sig.in_shapes)):
        if tuple(array.shape) != tuple(shape):
            raise KernelSpikeError(f"input {i} has shape {array.shape}, kernel expects {shape}")

    res = lower_model(text(module), work / "lower", targets=(), vectorize=vectorize,
                      transform_schedule=transform_schedule)
    _run([toolchain.clang(), "--target=riscv64-unknown-elf", *RVV_CFLAGS, "-c", res.ll_path,
          "-o", work / "kernel.o"])

    main_c = work / "main.c"
    main_c.write_text(generate_main(entry, inputs, sig.out_shapes, sig.out_dtypes),
                      encoding="utf-8")
    harness = runtime_dir() / "baremetal/spike"
    gcc = _spike.gcc_path()
    objects = []
    for name, src in (("main.o", main_c), ("crt.o", harness / "crt.S"),
                      ("htif.o", harness / "htif.c"), ("libc.o", harness / "libc_min.c")):
        _run([gcc, *RVV_CFLAGS, "-I", harness, "-c", src, "-o", work / name])
        objects.append(work / name)
    objects.append(work / "kernel.o")

    elf = work / "kernel.elf"
    _run([gcc, *RVV_CFLAGS, "-nostdlib", "-nostartfiles", "-T", harness / "link.ld",
          *objects, "-lm", "-o", elf])
    return {"elf": elf, "entry": entry, "signature": sig, "ll_path": res.ll_path,
            "object": work / "kernel.o"}


def vector_ops(object_path: str | Path) -> list[str]:
    """Which RVV instructions the kernel object actually contains — evidence, not inference."""
    from merlin.llvmlower.custom_isa import disassemble

    disassembly = disassemble(Path(object_path))
    candidates = ("vsetvli", "vsetivli", "vle8.v", "vle32.v", "vse8.v", "vse32.v", "vadd.vv",
                  "vmul.vv", "vfadd.vv", "vfmul.vv", "vfmacc.vv", "vwmacc.vv", "vredsum.vs",
                  "vmv.v.i", "vmv.v.x")
    return [op for op in candidates if op in disassembly]


def run(module, inputs: list[np.ndarray], workdir: str | Path, *, vectorize: bool = True,
        harts: int = 1, isa: str = "rv64gcv_zfh_zvfh", timeout: int = 900,
        transform_schedule: str | None = None) -> KernelRun:
    """Build and run ``module`` on spike; return the outputs as numpy arrays."""
    from merlin.runtime.backends.base import parse_console

    record = build(module, inputs, workdir, vectorize=vectorize,
                   transform_schedule=transform_schedule)
    console = _spike.run_elf(record["elf"], harts=harts, isa=isa, timeout=timeout)
    parsed, _ = parse_console(console, error_cls=KernelSpikeError)

    sig = record["signature"]
    outputs: list[np.ndarray] = []
    for i, (shape, dtype) in enumerate(zip(sig.out_shapes, sig.out_dtypes)):
        rows = parsed.get(f"out{i}")
        if rows is None:
            raise KernelSpikeError(f"kernel printed no `OUT out{i}` line:\n{console}")
        _, wire, view = _element_of(dtype)
        flat = np.array(rows[0], dtype=np.int64).astype(wire)
        if view is None:
            raise KernelSpikeError(
                f"output {i} is {dtype}, which has no numpy view — read the raw words instead")
        outputs.append(flat.view(view).reshape(shape))
    return KernelRun(elf=record["elf"], outputs=outputs, console=console,
                     vector_ops=vector_ops(record["object"]), workdir=Path(workdir))
