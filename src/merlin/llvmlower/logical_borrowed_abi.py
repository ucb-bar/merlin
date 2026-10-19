"""Checked logical-buffer borrowing with a byte-preserving packing fallback.

The primitive is an external implementation, never decoded or synthesized here.
Original software declares its layout/alignment/access contract and the caller
supplies independent qualification. Descriptor checks do not prove accessible
allocation, implementation equivalence or hardware support.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path

from merlin.compile.scheduling.dense_contracts import DenseLogicalPointerABI

from .borrowed_buffer_calls import BorrowedBufferEffects
from .device_shim import LogicalBuffer, ShimUnit, _memref_typedef


@dataclass(frozen=True)
class LogicalBorrowedABI:
    """Original conditional primitive premises and an explicit scratch budget.

    Qualification covers preservation of logical input bytes, exact bounded
    reads and complete bounded output writes. Neither these declarations nor a
    successful numerical test establish that independent premise.
    """

    declaration: DenseLogicalPointerABI
    qualified_effects: BorrowedBufferEffects
    format_qualification_identity: str
    max_stack_bytes: int
    observe_transfers: bool = False
    observer_effects: BorrowedBufferEffects | None = None
    observer_preserves_logical_state: bool = False
    readout_format_qualification_identity: str | None = None

    def __post_init__(self):
        if (
            type(self.declaration) is not DenseLogicalPointerABI
            or self.declaration.layout != "dense_row_major"
            or self.declaration.access != ("read_only", "read_only", "fully_written")
            or self.declaration.bounds != "logical_elements"
            or self.declaration.completion != "synchronous"
            or type(self.qualified_effects) is not BorrowedBufferEffects
            or type(self.format_qualification_identity) is not str
            or not self.format_qualification_identity.strip()
            or len(self.format_qualification_identity) > 4096
            or type(self.max_stack_bytes) is not int
            or not 0 < self.max_stack_bytes < 1 << 63
            or type(self.observe_transfers) is not bool
            or self.readout_format_qualification_identity is not None
            and (
                type(self.readout_format_qualification_identity) is not str
                or not self.readout_format_qualification_identity.strip()
                or len(self.readout_format_qualification_identity) > 4096
            )
        ):
            raise ValueError("logical borrowed ABI requires original exact format/access/effects and bounded scratch")
        self.qualified_effects.__post_init__()
        if type(self.observer_preserves_logical_state) is not bool:
            raise ValueError("logical ABI observer requires original logical-state preservation")
        if self.observe_transfers:
            if type(self.observer_effects) is not BorrowedBufferEffects or not self.observer_preserves_logical_state:
                raise ValueError("logical ABI observer requires separately qualified effects and state preservation")
            self.observer_effects.__post_init__()
        elif self.observer_effects is not None or self.observer_preserves_logical_state:
            raise ValueError("logical ABI observer effects require selected observation")
        alignments = self.declaration.alignment_bytes
        if (
            type(alignments) is not tuple
            or len(alignments) != 3
            or any(type(a) is not int or not 0 < a < 1 << 63 or a & (a - 1) for a in alignments)
        ):
            raise ValueError("logical borrowed ABI requires explicit power-of-two primitive alignment")


def implementation_paths():
    from . import device_shim

    return (Path(__file__), Path(device_shim.__file__))


def dense_buffers(problem, dtypes, *, output_dtype=None):
    """Use current source extents and selected element formats, never target names."""
    if output_dtype is not None:
        dtypes = (*dtypes[:2], output_dtype)
    return tuple(
        LogicalBuffer(name, kind, name, shape, dtype)
        for name, kind, shape, dtype in zip(
            ("lhs", "rhs", "out"),
            ("input", "input", "output"),
            ((problem.m, problem.k), (problem.k, problem.n), (problem.m, problem.n)),
            dtypes,
            strict=True,
        )
    )


def _geometry(buffers, capability):
    capability.__post_init__()
    if type(buffers) is not tuple or len(buffers) != 3 or any(type(b) is not LogicalBuffer for b in buffers):
        raise ValueError("logical dense ABI requires three original typed logical buffers")
    widths, scratch = [], 0
    for i, (buffer, alignment) in enumerate(zip(buffers, capability.declaration.alignment_bytes, strict=True)):
        if (
            type(buffer.shape) is not tuple
            or len(buffer.shape) != 2
            or any(type(n) is not int or n <= 0 for n in buffer.shape)
            or buffer.dtype not in ("i8", "i16", "i32")
            or buffer.kind != ("output" if i == 2 else "input")
        ):
            raise ValueError("logical dense ABI requires static positive rank-two integer buffers")
        width = int(buffer.dtype[1:]) // 8
        size = prod(buffer.shape) * width
        if size >= 1 << 63:
            raise ValueError("logical dense ABI byte extent exceeds the ranked caller's signed index range")
        # Typed scratch also satisfies its element's natural alignment. Round
        # alignment padding in elements, keeping the complete array in budget.
        scratch += size + max(width, alignment) - width
        widths.append(width)
    if buffers[0].shape[1] != buffers[1].shape[0] or buffers[2].shape != (buffers[0].shape[0], buffers[1].shape[1]):
        raise ValueError("logical dense ABI operand extents must describe one complete contraction")
    if scratch > capability.max_stack_bytes:
        raise ValueError("logical dense ABI packing fallback exceeds the explicit stack scratch budget")
    return tuple(widths), scratch


def scratch_bytes(buffers, capability):
    return _geometry(buffers, capability)[1]


def _symbol(value):
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or not (value[0].isalpha() or value[0] == "_")
        or any(not (c.isalnum() or c == "_") for c in value)
    ):
        raise ValueError("logical ABI requires an explicit portable C symbol")
    return value


_CHECKS = r"""
/* The current generated model caller uses signed 64-bit ranked descriptor
 * fields. This is its software ABI, not an inferred target pointer width. */
_Static_assert(sizeof(intptr_t)==sizeof(int64_t), "logical bridge requires the selected 64-bit ranked caller ABI");
static int span(const merlin_lmemref_2d *d,uintptr_t width,uintptr_t *address,uintptr_t *length) {
  if(!d || !d->aligned || d->offset<0 || d->sizes[0]<=0 || d->sizes[1]<=0 ||
     d->strides[0]<=0 || d->strides[1]<=0)return 0;
  uintptr_t limit=(uintptr_t)INTPTR_MAX/width;
  uintptr_t last=(uintptr_t)d->offset, r=(uintptr_t)d->sizes[0]-1,c=(uintptr_t)d->sizes[1]-1;
  uintptr_t a=(uintptr_t)d->strides[0],b=(uintptr_t)d->strides[1];
  if(last>=limit || r>(limit-last-1)/a)return 0;
  last+=r*a;
  if(c>(limit-last-1)/b)return 0;
  last+=c*b;
  uintptr_t base=(uintptr_t)d->aligned,delta=(uintptr_t)d->offset*width;
  uintptr_t bytes=(last-(uintptr_t)d->offset+1)*width;
  if(delta>UINTPTR_MAX-base)return 0;
  uintptr_t start=base+delta;
  if(bytes>UINTPTR_MAX-start)return 0;
  *address=start;*length=bytes;return 1;
}
static int disjoint(uintptr_t a,uintptr_t an,uintptr_t b,uintptr_t bn) {
  return a+an<=b || b+bn<=a;
}
static int injective(const merlin_lmemref_2d *d) {
  uintptr_t a=(uintptr_t)d->strides[0],b=(uintptr_t)d->strides[1],x=a,y=b;
  while(y){uintptr_t r=x%y;x=y;y=r;}
  return !((uintptr_t)d->sizes[0]>b/x && (uintptr_t)d->sizes[1]>a/x);
}
"""


def emit_borrowed_dense_translation_unit(
    capability,
    buffers,
    *,
    symbol,
    kernel,
    borrow_contiguous=True,
    observer_symbol=None,
):
    """Emit a ranked C bridge; actual descriptors decide borrowing per operand.

    Unsupported layout/alignment packs into private bounded scratch; output
    aliasing, malformed descriptors and address/index overflow refuse before
    the primitive. Read-only input aliasing is permitted. Accessible backing
    and qualified primitive behavior remain independent original premises.
    """
    widths, _scratch = _geometry(buffers, capability)
    symbol, kernel = _symbol(symbol), _symbol(kernel)
    if symbol == kernel or type(borrow_contiguous) is not bool:
        raise ValueError("logical ABI needs distinct explicit symbols and a Boolean optimization control")
    if capability.observe_transfers:
        observer_symbol = _symbol(observer_symbol)
        if observer_symbol in (symbol, kernel):
            raise ValueError("logical ABI observer must have its own explicit symbol")
    elif observer_symbol is not None:
        raise ValueError("logical ABI transfer observation was not selected")
    lines = ["#include <stdint.h>", _memref_typedef(2), _CHECKS, f"extern void {kernel}(void *,void *,void *);"]
    if observer_symbol:
        lines.append(f"extern void {observer_symbol}(uint64_t,uint64_t,uint64_t);")
    lines.extend(
        (
            f"void {symbol}(const merlin_lmemref_2d *d0,const merlin_lmemref_2d *d1,merlin_lmemref_2d *d2) {{",
            "  uintptr_t address[3],length[3];",
        )
    )
    for i, (buffer, width) in enumerate(zip(buffers, widths, strict=True)):
        rows, columns = buffer.shape
        lines.append(
            f"  if(!span(d{i},{width}u,&address[{i}],&length[{i}]) || "
            f"d{i}->sizes[0]!={rows} || d{i}->sizes[1]!={columns})__builtin_trap();"
        )
    lines.append(
        "  if(!injective(d2) || !disjoint(address[0],length[0],address[2],length[2]) || "
        "!disjoint(address[1],length[1],address[2],length[2]))__builtin_trap();"
    )
    for i, (buffer, width, alignment) in enumerate(
        zip(
            buffers,
            widths,
            capability.declaration.alignment_bytes,
            strict=True,
        )
    ):
        rows, columns = buffer.shape
        size = rows * columns * width
        storage_alignment = max(width, alignment)
        elements = (size + storage_alignment - width) // width
        condition = (
            f"d{i}->strides[0]=={columns} && d{i}->strides[1]==1 && address[{i}]%{alignment}u==0"
            if borrow_contiguous
            else "0"
        )
        lines.extend(
            (
                f"  int direct{i}=({condition});",
                f"  _Static_assert(_Alignof(int{width * 8}_t)=={width},"
                '"logical ABI scratch requires the selected native integer alignment");',
                f"  int{width * 8}_t raw{i}[direct{i}?1:{elements}];",
                f"  uintptr_t base{i}=(uintptr_t)raw{i};",
                f"  if(base{i}>UINTPTR_MAX-{storage_alignment - 1}u)__builtin_trap();",
                f"  unsigned char *p{i}=direct{i}?(unsigned char*)address[{i}]:"
                f"(unsigned char*)((base{i}+{storage_alignment - 1}u)&~(uintptr_t){storage_alignment - 1}u);",
            )
        )
    if observer_symbol:
        lines.append("  uint64_t read_bytes=0,write_bytes=0;")
    for i, (buffer, width) in enumerate(zip(buffers[:2], widths[:2], strict=True)):
        rows, columns = buffer.shape
        lines.extend(
            (
                f"  if(!direct{i})for(intptr_t r=0;r<{rows};r++)for(intptr_t c=0;c<{columns};c++)",
                f"    for(intptr_t b=0;b<{width};b++){{",
                f"      p{i}[(r*{columns}+c)*{width}+b]=((const unsigned char*)d{i}->aligned)"
                f"[(d{i}->offset+r*d{i}->strides[0]+c*d{i}->strides[1])*{width}+b];",
            )
        )
        if observer_symbol:
            lines.append("      read_bytes++;write_bytes++;")
        lines.append("    }")
    lines.append(f"  {kernel}(p0,p1,p2);")
    rows, columns = buffers[2].shape
    width = widths[2]
    lines.extend(
        (
            f"  if(!direct2)for(intptr_t r=0;r<{rows};r++)for(intptr_t c=0;c<{columns};c++)",
            f"    for(intptr_t b=0;b<{width};b++){{",
            "      ((unsigned char*)d2->aligned)[(d2->offset+r*d2->strides[0]+c*d2->strides[1])"
            f"*{width}+b]=p2[(r*{columns}+c)*{width}+b];",
        )
    )
    if observer_symbol:
        lines.append("      read_bytes++;write_bytes++;")
    lines.append("    }")
    if observer_symbol:
        lines.append(
            f"  {observer_symbol}(read_bytes,write_bytes,"
            "(uint64_t)direct0|((uint64_t)direct1<<1)|((uint64_t)direct2<<2));"
        )
    lines.append("}")
    return ShimUnit(text="\n".join(lines) + "\n", symbols=(symbol,), kernel=kernel, kernels=(kernel,))


def prepare_binding(binding, capability, directory, *, borrow_contiguous):
    """Bind one shared bridge to the current source and declared primitive ABI."""
    from dataclasses import replace

    binding.compilation.revalidate()
    source = binding.compilation.operation
    m, k = source.inputs[0].type.get_shape()
    _, n = source.inputs[1].type.get_shape()
    selected = binding.compilation.selected_contracts
    readout = getattr(binding.compilation, "output_readout", None)
    if readout is not None and (
        not capability.readout_format_qualification_identity
        or capability.readout_format_qualification_identity != binding.compilation.readout_qualification_identity
    ):
        raise ValueError("logical narrow output requires the original independently qualified readout format")
    buffers = dense_buffers(
        selected.dense.problem(m, k, n),
        selected.dense.dtypes,
        output_dtype=getattr(binding.compilation, "output_dtype", None),
    )
    kernel = binding.writer.symbol + "_logical_kernel"
    observer = binding.writer.symbol + "_abi_observe" if capability.observe_transfers else None
    unit = emit_borrowed_dense_translation_unit(
        capability,
        buffers,
        symbol="_mlir_ciface_" + binding.writer.borrowed_symbol,
        kernel=kernel,
        borrow_contiguous=borrow_contiguous,
        observer_symbol=observer,
    )
    directory.mkdir(exist_ok=True)
    path = directory / (binding.writer.symbol + ".c")
    if path.exists():
        raise ValueError("logical ABI bridge output already exists")
    path.write_text(unit.text)
    return replace(binding, logical_kernel_symbol=kernel, logical_abi_source=path, logical_observer_symbol=observer)


def compile_binding(binding, context):
    """Compile the shared bridge through the original provider command owner."""
    from merlin.common.digest import sha256_file
    from merlin.runtime.host_provider import HostProviderObject, read_host_function_abi

    binding.compilation.revalidate()
    source = binding.logical_abi_source
    llvm, obj = (context.workdir / (binding.writer.symbol + suffix) for suffix in ("_logical_abi.ll", "_logical_abi.o"))
    commands = []
    for flags, output in ((["-S", "-emit-llvm"], llvm), (["-c"], obj)):
        argv = [str(context.compiler), *context.compiler_flags, *flags, str(source), "-o", str(output)]
        context.compile(argv, inputs=(source,), output=output)
        commands.append(tuple(argv))
    required = (
        binding.logical_kernel_symbol,
        *((binding.logical_observer_symbol,) if binding.logical_observer_symbol is not None else ()),
    )
    return HostProviderObject(
        obj,
        sha256_file(obj),
        llvm,
        sha256_file(llvm),
        (read_host_function_abi(llvm, "_mlir_ciface_" + binding.writer.borrowed_symbol),),
        tuple(read_host_function_abi(llvm, symbol) for symbol in required),
        tuple((path.resolve(), sha256_file(path)) for path in (source, context.compiler)),
        tuple(commands),
        Path.cwd(),
    )
