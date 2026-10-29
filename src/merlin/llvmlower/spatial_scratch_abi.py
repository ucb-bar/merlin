"""Private caller-owned scratch for a source-bound staged spatial producer.

Reuse the ordinary fresh writer and workspace pool. Public tensor signatures
remain unchanged; only a private borrowed helper receives the byte descriptor.
The supplied source/effect contracts remain independently owned premises, not
qualifications inferred from pointer shape, source hashes or generated code.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from math import prod

from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan

from .borrowed_effects import BorrowedBufferEffects
from .fresh_tensor_writer import (
    FreshTensorWriterContract,
    PrivateWorkspaceContract,
    rewrite_fresh_tensor_writers,
    validate_private_workspaces,
)
from .private_workspace_pool import pool_private_writer_workspaces


@dataclass(frozen=True)
class SpatialScratchArtifact:
    module: object
    bridge_source: str
    writer_contract: FreshTensorWriterContract
    pointer_alignments: tuple[int, ...]
    owner_symbol: str
    selected_source_sha256: str
    plan_sha256: str
    receipt: dict


def _symbol(value):
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or not (value[0].isalpha() or value[0] == "_")
        or any(not (c.isalnum() or c == "_") for c in value)
    ):
        raise ValueError("spatial scratch requires a plain explicit C symbol")
    return value


def spatial_logical_buffers(plan):
    """Exact original logical pointer shapes; private scratch is excluded."""
    if type(plan) not in (SpatialProducerPlan, PackedSpatialProducerPlan):
        raise ValueError("spatial scratch requires an exact producer plan")
    plan.__post_init__()
    if type(plan) is PackedSpatialProducerPlan:
        plan = plan.source
    p, s = plan.producer, plan.spatial
    batch, _, _, ci = p.window.input_shape
    kh, kw = p.window.kernel
    oh, ow = s.output_hw
    buffers = [("input", p.window.input_shape, "i8"), ("weight", (kh * kw * ci, p.output_channels), "i8")]
    if p.bias:
        buffers.append(("bias", (p.output_channels,), "i32"))
    buffers.append(("output", (batch * oh * ow, p.output_channels), "i8"))
    return tuple(buffers)


def _transport_preamble(buffers, *, kernel_symbol=None):
    ranks = sorted({1, *(len(shape) for _, shape, _ in buffers)})
    descriptors = "\n".join(
        f"typedef struct {{ void *allocated,*aligned; int64_t offset,size[{rank}],stride[{rank}]; }} M{rank};"
        for rank in ranks
    )
    external = (
        f"extern void {_symbol(kernel_symbol)}({','.join('void*' for _ in range(len(buffers) + 1))});\n"
        if kernel_symbol is not None
        else ""
    )
    return f"""#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(uintptr_t)==8 && sizeof(intptr_t)==8,"selected ranked caller requires 64-bit pointers");
{descriptors}
{external}static int spatial_span(void *aligned,int64_t offset,uintptr_t width,uintptr_t elements,
                        uintptr_t *address,uintptr_t *bytes) {{
  if (!aligned || offset<0 || !width || !elements) return 0;
  uintptr_t limit=(uintptr_t)INTPTR_MAX/width, off=(uintptr_t)offset, base=(uintptr_t)aligned;
  if (elements>limit || off>limit-elements) return 0;
  uintptr_t delta=off*width, size=elements*width;
  if (delta>UINTPTR_MAX-base) return 0;
  uintptr_t first=base+delta;
  if (size>UINTPTR_MAX-first) return 0;
  *address=first;*bytes=size;return 1;
}}
static int spatial_disjoint(uintptr_t a,uintptr_t an,uintptr_t b,uintptr_t bn) {{
  if (an>UINTPTR_MAX-a || bn>UINTPTR_MAX-b) return 0;
  return a+an<=b || b+bn<=a;
}}
"""


def _bridge(plan, contract, workspace, kernel_symbol, alignments):
    buffers = spatial_logical_buffers(plan)
    parameters = [f"const M{len(shape)} *d{index}" for index, (_, shape, _) in enumerate(buffers)]
    parameters.append("const M1 *scratch")
    lines = [f"void _mlir_ciface_{_symbol(contract.borrowed_symbol)}({', '.join(parameters)}) {{"]
    count = len(buffers) + 1
    lines.extend(
        (f"  uintptr_t address[{count}], bytes[{count}];", f"  uintptr_t metadata[{count}], metadata_bytes[{count}];")
    )
    for index, (_, shape, dtype) in enumerate(buffers):
        name = f"d{index}"
        row_stride = 1
        checks = [f"!{name}"]
        for axis in reversed(range(len(shape))):
            checks.extend(
                (f"{name}->size[{axis}] != INT64_C({shape[axis]})", f"{name}->stride[{axis}] != INT64_C({row_stride})")
            )
            row_stride *= shape[axis]
        width = int(dtype[1:]) // 8
        lines.extend(
            (
                f"  if ({' || '.join(checks)}) __builtin_trap();",
                f"  if (!spatial_span({name}->aligned,{name}->offset,{width},UINT64_C({prod(shape)}),",
                f"       &address[{index}],&bytes[{index}]) ||",
                f"       address[{index}] % UINT64_C({alignments[index]})) __builtin_trap();",
                f"  metadata[{index}]=(uintptr_t){name}; metadata_bytes[{index}]=sizeof(*{name});",
            )
        )
    index = len(buffers)
    lines.extend(
        (
            "  if (!scratch || scratch->stride[0] != 1 ||",
            f"      scratch->size[0] < INT64_C({workspace.bytes}) ||",
            "      !spatial_span(scratch->aligned,scratch->offset,1,(uintptr_t)scratch->size[0],",
            f"          &address[{index}],&bytes[{index}]) ||",
            f"      address[{index}] % UINT64_C({workspace.alignment})) __builtin_trap();",
            f"  metadata[{index}]=(uintptr_t)scratch; metadata_bytes[{index}]=sizeof(*scratch);",
            f"  for (unsigned i=0;i<{count};++i) {{",
            "    if (metadata_bytes[i] > UINTPTR_MAX-metadata[i]) __builtin_trap();",
            f"    for (unsigned j=0;j<{count};++j) {{",
            "      if (!spatial_disjoint(address[i],bytes[i],metadata[j],metadata_bytes[j])) __builtin_trap();",
            "      if (i<j && !spatial_disjoint(address[i],bytes[i],address[j],bytes[j])) __builtin_trap();",
            "    }",
            "  }",
            f"  {_symbol(kernel_symbol)}({','.join(f'(void*)address[{n}]' for n in range(count))});",
            "}",
        )
    )
    preamble = _transport_preamble(buffers, kernel_symbol=kernel_symbol)
    return preamble + "\n".join(lines) + "\n"


def prepare_spatial_scratch_abi(
    module,
    plan,
    contract,
    workspace,
    *,
    kernel_symbol,
    pointer_alignments,
    source_binding_sha256,
    stack_budget_bytes=None,
):
    """Return a detached normal writer/pool preparation and its descriptor bridge.

    The original caller separately binds the exact source pair, numerical DAG,
    device image and effects. ``workspace`` carries that caller's explicit
    noescape/initialized-before-read/completion premises. Capacity validation
    and safe generated pointer transport cannot establish those premises.
    Accessible allocations and target DMA visibility remain runtime obligations.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ModuleOp, TensorType, i8, i32

    from merlin.xdsl_dialects._common import text

    if type(module) is not ModuleOp or type(plan) not in (SpatialProducerPlan, PackedSpatialProducerPlan):
        raise ValueError("spatial scratch requires current typed source and producer plan")
    plan.__post_init__()
    source_plan = plan.source if type(plan) is PackedSpatialProducerPlan else plan
    if source_plan.readout_mode not in ("staged_readouts", "single_readout"):
        raise ValueError("spatial scratch ABI requires a checked original readout realization")
    if (
        type(contract) is not FreshTensorWriterContract
        or contract.private_workspaces
        or type(contract.borrowed_effects) is not BorrowedBufferEffects
        or contract.declaration_abi != "ranked_c"
    ):
        raise ValueError("spatial writer needs a separate ranked borrowed contract and complete effects")
    contract.borrowed_effects.__post_init__()
    validate_private_workspaces((workspace,))
    if type(plan) is PackedSpatialProducerPlan:
        required = plan.workspace.start + plan.required_bytes
    elif source_plan.readout_mode == "staged_readouts":
        required = plan.intermediate.start + plan.temporary_rows * plan.producer.column_tile
    else:
        # Keep the existing private pointer ABI even when a complete numerical
        # proof eliminates the byte stripe. The caller still supplies checked
        # positive storage (which may be a one-byte bounded stack workspace).
        required = 0
    if type(workspace) is not PrivateWorkspaceContract or workspace.bytes < required:
        raise ValueError("private workspace does not cover the complete staged stripe and byte origin")
    if type(plan) is PackedSpatialProducerPlan and workspace.bytes < plan.workspace.start + plan.workspace.size:
        raise ValueError("private owner does not cover the complete assigned packed workspace view")
    if type(plan) is PackedSpatialProducerPlan and workspace.alignment % plan.panel_alignment:
        raise ValueError("private workspace does not guarantee the selected packed panel alignment")
    if (
        type(source_binding_sha256) is not str
        or len(source_binding_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_binding_sha256)
    ):
        raise ValueError("exact source-binding custody identity required")
    buffers = spatial_logical_buffers(plan)
    if (
        type(pointer_alignments) is not tuple
        or len(pointer_alignments) != len(buffers)
        or any(type(n) is not int or not 0 < n < 1 << 63 or n & (n - 1) for n in pointer_alignments)
    ):
        raise ValueError("every original logical pointer requires an explicit power-of-two alignment")
    _symbol(kernel_symbol)
    if contract.result_argument != len(buffers) - 1 or contract.fully_written_arguments != (len(buffers) - 1,):
        raise ValueError("spatial writer must retain its sole original fully written output")
    declaration = next(
        (f for f in module.body.block.ops if isinstance(f, func.FuncOp) and f.sym_name.data == contract.symbol), None
    )
    if declaration is None or tuple(declaration.function_type.inputs) != tuple(
        TensorType({"i8": i8, "i32": i32}[dtype], shape) for _, shape, dtype in buffers
    ):
        raise ValueError("spatial logical signature differs from original source extents and formats")
    before = text(module, generic=True)
    public_types = {
        f.sym_name.data: str(f.function_type)
        for f in module.body.block.ops
        if isinstance(f, func.FuncOp) and getattr(f.sym_visibility, "data", None) != "private"
    }
    selected = module.clone()
    selected_contract = replace(contract, private_workspaces=(workspace,))
    writer = rewrite_fresh_tensor_writers(selected, (selected_contract,))
    pool = pool_private_writer_workspaces(selected, writer, stack_budget_bytes=stack_budget_bytes)
    after_types = {
        f.sym_name.data: str(f.function_type)
        for f in selected.body.block.ops
        if isinstance(f, func.FuncOp) and getattr(f.sym_visibility, "data", None) != "private"
    }
    if public_types != after_types or text(module, generic=True) != before:
        raise ValueError("spatial scratch preparation changed the original public source signature")
    bridge = _bridge(plan, selected_contract, workspace, kernel_symbol, pointer_alignments)
    return SpatialScratchArtifact(
        selected,
        bridge,
        selected_contract,
        pointer_alignments,
        pool["owner_symbol"],
        hashlib.sha256(text(selected, generic=True).encode()).hexdigest(),
        plan.identity,
        dict(
            schema="merlin.spatial_scratch_abi.v1",
            plan_sha256=plan.identity,
            original_source_sha256=hashlib.sha256(before.encode()).hexdigest(),
            selected_source_sha256=hashlib.sha256(text(selected, generic=True).encode()).hexdigest(),
            source_binding_sha256=source_binding_sha256,
            public_signatures=public_types,
            private_workspace=asdict(workspace),
            required_stripe_bytes=required,
            logical_buffers=buffers,
            pointer_alignments=pointer_alignments,
            writer=writer,
            pool=pool,
            bridge_sha256=hashlib.sha256(bridge.encode()).hexdigest(),
            raw_private_helper=kernel_symbol,
            release="normal upstream owned deallocation; arena reclamation not inferred"
            if stack_budget_bytes is None
            else pool["release"],
            target_execution="not_established",
            source_and_effect_qualification="caller supplied; not inferred",
        ),
    )


def emit_spatial_raw_entry(artifact, plan, *, entry_symbol="kernel_entry", standalone=False):
    """Adapt the original flat logical pointers to the normal ranked public owner.

    By default concatenate with this artifact's descriptor bridge. Explicit
    ``standalone=True`` emits transport declarations for a separate object;
    the existing private bridge remains in the normal device shim. The caller must
    lower its actual detached owner through ordinary output-parameter conversion
    and link that exact C-interface definition. No private pointer is added to
    the public signature; allocation/release stay inside the compiled owner.
    """
    from xdsl.dialects import func

    from merlin.xdsl_dialects._common import text

    if (
        type(artifact) is not SpatialScratchArtifact
        or artifact.plan_sha256 != plan.identity
        or hashlib.sha256(text(artifact.module, generic=True).encode()).hexdigest() != artifact.selected_source_sha256
    ):
        raise ValueError("raw spatial entry requires its exact prepared owner and source plan")
    if type(standalone) is not bool:
        raise ValueError("standalone spatial transport requires an explicit Boolean")
    buffers = spatial_logical_buffers(plan)
    owner_symbol = artifact.owner_symbol
    owner = next(
        op for op in artifact.module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == owner_symbol
    )
    if (
        len(owner.function_type.inputs) != len(buffers) - 1
        or len(owner.function_type.outputs) != 1
        or tuple(owner.function_type.outputs)[0].get_shape() != buffers[-1][1]
    ):
        raise ValueError("raw spatial entry requires the complete original input and sole result owner")
    _symbol(entry_symbol)
    descriptor_types = [f"M{len(shape)}" for _, shape, _ in buffers]
    lines = [
        f"extern void _mlir_ciface_{_symbol(owner_symbol)}({','.join(t + '*' for t in descriptor_types)});",
        f"void {entry_symbol}({','.join('void*p' + str(i) for i in range(len(buffers)))}) {{",
        f"  uintptr_t address[{len(buffers)}],bytes[{len(buffers)}];",
    ]
    for index, (_, shape, dtype) in enumerate(buffers):
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        sizes = ",".join(str(n) for n in shape)
        steps = ",".join(str(n) for n in strides)
        lines.extend(
            (
                f"  {descriptor_types[index]} d{index}={{p{index},p{index},0,{{{sizes}}},{{{steps}}}}};",
                f"  if (!spatial_span(p{index},0,{int(dtype[1:]) // 8},UINT64_C({prod(shape)}),",
                f"      &address[{index}],&bytes[{index}]) ||",
                f"      address[{index}] % UINT64_C({artifact.pointer_alignments[index]})) __builtin_trap();",
            )
        )
    count = len(buffers)
    lines.extend(
        (
            f"  for(unsigned i=0;i<{count};++i) for(unsigned j=i+1;j<{count};++j)",
            "    if(!spatial_disjoint(address[i],bytes[i],address[j],bytes[j])) __builtin_trap();",
        )
    )
    for index in range(count):
        lines.extend(
            (
                f"  for(unsigned i=0;i<{count};++i)",
                f"    if(!spatial_disjoint(address[i],bytes[i],(uintptr_t)&d{index},sizeof(d{index})))",
                "      __builtin_trap();",
            )
        )
    lines.extend((f"  _mlir_ciface_{owner_symbol}({','.join('&d' + str(i) for i in range(count))});", "}"))
    return (_transport_preamble(buffers) if standalone else "") + "\n".join(lines) + "\n"
