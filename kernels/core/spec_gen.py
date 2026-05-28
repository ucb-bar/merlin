"""Generate a transform-dialect spec MLIR from a kernel manifest.

The output spec is consumed by IREE's preprocessing phase via
`--iree-preprocessing-transform-spec-filename=<path>`. It contains:

    1. One `hal.executable.source private @kb_<name>` per kernel, carrying
       `hal.executable.objects` that points at the per-target precompiled
       artifact produced by `precompile.py`. The export layout is derived
       from the manifest's `signature`. For SPIR-V kernels we set bindings as
       storage_buffers (one per operand) and one push constant per dynamic
       shape dim; CPU/ELF kernels use whatever the user's match.mlir
       declares.
    2. One `util.func private @call_<name>(...)` that performs a
       `flow.dispatch @kb_executable::@v::@<name>` — this is the call the
       transform spec hooks into the matched root.
    3. One `transform.named_sequence @match_<name>` per kernel: literally the
       contents of the manifest's `match.spec_path` file, wrapped in a
       canonical `transform.iree.match.cast_compatible_dag_from_root` block
       for `match.kind == "linalg_dag"`, or used verbatim for
       `match.kind == "named_sequence"`.
    4. One `transform.named_sequence @cast_and_call_<name>` that invokes
       `transform.util.cast_and_call %func(%ins) -> %out after %root`.
    5. The top-level `@__transform_main` driver running `transform.foreach
       %funcs : transform.foreach_match @match_<name> -> @cast_and_call_<name>`
       across the module.

We intentionally splice the user's match-body verbatim instead of re-deriving
it from the signature — KernelBlaster's match fragments encode body-level
match constraints (e.g. mul→add op chain, specific yield) that the manifest
signature can't represent.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import re
from collections.abc import Iterable

from . import manifest as _manifest
from . import precompile as _precompile

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    spec_path: pathlib.Path
    object_search_path: pathlib.Path  # parent dir of all object artifacts
    kernels: tuple[str, ...]
    # When the manifest contains any qnn-context-binary kernels, this is the
    # sidecar JSON path that maps `<export_symbol>` → `<.qnn-ctx absolute path>`.
    # The caller forwards it to iree-compile as
    # `--iree-hal-qnn-manifest=<path>` so the QNN passthrough plugin embeds
    # the right blob per executable export. None when no QNN kernels exist.
    qnn_manifest_path: pathlib.Path | None = None


# A single HAL executable target attribute string per HAL key. The full
# attribute is target-specific; we keep the templates close to the existing
# samples so generated MLIR is iree-opt-verifiable without any extra context.
#
# The canonical reference is samples/custom_dispatch/{vulkan,cpu}/...
_HAL_TARGET_ATTR = {
    # Minimal target attrs — just `<format, mnemonic>`. The richer form with
    # `iree_codegen.target_info = #iree_gpu.target<...>` from the
    # samples/custom_dispatch/vulkan/shaders/example.mlir reference doesn't
    # parse here because the MLIR pretty-dialect parser miscounts angle
    # brackets when iree_gpu.target's `wgp = <...>` block nests inside it.
    # Users who need richer target metadata can supply it via a hand-edited
    # spec; the embedding flow only needs the format key to match the
    # objects entry in `hal.executable.objects`.
    "qualcomm-adreno-vulkan": ('#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">'),
    "vulkan-cpu-lavapipe": ('#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">'),
    "llvm-cpu-x86_64": ('#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">'),
    "llvm-cpu-aarch64": ('#hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">'),
    "llvm-cpu-riscv64": ('#hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">'),
    "llvm-cpu-riscv64-rvv": ('#hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">'),
    "llvm-cpu-spacemit-x60": ('#hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64">'),
    # QNN backends consume opaque pre-built `.qnn-ctx` blobs. The
    # target's format string is "qnn-context-binary" (matches the QNN
    # plugin's kFormatName at compiler/plugins/target/QNN/QNNTarget.cpp).
    # The `qnn_backend` config attribute distinguishes gpu / hta at the
    # runtime HAL driver layer.
    "qnn-gpu": ('#hal.executable.target<"qnn", "qnn-context-binary", ' '{qnn_backend = "gpu", opaque_binary = true}>'),
    "qnn-hta": ('#hal.executable.target<"qnn", "qnn-context-binary", ' '{qnn_backend = "hta", opaque_binary = true}>'),
}


def hal_target_attr(target: str) -> str:
    if target not in _HAL_TARGET_ATTR:
        raise RuntimeError(
            f"no HAL target template for '{target}' — extend " f"kernels/core/spec_gen.py:_HAL_TARGET_ATTR"
        )
    return _HAL_TARGET_ATTR[target]


def _pipeline_layout_for(kernel: _manifest.KernelEntry) -> str:
    # One storage buffer binding per operand; mark `in` as ReadOnly. Push
    # constants are declared via `signature.constants` in the manifest and
    # appear before bindings in the dispatch arg list (IREE convention).
    bindings = []
    for op in kernel.signature.operands:
        if op.role == "in":
            bindings.append("#hal.pipeline.binding<storage_buffer, ReadOnly>")
        else:
            bindings.append("#hal.pipeline.binding<storage_buffer>")
    n_constants = len(kernel.signature.constants)
    return (
        f"#hal.pipeline.layout<constants = {n_constants}, bindings = [\n        " + ",\n        ".join(bindings) + "]>"
    )


def _pipeline_layout_symbol(kernel: _manifest.KernelEntry) -> str:
    return f"#pipeline_layout_{kernel.name}"


def _export_name(kernel: _manifest.KernelEntry) -> str:
    """Name used for `hal.executable.export` and the wrapper `func.func` inside
    the executable's `builtin.module`. Distinct from `kernel.entry_symbol`
    (which is the precompiled-object's C symbol) so the wrapper can call into
    the imported C function without name collision. Convention: drop a
    trailing `_workgroup` suffix when present (matches the IREE sample at
    `third_party/iree_bar/samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir`,
    where export `simple_mul_abs_negate` calls C symbol
    `simple_mul_abs_negate_workgroup`); otherwise append `_dispatch`.
    """
    sym = kernel.entry_symbol
    if sym.endswith("_workgroup"):
        return sym[: -len("_workgroup")]
    return sym + "_dispatch"


def _tensor_to_memref(tensor_ty: str) -> str:
    # Naive: replace the leading `tensor<` with `memref<`. Sufficient for the
    # signature shapes we currently emit (all-static or 1D-dynamic).
    if not tensor_ty.startswith("tensor<"):
        raise RuntimeError(f"expected tensor type, got '{tensor_ty}'")
    return "memref<" + tensor_ty[len("tensor<") :]


def _inner_module_for_c(kernel: _manifest.KernelEntry) -> str:
    """Inner `builtin.module` for C-source kernels — declares the precompiled
    C symbol with `hal.import.static` and emits the wrapper that materializes
    bindings and calls into it.

    Layout of the call: `<binding ptrs/types>, <push constants as index>, tid`.
    Push constants are loaded via `hal.interface.constant.load ordinal(N)` and
    cast from i32 to index so the C signature can be plain `size_t` for each.
    The `tid` is derived from `hal.interface.workgroup.id[0]` and is always
    the last argument. The kernel-level grid is `count == workload`, so for
    a 1D op this is one workgroup per element (no separate `dim` argument is
    passed when there are user-declared push constants — the kernel can
    recover `dim` from one of them).
    """
    if kernel.source_lang != "c":
        return ""

    layout_sym = _pipeline_layout_symbol(kernel)
    export = _export_name(kernel)
    csym = kernel.entry_symbol
    operands = kernel.signature.operands
    constants = kernel.signature.constants

    has_user_constants = bool(constants)
    # When the user declares push constants, derive the per-binding memref
    # `{%dim_X, %dim_Y, ...}` shape annotations from those constants. Otherwise
    # fall back to the implicit single `%dim = workgroup.count[0]` path that
    # the no-constant case used.
    subspan_lines: list[str] = []
    call_args: list[str] = []
    call_types: list[str] = []

    # Build a map from (input_index, dim_index) -> constant name for fast
    # lookup when annotating subspan memref shapes. Includes aliases — a
    # single logical dim (e.g. matmul's K) often appears in multiple input
    # tensor positions.
    dim_const_lookup: dict[tuple[int, int], str] = {}
    for c in constants:
        dim_const_lookup[(c.source.input_index, c.source.dim_index)] = c.name
        for alias in c.aliases:
            dim_const_lookup[(alias.input_index, alias.dim_index)] = c.name

    in_indices = [i for i, op in enumerate(operands) if op.role == "in"]
    explicit_out_dims = kernel.signature.output_dims

    def _resolve_out_dim(k: int) -> tuple[int, int]:
        """Resolve output dim k -> (input_index, dim_index) source. Honors
        the manifest's explicit output_dims list when present.
        """
        if explicit_out_dims:
            cname = explicit_out_dims[k]
            for c in constants:
                if c.name == cname:
                    return (c.source.input_index, c.source.dim_index)
            raise RuntimeError(f"kernel '{kernel.name}': output_dims '{cname}' not in " f"declared constants")
        # Legacy heuristic for unannotated kernels (matmul-shaped).
        if k == 0 and in_indices:
            return (in_indices[0], 0)
        if in_indices:
            return (in_indices[-1], k)
        return (0, k)

    for i, op in enumerate(operands):
        memref_ty = _tensor_to_memref(op.tensor)
        n_dynamic = memref_ty.count("?")
        if n_dynamic == 0:
            dim_suffix = ""
        elif has_user_constants:
            # Resolve each `?` via the user's constants. For inputs we look up
            # (i, k) directly. For outputs we synthesize from the inputs:
            # convention is that the output's leading dims come from inputs[0]
            # and trailing dims from inputs[-1]. This matches matmul/elementwise
            # shapes; manifests with exotic shape lattices need a hand-edited
            # spec.
            dim_refs: list[str] = []
            for k in range(n_dynamic):
                if op.role == "in":
                    src = (i, k)
                else:
                    src = _resolve_out_dim(k)
                cname = dim_const_lookup.get(src)
                if cname is None:
                    raise RuntimeError(
                        f"kernel '{kernel.name}': operand {i} dim {k} cannot "
                        f"be resolved from declared constants — add a "
                        f'`{{"name": ..., "from": {{"input": {src[0]}, '
                        f'"dim": {src[1]}}}}}` constant or hand-edit the '
                        f"generated spec"
                    )
                dim_refs.append(f"%{cname}")
            dim_suffix = "{" + ", ".join(dim_refs) + "}"
        else:
            # Legacy 1D-dynamic single-`?` path.
            if n_dynamic > 1:
                raise RuntimeError(
                    f"kernel '{kernel.name}': operand {i} has multiple "
                    f"dynamic dims but no `signature.constants` declared — "
                    f"add push constants for each runtime dim"
                )
            dim_suffix = "{%dim}"
        subspan_lines.append(
            f"        %binding{i} = hal.interface.binding.subspan "
            f"layout({layout_sym}) binding({i}) alignment(64) offset(%c0) "
            f": {memref_ty}{dim_suffix}"
        )
        call_args.append(f"%binding{i}")
        call_types.append(memref_ty)

    # Constant-load preamble.
    const_load_lines: list[str] = []
    const_call_args: list[str] = []
    const_call_types: list[str] = []
    for j, c in enumerate(constants):
        const_load_lines.append(
            f"          %{c.name}_i32 = hal.interface.constant.load " f"layout({layout_sym}) ordinal({j}) : i32"
        )
        const_load_lines.append(f"          %{c.name} = arith.index_cast %{c.name}_i32 " f": i32 to index")
        const_call_args.append(f"%{c.name}")
        const_call_types.append("index")

    csym_param_types = ", ".join(call_types + const_call_types + ["index"])
    csym_args = ", ".join(call_args + const_call_args + ["%tid"])

    if has_user_constants:
        dim_setup: list[str] = ["          %tid = hal.interface.workgroup.id[0] : index"]
    else:
        dim_setup = [
            "          %dim = hal.interface.workgroup.count[0] : index",
            "          %tid = hal.interface.workgroup.id[0] : index",
        ]

    inner = [
        "      builtin.module {",
        f"        func.func private @{csym}({csym_param_types})" " attributes {hal.import.static}",
        f"        func.func @{export}() {{",
        "          %c0 = arith.constant 0 : index",
        *const_load_lines,
        *dim_setup,
        *subspan_lines,
        f"          func.call @{csym}({csym_args}) " f": ({csym_param_types}) -> ()",
        "          return",
        "        }",
        "      }",
    ]
    return "\n".join(inner)


def _read_match_body(match_spec_path: pathlib.Path) -> str:
    text = match_spec_path.read_text()
    return text.strip()


# Match a top-level `^bb0(%name0: type0, %name1: type1, ...)` line. We
# deliberately anchor at column 0 so we don't pick up nested `^bb0(...)`
# blocks inside `linalg.generic` bodies (those declare scalar element
# arguments, not the matched DAG's operand types).
_BB0_LINE_RE = re.compile(r"^\^bb0\(([^)]*)\)\s*:")


def _matched_block_arg_types(
    kernel: _manifest.KernelEntry,
) -> tuple[str, ...] | None:
    """Return the operand types declared by the match's `^bb0(...)` block,
    or None if the kernel doesn't use a linalg-DAG match (no .match.mlir to
    inspect).

    These types drive the `util.func @call_<name>` wrapper signature so its
    arity matches the value count `cast_compatible_dag_from_root` returns
    via `%ins`. cast_and_call binds %ins → wrapper args by position, so any
    arity skew between the matched DAG and the wrapper is a hard error.
    """
    if kernel.match.kind != "linalg_dag":
        return None
    body = kernel.match.spec_path.read_text()
    types: list[str] = []
    for line in body.splitlines():
        if not line.startswith("^bb0("):
            continue
        m = _BB0_LINE_RE.match(line)
        if not m:
            continue
        decl = m.group(1).strip()
        if not decl:
            return ()
        # Split on commas not inside angle brackets — tensor types contain
        # commas (e.g., `tensor<1x16xf32>` has none, but `tensor<?x?xf32>`
        # could; future shapes like `vector<4x4xf32>` likewise). A simple
        # depth counter is enough.
        parts: list[str] = []
        depth = 0
        cur = ""
        for ch in decl:
            if ch in "<({[":
                depth += 1
            elif ch in ">)}]":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur.strip())
                cur = ""
                continue
            cur += ch
        if cur.strip():
            parts.append(cur.strip())
        for p in parts:
            # `%name : type` — pull the type half.
            if ":" not in p:
                continue
            types.append(p.split(":", 1)[1].strip())
        return tuple(types)
    return None


def _target_alias_name(target: str) -> str:
    return "kb_target_" + target.replace("-", "_").replace("/", "_")


def _executable_block(
    kernel: _manifest.KernelEntry,
    objects: dict[tuple[str, str], _precompile.ObjectArtifact],
    object_root: pathlib.Path,
) -> str:
    # `hal.executable.objects` keys must be top-level attribute aliases —
    # the parser rejects an inline `#hal.executable.target<...>` because it
    # contains nested angle brackets. We emit `#kb_target_<key> = ...` at the
    # top of the module and reference the short alias here. This mirrors the
    # custom_dispatch/{vulkan,cpu} samples.
    parts = []
    parts.append(f"  hal.executable.source private @kb_{kernel.name} attributes {{")
    parts.append("    objects = #hal.executable.objects<{")
    target_lines = []
    for target in kernel.targets:
        artifact = objects.get((kernel.name, target))
        if artifact is None:
            continue
        rel = artifact.path.relative_to(object_root) if artifact.path.is_relative_to(object_root) else artifact.path
        alias = _target_alias_name(target)
        target_lines.append(f"      #{alias} = [\n" f'        #hal.executable.object<{{path = "{rel}"}}>\n' f"      ]")
    parts.append(",\n".join(target_lines))
    parts.append("    }>")
    parts.append("  } {")
    parts.append(f"    hal.executable.export public @{_export_name(kernel)} ordinal(0)")
    parts.append(f"        layout({_pipeline_layout_symbol(kernel)})")
    # Default workgroup count: one workgroup per output tensor element, X-only.
    # Manifests can override by hand-editing the generated spec or supplying a
    # full named_sequence match.kind. We document the limitation here rather
    # than auto-deriving a bad count region for every shape.
    # Workgroup count is `(1, 1, 1)` for QNN-backed kernels — the entire
    # graph is dispatched as one work item and the QNN runtime handles its
    # own internal tiling. Returning `%workload` from the count region
    # would force HAL conversion's `calculateWorkgroupCountFromRegion` to
    # look up a value `stream.cmd.dispatch` doesn't carry (no workload
    # operands when the kernel has no dynamic dims), tripping IRMapping.
    parts.append("        count(%device: !hal.device, %workload: index)")
    parts.append("        -> (index, index, index) {")
    parts.append("      %c1 = arith.constant 1 : index")
    parts.append("      hal.return %c1, %c1, %c1 : index, index, index")
    parts.append("    }")
    inner = _inner_module_for_c(kernel)
    if inner:
        parts.append(inner)
    parts.append("  }  // hal.executable.source")
    return "\n".join(parts)


def _is_fully_static(tensor_ty: str) -> bool:
    return "?" not in tensor_ty


def _is_1d_dynamic(tensor_ty: str) -> bool:
    # Naive: matches "tensor<?x...>" with exactly one '?' as the leading dim.
    return tensor_ty.startswith("tensor<?x") and tensor_ty.count("?") == 1


def _call_wrapper(kernel: _manifest.KernelEntry) -> str:
    """Generates a `util.func` whose argument list mirrors the *inputs* of the
    matched DAG. The transform.util.cast_and_call op binds matched %ins (the
    inputs returned from the linalg-DAG match) to these args, and the function
    returns the dispatched output. The output tensor is not an explicit arg —
    it materializes from the flow.dispatch result, matching the pattern used
    by samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir.

    Three emission paths:
      - All-static signature, no constants: simple flow.dispatch.
      - 1D dynamic, no constants: emit `tensor.dim`, dispatch with [%dim]
        workload, annotate each tensor with {%dim}.
      - Multi-D with declared push constants: emit `tensor.dim` for each
        constant per its from-spec, materialize i32 versions for the dispatch
        constant operands, compute workload as the product of the output's
        dynamic dims, and annotate each tensor with the corresponding {%dim_*}
        list.
    """
    ins = [op for op in kernel.signature.operands if op.role == "in"]
    outs = [op for op in kernel.signature.operands if op.role == "out"]
    if len(outs) != 1:
        raise RuntimeError(f"kernel '{kernel.name}': expected exactly 1 'out' operand, " f"got {len(outs)}")
    out_ty = outs[0].tensor
    constants = kernel.signature.constants

    # When the kernel has a linalg-DAG match pattern, the wrapper must accept
    # *all* values that `cast_compatible_dag_from_root` returns from `%ins`,
    # not just the manifest's `in` operands. The matched ins always include
    # the linalg op's destination (`init`) tensors and any extra tensors the
    # match's `^bb0(...)` declares (e.g. weight/bias for fused conv kernels).
    # Mismatch surfaces as
    #   "mismatch between number of function arguments N and number of inputs M"
    # at the foreach_match action.
    #
    # We parse `^bb0(...)` from the .match.mlir to recover those types, then
    # synthesize the wrapper with one arg per matched value. Dispatch still
    # only forwards the manifest-declared ins; trailing init/aux args are
    # accepted and discarded — they exist purely so cast_and_call's arity
    # matches the matcher's `%ins` cardinality.
    matched_arg_tys = _matched_block_arg_types(kernel)

    if not constants and all(_is_fully_static(op.tensor) for op in kernel.signature.operands):
        if matched_arg_tys is not None:
            wrapper_arg_decl = ", ".join(f"%m{i}: {ty}" for i, ty in enumerate(matched_arg_tys))
        else:
            wrapper_arg_decl = ", ".join(f"%in{i}: {op.tensor}" for i, op in enumerate(ins))
        # Dispatch only consumes the manifest's `in` tensors — by convention
        # the first `len(ins)` matched values map 1:1 to the manifest ins.
        # The remaining matched args (linalg destinations, baked-in weights,
        # …) flow in as wrapper args but never reach the dispatch.
        in_types = ", ".join(op.tensor for op in ins)
        in_refs = ", ".join((f"%m{i}" if matched_arg_tys is not None else f"%in{i}") for i in range(len(ins)))
        return (
            f"  util.func private @call_{kernel.name}({wrapper_arg_decl}) -> {out_ty} {{\n"
            f"    %0 = flow.dispatch @kb_{kernel.name}::@{_export_name(kernel)}"
            f"({in_refs}) : ({in_types}) -> {out_ty}\n"
            f"    util.return %0 : {out_ty}\n"
            f"  }}"
        )

    if not constants and all(_is_1d_dynamic(op.tensor) for op in kernel.signature.operands):
        in_args = ", ".join(f"%in{i}: {op.tensor}" for i, op in enumerate(ins))
        in_refs = ", ".join(f"%in{i}" for i in range(len(ins)))
        in_types_dim = ", ".join(f"{op.tensor}{{%dim}}" for op in ins)
        return (
            f"  util.func private @call_{kernel.name}({in_args}) -> {out_ty} {{\n"
            f"    %c0 = arith.constant 0 : index\n"
            f"    %dim = tensor.dim %in0, %c0 : {ins[0].tensor}\n"
            f"    %0 = flow.dispatch @kb_{kernel.name}::@{_export_name(kernel)}"
            f"[%dim]({in_refs}) : ({in_types_dim}) -> {out_ty}{{%dim}}\n"
            f"    util.return %0 : {out_ty}\n"
            f"  }}"
        )

    if not constants:
        raise RuntimeError(
            f"kernel '{kernel.name}': multi-dim dynamic shape needs declared "
            f"`signature.constants` (one per runtime dim) so the wrapper can "
            f"derive shapes via tensor.dim and pass them as push constants."
        )

    # Constants path: derive each from tensor.dim, build the dispatch.
    # Build lookup including aliases (matches `_inner_module_for_c`).
    dim_const_lookup_call: dict[tuple[int, int], str] = {}
    for c in constants:
        dim_const_lookup_call[(c.source.input_index, c.source.dim_index)] = c.name
        for alias in c.aliases:
            dim_const_lookup_call[(alias.input_index, alias.dim_index)] = c.name
    in_args = ", ".join(f"%in{i}: {op.tensor}" for i, op in enumerate(ins))
    lines: list[str] = []
    lines.append(f"  util.func private @call_{kernel.name}({in_args}) -> {out_ty} {{")
    # Index constants for tensor.dim args. Emit one per unique dim_index used.
    seen_axes: set[int] = set()
    for c in constants:
        if c.source.dim_index not in seen_axes:
            lines.append(f"    %c_axis_{c.source.dim_index} = arith.constant " f"{c.source.dim_index} : index")
            seen_axes.add(c.source.dim_index)
    # tensor.dim per declared constant.
    for c in constants:
        in_op = ins[c.source.input_index]
        lines.append(
            f"    %{c.name} = tensor.dim %in{c.source.input_index}, " f"%c_axis_{c.source.dim_index} : {in_op.tensor}"
        )
    # Cast each to i32 for push-constant transport.
    for c in constants:
        lines.append(f"    %{c.name}_i32 = arith.index_cast %{c.name} : index to i32")
    # Per-input shape annotations: for each `?` in tensor type, look up the
    # constant whose source matches (input_idx, k).
    dim_const_lookup = dim_const_lookup_call
    in_indices = list(range(len(ins)))
    explicit_out_dims_call = kernel.signature.output_dims

    def _resolve_out_dim_call(k: int) -> tuple[int, int]:
        if explicit_out_dims_call:
            cname = explicit_out_dims_call[k]
            for c in constants:
                if c.name == cname:
                    return (c.source.input_index, c.source.dim_index)
            raise RuntimeError(f"kernel '{kernel.name}': output_dims '{cname}' not in " f"declared constants")
        if k == 0 and in_indices:
            return (in_indices[0], 0)
        if in_indices:
            return (in_indices[-1], k)
        return (0, k)

    def _shape_annot(op: _manifest.Operand, op_index: int, role: str) -> str:
        n_dynamic = op.tensor.count("?")
        if n_dynamic == 0:
            return ""
        refs: list[str] = []
        for k in range(n_dynamic):
            if role == "in":
                src = (op_index, k)
            else:
                src = _resolve_out_dim_call(k)
            cname = dim_const_lookup.get(src)
            if cname is None:
                raise RuntimeError(
                    f"kernel '{kernel.name}': cannot annotate operand "
                    f"shape for ({op_index}, {k}); declare a constant "
                    f"sourced from input {src[0]} dim {src[1]}"
                )
            refs.append(f"%{cname}")
        return "{" + ", ".join(refs) + "}"

    in_types_dim = ", ".join(f"{op.tensor}{_shape_annot(op, i, 'in')}" for i, op in enumerate(ins))
    out_annot = _shape_annot(outs[0], 0, "out")
    # Workload: product of the output tensor's dynamic dims. The kernel sees
    # one workgroup per output element via the auto-emitted
    # `count(%workload) -> (%workload, 1, 1)` region; the wrapper decodes
    # `tid` into per-axis indices using the push-constant dims.
    out_dynamic = outs[0].tensor.count("?")
    if out_dynamic == 0:
        workload = ""
    else:
        # Resolve each output dim to a constant name (same heuristic as
        # _shape_annot for outputs).
        out_dim_names: list[str] = []
        for k in range(out_dynamic):
            src = _resolve_out_dim_call(k)
            cname = dim_const_lookup.get(src)
            if cname is None:
                raise RuntimeError(
                    f"kernel '{kernel.name}': cannot derive output dim {k}; "
                    f"declare a constant sourced from input {src[0]} dim {src[1]}"
                )
            out_dim_names.append(cname)
        if out_dynamic == 1:
            workload = f"[%{out_dim_names[0]}]"
        else:
            # Chained arith.muli: %workload = dim0 * dim1 * dim2 * ...
            prev = f"%{out_dim_names[0]}"
            for j, name in enumerate(out_dim_names[1:], start=1):
                cur = f"%workload_{j}" if j < out_dynamic - 1 else "%workload"
                lines.append(f"    {cur} = arith.muli {prev}, %{name} : index")
                prev = cur
            workload = "[%workload]"

    in_refs = ", ".join(f"%in{i}" for i in range(len(ins)))
    const_i32_refs = ", ".join(f"%{c.name}_i32" for c in constants)
    const_i32_types = ", ".join("i32" for _ in constants)
    all_refs = ", ".join(filter(None, [const_i32_refs, in_refs]))
    all_types = ", ".join(filter(None, [const_i32_types, in_types_dim]))
    lines.append(
        f"    %0 = flow.dispatch @kb_{kernel.name}::@{_export_name(kernel)}"
        f"{workload}({all_refs}) : ({all_types}) -> {out_ty}{out_annot}"
    )
    lines.append(f"    util.return %0 : {out_ty}")
    lines.append("  }")
    return "\n".join(lines)


def dim_index_used_for_input(c: _manifest.Constant) -> int:
    return c.source.dim_index


def _named_op_match_body(kernel: _manifest.KernelEntry) -> str:
    """Synthesize a body for `match.kind: "named_op"` — a canonical scaffold
    matching any concrete instance of `kernel.match.op_name` regardless of
    shape. The signature.operands list determines the input/output structure
    of the matched DAG; cast_and_call re-binds them to the wrapper.

    For named ops with `outs(...)` semantics that consume a fill-initialized
    accumulator (matmul, conv, pooling), the scaffold includes a preceding
    `tensor.empty` + `linalg.fill` chain marked with `match.operation_name_only`
    so it matches whatever init the payload uses. For ops without such
    accumulator semantics (broadcast, transpose), the scaffold omits the
    fill chain.
    """
    op_name = kernel.match.op_name
    ins_all = [op for op in kernel.signature.operands if op.role == "in"]
    outs = [op for op in kernel.signature.operands if op.role == "out"]
    if len(outs) != 1:
        raise RuntimeError(f"kernel '{kernel.name}': named_op match requires exactly 1 'out' " f"operand")
    out_ty = outs[0].tensor

    # Decide whether `outs(...)` of the matched op is one of our `in` operands
    # (e.g. dronet's conv where a broadcasted bias is fed into `outs`) or is
    # synthesized via a `linalg.fill` scaffold.
    outs_idx = kernel.match.outs_from_input
    if outs_idx >= 0:
        if outs_idx >= len(ins_all):
            raise RuntimeError(
                f"kernel '{kernel.name}': outs_from_input={outs_idx} out of "
                f"range (kernel has {len(ins_all)} inputs)"
            )
        outs_operand = ins_all[outs_idx]
        # The `ins(...)` of the matched op is everything except the outs-input.
        ins_for_op = [op for i, op in enumerate(ins_all) if i != outs_idx]
        ins_for_op_indices = [i for i in range(len(ins_all)) if i != outs_idx]
    else:
        outs_operand = None
        ins_for_op = ins_all
        ins_for_op_indices = list(range(len(ins_all)))

    bb_args = ", ".join(f"%in{i}: {op.tensor}" for i, op in enumerate(ins_all))
    in_refs = ", ".join(f"%in{i}" for i in ins_for_op_indices)
    in_types = ", ".join(op.tensor for op in ins_for_op)

    body_lines: list[str] = ["^bb0(" + bb_args + "):"]
    op_attrs = kernel.match.op_attrs
    attrs_str = f" {op_attrs} " if op_attrs else " "

    if outs_operand is not None:
        # outs is one of the inputs; use it directly.
        outs_ref = f"%in{outs_idx}"
        body_lines.append(
            f"  %op = {op_name}{attrs_str}"
            f"ins({in_refs} : {in_types}) "
            f"outs({outs_ref} : {outs_operand.tensor}) -> {out_ty}"
        )
    else:
        # Synthesize a fill-initialized empty for ops with accumulator outs
        # (matmul, conv, pool). The `match.operation_name_only` annotation
        # makes the matcher treat tensor.empty / linalg.fill as wildcards
        # whose contents (constants, dim args) don't have to match.
        body_lines.append(f'  %empty = tensor.empty() {{"match.operation_name_only"}} : {out_ty}')
        body_lines.append("  %cst = arith.constant 0.000000e+00 : f32")
        body_lines.append(
            f'  %filled = linalg.fill {{"match.operation_name_only"}} '
            f"ins(%cst : f32) outs(%empty : {out_ty}) -> {out_ty}"
        )
        body_lines.append(
            f"  %op = {op_name}{attrs_str}" f"ins({in_refs} : {in_types}) " f"outs(%filled : {out_ty}) -> {out_ty}"
        )
    return "\n".join(body_lines)


def _match_named_sequence(kernel: _manifest.KernelEntry) -> str:
    if kernel.match.kind == "named_op":
        body = _named_op_match_body(kernel)
        return (
            f"  transform.named_sequence @match_{kernel.name}(\n"
            f"      %root: !transform.any_op {{transform.readonly}})\n"
            f"      -> (!transform.any_value, !transform.any_value) {{\n"
            f"    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {{\n"
            f"{body}\n"
            f"    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)\n"
            f"    transform.yield %ins, %outs : !transform.any_value, !transform.any_value\n"
            f"  }}"
        )

    body = _read_match_body(kernel.match.spec_path)
    if kernel.match.kind == "named_sequence":
        # The user supplied a complete named sequence; just splice.
        return body
    # linalg_dag: wrap the body in the canonical match scaffold.
    return (
        f"  transform.named_sequence @match_{kernel.name}(\n"
        f"      %root: !transform.any_op {{transform.readonly}})\n"
        f"      -> (!transform.any_value, !transform.any_value) {{\n"
        f"    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {{\n"
        f"{body}\n"
        f"    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)\n"
        f"    transform.yield %ins, %outs : !transform.any_value, !transform.any_value\n"
        f"  }}"
    )


def _cast_and_call_named_sequence(kernel: _manifest.KernelEntry) -> str:
    # `transform.type_conversion.tensor.cast_shape_dynamic_dims` lets
    # cast_and_call insert `tensor.cast` ops when the matched payload (in the
    # user's MLIR) has different staticness from the wrapper signature
    # (typically static payload → dynamic wrapper, mirroring
    # samples/custom_dispatch/cpu/embedded/example_transform_spec.mlir).
    return (
        f"  transform.named_sequence @cast_and_call_{kernel.name}(\n"
        f"      %ins: !transform.any_value {{transform.readonly}},\n"
        f"      %out: !transform.any_value {{transform.readonly}}) {{\n"
        f"    %root = transform.get_defining_op %out "
        f": (!transform.any_value) -> !transform.any_op\n"
        f"    %module = transform.util.get_nearest_symbol_table %root "
        f": (!transform.any_op) -> !transform.any_op\n"
        f"    %executable = transform.util.import_symbol "
        f"@kb_{kernel.name} into %module if undefined "
        f": (!transform.any_op) -> !transform.any_op\n"
        f"    %func = transform.util.import_symbol "
        f"@call_{kernel.name} into %module if undefined "
        f": (!transform.any_op) -> !transform.any_op\n"
        f"    transform.util.cast_and_call %func(%ins) -> %out after %root {{\n"
        f"        transform.type_conversion.tensor.cast_shape_dynamic_dims\n"
        f"    }} : (!transform.any_op, !transform.any_value, "
        f"!transform.any_value, !transform.any_op) -> !transform.any_op\n"
        f"    transform.yield\n"
        f"  }}"
    )


def _foreach_match_block(kernels: Iterable[_manifest.KernelEntry]) -> str:
    pairs = "\n        ".join(f"@match_{k.name} -> @cast_and_call_{k.name}," for k in kernels)
    pairs = pairs.rstrip(",")
    return (
        f"  transform.named_sequence @__transform_main(%module: !transform.any_op) {{\n"
        f'    %funcs = transform.structured.match ops{{["util.func"]}} in %module\n'
        f"        : (!transform.any_op) -> !transform.any_op\n"
        f"    transform.foreach %funcs : !transform.any_op {{\n"
        f"    ^bb1(%f: !transform.any_op):\n"
        f"      transform.foreach_match in %f\n"
        f"        {pairs}\n"
        f"        : (!transform.any_op) -> (!transform.any_op)\n"
        f"    }}\n"
        f"    transform.apply_dce to %module : !transform.any_op\n"
        f"    transform.yield\n"
        f"  }}"
    )


def emit(
    manifest: _manifest.Manifest,
    objects: dict[tuple[str, str], _precompile.ObjectArtifact],
    out_path: pathlib.Path,
    *,
    object_search_path: pathlib.Path | None = None,
) -> GenerationResult:
    """Generate a transform-dialect spec covering every kernel in `manifest`.

    `object_search_path` becomes the prefix that hal.executable.object paths
    are written relative to; the caller is expected to pass the same path to
    `iree-compile --iree-hal-executable-object-search-path=<dir>`. If
    omitted, defaults to the parent of the first object's path.
    """
    if not manifest.kernels:
        raise ValueError(f"manifest {manifest.path}: no kernels to emit")

    if object_search_path is None:
        any_artifact = next(iter(objects.values()), None)
        if any_artifact is None:
            raise ValueError(f"manifest {manifest.path}: no precompiled objects supplied")
        object_search_path = any_artifact.path.parent
    object_search_path = object_search_path.resolve()

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Top-level HAL-target attribute aliases. These have to live at module
    # scope so they can appear unbraced as keys in `hal.executable.objects`.
    needed_targets: dict[str, str] = {}
    for kernel in manifest.kernels:
        for target in kernel.targets:
            if (kernel.name, target) not in objects:
                continue
            alias = _target_alias_name(target)
            if alias not in needed_targets:
                needed_targets[alias] = hal_target_attr(target)

    parts: list[str] = []
    for alias, attr in needed_targets.items():
        parts.append(f"#{alias} = {attr}")
    # Per-kernel pipeline-layout aliases — referenced both by `hal.executable.export`
    # and by `hal.interface.binding.subspan` inside the wrapper.
    for kernel in manifest.kernels:
        parts.append(f"{_pipeline_layout_symbol(kernel)} = {_pipeline_layout_for(kernel)}")
    parts.append("module attributes {transform.with_named_sequence} {")
    for kernel in manifest.kernels:
        parts.append(_executable_block(kernel, objects, object_search_path))
    for kernel in manifest.kernels:
        parts.append(_call_wrapper(kernel))
    for kernel in manifest.kernels:
        parts.append(_match_named_sequence(kernel))
        parts.append(_cast_and_call_named_sequence(kernel))
    parts.append(_foreach_match_block(manifest.kernels))
    parts.append("}")

    out_path.write_text("\n\n".join(parts) + "\n")

    # Write a sidecar QNN manifest when any kernel produces a .qnn-ctx blob.
    # The QNN passthrough plugin keys by `hal.executable.export` symbol —
    # which `_export_name(kernel)` derives from `kernel.entry_symbol`.
    qnn_manifest_path: pathlib.Path | None = None
    qnn_entries: dict[str, str] = {}
    for kernel in manifest.kernels:
        for target in kernel.targets:
            artifact = objects.get((kernel.name, target))
            if artifact is None or not getattr(artifact, "qnn_context", False):
                continue
            export_sym = _export_name(kernel)
            qnn_entries[export_sym] = str(artifact.path)
    if qnn_entries:
        qnn_manifest_path = out_path.with_suffix(".qnn_manifest.json")
        import json as _json

        qnn_manifest_path.write_text(_json.dumps(qnn_entries, indent=2) + "\n")
        _LOG.info("qnn manifest -> %s (%d kernel(s))", qnn_manifest_path, len(qnn_entries))

    return GenerationResult(
        spec_path=out_path,
        object_search_path=object_search_path,
        kernels=tuple(k.name for k in manifest.kernels),
        qnn_manifest_path=qnn_manifest_path,
    )
