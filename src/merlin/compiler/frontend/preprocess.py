"""Compose shared source rewrites and preserve actual source-to-output accounting.

Preserve source ordering, numerical/effect obligations and explicit selection.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.compiler.frontend.repairs import (  # noqa: F401 -- exact historical preprocessing namespace
    _AXIS_HEAD,
    _AXIS_TAIL,
    _DEQUANT_HEAD,
    _EXTRACT_HEAD,
    _FUNC_HEAD,
    _IDENT_CHARS,
    _INSERT_HEAD,
    _OFFSETS_FIELD,
    _OPERAND_TYPES,
    _PROPS_CLOSE,
    _PROPS_OPEN,
    _RESULT_SEP,
    _SIZES_FIELD,
    _STRIDES_FIELD,
    _TYPES_SEP,
    _array_field,
    _attach_c_interface,
    _Dequant,
    _dequant_to_generic,
    _fix_extract_slices,
    _fix_insert_slices,
    _parse_dequant,
    _skip_space,
    _slice_op_span,
    _split_operand_types,
    preprocess_text_textual,
)
from merlin.compiler.ir.provenance import carry_provenance as carry_provenance
from merlin.compiler.lower.scalar.casts import (  # noqa: F401 -- exact historical preprocessing namespace
    fix_bool_fptosi,
    fix_bool_sitofp,
)
from merlin.compiler.quantization.linalg import (  # noqa: F401 -- exact historical preprocessing namespace
    _DEQUANT_KINDS,
    _QUANT_KINDS,
    lower_quant_ext,
)
from merlin.compiler.transforms.dead import (  # noqa: F401 -- exact historical preprocessing namespace
    _dead_tensor_operation_is_pure,
    prune_dead_pure_tensor_ops,
)
from merlin.compiler.transforms.matmul import (  # noqa: F401 -- exact historical preprocessing namespace
    collapse_overrank_matmul,
    lower_bf16_matmul_f32acc,
)
from merlin.frontends.linalg_mlir import (  # noqa: F401 -- exact historical preprocessing namespace
    make_context,
    parse_mlir_text,
)


def implementation_paths() -> tuple[Path, ...]:
    """Source custody of the original shared preprocessing implementation.

    Keep the complete former file envelope and the actual selected stage
    bindings. A patched stage is an input to the caller, rather than evidence
    that its disconnected default implementation ran.
    """
    from merlin.compiler.frontend import repairs
    from merlin.compiler.ir import provenance
    from merlin.compiler.lower.scalar import casts
    from merlin.compiler.quantization import linalg
    from merlin.compiler.transforms import dead, matmul

    owners = (repairs, provenance, casts, linalg, dead, matmul)
    selected = (globals()[name] for _, _, name in _PREPROCESS)
    legacy = Path(__file__).parents[2] / "llvmlower" / "passes_xdsl.py"
    return tuple(
        dict.fromkeys(
            (
                Path(__file__),
                *((legacy,) if legacy.is_file() else ()),
                *(Path(owner.__file__) for owner in owners),
                *(Path(function.__code__.co_filename) for function in selected),
            )
        )
    )


def add_c_interface(module) -> int:
    """Keep the host descriptor route, preserving private scalar callback ABIs.

    Existing shaped functions retain their descriptor-interface selection.
    Private scalar functions need no descriptor adaptation; adding an implicit
    interface to their declarations would request a different runtime symbol.
    Explicit interfaces remain untouched, including on private callbacks.
    """
    from xdsl.dialects.builtin import AnyFloat, IndexType, IntegerType, UnitAttr
    from xdsl.dialects.func import FuncOp

    n = 0
    for op in module.walk():
        if isinstance(op, FuncOp):
            private_scalar = (
                op.sym_visibility is not None
                and op.sym_visibility.data == "private"
                and all(
                    isinstance(t, IntegerType | IndexType | AnyFloat)
                    for t in (*op.function_type.inputs, *op.function_type.outputs)
                )
            )
            if private_scalar:
                continue
            op.attributes["llvm.emit_c_interface"] = UnitAttr()
            n += 1
    return n


_PREPROCESS = (
    ("dead_tensor_ops_pruned", "xdsl-pruned", "prune_dead_pure_tensor_ops"),
    ("quant_ext_lowered", "xdsl-quant-lowered", "lower_quant_ext"),
    ("c_interface_funcs", "xdsl-c-interface", "add_c_interface"),
)

PREPROCESS_STAGES = ("xdsl-parsed", *(stage for _, stage, _ in _PREPROCESS))


def _layer_norm_chunked_sums_selected() -> bool:
    """Whether the numerics-changing ``layer-norm-chunked-sums`` pass runs (default off)."""
    import os

    from merlin.llvmlower.optional_passes import switched

    return switched("layer-norm-chunked-sums", os.environ.get("MERLIN_LAYER_NORM_CHUNKED_SUMS") == "1")


def _preprocess_module(module, *, audit=None, on_quant_rewrite=None) -> dict:
    from functools import partial

    from merlin.compiler.passes.xdsl import Rewrite, run_rewrites
    from merlin.xdsl_dialects.ir_inspection import record_stage

    def selected():
        for statistic, stage, name in _PREPROCESS:
            transform = globals()[name]
            if statistic == "quant_ext_lowered" and on_quant_rewrite is not None:
                transform = partial(transform, on_rewrite=on_quant_rewrite)
            yield Rewrite(statistic, stage, transform)
            if statistic == "quant_ext_lowered" and _layer_norm_chunked_sums_selected():
                # Optional and default off. Its numerical permissions and transform-map
                # ownership remain here; the result lands in the next recorded stage.
                from merlin.llvmlower.normalization_reassociation import chunk_layer_norm_sums

                yield Rewrite(
                    "layer_norm_sums_chunked",
                    None,
                    partial(
                        chunk_layer_norm_sums,
                        allow_reassociation=True,
                        assume_finite_intermediates=True,
                        on_rewrite=on_quant_rewrite,
                    ),
                )

    return run_rewrites(
        module,
        selected(),
        observe=partial(record_stage, audit, generic=True) if audit is not None else None,
        initial_stage=PREPROCESS_STAGES[0],
    )


def preprocess_text(mlir_text: str, *, audit=None) -> tuple[str, dict]:
    """Run Merlin xDSL passes, optionally observing each completed rewrite without changing it."""
    from merlin.xdsl_dialects._common import text as module_to_text

    module = parse_mlir_text(mlir_text)
    transform_map = None
    if audit is not None and audit.directory is not None:
        supported = _SourceTransformMap.supports(module)
        audit.record["source_transform_map"] = {
            "status": "pending" if supported else "unavailable",
            "scope": "one defined, single-block func.func; top-level non-return operations",
        }
        if not supported:
            audit.record["source_transform_map"]["reason"] = (
                "module has zero or multiple defined functions, or a multi-block function"
            )
        audit._flush()
        if supported:
            transform_map = _SourceTransformMap(mlir_text, module)
    stats = _preprocess_module(
        module,
        audit=audit,
        on_quant_rewrite=transform_map.record if transform_map is not None else None,
    )
    # Generic form preserves source attributes on every operation and avoids
    # custom-printer grammar skew at the upstream MLIR boundary (notably yield
    # and rank-changing tensor operations). This is serialization, not lowering.
    upstream = module_to_text(module, generic=True)
    if transform_map is not None:
        descriptor = audit.accounting("source-transform-map", transform_map.finish(upstream))
        audit.record["source_transform_map"] = {
            "status": "recorded",
            "scope": "one defined, single-block func.func; top-level non-return operations",
            "receipt": descriptor["file"],
            "sha256": descriptor["sha256"],
        }
        audit._flush()
    return upstream, stats


def preprocess_text_with_transform_map(mlir_text: str) -> tuple[str, dict, dict]:
    """Preprocess one entry and receipt-map every top-level source op/result.

    This is an accounting map, not a semantics proof. The normal audited path
    records the same map as an accounting sidecar without a second parse/rewrite.
    A rewrite must explicitly register every generated helper and its result
    replacement; an unowned emitted operation causes refusal.
    """
    from merlin.xdsl_dialects._common import text as module_to_text

    module = parse_mlir_text(mlir_text)
    transform_map = _SourceTransformMap(mlir_text, module)
    stats = _preprocess_module(module, on_quant_rewrite=transform_map.record)
    upstream = module_to_text(module, generic=True)
    return upstream, stats, transform_map.finish(upstream)


class _SourceTransformMap:
    """Track raw top-level operation identities through the structural rewrite."""

    @staticmethod
    def supports(module) -> bool:
        functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
        return len(functions) == 1 and len(functions[0].body.blocks) == 1

    def __init__(self, mlir_text: str, module):
        self.source_sha256 = sha256(mlir_text.encode()).hexdigest()
        functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
        if len(functions) != 1 or len(functions[0].body.blocks) != 1:
            raise ValueError("transform map requires exactly one single-block function")
        self.function = functions[0]
        self.original = [op for op in self.function.body.block.ops if op.name != "func.return"]
        self.source_by_id = {id(op): index for index, op in enumerate(self.original)}
        self.generated_owner: dict[int, int] = {}
        self.replacements: dict[int, Any] = {}

    def record(self, source, generated: tuple[Any, ...], result_owner) -> None:
        index = self.source_by_id.get(id(source))
        if index is None or index in self.replacements:
            raise ValueError("quantization rewrite lacks a unique top-level source operation")
        if len(source.results) != len(result_owner.results):
            raise ValueError("quantization rewrite changes source result count")
        for old, new in zip(source.results, result_owner.results):
            if old.type != new.type:
                raise ValueError("quantization rewrite changes source result type")
        for op in generated:
            if id(op) in self.source_by_id or id(op) in self.generated_owner:
                raise ValueError("generated operation has conflicting source ownership")
            self.generated_owner[id(op)] = index
        self.replacements[index] = result_owner

    def finish(self, upstream: str) -> dict:
        processed = [op for op in self.function.body.block.ops if op.name != "func.return"]
        emitted_by_source: list[list[int]] = [[] for _ in self.original]
        processed_index_by_id = {id(op): index for index, op in enumerate(processed)}
        for index, op in enumerate(processed):
            owner = self.source_by_id.get(id(op), self.generated_owner.get(id(op)))
            if owner is None:
                raise ValueError(f"preprocessed operation {index} has no raw source owner")
            emitted_by_source[owner].append(index)
        operations = []
        for index, source in enumerate(self.original):
            replacement = self.replacements.get(index, source)
            emitted = emitted_by_source[index]
            if id(replacement) not in processed_index_by_id and emitted:
                raise ValueError(f"source operation {index} lost its result replacement")
            result_map = []
            if id(replacement) in processed_index_by_id:
                result_map = [
                    {
                        "source_result_index": result_index,
                        "preprocessed_op_index": processed_index_by_id[id(replacement)],
                        "preprocessed_result_index": result_index,
                    }
                    for result_index in range(len(source.results))
                ]
            elif source.results and any(result.uses for result in source.results):
                raise ValueError(f"source operation {index} was elided with live results")
            operations.append(
                {
                    "source_op_index": index,
                    "source_op_name": source.op_name.data if source.name == "builtin.unregistered" else source.name,
                    "preprocessed_op_indices": emitted,
                    "result_map": result_map,
                }
            )
        return {
            "schema": "source_transform_map_v1",
            "source_sha256": self.source_sha256,
            "preprocessed_sha256": sha256(upstream.encode()).hexdigest(),
            "entry": self.function.sym_name.data,
            "source_op_count": len(self.original),
            "preprocessed_op_count": len(processed),
            "operations": operations,
        }
