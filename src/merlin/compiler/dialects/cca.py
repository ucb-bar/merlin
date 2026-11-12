"""The ``cca`` dialect (xDSL): a Common Compute Abstraction expressed IN MLIR.

A CCA is the DESCRIPTIVE, target-agnostic record of what a kernel does — composed by deterministic
analyzers (asm decode + flat-graph inspection), never authored by an LLM. This dialect is the MLIR
representation of that record so a CCA is inspectable, versionable, diffable, and composable in the
compiler ecosystem. It is a CAPTURE/analysis dialect (distinct from the prescriptive ``schedule``
dialect of chosen compiler decisions), so it is deliberately NOT part of the lowering
``CORE_DIALECT_MODULES`` — callers load it explicitly via ``get_dialect()``.

``cca.kernel`` holds identity, scope and complete provenance. Its ``cca.facet`` children
preserve every analyzer facet and field as typed JSON in StringAttrs, including unknown values.
The serializer derives the facet universe from the CCA dataclass instead of a separate field map.
Legacy ``cca.compute``, ``cca.vector`` and ``cca.memory`` views remain readable and inspectable;
the reader cross-checks them against the complete record and refuses contradictions.
"""

from __future__ import annotations

from merlin.compiler.dialects._common import HAS_XDSL

DIALECT_NAME = "cca"
OPS = ["kernel", "compute", "vector", "memory", "facet"]
TYPES: list[str] = []

# The facet fields each op carries (also the serializer's contract). Kept here so the dialect and the
# serializer share ONE source of truth for the field set.
COMPUTE_FIELDS = (
    "contraction_form",
    "accumulator_dtype",
    "widening",
    "reduction_form",
    "register_block_mr",
    "epilogue",
    "accumulator_resident",
    "nr_is_vsetvlmax",
    "activation_vectorization",
)
VECTOR_FIELDS = ("sew", "lmul", "vl_strategy", "tail")
MEMORY_FIELDS = ("access_pattern", "panel_reuse", "a_broadcast_vf")

if HAS_XDSL:
    from xdsl.dialects.builtin import DictionaryAttr, StringAttr
    from xdsl.ir import Dialect
    from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_prop_def, prop_def, region_def, traits_def
    from xdsl.traits import NoTerminator

    @irdl_op_definition
    class ComputeOp(IRDLOperation):
        """cca.compute — the target-agnostic compute facet (one optional StringAttr per field)."""

        name = "cca.compute"
        contraction_form = opt_prop_def(StringAttr)
        accumulator_dtype = opt_prop_def(StringAttr)
        widening = opt_prop_def(StringAttr)
        reduction_form = opt_prop_def(StringAttr)
        register_block_mr = opt_prop_def(StringAttr)
        epilogue = opt_prop_def(StringAttr)
        accumulator_resident = opt_prop_def(StringAttr)
        nr_is_vsetvlmax = opt_prop_def(StringAttr)
        activation_vectorization = opt_prop_def(StringAttr)

    @irdl_op_definition
    class VectorOp(IRDLOperation):
        """cca.vector — the RVV/SIMD vector facet."""

        name = "cca.vector"
        sew = opt_prop_def(StringAttr)
        lmul = opt_prop_def(StringAttr)
        vl_strategy = opt_prop_def(StringAttr)
        tail = opt_prop_def(StringAttr)

    @irdl_op_definition
    class MemoryOp(IRDLOperation):
        """cca.memory — the data-movement / packing facet (the #1 expert GEMM lever)."""

        name = "cca.memory"
        access_pattern = opt_prop_def(StringAttr)
        panel_reuse = opt_prop_def(StringAttr)
        a_broadcast_vf = opt_prop_def(StringAttr)

    @irdl_op_definition
    class FacetOp(IRDLOperation):
        """Complete analyzer facet, including fields beyond the legacy views.

        Keys are the analyzer's actual dataclass fields; each value is JSON in
        a StringAttr so tuples, booleans, numbers and unknowns remain distinct.
        The analysis serializer validates kinds, fields and value types.
        """

        name = "cca.facet"
        facet = prop_def(StringAttr)
        values = prop_def(DictionaryAttr)

    @irdl_op_definition
    class KernelOp(IRDLOperation):
        """cca.kernel — a captured CCA: identity props + a region holding the facet ops."""

        name = "cca.kernel"
        op = opt_prop_def(StringAttr)
        backend = opt_prop_def(StringAttr)
        source = opt_prop_def(StringAttr)
        level = opt_prop_def(StringAttr)
        scope = opt_prop_def(StringAttr)
        provenance = opt_prop_def(StringAttr)
        backend_values = opt_prop_def(StringAttr)
        body = region_def()
        traits = traits_def(NoTerminator())

    _OP_CLASSES = [KernelOp, ComputeOp, VectorOp, MemoryOp, FacetOp]
    CCA_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, [])

    def get_dialect() -> Dialect:
        return CCA_DIALECT

else:  # pragma: no cover - xDSL absent
    ComputeOp = VectorOp = MemoryOp = FacetOp = KernelOp = None

    def get_dialect():
        return None
