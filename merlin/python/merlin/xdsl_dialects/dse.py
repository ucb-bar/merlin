"""The ``dse`` dialect (xDSL): interface candidates, variant runs, regimes.

Minimal and descriptive by design: it mirrors the ``interface_candidate``,
``dse_result``, and ``exploitability_report`` schemas as IR so candidates and measured
variant results can live next to the pipeline. It never participates in lowering.
(Created by explicit decision; ``kernel``/``search`` remain schemas-first.)
See docs/core_dialects.md and docs/dialects.md.
"""
from __future__ import annotations

from ._common import HAS_XDSL, Visibility

DIALECT_NAME = "dse"
OPS = ["candidate", "result", "regime_tag"]
TYPES = ["interface_candidate", "variant", "exploitability"]

if HAS_XDSL:
    from xdsl.ir import (Dialect, EnumAttribute, ParametrizedAttribute,
                         SpacedOpaqueSyntaxAttribute, TypeAttribute)
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_prop_def, prop_def, result_def)
    from xdsl.utils.exceptions import VerifyException
    from xdsl.utils.str_enum import StrEnum
    from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, IntegerAttr,
                                       StringAttr)

    # -- enums ---------------------------------------------------------------

    class Regime(StrEnum):
        """Exploitability buckets for regime_map/exploitability_report."""

        IRRELEVANT = "irrelevant"
        MARGINAL = "marginal"
        EXPLOITABLE = "exploitable"
        DOMINANT = "dominant"

    @irdl_attr_definition
    class VariantAttr(EnumAttribute[Visibility], SpacedOpaqueSyntaxAttribute):
        """#dse.variant — which visibility variant a result was measured under."""
        name = "dse.variant"

    @irdl_attr_definition
    class RegimeAttr(EnumAttribute[Regime], SpacedOpaqueSyntaxAttribute):
        """#dse.regime — exploitability regime assigned to a candidate."""
        name = "dse.regime"

    # -- types ---------------------------------------------------------------

    @irdl_attr_definition
    class InterfaceCandidateType(ParametrizedAttribute, TypeAttribute):
        """!dse.interface_candidate<"name"> — a proposed interface abstraction."""
        name = "dse.interface_candidate"
        candidate: StringAttr

    @irdl_attr_definition
    class VariantType(ParametrizedAttribute, TypeAttribute):
        """!dse.variant — a variant value (baseline/software_visible/...)."""
        name = "dse.variant_t"

    @irdl_attr_definition
    class ExploitabilityType(ParametrizedAttribute, TypeAttribute):
        """!dse.exploitability — an exploitability record."""
        name = "dse.exploitability"

    # -- ops -----------------------------------------------------------------

    @irdl_op_definition
    class CandidateOp(IRDLOperation):
        """dse.candidate — declares an interface candidate (interface_candidate.yaml)."""
        name = "dse.candidate"
        candidate_name = prop_def(StringAttr)
        interface_ops = prop_def(ArrayAttr)
        interface_types = opt_prop_def(ArrayAttr)
        justified_by = opt_prop_def(ArrayAttr)
        candidate = result_def(InterfaceCandidateType)

        def verify_(self) -> None:
            if not self.candidate_name.data:
                raise VerifyException("dse.candidate needs a name")
            if not self.interface_ops:
                raise VerifyException(
                    "dse.candidate must reference at least one interface op")
            for entry in self.interface_ops:
                ref = entry.data if isinstance(entry, StringAttr) else None
                if not ref or not ref.startswith("interface."):
                    raise VerifyException(
                        "dse.candidate interface_ops entries must be interface.* op "
                        "names, got %r" % (ref,))
            if self.candidate.type.candidate.data != self.candidate_name.data:
                raise VerifyException(
                    "dse.candidate result type %r does not match candidate name %r"
                    % (self.candidate.type.candidate.data, self.candidate_name.data))

    @irdl_op_definition
    class ResultOp(IRDLOperation):
        """dse.result — records one measured variant run (dse_result.yaml row)."""
        name = "dse.result"
        candidate = operand_def(InterfaceCandidateType)
        variant = prop_def(VariantAttr)
        workload = prop_def(StringAttr)
        backend = prop_def(StringAttr)
        metrics = prop_def(DictionaryAttr)

        def verify_(self) -> None:
            if not self.workload.data:
                raise VerifyException("dse.result needs a workload name")
            for key, val in self.metrics.data.items():
                if not isinstance(val, IntegerAttr):
                    raise VerifyException(
                        "dse.result metric %r must be an integer count/cycles" % key)

    @irdl_op_definition
    class RegimeTagOp(IRDLOperation):
        """dse.regime_tag — assigns an exploitability regime to a candidate."""
        name = "dse.regime_tag"
        candidate = operand_def(InterfaceCandidateType)
        regime = prop_def(RegimeAttr)
        reason = opt_prop_def(StringAttr)

    _OP_CLASSES = [CandidateOp, ResultOp, RegimeTagOp]
    _ATTR_CLASSES = [InterfaceCandidateType, VariantType, ExploitabilityType,
                     VariantAttr, RegimeAttr]
    DSE_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, _ATTR_CLASSES)

    def get_dialect() -> Dialect:
        return DSE_DIALECT

    def build_example():
        """A small, verifiable module: declare a candidate, record two variant runs."""
        from xdsl.ir import Block, Region
        from xdsl.dialects.builtin import FunctionType, ModuleOp
        from xdsl.dialects.func import FuncOp, ReturnOp

        blk = Block()
        cand = CandidateOp(
            result_types=[InterfaceCandidateType(StringAttr("resident_packed_tensor"))],
            properties={
                "candidate_name": StringAttr("resident_packed_tensor"),
                "interface_ops": ArrayAttr([StringAttr("interface.resident_pack"),
                                            StringAttr("interface.matmul"),
                                            StringAttr("interface.resident_evict")]),
                "interface_types": ArrayAttr([StringAttr("resident_tensor")]),
                "justified_by": ArrayAttr([StringAttr("repeated_rhs_matmul")]),
            })
        base = ResultOp(operands=[cand.candidate], properties={
            "variant": VariantAttr(Visibility.BASELINE),
            "workload": StringAttr("repeated_rhs_matmul"),
            "backend": StringAttr("simulator"),
            "metrics": DictionaryAttr({"cycles": IntegerAttr(11008, 64),
                                       "bytes_moved": IntegerAttr(73728, 64)})})
        sv = ResultOp(operands=[cand.candidate], properties={
            "variant": VariantAttr(Visibility.SOFTWARE_VISIBLE),
            "workload": StringAttr("repeated_rhs_matmul"),
            "backend": StringAttr("simulator"),
            "metrics": DictionaryAttr({"cycles": IntegerAttr(8256, 64),
                                       "bytes_moved": IntegerAttr(49152, 64)})})
        tag = RegimeTagOp(operands=[cand.candidate], properties={
            "regime": RegimeAttr(Regime.EXPLOITABLE),
            "reason": StringAttr("software_visible cuts bytes_moved 33% at reuse=4")})
        ret = ReturnOp()
        blk.add_ops([cand, base, sv, tag, ret])
        fn = FuncOp("dse_records", FunctionType.from_lists([], []), Region([blk]))
        return ModuleOp([fn])

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None

    def build_example():
        return None
