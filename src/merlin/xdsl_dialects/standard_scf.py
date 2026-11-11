"""Preserve standard execute-region fields across xDSL custom syntax.

SCFOps.td/SCF.cpp at a47bddccec30255619bb8c37fa59700e661d4e66 own the
native syntax and no-inline rule. Ordinary unmarked single-block inlining
delegates to xDSL's existing pattern. No memory or arithmetic permission follows.
"""

from functools import cache

from xdsl.dialects.builtin import UnitAttr
from xdsl.dialects.scf import Scf, ExecuteRegionOp as XDSLExecuteRegionOp
from xdsl.ir import Dialect, OpTraits
from xdsl.irdl import (irdl_op_definition, opt_attr_def, region_def, traits_def, var_result_def)
from xdsl.pattern_rewriter import RewritePattern, op_type_rewrite_pattern
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.utils.exceptions import VerifyException


class ExecuteRegionInliner(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: XDSLExecuteRegionOp, rewriter):
        if "no_inline" in op.attributes:
            return
        from xdsl.transforms.canonicalization_patterns.scf import SingleBlockExecuteInliner
        SingleBlockExecuteInliner().match_and_rewrite(op, rewriter)


class ExecuteRegionPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls):
        return (ExecuteRegionInliner(),)


@irdl_op_definition
class ExecuteRegionOp(XDSLExecuteRegionOp):
    outs = var_result_def()
    region = region_def()
    no_inline = opt_attr_def(UnitAttr)
    traits = traits_def(ExecuteRegionPatterns())

    def __init__(self, result_types, region, attr_dict=None):
        super().__init__(result_types, region)
        self.attributes.update(attr_dict or {})

    @classmethod
    def parse(cls, parser):
        results = []
        if parser.parse_optional_punctuation("->"):
            results = parser.parse_optional_comma_separated_list(parser.Delimiter.PAREN, parser.parse_type)
            if results is None:
                results = [parser.parse_type()]
        no_inline = parser.parse_optional_keyword("no_inline") is not None
        region = parser.parse_region()
        attributes = parser.parse_optional_attr_dict()
        if no_inline:
            if "no_inline" in attributes:
                parser.raise_error("duplicate execute-region no_inline attribute")
            attributes["no_inline"] = UnitAttr()
        return cls(results, region, attributes)

    def print(self, printer):
        if self.outs:
            printer.print_string(" -> (")
            printer.print_list(self.outs.types, printer.print_attribute)
            printer.print_string(")")
        printer.print_string(" ")
        if self.no_inline is not None:
            printer.print_string("no_inline ")
        printer.print_region(self.region, print_entry_block_args=False, print_block_terminators=True)
        printer.print_op_attributes(self.attributes, reserved_attr_names=("no_inline",))

    def verify_(self):
        super().verify_()
        if self.region.first_block is not None and self.region.first_block.args:
            raise VerifyException("execute_region entry cannot have arguments")


# IRDL combines inherited traits. Replace only the inherited canonicalization
# behavior, whose unguarded inliner otherwise still runs beside our guarded one.
# Preserve the other inherited traits and keep the owned definition consistent.
_INHERITED_TRAITS = tuple(trait for trait in XDSLExecuteRegionOp.traits
    if not isinstance(trait, HasCanonicalizationPatternsTrait))
ExecuteRegionOp.traits = OpTraits(lambda: (*_INHERITED_TRAITS, ExecuteRegionPatterns()))
ExecuteRegionOp.get_irdl_definition().traits = ExecuteRegionOp.traits


@cache
def scf_dialect():
    return Dialect("scf", [ExecuteRegionOp if op.name == ExecuteRegionOp.name else op for op in Scf.operations],
        Scf.attributes)
