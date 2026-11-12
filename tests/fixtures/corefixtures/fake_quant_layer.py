from __future__ import annotations

from merlin.compiler.dialects.lowering import compute_groups as CG


class Oracle:
    """An integer unit that takes elementwise stages only fused, at the granularities it holds.

    ``readout`` is the target's derived readout, asked only about what no capability row states:
    whether a unit's load can sum separately scaled operands.
    """

    def __init__(
        self,
        holds=("tensor",),
        families=("contraction", "elementwise_map"),
        readout=None,
        applies=("bias_add", "bias", "requant", "acc_scale", "relu", "maxpool"),
        standalone=(),
    ):
        self.holds, self.families, self.target, self.readout = (holds, families, "synthetic", readout)
        self.standalone = tuple(standalone)
        self.asked: list[dict] = []
        self.applies = applies

    def unit_for(self, *_args) -> str:
        return "unit0"

    def absorbs(self, kind: str):
        return CG.readout_absorbs(kind, self.declared_readout(), target=self.target)

    def declared_readout(self):
        """This synthetic target's ``readout_epilogue_capability``, as the facet a real one derives."""
        from merlin.development.targetgen import readout_facet as RF

        declared = ({"selector": "out", "applies": list(self.applies)},) if self.applies is not None else ()
        return RF.TargetReadout((RF.ReadoutFacet(target=self.target, readouts=declared),))

    def ask(self, *, op, family, in_dtype, weight_dtype=None, rank=None, attached, granularity=None, form=None):
        self.asked.append({"op": op, "family": family, "in_dtype": in_dtype, "attached": attached})
        if family not in self.families:
            return CG.Admission(False, "undeclared_family", f"no capability for {family}")
        if in_dtype != "int8":
            return CG.Admission(False, "input_dtype", f"{in_dtype} is not an integer format here")
        if family != "contraction" and family not in self.standalone and (not attached):
            return CG.Admission(False, "fused_only", "available only fused with a contraction")
        if granularity is not None and granularity not in self.holds:
            return CG.Admission(
                False, "scale_granularity", f"scales per {granularity}; the readout holds {list(self.holds)}"
            )
        return CG.Admission(True, units=("unit0",))
