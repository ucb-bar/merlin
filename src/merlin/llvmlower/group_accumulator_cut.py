"""Preserve a proved host readout after a device accumulator-domain bias preload.

The output theorem permits removing selected additions inside a single-use host
chain, while the device returns the adjusted integer root. Intermediate floats
are not independently observed: the original quantize frontier stays in place.
"""

from dataclasses import dataclass

from merlin.xdsl_dialects.lowering import output_fusion as OF


@dataclass(frozen=True)
class AccumulatorCut:
    prefix: tuple
    bypasses: tuple

    def apply(self):
        """Remove only the checked additions, after the new integer root is wired."""
        from xdsl.ir import Operation

        feeding = []
        for op, operand_index in self.bypasses:
            feeding.extend(value.owner for value in op.operands if isinstance(value.owner, Operation))
            op.results[0].replace_all_uses_with(op.operands[operand_index])
            op.detach()
            op.erase()
        return feeding


def prepare(group, capture):
    """Plan a source-closed split without mutating it; unsupported shapes refuse."""
    fusion = group.fusion or {}
    if fusion.get("cut") != "accumulator":
        return None
    proof = fusion.get("proof") or {}
    if proof.get("readout") != "full_width" or set(proof.get("host_skips") or ()) != {"bias"}:
        raise ValueError("the accumulator cut has no supported source-bound bias-only host split")
    chain = OF.source_chain(group.root, OF.Values(capture))
    if chain.unread or not chain.stages or chain.stages[-1].kind != "quantize":
        raise ValueError("the accumulator cut no longer has its closed single-use quantize frontier")
    selected = {id(stage.op) for stage in chain.stages if stage.kind == "bias"}
    prefix = tuple(group.members[:group.members.index(group.root) + 1])
    if selected != {id(op) for op in group.members[len(prefix):]}:
        raise ValueError("the accumulator cut's additions differ from the proved host-skipped stages")
    bypasses = []
    data = group.root.results[0]
    for op in chain.ops:
        if id(op) in selected:
            slots = [i for i, operand in enumerate(op.operands) if operand is data]
            if len(slots) != 1 or len(op.results) != 1 or op.results[0].type != data.type:
                raise ValueError("a host-skipped addition cannot preserve its incoming tensor type")
            bypasses.append((op, slots[0]))
        if len(op.results) != 1 or op.parent is not group.root.parent:
            raise ValueError("the accumulator host readout is not a single-result chain in the root's block")
        data = op.results[0]
    if len(bypasses) != len(selected):
        raise ValueError("the accumulator host readout does not cover every proved addition")
    return AccumulatorCut(prefix, tuple(bypasses))
