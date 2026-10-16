"""Original full-coverage placement over typed source compute groups."""

from __future__ import annotations

from merlin.common import mlir_query as mq
from merlin.llvmlower.device_build import DeviceRouting, _accum_from_facts
from merlin.llvmlower.source_compute_groups import form_source_groups
from merlin.xdsl_dialects.lowering import compute_groups as CG


def select_complete_group(_shape):
    """Every source-admitted group must be requested; legality remains source-owned."""
    return True


def routing(target, source, package, *, capture, model, linked_elf_admission):
    module = mq.parse(source)
    groups = form_source_groups(module, target)
    admitted = [group for group in groups if group.placement != CG.HOST]
    if not admitted:
        raise ValueError("complete source placement has no admitted accelerator groups")
    pairs = set()
    for group in admitted:
        if group.weight_dtype is not None:
            pair = group.in_dtype, group.weight_dtype
        elif group.operand_sum is not None or group.window_mean is not None:
            # These closed source forms explicitly restate same-format integer
            # inputs or a same-format generated unit-weight reduction.
            pair = group.in_dtype, group.in_dtype
        elif group.root is not None and len(group.root.operands) >= 2:
            pair = tuple(CG._dtype_token(mq.type_shape_dtype(value.type)[1])  # noqa: PLC2701 -- source format owner
                         for value in list(group.root.operands)[:2])
            if pair[0] != group.in_dtype:
                raise ValueError("complete source placement input type differs from its root")
        else:
            raise ValueError("complete source placement has no proven operand pair")
        pairs.add(pair)
    if len(pairs) != 1 or any(not isinstance(value, str) or not value for pair in pairs for value in pair):
        raise ValueError("complete source placement needs separately supported typed datapaths")
    operand, weight = pairs.pop()
    return DeviceRouting(
        device=target, package_dir=package, operand_dtype=operand,
        accum_dtype=_accum_from_facts(target, operand, weight), select=select_complete_group,
        granularity="group", capture=capture, model=model, linked_elf_admission=linked_elf_admission,
        require_complete_groups=True,
    )
