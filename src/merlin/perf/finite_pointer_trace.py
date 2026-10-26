"""Bounded emitted-LLVM operand projection with symbolic ABI pointer identity.

No concrete addresses or tensor values are invented. This source-only observer
supports constant metadata control and byte-addressed pointer arithmetic. Calls,
poison, data-dependent control and exhausted bounds refuse the complete trace.
It is not a native execution, alias authority or numerical qualification.
"""

from dataclasses import dataclass

from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerType

from merlin.perf.constant_control_cfg import _poison_flags, integer_operation


@dataclass(frozen=True)
class ArgumentAddress:
    argument: int
    offset: int = 0


def _value(op, args, pointer_bits):
    if _poison_flags(op):
        raise ValueError("pointer projection refuses unproved poison assertions")
    if op.name == "llvm.getelementptr":
        # Only an actual one-dimensional byte displacement has a layout-free
        # interpretation. Other element layouts, assertion flags and indices
        # remain unknown; no object bounds or nonalias premise follows here.
        element = op.properties.get("elem_type")
        flags = op.properties.get("noWrapFlags")
        indices = op.properties.get("rawConstantIndices")
        if (not isinstance(element, IntegerType) or element.width.data != 8
                or flags is None or flags.value.data != 0 or indices is None
                or type(args[0]) is not ArgumentAddress):
            return None
        raw = tuple(indices.get_values())
        if len(raw) != 1:
            return None
        index = raw[0]
        if index == llvm.GEP_USE_SSA_VAL:
            if (len(args) != 2 or type(args[1]) is not int
                    or not isinstance(op.operands[1].type, IntegerType)
                    or op.operands[1].type.width.data != pointer_bits):
                return None
            index = args[1] % (1 << pointer_bits)
            if index >= 1 << (pointer_bits - 1):
                index -= 1 << pointer_bits
        elif len(args) != 1:
            return None
        return ArgumentAddress(args[0].argument, args[0].offset + index)
    if op.name == "llvm.ptrtoint":
        if (
            len(args) != 1
            or not isinstance(op.results[0].type, IntegerType)
            or op.results[0].type.width.data != pointer_bits
        ):
            raise ValueError("pointer projection requires the original exact pointer integer width")
        return args[0] if type(args[0]) is ArgumentAddress else None
    if op.name == "llvm.inttoptr":
        return args[0] if len(args) == 1 and type(args[0]) is ArgumentAddress else None
    if op.name == "llvm.select":
        if type(args[0]) is not int:
            return None
        return args[1] if args[0] else args[2]
    if op.name in ("llvm.add", "llvm.sub") and len(args) == 2:
        a, b = args
        if type(a) is ArgumentAddress and type(b) is int:
            return ArgumentAddress(a.argument, a.offset + (b if op.name == "llvm.add" else -b))
        if op.name == "llvm.add" and type(a) is int and type(b) is ArgumentAddress:
            return ArgumentAddress(b.argument, b.offset + a)
        if op.name == "llvm.sub" and type(a) is type(b) is ArgumentAddress and a.argument == b.argument:
            return a.offset - b.offset
    if any(type(a) is ArgumentAddress for a in args):
        return None
    return integer_operation(op, args)


def project_finite_pointer_trace(function, *, observe, pointer_bits, max_operations=200_000, max_blocks=20_000):
    """Observe the actual scalar operands at every visited original operation.

    A pointer is its original formal argument identity plus a signed byte
    displacement. Selected addresses still need original array-bound and
    nonalias checks. Unsupported pointer transforms stay unknown. No partial
    observation is returned when any later control or resource bound refuses.
    """
    if (
        function.name != "llvm.func"
        or not function.body.blocks
        or not callable(observe)
        or type(pointer_bits) is not int
        or not 1 <= pointer_bits <= 128
        or type(max_operations) is not int
        or not 1 <= max_operations <= 1_000_000
        or type(max_blocks) is not int
        or not 1 <= max_blocks <= 100_000
    ):
        raise ValueError("bounded original function, pointer width and observer required")
    function.verify()
    blocks = tuple(function.body.blocks)
    block_set = set(blocks)
    values = {
        arg: ArgumentAddress(i) if type(arg.type) is llvm.LLVMPointerType else None
        for i, arg in enumerate(blocks[0].args)
    }
    pending, operations, visits, observations = blocks[0], 0, 0, []
    while pending is not None:
        block = pending
        visits += 1
        if visits > max_blocks or block not in block_set:
            raise ValueError("finite pointer projection block budget exhausted or branch leaves function")
        pending = None
        for op in block.ops:
            operations += 1
            if operations > max_operations:
                raise ValueError("finite pointer projection operation budget exhausted")
            if op.regions or op.name in ("llvm.call", "llvm.invoke"):
                raise ValueError("finite pointer projection has opaque control/effects")
            args = tuple(values.get(value) for value in op.operands)
            observed = observe(op, args)
            if observed is not None:
                observations.append(observed)
            if op.name == "llvm.br":
                incoming = args
                pending = op.successors[0]
            elif op.name == "llvm.cond_br":
                if type(args[0]) is not int:
                    raise ValueError(
                        "finite pointer projection control depends on tensor data or unknown pointer facts"
                    )
                sizes = tuple(op.properties["operandSegmentSizes"].get_values())
                taken = bool(args[0])
                pending = op.successors[0 if taken else 1]
                incoming = args[1 : 1 + sizes[1]] if taken else args[1 + sizes[1] :]
            elif op.name == "llvm.return":
                continue
            elif len(op.results) == 1:
                values[op.results[0]] = _value(op, args, pointer_bits)
            else:
                values.update((v, None) for v in op.results)
        if block.last_op is None or block.last_op.name not in ("llvm.br", "llvm.cond_br", "llvm.return"):
            raise ValueError("finite pointer projection requires explicit original LLVM control")
        if pending is not None:
            if pending not in block_set or len(incoming) != len(pending.args):
                raise ValueError("finite pointer projection has malformed branch arguments")
            values.update(zip(pending.args, incoming, strict=True))
    return tuple(observations)
