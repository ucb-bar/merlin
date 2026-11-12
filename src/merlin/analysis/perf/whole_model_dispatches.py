"""Externalize an accelerator compute group as a host/device dispatch."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from merlin.compiler.ir.boundary import free_values

#: The symbol stems of the cut: the call the host code makes, and the group's own lowered body.
DISPATCH_PREFIX = "merlin_dispatch_g"
#: The argument attribute one-shot bufferization reads an external function's access to an operand from.
BUFFER_ACCESS = "bufferization.access"
HOST_PREFIX = "merlin_host_g"


class OpenModelError(RuntimeError):
    """The open model cannot be built as one program, and the message says which part stopped it."""


@dataclasses.dataclass(frozen=True)
class Dispatch:
    """One device group, cut out of the host code: its symbols and the values that cross the cut."""

    group: int
    dispatch_symbol: str
    host_symbol: str
    #: ``(shape, dtype)`` of each argument, in call order, and of the committed result.
    arguments: tuple[tuple[tuple[int, ...], str], ...]
    result: tuple[tuple[int, ...], str]
    #: Which argument is the group's root operand ``i`` (``None`` when a root operand is produced
    #: inside the cut), so a caller can say which argument is the activation and which the weight.
    root_operands: tuple[int | None, ...]
    region: str | None = None
    #: The argument the contraction accumulates INTO, and whether the IR states it zero (a splat or
    #: fill of a constant zero): only then is ``lhs @ rhs`` the dispatch's whole answer.
    init_argument: int | None = None
    zero_init: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "dispatch_symbol": self.dispatch_symbol,
            "host_symbol": self.host_symbol,
            "arguments": [{"shape": list(s), "dtype": d} for s, d in self.arguments],
            "result": {"shape": list(self.result[0]), "dtype": self.result[1]},
            "root_operands": list(self.root_operands),
            "region": self.region,
            "init_argument": self.init_argument,
            "zero_init": self.zero_init,
        }


def _shape_dtype(value) -> tuple[tuple[int, ...], str]:
    from merlin.common import mlir_query as mq

    shape, dtype = mq.type_shape_dtype(value.type)
    return tuple(int(e) for e in shape), str(dtype)


def _committed_members(group) -> list[Any]:
    """The members up to the value the device COMMITS: a trailing widening cast (an integer result put
    in a float box) stays on the host side of the cut, because the device commits the integers."""
    from merlin.compiler.lower.model.whole_program import _pure_retype

    members = list(group.members)
    retype = _pure_retype(group)
    if retype is not None and members and members[-1] is retype:
        members = members[:-1]
    return members


def externalize_dispatches(
    module, groups: Sequence[Any], *, function: str = "forward"
) -> tuple[Any, Any, list[Dispatch]]:
    """Cut each device group of ``groups`` out of ``module``'s ``function``. Mutates ``module``.

    Returns ``(module, host_module, dispatches)``: ``module`` now calls ``merlin_dispatch_g<i>`` (an
    external declaration with the C interface) where each group's committed value was computed, and
    ``host_module`` defines ``merlin_host_g<i>`` -- the group's own members, with the cheap producers
    they read cloned in -- so any dispatch can be answered by the group's own IR. A group whose
    members are read outside the cut (other than through its committed value) is refused by name:
    cutting it would drop a value the host code still reads.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, ModuleOp, StringAttr, UnitAttr
    from xdsl.ir import Block, Region

    from merlin.compiler.dialects.lowering import outline as OL

    functions = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    functions = [f for f in functions if f.sym_name.data == function] or functions[:1]
    if not functions:
        raise OpenModelError(f"the module has no function @{function} with a body")
    block = functions[0].body.blocks[0]
    order = {id(op): index for index, op in enumerate(block.ops)}
    host_functions: list[Any] = []
    dispatches: list[Dispatch] = []
    declarations: list[Any] = []
    for group in groups:
        members = sorted(_committed_members(group), key=lambda m: order.get(id(m), -1))
        if not members or any(id(m) not in order for m in members):
            raise OpenModelError(f"group {group.index} names an operation outside @{function}")
        committed = members[-1].results[0]
        inside = {id(r) for m in members for r in m.results}
        for m in members[:-1]:
            for result in m.results:
                for use in result.uses:
                    if id(use.operation) not in {id(x) for x in members} and use.operation.parent_block() is block:
                        raise OpenModelError(
                            f"group {group.index}: {m.name} is read outside the group, so cutting the group "
                            f"out would drop a value the host code still reads"
                        )
        closure = OL._group_closure(members)
        params = [p for p in free_values(closure) if id(p) not in inside]
        arg_types = [p.type for p in params]
        result_type = committed.type
        index = int(group.index)
        dispatch_symbol, host_symbol = f"{DISPATCH_PREFIX}{index}", f"{HOST_PREFIX}{index}"

        # The group's own body, as a function of the values that cross the cut.
        kblock = Block(arg_types=arg_types)
        mapping = dict(zip(params, kblock.args, strict=True))
        for op in closure:
            clone = op.clone(value_mapper=mapping)
            kblock.add_op(clone)
            for old, new in zip(op.results, clone.results, strict=True):
                mapping[old] = new
        kblock.add_op(func.ReturnOp(mapping[committed]))
        body = func.FuncOp(host_symbol, (arg_types, [result_type]), Region([kblock]))
        body.attributes["llvm.emit_c_interface"] = UnitAttr()
        host_functions.append(body)

        # EVERY ARGUMENT IS ONLY READ: a dispatch writes its result into a fresh buffer it returns, never
        # into an operand (and the group's own body, the host route, copies before any write). Stated, so
        # the bufferization does not copy each operand -- a weight, the zero accumulator -- ahead of every
        # call to protect it from a callee it would otherwise have to assume writes it.
        read = DictionaryAttr({BUFFER_ACCESS: StringAttr("read")})
        declaration = func.FuncOp(
            dispatch_symbol,
            (arg_types, [result_type]),
            Region(),
            visibility="private",
            arg_attrs=ArrayAttr([read for _ in arg_types]),
        )
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        declarations.append(declaration)

        call = func.CallOp(dispatch_symbol, list(params), [result_type])
        block.insert_op_before(call, members[-1])
        committed.replace_all_uses_with(call.results[0])
        for op in reversed(members):
            op.detach()
            op.erase()

        root_operands = []
        for operand in list(group.root.operands)[:2] if group.root is not None else ():
            root_operands.append(next((i for i, p in enumerate(params) if p is operand), None))
        inits = list(getattr(group.root, "outputs", ()) or ()) if group.root is not None else []
        init_argument = next((i for i, p in enumerate(params) if inits and p is inits[0]), None)
        region = group.root.attributes.get("prov.region_id") if group.root is not None else None
        dispatches.append(
            Dispatch(
                group=index,
                dispatch_symbol=dispatch_symbol,
                host_symbol=host_symbol,
                arguments=tuple(_shape_dtype(p) for p in params),
                result=_shape_dtype(committed),
                root_operands=tuple(root_operands),
                region=region.data if isinstance(region, StringAttr) else None,
                init_argument=init_argument,
                zero_init=bool(inits) and _is_zero(inits[0]),
            )
        )
    # AFTER the function: the whole-model lowering attaches its C interface to the first function
    # WITH A BODY, and a declaration placed first is only skipped by that step, never needed first.
    for declaration in declarations:
        module.body.block.add_op(declaration)
    host_module = ModuleOp(host_functions)
    return module, host_module, dispatches


def _is_zero(value) -> bool:
    """Whether ``value`` is a tensor the IR states is all zeros: a splat or a fill of a constant 0,
    through views. Anything else -- including a zero computed at run time -- is not known zero."""
    from merlin.compiler.dialects.lowering import compute_groups as CG

    for _ in range(8):
        owner = getattr(value, "owner", None)
        if owner is None:
            return False
        if owner.name in ("tensor.splat", "linalg.fill"):
            scalar = getattr(owner.operands[0], "owner", None)
            if scalar is None or scalar.name != "arith.constant":
                return False
            attr = scalar.properties.get("value")
            raw = getattr(getattr(attr, "value", None), "data", None)
            return raw is not None and float(raw) == 0.0
        if owner.name == "arith.constant":
            attr = owner.properties.get("value")
            values = list(attr.get_values()) if hasattr(attr, "get_values") else []
            return bool(values) and all(float(v) == 0.0 for v in values)
        stage = CG.classify(owner)
        if stage is None or stage.kind != CG.VIEW or not owner.operands:
            return False
        value = owner.operands[0]
    return False
