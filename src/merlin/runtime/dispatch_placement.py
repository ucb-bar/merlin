"""Honor selected outline placement without assigning hardware execution roles.

These are structural caller requirements, not an issuer of device capabilities,
command acceptance/completion, storage ownership or synchronization semantics.
"""

from dataclasses import replace

from merlin.xdsl_dialects.lowering.dispatch_program import _check_dispatch_definition
from merlin.xdsl_dialects.lowering.outline import OutlineError, _external_declarations


class DispatchRuntimeError(RuntimeError):
    pass


def _selection(row):
    if (
        type(row.index) is not int
        or type(row.symbol) is not str
        or not row.symbol
        or type(row.n_operands) is not int
        or type(row.result_types) is not list
        or any(type(value) is not str for value in row.result_types)
        or row.placement is not None
        and (type(row.placement) is not str or not row.placement.strip())
        or row.group is not None
        and (type(row.group) is not int or row.group < 0)
    ):
        raise DispatchRuntimeError("runtime dispatch has malformed original placement/ABI membership")
    return row.index, row.symbol, row.n_operands, tuple(row.result_types), row.placement, row.group


class DispatchPlacements:
    """Bind the original static calls/table and recheck the actual selected call.

    A repeated dynamic call uses its original static selection. A successful
    host dispatch may satisfy only an unspecified or explicitly host selection.
    The existing device runner's result supplies no hardware authority here.
    """

    def __init__(self, outlined, driver, *, host_only):
        self.outlined = outlined
        self.rows = {}
        try:
            declarations = _external_declarations(outlined.module, outlined.external_symbols)
            calls = [
                op for op in driver.walk() if op.name == "func.call" and op.callee.string_value() not in declarations
            ]
            if len(calls) != len(outlined.dispatches):
                raise DispatchRuntimeError("runtime dispatch table omitted or added original calls")
            for index, (call, row) in enumerate(zip(calls, outlined.dispatches, strict=True)):
                selected = _selection(row)
                if row.index != index or call.callee.string_value() != row.symbol:
                    raise DispatchRuntimeError("runtime dispatch table differs from original call order/symbol")
                # Copy the complete ordered result type slots; the caller's mutable
                # table cannot later remove the selected nonhost requirement.
                snapshot = replace(row, result_types=list(row.result_types))
                _check_dispatch_definition(outlined.module, call, snapshot)
                self.rows[call] = snapshot, selected
                if host_only:
                    self.require_host(call)
        except OutlineError as error:
            raise DispatchRuntimeError(str(error)) from error
        self.host_symbols = frozenset(row.symbol for row, _selected in self.rows.values() if row.placement == "host")

    def verify_call(self, call):
        if call not in self.rows:
            raise DispatchRuntimeError("runtime call has no original defined dispatch selection")
        row, selected = self.rows[call]
        if (
            row.index >= len(self.outlined.dispatches)
            or _selection(self.outlined.dispatches[row.index]) != selected
            or call.callee.string_value() != row.symbol
        ):
            raise DispatchRuntimeError("runtime original dispatch selection changed")
        try:
            _check_dispatch_definition(self.outlined.module, call, row)
        except OutlineError as error:
            raise DispatchRuntimeError(str(error)) from error
        return row.placement

    def is_device(self, call):
        return self.verify_call(call) not in (None, "host")

    def require_host(self, call):
        if self.is_device(call):
            row = self.rows[call][0]
            raise DispatchRuntimeError(
                f"dispatch {row.symbol!r} explicitly requires placement {row.placement!r}; "
                "host execution is unavailable"
            )
