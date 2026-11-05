"""Bounded parser windows reconstruct the complete original typed SSA module.

No operation is summarized or discarded. Each window uses the original parser,
then exact SSA values are joined into the original function and the complete
module is verified. Unsupported framing/forward references refuse. This is a
private intake allocation strategy, never a numerical or publication grant.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping

from xdsl.dialects.func import FuncOp
from xdsl.ir import Block
from xdsl.utils.mlir_lexer import MLIRTokenKind

from merlin.common.dense_float_ir import DenseFloatParser
from merlin.common.ir_lock import IR_LOCK
from merlin.frontends.linalg_mlir import make_context, strip_paren_results

SCHEMA = "merlin.phase0.complete_typed_ssa_windows.v1"


class _HeaderParser(DenseFloatParser):
    def __init__(self, *args):
        super().__init__(*args)
        self.definitions = {}
        self.function_offsets = []

    def parse_operation(self):
        start = self._current_token.span.start
        operation = super().parse_operation()
        if type(operation) is FuncOp:
            self.function_offsets.append(start)
        return operation

    def _register_ssa_definition(self, name, values, span):
        super()._register_ssa_definition(name, values, span)
        self.definitions[name] = tuple(values)


class _Operands(Mapping):
    """Transactional typed proxies: failed parses cannot change original uses."""

    def __init__(self, actual):
        self.actual, self.proxies = actual, {}

    def __contains__(self, name):
        return name in self.actual

    def __getitem__(self, name):
        values = self.actual[name]
        if name not in self.proxies:
            self.proxies[name] = tuple(Block(arg_types=[v.type for v in values]).args)
        return self.proxies[name]

    def __iter__(self):
        return iter(self.actual)

    def __len__(self):
        return len(self.actual)

    def bind(self):
        for name, proxies in self.proxies.items():
            for proxy, actual in zip(proxies, self.actual[name], strict=True):
                proxy.replace_all_uses_with(actual)


def _function(module, *, external=False):
    operations = list(module.body.block.ops)
    if (len(module.body.blocks) != 1 or len(operations) != 1 or type(operations[0]) is not FuncOp
            or len(operations[0].body.blocks) != (0 if external else 1)):
        raise ValueError("windowed typed intake requires complete registered function framing")
    return operations[0]


def _header(context, raw, *, external=False):
    text = strip_paren_results(raw.decode("utf-8"))
    parser = _HeaderParser(context, text)
    module = parser.parse_module()
    function = _function(module, external=external)
    arguments = () if external else function.body.block.args
    if ((not external and function.body.block.ops) or len(parser.function_offsets) != 1
            or set(v for values in parser.definitions.values() for v in values) != set(arguments)):
        raise ValueError("windowed typed intake has unowned header SSA definitions or operations")
    return module, function, parser


def _discover(context, raw):
    # Original generic/custom grammars only. The registered parser establishes
    # every boundary; temporary suffixes never become source operations.
    if raw.rstrip().endswith((b"{", b":")):
        for suffix in (b"}\n}\n", b"}) : () -> ()\n}) : () -> ()\n"):
            try:
                return (*_header(context, raw + suffix), False)
            except Exception:
                continue
    for suffix in (b"}\n", b"}) : () -> ()\n"):
        try:
            return (*_header(context, raw + suffix, external=True), True)
        except Exception:
            continue
    return None


def _close_function(context, prefix, footer, aliases, resources):
    text = (prefix + footer).decode("utf-8")
    parser = _HeaderParser(context, text)
    parser.attribute_aliases = dict(aliases)
    parser.dialect_resources = resources
    parser._parser_state.dialect_stack.append("builtin")
    function = parser.parse_operation()
    if (type(function) is not FuncOp or len(function.body.blocks) != 1
            or function.body.block.ops or parser.forward_ssa_references or parser.forward_block_references):
        raise ValueError("windowed typed intake has an incomplete original function footer")
    consumed = len(text[:parser._current_token.span.start].encode("utf-8")) - len(prefix)
    if consumed < 1:
        raise ValueError("windowed typed intake did not consume the original function footer")
    return function, footer[consumed:]


def parse(stream, *, max_window_bytes, max_total_bytes, max_operations=200000):
    """Read bounded physical lines; parser success alone defines op boundaries.

    Failed candidates retain no uses on actual SSA values. Actual original
    operands/results, nested regions, aliases, attributes and all operations
    remain in the final AST; final whole-module verification is mandatory.
    """
    if (any(type(n) is not int or n < 1 for n in (max_window_bytes, max_total_bytes, max_operations))
            or max_total_bytes < max_window_bytes):
        raise ValueError("windowed typed intake needs finite parser and aggregate bounds")
    with IR_LOCK:
        return _parse(stream, max_window_bytes, max_total_bytes, max_operations)


def _parse(stream, window_limit, total_limit, operation_limit):
    context = make_context()
    total = operations = 0
    prefix, pending = bytearray(), bytearray()
    header = None

    def line():
        nonlocal total
        value = stream.readline(window_limit + 1)
        if not isinstance(value, bytes) or len(value) > window_limit:
            raise ValueError("windowed typed intake physical line exceeds parser ceiling")
        total += len(value)
        if total > total_limit:
            raise ValueError("windowed typed intake aggregate source exceeds its ceiling")
        return value

    while value := line():
        prefix.extend(value)
        if len(prefix) + 64 > window_limit:
            raise ValueError("windowed typed intake header exceeds parser ceiling")
        header = _discover(context, bytes(prefix))
        if header is not None:
            break
    if header is None:
        raise ValueError("windowed typed intake has no complete registered function header")
    _, _, first_parser, _ = header
    first_text = prefix.decode("utf-8")
    module_prefix = first_text[:first_parser.function_offsets[0]].encode("utf-8")
    functions = []
    module_footer = None
    function_prefix = bytes(prefix[len(module_prefix):])
    while header is not None:
        _, function, parser, external = header
        if external:
            functions.append(function)
        else:
            values = dict(parser.definitions)
            aliases, resources = parser.attribute_aliases, parser.dialect_resources
            block = function.body.block
            complete = False
            while True:
                if not pending:
                    value = line()
                    if not value:
                        break
                    pending.extend(value)
                if len(pending) + len(function_prefix) > window_limit:
                    raise ValueError("windowed typed intake operation exceeds parser ceiling")
                text = strip_paren_results(pending.decode("utf-8"))
                candidate = DenseFloatParser(context, text)
                candidate._parser_state.dialect_stack.extend(("builtin", "func"))
                if candidate._current_token.kind == MLIRTokenKind.EOF:
                    pending.clear()
                    continue
                if candidate._current_token.kind == MLIRTokenKind.R_BRACE:
                    try:
                        original, remainder = _close_function(
                            context, function_prefix, bytes(pending), aliases, resources)
                    except Exception:
                        value = line()
                        if not value:
                            break
                        pending.extend(value)
                        continue
                    original_block = original.body.block
                    for old, new in zip(block.args, original_block.args, strict=True):
                        old.replace_all_uses_with(new)
                    for op in list(block.ops):
                        op.detach()
                        original_block.add_op(op)
                    functions.append(original)
                    pending = bytearray(remainder)
                    complete = True
                    break
                operands = _Operands(values)
                candidate.ssa_values = ChainMap({}, operands)
                candidate.attribute_aliases = dict(aliases)
                candidate.dialect_resources = resources
                try:
                    op = candidate.parse_operation()
                    if (candidate._current_token.kind != MLIRTokenKind.EOF or candidate.forward_ssa_references
                            or candidate.forward_block_references or candidate.blocks):
                        raise ValueError("windowed typed intake has incomplete or forward SSA/block references")
                except Exception:
                    value = line()
                    if not value:
                        break
                    pending.extend(value)
                    continue
                operands.bind()
                block.add_op(op)
                values.update(candidate.ssa_values.maps[0])
                operations += sum(1 for _ in op.walk())
                if operations + len(functions) > operation_limit:
                    raise ValueError("windowed typed intake operation roster exceeds its ceiling")
                pending.clear()
            if not complete:
                raise ValueError("windowed typed intake has no complete original function footer")
        # The next complete source item is another registered function, or the
        # actual module footer. SSA definitions reset at every function boundary.
        header = None
        while True:
            if pending:
                probe = DenseFloatParser(context, pending.decode("utf-8"))
                if probe._current_token.kind == MLIRTokenKind.R_BRACE:
                    module_footer = bytearray(pending)
                    while value := line():
                        module_footer.extend(value)
                        if len(module_prefix) + len(module_footer) > window_limit:
                            raise ValueError("windowed typed intake module footer exceeds parser ceiling")
                    break
                header = _discover(context, module_prefix + bytes(pending))
                if header is not None:
                    function_prefix = bytes(pending)
                    pending.clear()
                    break
            value = line()
            if not value:
                break
            pending.extend(value)
            if len(module_prefix) + len(pending) + 64 > window_limit:
                raise ValueError("windowed typed intake function header exceeds parser ceiling")
    if module_footer is None or not functions or operations + len(functions) > operation_limit:
        raise ValueError("windowed typed intake has no complete original module footer")
    final_parser = DenseFloatParser(context, (module_prefix + bytes(module_footer)).decode("utf-8"))
    original = final_parser.parse_module()
    if not original.body.blocks:
        original.body.add_block(Block())
    if len(original.body.blocks) != 1 or original.body.block.ops:
        raise ValueError("windowed typed intake final module contains unconsumed source operations")
    for function in functions:
        if function.parent is not None:
            function.detach()
        original.body.block.add_op(function)
    original.verify()
    return original, dict(compact_bytes=total, operations=operations + len(functions), functions=len(functions))
