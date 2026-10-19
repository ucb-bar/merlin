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


def _function(module):
    operations = list(module.body.block.ops)
    if (len(module.body.blocks) != 1 or len(operations) != 1 or type(operations[0]) is not FuncOp
            or len(operations[0].body.blocks) != 1):
        raise ValueError("windowed typed intake requires one complete single-block function")
    return operations[0]


def _header(context, raw):
    parser = _HeaderParser(context, strip_paren_results(raw.decode("utf-8")))
    module = parser.parse_module()
    function = _function(module)
    if function.body.block.ops:
        raise ValueError("windowed typed intake header contains body operations")
    if set(v for values in parser.definitions.values() for v in values) != set(function.body.block.args):
        raise ValueError("windowed typed intake has unowned header SSA definitions")
    return module, function, parser


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
        if len(prefix) + 4 > window_limit:
            raise ValueError("windowed typed intake header exceeds parser ceiling")
        if not prefix.rstrip().endswith((b"{", b":")):
            continue
        # Both are original registered MLIR grammars. No source operations or
        # signatures are reprinted: only the temporary empty header is closed.
        for suffix in (b"}\n}\n", b"}) : () -> ()\n}) : () -> ()\n"):
            try:
                header = _header(context, bytes(prefix) + suffix)
            except Exception:
                continue
            break
        if header is not None:
            break
    if header is None:
        raise ValueError("windowed typed intake has no complete registered function header")
    module, function, parser = header
    values = dict(parser.definitions)
    aliases, resources = parser.attribute_aliases, parser.dialect_resources
    block = function.body.block
    footer = None
    while value := line():
        pending.extend(value)
        if len(pending) > window_limit:
            raise ValueError("windowed typed intake operation exceeds parser ceiling")
        text = strip_paren_results(pending.decode("utf-8"))
        candidate = DenseFloatParser(context, text)
        candidate._parser_state.dialect_stack.extend(("builtin", "func"))
        if candidate._current_token.kind == MLIRTokenKind.EOF:
            pending.clear()
            continue
        if candidate._current_token.kind == MLIRTokenKind.R_BRACE:
            footer = bytearray(pending)
            while value := line():
                footer.extend(value)
                if len(prefix) + len(footer) > window_limit:
                    raise ValueError("windowed typed intake footer exceeds parser ceiling")
            break
        operands = _Operands(values)
        candidate.ssa_values = ChainMap({}, operands)
        candidate.attribute_aliases = dict(aliases)
        candidate.dialect_resources = resources
        try:
            op = candidate.parse_operation()
            if (candidate._current_token.kind != MLIRTokenKind.EOF or candidate.forward_ssa_references
                    or candidate.forward_block_references or candidate.blocks):
                raise ValueError("windowed typed intake has incomplete or forward SSA references")
        except Exception:
            continue
        operands.bind()
        block.add_op(op)
        values.update(candidate.ssa_values.maps[0])
        operations += sum(1 for _ in op.walk())
        if operations > operation_limit:
            raise ValueError("windowed typed intake operation roster exceeds its ceiling")
        pending.clear()
    if footer is None or not operations:
        raise ValueError("windowed typed intake has no complete original function footer")
    original, original_function, _ = _header(context, bytes(prefix) + bytes(footer))
    original_block = original_function.body.block
    for old, new in zip(block.args, original_block.args, strict=True):
        old.replace_all_uses_with(new)
    for op in list(block.ops):
        op.detach()
        original_block.add_op(op)
    original.verify()
    return original, dict(compact_bytes=total, operations=operations)
