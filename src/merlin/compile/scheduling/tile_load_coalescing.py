"""Coalesce contiguous matrix columns into explicitly strided local tile rows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TileLoadCoalescing:
    rows: int
    columns: int
    tile: int
    max_columns: int
    local_base: int
    block_stride: int
    capacity_rows: int

    def __post_init__(self):
        if (any(type(v) is not int or not 0 < v < 1 << 63 for v in (
                self.rows, self.columns, self.tile, self.max_columns, self.block_stride, self.capacity_rows))
                or type(self.local_base) is not int or not 0 <= self.local_base < 1 << 63
                or self.rows > self.tile or self.max_columns % self.tile
                or self.block_stride < self.rows
                or self.local_base + (self.columns - 1) // self.tile * self.block_stride + self.rows
                > self.capacity_rows):
            raise ValueError("coalesced loads need complete bounded nonoverlapping local tile rows")

    def commands(self):
        """Yield bounded-width logical source start, width and local base."""
        for column in range(0, self.columns, self.max_columns):
            local = self.local_base + column // self.tile * self.block_stride
            yield column, min(self.max_columns, self.columns - column), local

    def location(self, row, column):
        if (any(type(v) is not int for v in (row, column))
                or not 0 <= row < self.rows or not 0 <= column < self.columns):
            raise ValueError("load element leaves its complete logical rectangle")
        return self.local_base + column // self.tile * self.block_stride + row, column % self.tile


@dataclass(frozen=True)
class TileLoadAddressLaw:
    """Exact row-to-local-tile addressing, without any completion premise.

    Source row pitch is bytes; destination block stride is local rows. These
    are different quantities. Numeric limits must describe decoded hardware,
    not unused ISA packet space. This declaration is conditional input to a
    proposal, not physical qualification or permission to remove completion.
    """

    tile_columns: int
    max_rows: int
    max_columns: int
    element_bytes: int
    source_pitch_bytes: int
    local_block_stride: int
    identity: str
    exact_copy: bool = True

    def __post_init__(self):
        if (any(type(v) is not int or not 0 < v < 1 << 63 for v in (
                self.tile_columns, self.max_rows, self.max_columns, self.element_bytes,
                self.source_pitch_bytes, self.local_block_stride))
                or self.max_columns < self.tile_columns
                or type(self.identity) is not str or not self.identity
                or self.exact_copy is not True):
            raise ValueError("tile load addressing requires explicit bounded exact mapping")


@dataclass(frozen=True)
class TileLoadLaw(TileLoadAddressLaw):
    """Original synchronous load law; addressing alone never admits this route."""

    synchronous: bool = True

    def __post_init__(self):
        try:
            super().__post_init__()
        except ValueError as error:
            raise ValueError("tile load rewriting requires explicit bounded exact synchronous mapping") from error
        if self.synchronous is not True:
            raise ValueError("tile load rewriting requires explicit bounded exact synchronous mapping")


@dataclass(frozen=True)
class TileLoadMapping:
    """Actual projected operands; source address uses bytes, local address rows.

    Lane ``c`` maps to local row ``local_row + floor(c/tile)*block_stride``
    and lane ``c % tile``. The target must independently establish that this
    describes ALL effects of the selected call, including numerical identity.
    """

    source: str
    destination: str
    source_byte: object
    local_row: object
    rows: object
    columns: object
    law: TileLoadAddressLaw

    def __post_init__(self):
        from merlin.sched.ir import Expr

        if (any(type(s) is not str or not s for s in (self.source, self.destination))
                or self.source == self.destination or type(self.law) not in (TileLoadLaw, TileLoadAddressLaw)
                or any(not isinstance(x, Expr) for x in (
                    self.source_byte, self.local_row, self.rows, self.columns))):
            raise ValueError("tile load mapping requires distinct stores and explicit index operands")


def _linear(expression):
    """Canonical affine coefficients; unsupported expressions keep their route."""
    from merlin.sched.ir import Const, Scaled, Sum, Var

    if type(expression) is Const:
        return expression.value, {}
    if type(expression) is Var:
        return 0, {expression.name: 1}
    if type(expression) is Scaled:
        constant, coefficients = _linear(expression.expr)
        return constant * expression.factor, {
            name: value * expression.factor for name, value in coefficients.items() if value * expression.factor
        }
    if type(expression) is Sum:
        constant, coefficients = 0, {}
        for part in expression.terms:
            c, terms = _linear(part)
            constant += c
            for name, value in terms.items():
                coefficients[name] = coefficients.get(name, 0) + value
        return constant, {name: value for name, value in coefficients.items() if value}
    raise ValueError("tile coalescing retains nonaffine operands")


def _substitute(expression, variable, replacement):
    from merlin.sched.ir import Const, Scaled, Select, Sum, Var, add, mul, select

    if type(expression) is Const:
        return expression
    if type(expression) is Var:
        return replacement if expression.name == variable else expression
    if type(expression) is Sum:
        return add(*(_substitute(x, variable, replacement) for x in expression.terms))
    if type(expression) is Scaled:
        return mul(_substitute(expression.expr, variable, replacement), expression.factor)
    if type(expression) is Select:
        if expression.var == variable:
            if type(replacement) is not Const:
                raise ValueError("tile coalescing retains variable tail selectors")
            return _substitute(expression.then if replacement.value == expression.equals else expression.other,
                               variable, replacement)
        return select(expression.var, expression.equals,
                      _substitute(expression.then, variable, replacement),
                      _substitute(expression.other, variable, replacement))
    raise ValueError("tile coalescing retains unknown operands")


def _replace_call(call, variable, replacement):
    from dataclasses import replace

    from merlin.sched.ir import Expr, Ptr

    def value(x):
        if isinstance(x, Expr):
            return _substitute(x, variable, replacement)
        if type(x) is Ptr:
            return replace(x, offset=_substitute(x.offset, variable, replacement))
        return x
    return replace(call, args=tuple((name, value(x)) for name, x in call.args))


def _plain(call):
    return not (call.unit or call.produces or call.awaits or call.stages or call.sets or call.assumes)


def _same_mapping(actual, expected):
    if type(actual) is not TileLoadMapping:
        return False
    try:
        return (actual.source == expected.source and actual.destination == expected.destination
                and actual.law == expected.law
                and all(getattr(actual, name) == getattr(expected, name)
                        or _linear(getattr(actual, name)) == _linear(getattr(expected, name))
                        for name in ("source_byte", "local_row", "rows", "columns")))
    except ValueError as error:
        raise ValueError("coalesced actual target operands have an unsupported mapping") from error


def coalesce_tile_load_loops(kernel, *, project, encode, max_tiles, max_static_nodes=4096):
    """Consume actual counted load loops and emit independently decoded wide calls.

    Only innermost loops containing one synchronous exact load are eligible.
    No load moves across another effect, configuration or dependency. Original
    compact/source checks must run first; this rewrite preserves their exact
    byte-to-local-cell mapping. Partial/nonaffine tiles retain the old route.
    ``project`` decodes actual operands; ``encode(first_call, columns)`` emits
    the consumer's real wide instruction, which is projected and checked again.
    The budget counts static nodes, never dynamic tile instances.
    """
    return _coalesce_tile_load_loops(
        kernel, project=project, encode=encode, max_tiles=max_tiles,
        max_static_nodes=max_static_nodes, law_type=TileLoadLaw,
    )


def propose_tile_load_coalescing(kernel, *, project, encode, max_tiles, max_static_nodes=4096):
    """Preserve exact addresses; a separate temporal consumer must admit issue.

    An address-only declaration cannot use the synchronous rewrite entrypoint.
    This proposal retains every non-load effect and completion boundary, but
    supplies no permission to issue an overlapping command. A consumer must
    decode the final calls, expand ALL bands and recheck their actual epochs.
    """
    return _coalesce_tile_load_loops(
        kernel, project=project, encode=encode, max_tiles=max_tiles,
        max_static_nodes=max_static_nodes, law_type=TileLoadAddressLaw,
    )


def _coalesce_tile_load_loops(kernel, *, project, encode, max_tiles, max_static_nodes, law_type):
    from dataclasses import replace

    from merlin.sched.ir import Call, Const, Kernel, Loop, Var
    from merlin.sched.ir.integer_domain import check_integer_domain
    from merlin.sched.ir.kernel import check_structure

    if (type(kernel) is not Kernel or check_structure(kernel)
            or not callable(project) or not callable(encode)
            or any(type(x) is not int or x <= 0 for x in (max_tiles, max_static_nodes))):
        raise ValueError("tile coalescing requires valid counted IR, actual codecs and finite bounds")
    check_integer_domain(kernel, 64)
    nodes, changes, refusals = 0, [], []

    def rewrite(loop):
        if loop.extent < 2 or len(loop.body) != 1 or type(loop.body[0]) is not Call:
            return None
        call = loop.body[0]
        if not _plain(call):
            return None
        mapping = project(call)
        if mapping is None:
            return None
        if type(mapping) is not TileLoadMapping:
            raise ValueError("tile load decoder returned untyped effects")
        if type(mapping.law) is not law_type:
            raise ValueError("tile load rewrite requires its distinct selected completion or address-only law")
        try:
            rows, columns = _linear(mapping.rows), _linear(mapping.columns)
            source, local = _linear(mapping.source_byte), _linear(mapping.local_row)
            law = mapping.law
            if (rows[1] or not 0 < rows[0] <= min(law.max_rows, law.local_block_stride)
                    or columns != (law.tile_columns, {})
                    or source[1].get(loop.var, 0) != law.tile_columns * law.element_bytes
                    or local[1].get(loop.var, 0) != law.local_block_stride):
                return None
            width = min(max_tiles, law.max_columns // law.tile_columns, loop.extent)
            if width < 2:
                return None
            trips, remainder = divmod(loop.extent, width)
            first = _replace_call(call, loop.var, Var(loop.var) * width)
            first_mapping = project(first)
            if type(first_mapping) is not TileLoadMapping:
                raise ValueError("substituted tile load lost its actual mapping")
            actual = encode(first, width * law.tile_columns)
            if type(actual) is not Call or not _plain(actual):
                raise ValueError("coalesced target encoder added unknown or asynchronous effects")
            expected = replace(first_mapping, columns=Const(width * law.tile_columns))
            if not _same_mapping(project(actual), expected):
                raise ValueError("coalesced actual target operands change source or local mapping")
            body = [Loop(loop.var, trips, (actual,))]
            if remainder:
                body.append(Loop(loop.var, remainder,
                                 (_replace_call(call, loop.var, Var(loop.var) + trips * width),)))
            changes.append(dict(variable=loop.var, original_trips=loop.extent,
                                tiles_per_command=width, full_commands=trips,
                                retained_tail_tiles=remainder, original_commands=loop.extent,
                                actual_commands=trips + remainder, law_identity=law.identity,
                                source_pitch_bytes=law.source_pitch_bytes,
                                local_block_stride=law.local_block_stride,
                                actual_access=dict(source=expected.source, destination=expected.destination,
                                    source_byte_affine=_linear(expected.source_byte),
                                    local_row_affine=_linear(expected.local_row), rows=rows[0],
                                    columns=width*law.tile_columns),
                                addressing_law=("local_row+floor(column/tile_columns)*block_stride+row; "
                                                "lane=column%tile_columns")))
            return tuple(body)
        except ValueError as error:
            # Unknown original expressions retain the old program. A malformed
            # actual emitted mapping is a compiler error, never a fallback.
            if str(error).startswith("tile coalescing retains"):
                refusals.append(str(error))
                return None
            raise

    def walk(body):
        nonlocal nodes
        result = []
        for statement in body:
            nodes += 1
            if nodes > max_static_nodes:
                raise ValueError("tile coalescing static work budget exceeded")
            if type(statement) is Loop:
                replacement = rewrite(statement)
                result.extend(replacement if replacement is not None else
                              (replace(statement, body=walk(statement.body)),))
            else:
                result.append(statement)
        return tuple(result)
    actual = replace(kernel, body=walk(kernel.body))
    if check_structure(actual):
        raise ValueError("coalesced target schedule is structurally invalid")
    check_integer_domain(actual, 64)
    return actual, dict(schema="merlin.counted_tile_load_coalescing.v1",
                        source_sha256=kernel.digest(), actual_sha256=actual.digest(),
                        changes=changes, refusals=refusals, static_nodes=nodes,
                        target_execution="not_run",
                        scope="exact actual load mapping; no asynchronous completion or target qualification grant")


def expand_tile_load_mapping(mapping):
    """Project an ACTUAL wide load into local tile-band accesses for hazards.

    This is an addressing expansion, never a reconstructed target command or
    asynchronous proof. Hazard consumers must project the final actual call,
    retain its real completion epoch and inspect every returned band.
    """
    from dataclasses import replace

    from merlin.sched.ir import Const

    if type(mapping) is not TileLoadMapping or type(mapping.columns) is not Const:
        raise ValueError("tile-band accesses need an actual typed constant-width mapping")
    law = mapping.law
    if not 0 < mapping.columns.value <= law.max_columns:
        raise ValueError("actual tile-band accesses exceed the selected column bound")
    return tuple(replace(mapping,
                         source_byte=mapping.source_byte + first * law.element_bytes,
                         local_row=mapping.local_row + first // law.tile_columns * law.local_block_stride,
                         columns=Const(min(law.tile_columns, mapping.columns.value - first)))
                 for first in range(0, mapping.columns.value, law.tile_columns))
