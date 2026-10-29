"""Check final wide load bands within the original explicit panel epochs.

The addressing proposal supplies no synchronous completion fiction. The target
must decode the final command's actual source pitch, local block stride, width
and row count. Every band is then checked by the existing source/lifetime and
panel noninterference verifiers. Emission retains the wide command, never its
checking projection; original full-device drain premises remain mandatory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import prod

from merlin.compile.scheduling.compact_dense import _digest
from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadAddressLaw,
    TileLoadMapping,
    _plain,
    _same_mapping,
    _substitute,
    expand_tile_load_mapping,
)
from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Ptr, Scaled, Select, Sum, Var, check_structure
from merlin.sched.ir.expr import add, free_vars
from merlin.sched.ir.integer_domain import check_integer_domain


def _offset_form(value):
    """Exact additive form, retaining equality-select trees as opaque atoms."""
    if type(value) is Const:
        return value.value, {}
    if type(value) in (Var, Select):
        return 0, {value: 1}
    if type(value) is Scaled:
        constant, terms = _offset_form(value.expr)
        return constant * value.factor, {term: coefficient * value.factor for term, coefficient in terms.items()}
    if type(value) is Sum:
        constant, terms = 0, {}
        for item in value.terms:
            c, ts = _offset_form(item)
            constant += c
            for term, coefficient in ts.items():
                terms[term] = terms.get(term, 0) + coefficient
        return constant, {term: coefficient for term, coefficient in terms.items() if coefficient}
    raise ValueError("unsupported counted band expression")


def _map_call(call, transform):
    def value(item):
        if isinstance(item, Expr):
            return transform(item)
        if type(item) is Ptr:
            return replace(item, offset=transform(item.offset))
        return item
    return replace(call, args=tuple((key, value(item)) for key, item in call.args))


def _interpolate_calls(first, second, index):
    """Construct a candidate; the independent decoder checks every symbolic band."""
    if (replace(first, args=()) != replace(second, args=())
            or tuple(k for k, _ in first.args) != tuple(k for k, _ in second.args)):
        raise ValueError("counted band codec changes non-address fields")

    def value(a, b):
        if a == b:
            return a
        if isinstance(a, Expr) and isinstance(b, Expr):
            ac, av = _offset_form(a)
            bc, bv = _offset_form(b)
            if av == bv:
                return a + index * (bc - ac)
        if type(a) is Ptr and type(b) is Ptr and a.tensor == b.tensor:
            return replace(a, offset=value(a.offset, b.offset))
        raise ValueError("counted band codec is not an affine constant stride")

    return replace(first, args=tuple((key, value(a, b))
                                     for (key, a), (_, b) in zip(first.args, second.args, strict=True)))


def _same_band_call(a, b):
    """Complete call equality modulo exact additive expression normalization."""
    if (replace(a, args=()) != replace(b, args=())
            or tuple(k for k, _ in a.args) != tuple(k for k, _ in b.args)):
        return False

    def same(x, y):
        if x == y:
            return True
        if isinstance(x, Expr) and isinstance(y, Expr):
            return _offset_form(x) == _offset_form(y)
        return type(x) is Ptr and type(y) is Ptr and x.tensor == y.tensor and same(x.offset, y.offset)

    return all(same(x, y) for (_, x), (_, y) in zip(a.args, b.args, strict=True))


def _canonical_load_runs(kernel, project):
    """Exactly flatten affine mixed-radix load nests and join split runs.

    Only a one-load body is considered. No completion, compute, configuration,
    or other effect moves. For every operand, the two indices must occur ONLY
    as outer*inner_extent+inner; that map is a lexicographic-order bijection.
    Sequential runs join only if their complete calls match after an exact
    index shift. Unknown expressions retain their original checked structure.
    """
    changes = []

    def load_loop(statement):
        return (type(statement) is Loop and len(statement.body) == 1
                and type(statement.body[0]) is Call and _plain(statement.body[0])
                and project(statement.body[0]) is not None)

    def collapse(outer, inner):
        def expression(value):
            constant, terms = _offset_form(value)
            a, b = terms.pop(Var(outer.var), 0), terms.pop(Var(inner.var), 0)
            if a != b * inner.extent or any({outer.var, inner.var} & free_vars(atom) for atom in terms):
                raise ValueError("operand distinguishes mixed-radix coordinates")
            return add(Const(constant), *(atom * coefficient for atom, coefficient in terms.items()),
                       Var(outer.var) * b)
        return replace(outer, extent=outer.extent * inner.extent,
                       body=(_map_call(inner.body[0], expression),))

    def shifted(call, variable, name, offset):
        return _map_call(call, lambda value: _substitute(value, variable, Var(name) + offset))

    def walk(body):
        result = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=walk(statement.body))
                if len(statement.body) == 1 and load_loop(statement.body[0]):
                    try:
                        collapsed = collapse(statement, statement.body[0])
                    except ValueError:
                        pass
                    else:
                        changes.append(dict(kind="mixed_radix_load_run", outer=statement.var,
                                            inner=statement.body[0].var, extent=collapsed.extent))
                        statement = collapsed
            if result and load_loop(result[-1]) and load_loop(statement):
                previous = result[-1]
                try:
                    continuation = shifted(previous.body[0], previous.var, previous.var, previous.extent)
                    following = shifted(statement.body[0], statement.var, previous.var, 0)
                    matches = _same_band_call(continuation, following)
                except ValueError:
                    matches = False
                if matches:
                    changes.append(dict(kind="contiguous_split_load_run", variable=previous.var,
                                        extents=[previous.extent, statement.extent]))
                    result[-1] = replace(previous, extent=previous.extent + statement.extent)
                    continue
            result.append(statement)
        return tuple(result)

    return replace(kernel, body=walk(kernel.body)), changes


def expand_panel_tile_loads(kernel, *, project, encode_band, max_static_nodes=4096, counted_bands=True):
    """Decode actual calls into all checked bands, retaining original epochs.

    ``encode_band`` is only a checking representation. Its actual operands must
    decode back to exactly the expanded source byte and local band. Neither a
    retained original request nor the first band alone establishes this join.
    Unknown non-load calls remain present for the ordinary panel verifier.
    """
    if (type(kernel) is not Kernel or check_structure(kernel)
            or not callable(project) or not callable(encode_band)
            or type(max_static_nodes) is not int or max_static_nodes <= 0 or type(counted_bands) is not bool):
        raise ValueError("panel bands require actual counted commands, codecs and a finite bound")
    check_integer_domain(kernel, 64)
    nodes, accesses = 0, []
    physical, logical = 0, 0
    variables = set()

    def names(body):
        for statement in body:
            if type(statement) is Loop:
                variables.add(statement.var)
                names(statement.body)
    names(kernel.body)

    def bounded_choice(expression, limit):
        if type(expression) is Const:
            return 0 < expression.value <= limit
        return (type(expression) is Select and bounded_choice(expression.then, limit)
                and bounded_choice(expression.other, limit))

    def walk(body, scopes=()):
        nonlocal nodes, physical, logical
        result = []
        for statement in body:
            nodes += 1
            if nodes > max_static_nodes:
                raise ValueError("panel band projection exceeded its static bound")
            if type(statement) is Loop:
                result.append(replace(statement, body=walk(statement.body, (*scopes, statement.extent))))
                continue
            mapping = project(statement)
            if mapping is None:
                result.append(statement)
                continue
            if (type(statement) is not Call or not _plain(statement)
                    or type(mapping) is not TileLoadMapping or type(mapping.law) is not TileLoadAddressLaw
                    or not bounded_choice(mapping.rows, min(mapping.law.max_rows, mapping.law.local_block_stride))):
                raise ValueError("panel load needs a plain exact address-only mapping with bounded rows")
            if type(mapping.columns) is Const:
                count = (mapping.columns.value + mapping.law.tile_columns - 1) // mapping.law.tile_columns
                if count > max_static_nodes - nodes + 1:
                    raise ValueError("panel band projection exceeded its static bound")
                bands = expand_tile_load_mapping(mapping)
            elif bounded_choice(mapping.columns, mapping.law.tile_columns):
                # Every possible tail fits the original one-band primitive.
                # Wider dynamic widths are refused, never truncated to band0.
                bands = tuple(replace(b, columns=mapping.columns) for b in expand_tile_load_mapping(
                    replace(mapping, columns=Const(mapping.law.tile_columns))))
            else:
                raise ValueError("actual wide panel width must be constant; only one-band tails may select width")
            nodes += len(bands) - 1
            physical += prod(scopes)
            logical += prod(scopes) * len(bands)
            checked_bands = []
            for band in bands:
                checked = encode_band(statement, band)
                if type(checked) is not Call or not _plain(checked) or not _same_mapping(project(checked), band):
                    raise ValueError("actual checking band changes the final wide command mapping")
                checked_bands.append(checked)
            mode = "static_bands"
            if counted_bands and len(bands) > 1:
                whole = mapping.columns.value // mapping.law.tile_columns
                if whole >= 2:
                    variable = "panel_wide_band_" + str(len(accesses))
                    while variable in variables:
                        variable += "_"
                    variables.add(variable)
                    try:
                        candidate = _interpolate_calls(checked_bands[0], checked_bands[1], Var(variable))
                        expected = replace(mapping,
                            source_byte=mapping.source_byte + Var(variable) * (
                                mapping.law.tile_columns * mapping.law.element_bytes),
                            local_row=mapping.local_row + Var(variable) * mapping.law.local_block_stride,
                            columns=Const(mapping.law.tile_columns))
                        exact = _same_mapping(project(candidate), expected) and all(
                            _same_band_call(_map_call(candidate, lambda value: _substitute(
                                value, variable, Const(index))), checked_bands[index])
                            for index in range(whole))
                    except ValueError:
                        exact = False
                    if exact:
                        result.append(Loop(variable, whole, (candidate,)))
                        if whole < len(bands):
                            result.append(checked_bands[-1])
                        mode = "counted_affine_bands"
                    else:
                        result.extend(checked_bands)
                else:
                    result.extend(checked_bands)
            else:
                result.extend(checked_bands)
            def record(value):
                return dict(source=value.source, destination=value.destination, law=asdict(value.law),
                            **{key: str(getattr(value, key)) for key in
                               ("source_byte", "local_row", "rows", "columns")})

            accesses.append(dict(call=str(statement), mapping=record(mapping), bands=[record(b) for b in bands],
                                 representation=mode))
        return tuple(result)

    expanded = replace(kernel, body=walk(kernel.body))
    canonical = []
    if counted_bands:
        expanded, canonical = _canonical_load_runs(expanded, project)
    if check_structure(expanded):
        raise ValueError("panel band projection changed valid completion scopes")
    check_integer_domain(expanded, 64)
    return expanded, dict(
        schema="merlin.panel_tile_load_projection.v1",
        actual_kernel_sha256=kernel.digest(),
        expanded_kernel_sha256=expanded.digest(),
        actual_mapping_sha256=_digest(accesses),
        static_load_sites=len(accesses),
        physical_load_commands=physical,
        expanded_logical_tile_bands=logical,
        actual_accesses=accesses,
        counted_band_projection=counted_bands,
        canonical_load_runs=canonical,
        completion="original explicit drained epochs retained; no synchronous load law supplied",
    )


@dataclass(frozen=True)
class PanelTileLoadResult:
    kernel: Kernel
    original: object
    realization: dict
    checked_kernel_sha256: str
    checked_projection_sha256: str
    project: object
    encode_band: object
    max_static_nodes: int

    @property
    def admission(self):
        """Retain the selected execution premises across address-only passes."""
        return self.original.admission

    def revalidate(self):
        self.original.revalidate()
        _, projection = expand_panel_tile_loads(
            self.kernel, project=self.project, encode_band=self.encode_band, max_static_nodes=self.max_static_nodes
        )
        if self.kernel.digest() != self.checked_kernel_sha256 or _digest(projection) != self.checked_projection_sha256:
            raise ValueError("actual wide panel commands or their decoded mapping changed after checking")

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands

        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        self.revalidate()
        return module


def admit_panel_tile_loads(
    original, proposal, problem, block, resources, storage, adapter, *,
    project, encode_band, static_budget=128, timeout_ms=5000, max_static_nodes=4096,
    external_sources=None,
):
    """Check the actual final wide command's complete effects before emission.

    Address mapping is an independent target premise. Original overlap admission,
    source custody, accumulator semantics and epoch drains remain prerequisites.
    The checking expansion counts logical bands; it never claims they are extra
    physical queue entries. Finite issued-load bounds are conservatively checked
    using those bands, so no capacity or proof bound is enlarged here.
    """
    from merlin.compile.scheduling.dense_panel_pipeline import (
        DensePanelPipelineResult,
        DenseRetainedReductionAdmission,
        verify_dense_panel_pipeline,
        verify_dense_retained_reduction,
    )

    if type(original) is not DensePanelPipelineResult:
        raise ValueError("wide panel admission requires the original checked panel result")
    verifier = (verify_dense_retained_reduction if type(original.admission) is DenseRetainedReductionAdmission
                else verify_dense_panel_pipeline)
    return _admit_dense_tile_loads(
        original, proposal, problem, block, resources, storage, adapter,
        verifier=verifier, project=project, encode_band=encode_band,
        static_budget=static_budget, timeout_ms=timeout_ms, max_static_nodes=max_static_nodes,
        external_sources=external_sources,
    )


def _admit_dense_tile_loads(
    original, proposal, problem, block, resources, storage, adapter, *,
    verifier, project, encode_band, static_budget, timeout_ms, max_static_nodes,
    external_sources=None,
):
    """Shared exact address projection; each typed owner selects its verifier.

    This helper grants no execution capability. Public consumers must require
    their own checked result type and preserve its original admission unchanged.
    """
    original.revalidate()
    expanded, projection = expand_dense_panel_tile_loads(
        proposal, problem, storage, adapter, project=project, encode_band=encode_band,
        max_static_nodes=max_static_nodes, external_sources=external_sources,
    )
    report = verifier(
        expanded, problem, block, resources, storage, adapter, original.admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=original.output_readout, bias_source=original.bias_source,
    )
    original.revalidate()
    report["tile_load_projection"] = projection
    report["schedule_ir"] = original.realization["schedule_ir"]
    report["kernel_sha256"] = proposal.digest()
    report["checking_expansion_sha256"] = expanded.digest()
    report["qualification"] = "original panel execution premises plus independently selected exact wide addressing"
    report["original_boundary"] = dict(
        output_readout_identity=None if original.output_readout is None else original.output_readout.identity,
        bias_source=None if original.bias_source is None else asdict(original.bias_source),
        readout_qualification_identity=original.readout_qualification_identity,
        qualification="retained original boundary premise; no new target readout qualification",
    )
    result = PanelTileLoadResult(
        proposal, original, report, proposal.digest(), _digest(projection),
        _checked_dense_tile_project(problem, storage, adapter, project, external_sources=external_sources),
        encode_band, max_static_nodes
    )
    result.revalidate()
    return result


def _checked_dense_tile_project(problem, storage, adapter, project, *, external_sources=None):
    from merlin.compile.scheduling.block_schedule import Load
    from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
    from merlin.compile.scheduling.panel_transfers import AffineMatrixSource

    if external_sources is None:
        sources = {}
    elif type(external_sources) is not dict or set(external_sources) - {"lhs", "weight"}:
        raise ValueError("dense external sources must name original operand roles")
    else:
        sources = dict(external_sources)
    for role, source in sources.items():
        left = role == "lhs"
        expected_shape = (problem.m, problem.k) if left else (problem.k, problem.n)
        width = problem.lhs_bytes if left else problem.rhs_bytes
        if (type(source) is not AffineMatrixSource or source.shape != expected_shape
                or source.element_bytes != width or source.column_stride_bytes != width):
            raise ValueError("dense external view must preserve original shape, width and adjacent columns")
        source.__post_init__()

    def checked_project(call):
        mapping = project(call)
        if mapping is None:
            if call.instr != DRAIN and type(adapter.project(call)) is Load:
                raise ValueError("every actual panel load requires its complete decoded tile mapping")
            return None
        if type(mapping) is not TileLoadMapping:
            raise ValueError("actual panel decoder returned untyped load effects")
        operation = adapter.project(call)
        if type(operation) is not Load or operation.role not in ("lhs", "weight"):
            raise ValueError("actual tile mapping has no original dense source load")
        left = operation.role == "lhs"
        width = problem.lhs_bytes if left else problem.rhs_bytes
        pitch = problem.k if left else problem.n
        source = sources.get(operation.role)
        source_pitch = pitch * width if source is None else source.row_stride_bytes
        tile = problem.tile_k if left else problem.tile_n
        view = storage.lhs if left else storage.rhs
        if (mapping.law.element_bytes != width or mapping.law.source_pitch_bytes != source_pitch
                or mapping.law.tile_columns != tile or operation.step != Const(1)):
            raise ValueError("wide mapping differs from original contiguous dense source layout")
        expected = TileLoadMapping(
            operation.role if source is None else source.backing, view.backing,
            ((operation.dram_row * pitch + operation.dram_col) * width if source is None else
             source.first_byte + operation.dram_row * source.row_stride_bytes + operation.dram_col * width),
            operation.row, operation.rows, operation.cols, mapping.law,
        )
        if not _same_mapping(mapping, expected):
            raise ValueError("actual wide mapping differs from original decoded source/local operands")
        return mapping

    return checked_project


def expand_dense_panel_tile_loads(
    kernel, problem, storage, adapter, *, project, encode_band, max_static_nodes=4096,
    external_sources=None,
):
    """Expand every final wide band after checking its original dense layout.

    This is address projection only. Callers must independently check the
    complete expanded source and execution effects under their own typed law.
    Optional existing ``AffineMatrixSource`` declarations bind exact external
    strides/offsets; their source ownership and target effects remain caller
    obligations. Omitting them retains the original contiguous layout check.
    No target capability or numerical/completion obligation is supplied here.
    """
    return expand_panel_tile_loads(
        kernel, project=_checked_dense_tile_project(problem, storage, adapter, project,
                                                   external_sources=external_sources),
        encode_band=encode_band, max_static_nodes=max_static_nodes,
    )
