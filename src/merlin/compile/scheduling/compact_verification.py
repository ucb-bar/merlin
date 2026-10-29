"""Symbolic checks over projected *actual* counted-loop dense instructions.

Coverage is proved by bounded domains, exact cardinality and injectivity.
Residency is a latest-write condition over physical byte intervals and program
order. No loop is expanded and no tensor is allocated by this checker.
Completion is synchronous in this family; asynchronous adapters must not use it.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import product
from math import prod
from threading import Lock

from merlin.sched.ir import Const, Loop, Scaled, Select, Sum, Var, evaluate

from .block_schedule import Compute, InitializeAccumulator, Load, Preload, Store


class DenseVerificationUnavailable(ValueError):
    """The selected checking tool or budget could not establish the result."""


_SOURCE_PROOFS = OrderedDict()
_SOURCE_PROOF_LOCK = Lock()
_SOURCE_PROOF_LIMIT = 64


def _source_proof_key(sites, kernel, problem, resources, storage, adapter, output_readout, bias_source):
    """Exact typed projected program, ignoring only vacuous scope numbering.

    The ordered sites and their complete counted scope paths reconstruct the
    program. Singleton scopes were already substituted by _project. Actual
    Python types remain in the key: unrelated objects with similar fields do
    not borrow a proof. No caller digest or previous timing score is trusted.
    """
    def typed(value):
        if value is None or type(value) in (str, int, bool):
            return type(value), value
        if type(value) is tuple:
            return tuple, tuple(typed(item) for item in value)
        if is_dataclass(value):
            return type(value), tuple((field.name, typed(getattr(value, field.name))) for field in fields(value))
        raise TypeError("unsupported source-proof cache value")

    scopes = {}
    program = []
    for site in sites:
        path = tuple((scopes.setdefault(scope, len(scopes)), name, extent) for scope, name, extent in site.scopes)
        program.append((path, typed(site.operation)))
    return (
        typed(kernel.args), typed(problem), typed(resources), typed(storage),
        adapter.configuration_identity, adapter.numerical_policy_identity, adapter.index_bits,
        adapter.synchronous_completion, adapter.exact_accumulator_continuation,
        None if output_readout is None else (type(output_readout), output_readout.identity),
        typed(bias_source), tuple(program),
    )


def _defined_exists(variables, claim, *, instances=()):
    """Exactly eliminate integer witnesses defined by conjunctive equations.

    A writer coordinate often obeys ``a*w + rest == 0``. Replacing its
    existential with ``rest % abs(a) == 0`` and the corresponding quotient
    preserves the complete claim, including every nested overwrite condition.
    Return the complete formula and an optional constructive witness formula.
    Unknown/nonlinear definitions retain their original quantifier. No equation
    under a disjunction or a quantifier is assumed.
    """
    import z3

    occurrences, affine_parts = {}, {}

    def contains(expression, variable):
        key = expression, variable
        if key not in occurrences:
            occurrences[key] = z3.eq(expression, variable) or any(
                contains(child, variable) for child in expression.children()
            )
        return occurrences[key]

    def affine(expression, variable):
        key = expression, variable
        if key not in affine_parts:
            affine_parts[key] = linear_part(expression, variable)
        return affine_parts[key]

    def linear_part(expression, variable):
        if z3.eq(expression, variable):
            return 1, z3.IntVal(0)
        if not contains(expression, variable):
            return 0, expression
        kind = expression.decl().kind() if z3.is_app(expression) else None
        if kind in (z3.Z3_OP_ADD, z3.Z3_OP_SUB, z3.Z3_OP_UMINUS):
            parts = [affine(child, variable) for child in expression.children()]
            if any(part is None for part in parts):
                return None
            signs = [-1] if kind == z3.Z3_OP_UMINUS else [1] + [-1 if kind == z3.Z3_OP_SUB else 1] * (len(parts) - 1)
            return sum(sign * part[0] for sign, part in zip(signs, parts)), sum(
                sign * part[1] for sign, part in zip(signs, parts)
            )
        if kind == z3.Z3_OP_MUL:
            scale, dependent = 1, []
            for child in expression.children():
                if z3.is_int_value(child):
                    scale *= child.as_long()
                else:
                    dependent.append(child)
            if len(dependent) == 1 and (part := affine(dependent[0], variable)) is not None:
                return scale * part[0], scale * part[1]
        return None

    def conjuncts(expression):
        if z3.is_and(expression):
            return [child for item in expression.children() for child in conjuncts(item)]
        return [expression]

    remaining = list(variables)
    claim = z3.simplify(claim)
    while remaining:
        definitions = []
        for condition in conjuncts(claim):
            if not z3.is_eq(condition) or not z3.is_int(condition.arg(0)):
                continue
            difference = z3.simplify(condition.arg(0) - condition.arg(1))
            for variable in remaining:
                part = affine(difference, variable)
                if part is not None and part[0]:
                    definitions.append((variable, *part))
        if not definitions:
            break
        # Resolve the smallest integer coefficient first. Choosing a large
        # outer-loop stride before a unit/inner coordinate can needlessly put
        # still-bound variables inside quotients and obscure later definitions.
        variable, coefficient, rest = min(definitions, key=lambda part: abs(part[1]))
        # Normalize to a positive divisor; Z3 integer division then has the
        # required exact quotient precisely when the retained remainder is zero.
        numerator, divisor = (-rest if coefficient > 0 else rest), abs(coefficient)
        value = z3.simplify(numerator / divisor)
        claim = z3.simplify(z3.And(numerator % divisor == 0, z3.substitute(claim, (variable, value))))
        remaining = [other for other in remaining if not z3.eq(other, variable)]
    if not remaining:
        return claim, claim
    quantified = z3.Exists(remaining, claim)
    # A concrete common-scope witness is an exact redundant disjunct of the
    # existential, not an assumption about the latest writer. Retain the full
    # original quantifier as well, so cross-iteration witnesses stay available.
    # This exposes the usual same-output-block witness before Presburger
    # search considers every preceding block that reused the physical rows.
    substitutions = [(variable, value) for variable, value in instances if any(z3.eq(variable, v) for v in remaining)]
    if not substitutions:
        return quantified, quantified
    unassigned = [v for v in remaining if not any(z3.eq(v, variable) for variable, _value in substitutions)]
    instance = z3.simplify(z3.substitute(claim, *substitutions))
    if unassigned:
        instance = z3.Exists(unassigned, instance)
    return quantified, instance


@dataclass(frozen=True)
class _Site:
    ordinal: int
    operation: object
    scopes: tuple[tuple[int, str, int], ...]
    path: tuple

    @property
    def instances(self):
        return prod(extent for _scope, _name, extent in self.scopes)


def _requested_elements(site, *dimensions):
    """Sum piecewise-constant extents without visiting dynamic instances.

    The supported dimension language is constants and finite equality selects.
    Addresses retain the full affine language. Unknown dimension forms decline
    this accounting instead of substituting a plan's traffic estimate.
    """
    cuts = {}

    def collect(value):
        if type(value) is int or isinstance(value, Const):
            return
        if not isinstance(value, Select):
            raise ValueError("dense traffic requires piecewise-constant extent operands")
        cuts.setdefault(value.var, set()).add(value.equals)
        collect(value.then)
        collect(value.other)

    for dimension in dimensions:
        collect(dimension)
    categories = []
    unused = 1
    for _scope, name, extent in site.scopes:
        if name not in cuts:
            unused *= extent
            continue
        exceptional = sorted(point for point in cuts[name] if 0 <= point < extent)
        choices = [(point, 1) for point in exceptional]
        if len(exceptional) < extent:
            representative = 0
            while representative in exceptional:
                representative += 1
            choices.append((representative, extent - len(exceptional)))
        categories.append((name, choices))
    if prod(len(choices) for _name, choices in categories) > 4096:
        raise DenseVerificationUnavailable("dense traffic select partition exceeds its checking budget")
    total = 0
    for selected in product(*(choices for _name, choices in categories)):
        env = {name: choice[0] for (name, _choices), choice in zip(categories, selected, strict=True)}
        multiplicity = unused * prod(choice[1] for choice in selected)
        total += multiplicity * prod(value if type(value) is int else evaluate(value, env) for value in dimensions)
    return total


def _project(kernel, adapter, *, fold_singleton_loops=True):
    from .dense_contracts import TiledIntegerStore

    sites = []
    scope_serial = 0

    def fold_singletons(value, constants):
        if isinstance(value, Var):
            return Const(constants[value.name]) if value.name in constants else value
        if isinstance(value, Select):
            if value.var in constants:
                return fold_singletons(value.then if constants[value.var] == value.equals else value.other, constants)
            return replace(
                value, then=fold_singletons(value.then, constants), other=fold_singletons(value.other, constants)
            )
        if isinstance(value, Sum):
            from merlin.sched.ir.expr import add

            return add(*(fold_singletons(term, constants) for term in value.terms))
        if isinstance(value, Scaled):
            return fold_singletons(value.expr, constants) * value.factor
        if isinstance(value, Const):
            return value
        if type(value) is tuple:
            return tuple(fold_singletons(item, constants) for item in value)
        if is_dataclass(value):
            return replace(
                value, **{field.name: fold_singletons(getattr(value, field.name), constants) for field in fields(value)}
            )
        return value

    def walk(body, scopes=(), path=(), singleton_values=None):
        singleton_values = {} if singleton_values is None else singleton_values
        nonlocal scope_serial
        for index, statement in enumerate(body):
            position = (*path, index)
            if isinstance(statement, Loop):
                scope_serial += 1
                scope = (scope_serial, statement.var, statement.extent)
                if statement.extent == 1 and fold_singleton_loops:
                    # The only dynamic value is exactly zero. Keep its position
                    # in the execution order, while eliminating a vacuous
                    # quantified axis from every source/lifetime obligation.
                    walk(statement.body, scopes, (*position, 0), {**singleton_values, statement.var: 0})
                else:
                    walk(statement.body, (*scopes, scope), (*position, (scope_serial, statement.var)), singleton_values)
            else:
                if statement.produces is not None or statement.awaits:
                    raise ValueError("synchronous dense verification cannot erase completion tokens")
                if statement.unit or statement.stages or statement.sets or statement.assumes:
                    raise ValueError("dense verification cannot erase unmodeled execution or configuration effects")
                operation = adapter.project(statement)
                if type(operation) not in (Load, Preload, Compute, Store, TiledIntegerStore, InitializeAccumulator):
                    raise ValueError("unknown projected dense operation is a conservative barrier")
                operation = fold_singletons(operation, singleton_values)
                sites.append(_Site(len(sites), operation, scopes, position))

    walk(kernel.body)
    return sites


def verify_compact_dense(
    kernel, problem, resources, storage, adapter, *, static_budget, timeout_ms=2000,
    output_readout=None, bias_source=None,
):
    """Verify current loop bounds, tails, source addresses and physical lifetimes.

    The projection adapter remains responsible for independently qualified
    instruction semantics. This result is not a final machine-emission receipt.
    """
    try:
        import z3
    except ImportError as exc:
        raise DenseVerificationUnavailable("symbolic compact dense checking requires z3-solver") from exc

    from merlin.sched.ir.kernel import check_structure

    if type(timeout_ms) is not int or timeout_ms <= 0 or type(static_budget) is not int or static_budget < 0:
        raise ValueError("dense verification requires explicit checking and static command budgets")
    if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation or storage.accumulator is None:
        raise ValueError("dense realization lacks its completion or accumulator contract")
    from .compact_dense import DenseBiasSource
    if problem.bias != (bias_source is not None):
        raise ValueError("bias presence requires explicit source initialization")
    if bias_source is not None:
        if type(bias_source) is not DenseBiasSource:
            raise ValueError("typed dense bias source required")
        bias_source.validate(problem, (kernel.args[0].dtype,kernel.args[1].dtype,kernel.args[2].dtype))
    if not resources.partial_output_tiles and (problem.m % problem.tile_m or problem.n % problem.tile_n):
        raise ValueError("dense resources do not declare output tail support")
    structure = check_structure(kernel)
    if structure:
        raise ValueError("invalid dense kernel: " + "; ".join(structure))
    from merlin.sched.ir.kernel import DTYPE_BYTES

    from .dense_contracts import DenseIntegerReadout, DenseTiledIntegerReadout, TiledIntegerStore

    if output_readout is not None and type(output_readout) not in (DenseIntegerReadout, DenseTiledIntegerReadout):
        raise ValueError("dense output verification requires a selected typed readout")
    tiled_readout = type(output_readout) is DenseTiledIntegerReadout
    if tiled_readout:
        output_readout.validate_problem(problem)
    output_bytes = problem.accumulator_bytes if output_readout is None else DTYPE_BYTES[output_readout.output_dtype]
    expected_args = (
        ((problem.m, problem.k), problem.lhs_bytes, "read"),
        ((problem.k, problem.n), problem.rhs_bytes, "read"),
        *((((problem.n,), problem.accumulator_bytes, "read"),) if bias_source is not None else ()),
        ((problem.m, problem.n), output_bytes, "write"),
    )
    from merlin.sched.ir.kernel import DTYPE_BYTES

    if (
        kernel.attrs
        or tuple(arg.name for arg in kernel.args) != (
            ("lhs", "rhs", "bias", "output") if bias_source is not None else ("lhs", "rhs", "output")
        )
        or any(
            (arg.shape, DTYPE_BYTES.get(arg.dtype), arg.access) != expected
            for arg, expected in zip(kernel.args, expected_args)
        )
    ):
        raise ValueError("actual dense boundary differs from the complete contraction")
    sites = _project(kernel, adapter)
    if len(sites) > static_budget:
        raise ValueError("actual compact dense instructions exceed the static budget")
    # Cache only successful source proofs after current boundaries, budgets and
    # actual projections have been checked. Asynchronous epoch checks remain
    # with their caller and always execute, even when this serial proof repeats.
    try:
        proof_key = _source_proof_key(sites, kernel, problem, resources, storage, adapter, output_readout, bias_source)
    except TypeError:
        proof_key = None
    if proof_key is not None:
        with _SOURCE_PROOF_LOCK:
            previous = _SOURCE_PROOFS.get(proof_key)
            if previous is not None:
                _SOURCE_PROOFS.move_to_end(proof_key)
                report = deepcopy(previous)
                report.update(kernel_sha256=kernel.digest(), solver_queries=0,
                              reused_solver_queries=previous["solver_queries"], source_proof_reused=True)
                return report
    deadline = time.monotonic() + timeout_ms / 1000
    query_serial = 0

    def prove(constraints, claim, reason, *, witness=None):
        nonlocal query_serial
        # All supported addresses are affine, dimensions are finite equality
        # selects, and ownership/lifetime queries quantify integer loop indices.
        # Use the quantified linear-arithmetic procedure directly: the default
        # SMT heuristic can spend the entire budget instantiating these exact
        # Presburger queries even for a five-instruction rectangular schedule.
        # SAT and UNKNOWN retain the same refusal paths below.
        # Constructive common-scope witnesses retain *all* interval/order and
        # overwrite obligations. Their success proves the original existential.
        # If not established, check the complete formula in the SAME remaining
        # budget; a failed preferred witness is never an invariant failure.
        preferred = witness is not None and not z3.eq(witness, claim)
        attempts = (witness, claim) if preferred else (claim,)
        for index, selected in enumerate(attempts):
            query_serial += 1
            remaining = int((deadline - time.monotonic()) * 1000)
            if remaining <= 0:
                raise DenseVerificationUnavailable("compact dense verification budget exhausted: " + reason)
            solver = z3.Tactic("qsat").solver()
            solver.set(timeout=max(1, remaining // 2 if preferred and index == 0 else remaining))
            solver.add(*constraints, z3.Not(selected))
            result = solver.check()
            if result == z3.unsat:
                return
            if preferred and index == 0:
                continue
            if result == z3.sat:
                raise ValueError("actual compact dense invariant failed: " + reason)
            raise DenseVerificationUnavailable(
                "compact dense proof unavailable: " + reason + ": " + solver.reason_unknown()
            )

    def domain(site, tag):
        values = {name: z3.Int(f"{tag}:{scope}:{name}") for scope, name, _extent in site.scopes}
        constraints = [z3.And(values[name] >= 0, values[name] < extent) for _scope, name, extent in site.scopes]
        return values, constraints

    def expression(value, env):
        if type(value) is bool:
            return z3.IntVal(int(value))
        if type(value) is int:
            return z3.IntVal(value)
        if isinstance(value, Const):
            return z3.IntVal(value.value)
        if isinstance(value, Var):
            return env[value.name]
        if isinstance(value, Scaled):
            return expression(value.expr, env) * value.factor
        if isinstance(value, Sum):
            return sum(expression(term, env) for term in value.terms)
        if isinstance(value, Select):
            return z3.If(env[value.var] == value.equals, expression(value.then, env), expression(value.other, env))
        raise ValueError("unknown dense address or loop expression")

    def position(site, env):
        return tuple(z3.IntVal(value) if type(value) is int else env[value[1]] for value in site.path)

    def before(a, ae, b, be):
        left, right = position(a, ae), position(b, be)
        return z3.Or(
            *(
                z3.And(*(left[j] == right[j] for j in range(i)), left[i] < right[i])
                for i in range(min(len(left), len(right)))
            )
        )

    def extent(index, logical, tile):
        return z3.If(index == (logical - 1) // tile, (logical - 1) % tile + 1, tile)

    def equal(left, right):
        return z3.And(*(a == b for a, b in zip(left, right, strict=True)))

    def rectangle(view, row, rows, cols, element_bytes):
        return row * view.unit_bytes, row * view.unit_bytes + (rows - 1) * view.unit_bytes + cols * element_bytes

    def bounds(view, interval):
        return z3.And(
            interval[0] >= view.base * view.unit_bytes,
            interval[1] <= (view.base + view.capacity) * view.unit_bytes,
            interval[1] > interval[0],
        )

    def aligned(view, row):
        return row % view.alignment == 0

    def overlap(left, right):
        return z3.And(left[0] < right[1], right[0] < left[1])

    # Build semantic read/write records from actual projected operands. Compute
    # consumes exactly its immediately preceding preload in the same loop body.
    loads = []
    initializers = []
    computes = []
    stores = []
    consumed = set()
    mt, kt, nt = problem.tiles

    def facts(site, env):
        op = site.operation

        def x(value):
            return expression(value, env)

        if isinstance(op, InitializeAccumulator):
            if bias_source is None:
                raise ValueError("undeclared accumulator initializer")
            m,n=map(x,op.block)
            rows,cols=x(op.rows),x(op.cols)
            interval=rectangle(storage.accumulator,x(op.row),rows,cols,problem.accumulator_bytes)
            valid=z3.And(m>=0,m<mt,n>=0,n<nt,
                x(op.dram_col)==n*problem.tile_n,
                x(op.source_row_stride)==0,x(op.source_col_stride)==1,
                rows==extent(m,problem.m,problem.tile_m),
                cols==extent(n,problem.n,problem.tile_n),
                bounds(storage.accumulator,interval),aligned(storage.accumulator,x(op.row)),
                cols*problem.accumulator_bytes<=storage.accumulator.unit_bytes)
            return {"owner":(m,z3.IntVal(-1),n),"interval":interval,"view":storage.accumulator,
                    "valid":valid,"rows":rows,"cols":cols,
                    "reserved_end":(x(op.row)+rows-storage.accumulator.base)*storage.accumulator.unit_bytes}
        if isinstance(op, Load):
            if op.role not in {"lhs", "weight"} or op.gather is not None:
                raise ValueError("unknown dense transfer semantics or gathered layout")
            a, b = map(x, op.block)
            row, col, rows, cols = map(x, (op.dram_row, op.dram_col, op.rows, op.cols))
            if op.role == "lhs":
                view, shape, tile, count, width = (
                    storage.lhs,
                    (problem.m, problem.k),
                    (problem.tile_m, problem.tile_k),
                    (mt, kt),
                    problem.lhs_bytes,
                )
            else:
                view, shape, tile, count, width = (
                    storage.rhs,
                    (problem.k, problem.n),
                    (problem.tile_k, problem.tile_n),
                    (kt, nt),
                    problem.rhs_bytes,
                )
            interval = rectangle(view, x(op.row), rows, cols, width)
            valid = z3.And(
                a >= 0,
                a < count[0],
                b >= 0,
                b < count[1],
                row == a * tile[0],
                col == b * tile[1],
                rows == extent(a, shape[0], tile[0]),
                cols == extent(b, shape[1], tile[1]),
                x(op.step) == 1,
                bounds(view, interval),
                aligned(view, x(op.row)),
                cols * width <= view.unit_bytes,
            )
            return {
                "owner": (a, b),
                "interval": interval,
                "view": view,
                "valid": valid,
                "rows": rows,
                "cols": cols,
                "reserved_end": (x(op.row) + rows - view.base) * view.unit_bytes,
            }
        if isinstance(op, Compute):
            preload = sites[site.ordinal - 1].operation
            m, k = map(x, op.input_block)
            wk, n = map(x, preload.weight_block)
            rows, depth, cols = x(op.rows), x(op.cols), x(preload.cols)
            left = rectangle(storage.lhs, x(op.input_row), rows, depth, problem.lhs_bytes)
            right = rectangle(storage.rhs, x(preload.weight_row), depth, cols, problem.rhs_bytes)
            output = rectangle(storage.accumulator, x(preload.accumulator_row), rows, cols, problem.accumulator_bytes)
            valid = z3.And(
                m >= 0,
                m < mt,
                n >= 0,
                n < nt,
                k >= 0,
                k < kt,
                wk == k,
                rows == extent(m, problem.m, problem.tile_m),
                depth == extent(k, problem.k, problem.tile_k),
                cols == extent(n, problem.n, problem.tile_n),
                x(preload.rows) == rows,
                x(preload.weight_rows) == depth,
                x(preload.accumulate) == (1 if bias_source is not None else z3.If(k == 0, 0, 1)),
                bounds(storage.lhs, left),
                bounds(storage.rhs, right),
                bounds(storage.accumulator, output),
                aligned(storage.lhs, x(op.input_row)),
                aligned(storage.rhs, x(preload.weight_row)),
                aligned(storage.accumulator, x(preload.accumulator_row)),
                depth * problem.lhs_bytes <= storage.lhs.unit_bytes,
                cols * problem.rhs_bytes <= storage.rhs.unit_bytes,
                cols * problem.accumulator_bytes <= storage.accumulator.unit_bytes,
            )
            return {
                "owner": (m, k, n),
                "interval": output,
                "view": storage.accumulator,
                "valid": valid,
                "lhs": left,
                "rhs": right,
                "rows": rows,
                "cols": cols,
                "depth": depth,
                "reserved_end": (x(preload.accumulator_row) + rows - storage.accumulator.base)
                * storage.accumulator.unit_bytes,
            }
        if isinstance(op, Store):
            row, col, rows, cols = map(x, (op.dram_row, op.dram_col, op.rows, op.cols))
            m, n = row / problem.tile_m, col / problem.tile_n
            interval = rectangle(storage.accumulator, x(op.accumulator_row), rows, cols, problem.accumulator_bytes)
            valid = z3.And(
                row % problem.tile_m == 0,
                col % problem.tile_n == 0,
                m >= 0,
                m < mt,
                n >= 0,
                n < nt,
                rows == extent(m, problem.m, problem.tile_m),
                cols == extent(n, problem.n, problem.tile_n),
                bounds(storage.accumulator, interval),
                aligned(storage.accumulator, x(op.accumulator_row)),
            )
            if tiled_readout:
                valid = z3.And(valid, x(op.readout_tile) == n)
            return {
                "owner": (m, n),
                "interval": interval,
                "view": storage.accumulator,
                "valid": valid,
                "rows": rows,
                "cols": cols,
                "reserved_end": (x(op.accumulator_row) + rows - storage.accumulator.base)
                * storage.accumulator.unit_bytes,
            }
        raise ValueError("preload cannot independently establish a dense effect")

    for site in sites:
        op = site.operation
        if isinstance(op, Preload):
            continue
        if isinstance(op, Compute):
            if site.ordinal == 0:
                raise ValueError("dense compute has no actual preload")
            previous = sites[site.ordinal - 1]
            if (
                not isinstance(previous.operation, Preload)
                or previous.scopes != site.scopes
                or previous.path[:-1] != site.path[:-1]
                or previous.path[-1] + 1 != site.path[-1]
                or previous.operation.weight_block is None
                or previous.operation.weight_row is None
                or previous.operation.weight_rows is None
                or op.fresh_weights is not True
                or op.input_gather is not None
            ):
                raise ValueError("dense compute does not consume exactly one adjacent preload")
            consumed.add(previous.ordinal)
            computes.append(site)
        elif isinstance(op, InitializeAccumulator):
            initializers.append(site)
        elif isinstance(op, Load):
            loads.append(site)
        elif isinstance(op, Store):
            if (type(op) is TiledIntegerStore) != tiled_readout:
                raise ValueError("actual dense store must retain the source-indexed readout kind")
            if op.readout != (None if output_readout is None else output_readout.identity):
                raise ValueError("actual dense store readout differs from the checked output")
            stores.append(site)
        env, conditions = domain(site, f"bounds{site.ordinal}")
        prove(conditions, facts(site, env)["valid"], "source coordinates, tails, initialization or storage bounds")
    if {site.ordinal for site in sites if isinstance(site.operation, Preload)} != consumed:
        raise ValueError("dense preload was omitted, overwritten or left unconsumed")

    def coverage(rows, expected, name):
        if sum(site.instances for site in rows) != expected:
            raise ValueError("actual dense coverage differs: " + name)
        for i, a in enumerate(rows):
            for b in rows[i:]:
                ae, ac = domain(a, f"{name}left{a.ordinal}")
                be, bc = domain(b, f"{name}right{b.ordinal}")
                distinct = z3.BoolVal(True) if a is not b else z3.Or(*(ae[key] != be[key] for key in ae))
                prove(
                    (*ac, *bc, distinct),
                    z3.Not(equal(facts(a, ae)["owner"], facts(b, be)["owner"])),
                    name + " injective complete domain",
                )

    coverage(computes, mt * kt * nt, "contraction")
    coverage(stores, mt * nt, "output")
    coverage(initializers, mt * nt if bias_source is not None else 0, "bias initialization")

    # Conservatively separate the three simultaneously live roles through the
    # caller's explicit byte backings. No interchangeable-memory assumption.
    views = (storage.lhs, storage.rhs, storage.accumulator)
    for i, a in enumerate(views):
        for b in views[i + 1 :]:
            if (
                a.backing == b.backing
                and a.base * a.unit_bytes < (b.base + b.capacity) * b.unit_bytes
                and b.base * b.unit_bytes < (a.base + a.capacity) * a.unit_bytes
            ):
                raise ValueError("dense assigned live storage views alias through physical backing")

    # Reconstruct padded high-water marks from emitted addresses and loop
    # domains. Oversized assigned windows cannot bypass the resource budget.
    high_water = {view.name: 0 for view in views}
    for site in (*loads, *initializers, *computes):
        env, conditions = domain(site, f"high_water{site.ordinal}")
        fact = facts(site, env)
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise DenseVerificationUnavailable("dense high-water checking budget exhausted")
        solver = z3.Optimize()
        solver.set(timeout=remaining)
        solver.add(*conditions)
        objective = solver.maximize(fact["reserved_end"])
        if solver.check() != z3.sat:
            raise DenseVerificationUnavailable("dense high-water maximum unavailable")
        lower, upper = objective.lower(), objective.upper()
        if not z3.is_int_value(lower) or not z3.is_int_value(upper) or lower.as_long() != upper.as_long():
            raise DenseVerificationUnavailable("dense high-water optimum was not established")
        key = fact["view"].name
        high_water[key] = max(high_water[key], upper.as_long())
    if (
        high_water[storage.lhs.name] + high_water[storage.rhs.name] > resources.operand_bytes
        or high_water[storage.accumulator.name] > resources.accumulator_bytes
    ):
        raise ValueError("actual dense padded storage exceeds the selected resource budget")

    def prior_value(reader, renv, candidates, desired_owner, interval):
        witnesses, preferred = [], []
        for writer in candidates:
            we, wc = domain(writer, f"w{reader.ordinal}_{writer.ordinal}")
            wf = facts(writer, we)
            # Simplify only exact formulas. In particular, an initializer at
            # reduction -1 cannot witness a final K>=0 value. Excluding a
            # statically false witness avoids constructing its nested overwrite
            # quantifiers; every remaining lifetime obligation is unchanged.
            matches = z3.simplify(z3.And(
                equal(wf["owner"], desired_owner), equal(wf["interval"], interval),
                before(writer, we, reader, renv),
            ))
            if z3.is_false(matches):
                continue
            no_overwrites = []
            for other in candidates:
                oe, oc = domain(other, f"o{reader.ordinal}_{writer.ordinal}_{other.ordinal}")
                of = facts(other, oe)
                if of["view"].backing != wf["view"].backing:
                    continue
                intervenes = z3.simplify(z3.And(
                    *oc,
                    before(writer, we, other, oe),
                    before(other, oe, reader, renv),
                    overlap(of["interval"], interval),
                ))
                if z3.is_false(intervenes):
                    continue
                no_overwrites.append(z3.Not(z3.Exists(list(oe.values()), intervenes)) if oe else z3.Not(intervenes))
            claim = z3.And(
                *wc,
                matches,
                *no_overwrites,
            )
            reader_scopes = {scope: name for scope, name, _extent in reader.scopes}
            common = tuple(
                (we[name], renv[reader_scopes[scope]])
                for scope, name, _extent in writer.scopes
                if scope in reader_scopes
            )
            complete, instance = _defined_exists(list(we.values()), claim, instances=common)
            witnesses.append(complete)
            preferred.append(instance)
        return z3.Or(*witnesses), z3.Or(*preferred)

    for reader in computes:
        re, rc = domain(reader, f"read{reader.ordinal}")
        rf = facts(reader, re)
        m, k, n = rf["owner"]
        for role, owner, interval in (("lhs", (m, k), rf["lhs"]), ("weight", (k, n), rf["rhs"])):
            writers = [site for site in loads if site.operation.role == role]
            available, witness = prior_value(reader, re, writers, owner, interval)
            prove(rc, available, role + " residency and last completed load", witness=witness)
        available, witness = prior_value(reader, re, [*initializers,*computes], (m, k - 1, n), rf["interval"])
        prove(
            (*rc, z3.BoolVal(True) if bias_source is not None else k > 0),
            available,
            "exact ordered accumulator continuation or initial bias",
            witness=witness,
        )

    for reader in stores:
        re, rc = domain(reader, f"store{reader.ordinal}")
        rf = facts(reader, re)
        m, n = rf["owner"]
        available, witness = prior_value(reader, re, [*initializers,*computes], (m, kt - 1, n), rf["interval"])
        prove(rc, available, "complete reduction before output and no accumulator overwrite", witness=witness)

    report = {
        "schema": "merlin.compact_dense_realization.v1",
        "kernel_sha256": kernel.digest(),
        "configuration_identity": adapter.configuration_identity,
        "numerical_policy_identity": adapter.numerical_policy_identity,
        "c_index_bits": adapter.index_bits,
        "static_instruction_calls": len(sites),
        "executed_instruction_calls": sum(site.instances for site in sites),
        "lhs_loads": sum(site.instances for site in loads if site.operation.role == "lhs"),
        "rhs_loads": sum(site.instances for site in loads if site.operation.role == "weight"),
        "bias_initializations": sum(site.instances for site in initializers),
        "bias_source": None if bias_source is None else {
            "columns": bias_source.columns, "dtype": bias_source.dtype, "arithmetic": bias_source.arithmetic,
            "row_stride": 0, "column_stride": 1,
        },
        "requested_bias_load_bytes": sum(
            _requested_elements(site, site.operation.cols) * problem.accumulator_bytes for site in initializers
        ),
        "requested_lhs_bytes": sum(
            _requested_elements(site, site.operation.rows, site.operation.cols) * problem.lhs_bytes
            for site in loads if site.operation.role == "lhs"
        ),
        "requested_rhs_bytes": sum(
            _requested_elements(site, site.operation.rows, site.operation.cols) * problem.rhs_bytes
            for site in loads if site.operation.role == "weight"
        ),
        "requested_load_bytes": sum(
            _requested_elements(site, site.operation.rows, site.operation.cols)
            * (problem.lhs_bytes if site.operation.role == "lhs" else problem.rhs_bytes)
            for site in loads
        ) + sum(_requested_elements(site,site.operation.cols)*problem.accumulator_bytes for site in initializers),
        "requested_store_bytes": sum(
            _requested_elements(site, site.operation.rows, site.operation.cols) * output_bytes for site in stores
        ),
        "padded_high_water_bytes": high_water,
        "output_tiles": sum(site.instances for site in stores),
        **({"tiled_readout": {"identity": output_readout.identity,
                               "columns": output_readout.columns,
                               "tile_columns": output_readout.tile_columns,
                               "parameter_words": len(output_readout.table_words),
                               "selection": "actual readout_tile equals original output column tile"}}
           if tiled_readout else {}),
        "solver_queries": query_serial,
        "completion": "declared synchronous",
        "machine_emission": "not_established",
        "timing": "UNKNOWN",
    }
    if proof_key is not None:
        with _SOURCE_PROOF_LOCK:
            _SOURCE_PROOFS[proof_key] = deepcopy(report)
            _SOURCE_PROOFS.move_to_end(proof_key)
            while len(_SOURCE_PROOFS) > _SOURCE_PROOF_LIMIT:
                _SOURCE_PROOFS.popitem(last=False)
    return report
