"""Symbolic checks over projected *actual* counted-loop dense instructions.

Coverage is proved by bounded domains, exact cardinality and injectivity.
Residency is a latest-write condition over physical byte intervals and program
order. No loop is expanded and no tensor is allocated by this checker.
Completion is synchronous in this family; asynchronous adapters must not use it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from math import prod

from merlin.sched.ir import Const, Loop, Scaled, Select, Sum, Var, evaluate

from .block_schedule import Compute, Load, Preload, Store


class DenseVerificationUnavailable(ValueError):
    """The selected checking tool or budget could not establish the result."""


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


def _project(kernel, adapter):
    from .compact_dense import _adapter_guard

    fresh = _adapter_guard(adapter)
    sites = []
    scope_serial = 0

    def walk(body, scopes=(), path=()):
        nonlocal scope_serial
        for index, statement in enumerate(body):
            position = (*path, index)
            if isinstance(statement, Loop):
                scope_serial += 1
                scope = (scope_serial, statement.var, statement.extent)
                walk(statement.body, (*scopes, scope), (*position, (scope_serial, statement.var)))
            else:
                if statement.produces is not None or statement.awaits:
                    raise ValueError("synchronous dense verification cannot erase completion tokens")
                if statement.unit or statement.stages or statement.sets or statement.assumes:
                    raise ValueError("dense verification cannot erase unmodeled execution or configuration effects")
                fresh()
                operation = adapter.project(statement)
                fresh()
                if type(operation) not in (Load, Preload, Compute, Store):
                    raise ValueError("unknown projected dense operation is a conservative barrier")
                sites.append(_Site(len(sites), operation, scopes, position))

    walk(kernel.body)
    return sites


def verify_compact_dense(
    kernel, problem, resources, storage, adapter, *, static_budget, timeout_ms=2000, output_readout=None
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
    if problem.bias:
        raise ValueError("compact dense verification supports only the zero seed")
    if not resources.partial_output_tiles and (problem.m % problem.tile_m or problem.n % problem.tile_n):
        raise ValueError("dense resources do not declare output tail support")
    structure = check_structure(kernel)
    if structure:
        raise ValueError("invalid dense kernel: " + "; ".join(structure))
    from merlin.sched.ir.kernel import DTYPE_BYTES

    from .dense_contracts import DenseIntegerReadout

    if output_readout is not None and type(output_readout) is not DenseIntegerReadout:
        raise ValueError("dense output verification requires a selected typed readout")
    output_bytes = problem.accumulator_bytes if output_readout is None else DTYPE_BYTES[output_readout.output_dtype]
    expected_args = (
        ((problem.m, problem.k), problem.lhs_bytes, "read"),
        ((problem.k, problem.n), problem.rhs_bytes, "read"),
        ((problem.m, problem.n), output_bytes, "write"),
    )
    from merlin.sched.ir.kernel import DTYPE_BYTES

    if (
        kernel.attrs
        or tuple(arg.name for arg in kernel.args) != ("lhs", "rhs", "output")
        or any(
            (arg.shape, DTYPE_BYTES.get(arg.dtype), arg.access) != expected
            for arg, expected in zip(kernel.args, expected_args)
        )
    ):
        raise ValueError("actual dense boundary differs from the complete contraction")
    sites = _project(kernel, adapter)
    if len(sites) > static_budget:
        raise ValueError("actual compact dense instructions exceed the static budget")
    deadline = time.monotonic() + timeout_ms / 1000
    query_serial = 0

    def prove(constraints, claim, reason):
        nonlocal query_serial
        query_serial += 1
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise DenseVerificationUnavailable("compact dense verification budget exhausted: " + reason)
        solver = z3.Solver()
        solver.set(timeout=max(1, remaining))
        solver.add(*constraints, z3.Not(claim))
        result = solver.check()
        if result == z3.sat:
            raise ValueError("actual compact dense invariant failed: " + reason)
        if result != z3.unsat:
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
    computes = []
    stores = []
    consumed = set()
    mt, kt, nt = problem.tiles

    def facts(site, env):
        op = site.operation

        def x(value):
            return expression(value, env)

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
            right = (
                rectangle(storage.rhs, x(preload.weight_row), depth, cols, problem.rhs_bytes)
                if op.fresh_weights
                else None
            )
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
                x(preload.accumulate) == z3.If(k == 0, 0, 1),
                bounds(storage.lhs, left),
                bounds(storage.rhs, right) if right is not None else z3.BoolVal(True),
                bounds(storage.accumulator, output),
                aligned(storage.lhs, x(op.input_row)),
                aligned(storage.rhs, x(preload.weight_row)) if right is not None else z3.BoolVal(True),
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
                or previous.operation.weight_rows is None
                or type(op.fresh_weights) is not bool
                or op.input_gather is not None
            ):
                raise ValueError("dense compute does not consume exactly one adjacent preload")
            if op.fresh_weights:
                if previous.operation.weight_row is None:
                    raise ValueError("fresh dense compute has no actual weight preload")
            elif (
                adapter.retained_weights is not True
                or not adapter.qualification_identity
                or previous.operation.weight_row is not None
            ):
                raise ValueError("retained dense compute lacks qualified descriptor-only preload semantics")
            consumed.add(previous.ordinal)
            computes.append(site)
        elif isinstance(op, Load):
            loads.append(site)
        elif isinstance(op, Store):
            if op.readout != (None if output_readout is None else output_readout.identity):
                raise ValueError("actual dense store readout differs from the checked scalar output")
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
    for site in (*loads, *computes):
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

    def prior_value(reader, renv, candidates, desired_owner, interval, *, pe=False):
        witnesses = []
        for writer in candidates:
            we, wc = domain(writer, f"w{reader.ordinal}_{writer.ordinal}")
            wf = facts(writer, we)
            no_overwrites = []
            for other in candidates:
                oe, oc = domain(other, f"o{reader.ordinal}_{writer.ordinal}_{other.ordinal}")
                of = facts(other, oe)
                if not pe and of["view"].backing != wf["view"].backing:
                    continue
                intervenes = z3.And(
                    *oc,
                    before(writer, we, other, oe),
                    before(other, oe, reader, renv),
                    z3.BoolVal(True) if pe else overlap(of["interval"], interval),
                )
                no_overwrites.append(z3.Not(z3.Exists(list(oe.values()), intervenes)) if oe else z3.Not(intervenes))
            claim = z3.And(
                *wc,
                equal(wf["owner"][1:] if pe else wf["owner"], desired_owner),
                z3.BoolVal(True) if pe else equal(wf["interval"], interval),
                before(writer, we, reader, renv),
                *no_overwrites,
            )
            witnesses.append(z3.Exists(list(we.values()), claim) if we else claim)
        return z3.Or(*witnesses)

    for reader in computes:
        re, rc = domain(reader, f"read{reader.ordinal}")
        rf = facts(reader, re)
        m, k, n = rf["owner"]
        reads = (("lhs", (m, k), rf["lhs"]),) + (
            (("weight", (k, n), rf["rhs"]),) if reader.operation.fresh_weights else ()
        )
        for role, owner, interval in reads:
            writers = [site for site in loads if site.operation.role == role]
            available = prior_value(reader, re, writers, owner, interval)
            prove(rc, available, role + " residency and last completed load")
        if not reader.operation.fresh_weights:
            writers = [site for site in computes if site.operation.fresh_weights]
            available = prior_value(reader, re, writers, (k, n), None, pe=True)
            prove(rc, available, "retained PE weights have the exact latest fresh weight owner")
        available = prior_value(reader, re, computes, (m, k - 1, n), rf["interval"])
        prove((*rc, k > 0), available, "exact ordered accumulator continuation")

    for reader in stores:
        re, rc = domain(reader, f"store{reader.ordinal}")
        rf = facts(reader, re)
        m, n = rf["owner"]
        available = prior_value(reader, re, computes, (m, kt - 1, n), rf["interval"])
        prove(rc, available, "complete reduction before output and no accumulator overwrite")

    return {
        "schema": "merlin.compact_dense_realization.v1",
        "kernel_sha256": kernel.digest(),
        "configuration_identity": adapter.configuration_identity,
        "numerical_policy_identity": adapter.numerical_policy_identity,
        "c_index_bits": adapter.index_bits,
        "static_instruction_calls": len(sites),
        "executed_instruction_calls": sum(site.instances for site in sites),
        "lhs_loads": sum(site.instances for site in loads if site.operation.role == "lhs"),
        "rhs_loads": sum(site.instances for site in loads if site.operation.role == "weight"),
        "requested_load_bytes": sum(
            _requested_elements(site, site.operation.rows, site.operation.cols)
            * (problem.lhs_bytes if site.operation.role == "lhs" else problem.rhs_bytes)
            for site in loads
        ),
        "requested_store_bytes": sum(
            _requested_elements(site, site.operation.rows, site.operation.cols) * output_bytes for site in stores
        ),
        "padded_high_water_bytes": high_water,
        "output_tiles": sum(site.instances for site in stores),
        "solver_queries": query_serial,
        "completion": "declared synchronous",
        "machine_emission": "not_established",
        "timing": "UNKNOWN",
        **(
            {
                "pe_weight_loads": sum(site.instances for site in computes if site.operation.fresh_weights),
                "retained_weight_computes": sum(
                    site.instances for site in computes if not site.operation.fresh_weights
                ),
            }
            if any(not site.operation.fresh_weights for site in computes)
            else {}
        ),
    }
