"""Actual pinned registered schedule route and independent complete-IR intake.

All identity/runtime/author failures are UNAVAILABLE. Only explicit domain
exceptions yield a refusal marker; no arbitrary ValueError is admitted as one.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

if __package__:
    from .lean_graph_state_integer_capture import validate_calls
    from .lean_graph_state_pipeline_capture import loaded_sources, project_modules, selected_source
    from .lean_graph_state_scalar_capture import canonical, file_ref, project
    from .lean_graph_state_schedule_errors import ScheduleDomainRefusal, ScheduleUnavailable, ScheduleIdentityUnavailable, require
    from .lean_graph_state_schedule_projection import dependencies, describe, domain_call, match_step
else:
    from lean_graph_state_integer_capture import validate_calls
    from lean_graph_state_pipeline_capture import loaded_sources, project_modules, selected_source
    from lean_graph_state_scalar_capture import canonical, file_ref, project
    from lean_graph_state_schedule_errors import ScheduleDomainRefusal, ScheduleUnavailable, ScheduleIdentityUnavailable, require
    from lean_graph_state_schedule_projection import dependencies, describe, domain_call, match_step


def identity_call(function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise ScheduleIdentityUnavailable("source/runtime/identity failure: " + str(error)) from error


def parse(data):
    from merlin.transforms.certificates import parse_ir
    from xdsl.utils.exceptions import ParseError, VerifyException

    try:
        return parse_ir(data.decode())
    except (ParseError, VerifyException, UnicodeDecodeError) as error:
        raise ScheduleDomainRefusal("malformed actual IR: " + str(error)) from error


def checked_calls(modules):
    # This READONLY helper contains IR-only signature checks. Its exact known
    # domain failures are enumerated; runtime/identity ValueError is unavailable.
    for module in modules:
        describe(module)
    try:
        validate_calls(modules)
    except ValueError as error:
        if str(error) in ("explicit nonnested external declaration required", "actual call signature differs from declaration"):
            raise ScheduleDomainRefusal(str(error)) from error
        raise ScheduleUnavailable("unexpected call-projection author failure: " + str(error)) from error


def write_projection(reference, buffers, order_buffers, requests, destination, *, factor, tile, bits, mode, diagnostics=()):
    revision = identity_call(selected_source, reference)
    require(len(buffers) in (2, 7), "two standalone or SEVEN composed buffers required")
    require(type(requests) is list and 1 <= len(requests) <= 8 and len(order_buffers) == len(requests) + 1,
            "complete current-order trace required")
    modules, steps = [parse(data) for data in buffers], [parse(data) for data in order_buffers]
    schedule_index = 0 if len(buffers) == 2 else 5
    require(buffers[schedule_index] == order_buffers[0] and buffers[-1] == order_buffers[-1],
            "complete source/schedule/final step-buffer joins differ")
    checked_calls([*modules, *steps])
    matched = [match_step(first, second, request) for first, second, request in zip(steps, steps[1:], requests, strict=False)]
    context = tuple([*modules, *steps])
    pairs = [domain_call(project, first, second, context=context) for first, second in zip(steps, steps[1:])]
    require(all(row["registry"] == pairs[0]["registry"] and row["types"] == pairs[0]["types"] for row in pairs),
            "one actual complete registry/type table required for scheduling trace")
    prefix = None
    if len(buffers) == 7:
        integer = domain_call(project, modules[0], modules[1], context=context)
        stages, controls = domain_call(project_modules, modules[1:6])
        require(all(row["registry"] == pairs[0]["registry"] and row["types"] == pairs[0]["types"]
                    for row in [integer, *stages]), "exact registered prefix/schedule registry join required")
        prefix = {"prefix": integer, "pipeline": {"stages": stages, "index_operations": controls,
                  "factor": factor, "tile": tile, "index_bits": bits}}
    producer, runtime = identity_call(loaded_sources, reference)
    result = {
        "schema": "merlin.graph-state-schedule.actual-boundary.v1", "revision": revision,
        "pairs": pairs, "prefix": prefix, "matched_orders": matched,
        "requests": requests, "buffers": [file_ref(destination / f"stage{i}.mlir") for i in range(len(buffers))],
        "journal": file_ref(destination / "REQUESTS.json"),
        "order_buffers": [file_ref(destination / f"order{i}.mlir") for i in range(len(order_buffers))],
        "each_stage_changed": [a != b for a, b in zip(buffers, buffers[1:])],
        "each_order_changed": [a != b for a, b in zip(order_buffers, order_buffers[1:])],
        "settings": {"factor": factor, "tile": tile, "index_bits": bits},
        "producer_sources": producer, "python_runtime": runtime,
        "owned_projection": [file_ref(__file__), file_ref(Path(__file__).with_name("lean_graph_state_schedule_projection.py")),
                             file_ref(Path(__file__).with_name("lean_graph_state_schedule_errors.py"))],
        "mode": mode, "diagnostic_only": list(diagnostics), "projection_trusted": True,
        "Lean_checked": False, "qualified": False, "review_approved": False,
        "empty_effects_imply_totality": False, "region_motion_admitted": False,
        "native_or_poison_fidelity_proved": False,
    }
    if any((destination / f"stage{i}.mlir").read_bytes() != data for i, data in enumerate(buffers)) or any(
        (destination / f"order{i}.mlir").read_bytes() != data for i, data in enumerate(order_buffers)
    ):
        raise ScheduleIdentityUnavailable("submitted/snapshot buffers changed during intake")
    if identity_call(loaded_sources, reference) != (producer, runtime):
        raise ScheduleIdentityUnavailable("producer/runtime identity changed during intake")
    (destination / "CAPTURE.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def choose_request(module, blocked, minimum_depth):
    description = describe(module)
    for index, block in enumerate(description["blocks"]):
        if index == 0 or len(block["scope"]) < minimum_depth:
            continue
        rows = block["operations"]
        block_token = canonical(["selected-block", block["scope"]])
        if block_token in blocked:
            continue
        dependency = dependencies(module, index)
        for position, row in enumerate(rows):
            if row["eligible_total_primitive"] is None:
                continue
            token = canonical([block["scope"], row])
            if token in blocked:
                continue
            earliest = max([a + 1 for a, b in dependency["edges"] if b == position], default=0)
            if earliest >= position:
                continue
            order = list(range(len(rows)))
            order.remove(position)
            order.insert(earliest, position)
            blocked.add(token)
            blocked.add(block_token)
            return index, order
    raise ScheduleDomainRefusal("no new admitted total-primitive reorder in current program")


def capture(reference, source, destination, *, chain=False, count=1, minimum_depth=1,
            factor=2, tile=3, bits=32, explicit_orders=None):
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
    from merlin.transforms.planning import apply_order

    reference, source, destination = reference.resolve(), source.resolve(), destination.resolve()
    identity_call(selected_source, reference)
    destination.mkdir(parents=True, exist_ok=False)
    original = source.read_bytes()
    (destination / "input.mlir").write_bytes(original)
    module = parse(original)
    buffers, diagnostics = [ir_text(module).encode()], []
    if chain:
        from merlin.transforms.normalization import simplify_integers
        from merlin.transforms.pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants
        from merlin.transforms.unrolling import unroll_ordered_loops
        from merlin.transforms.loops import segment_loops

        for name, run in (
            ("integer", lambda: simplify_integers(module)), ("cse-dce", lambda: eliminate_scalar_redundancy(module)),
            ("licm", lambda: hoist_scalar_invariants(module)),
            ("unroll", lambda: unroll_ordered_loops(module, (factor,), index_bits=bits)),
            ("segment", lambda: segment_loops(module, (tile,), index_bits=bits)),
        ):
            witnesses = run()
            module.verify()
            diagnostics.append({"stage": name, "witnesses": [asdict(w) for w in witnesses]})
            buffers.append(ir_text(module).encode())
    require(type(count) is int and 1 <= count <= 8 and type(minimum_depth) is int and minimum_depth >= 1,
            "bounded actual order count/depth required")
    probe, blocked, selected, simulated = module.clone(), set(), [], [ir_text(module).encode()]
    if explicit_orders is not None:
        require(type(explicit_orders) is list and len(explicit_orders) == count,
                "explicit order list must match selected bounded count")
    for step in range(count if explicit_orders is None else len(explicit_orders)):
        if explicit_orders is None:
            index, order = choose_request(probe, blocked, minimum_depth)
        else:
            row = explicit_orders[step]
            require(type(row) is dict and set(row) == {"block_index", "order"}, "closed explicit producer order required")
            index, order = row["block_index"], row["order"]
            require(type(index) is int and type(order) is list and all(type(i) is int for i in order), "exact producer order integers required")
            facts = dependencies(probe, index)
            require(sorted(order) == list(range(len(facts["operation_sha256"]))), "explicit current order is not a permutation")
            positions = {op: i for i, op in enumerate(order)}
            require(all(positions[a] < positions[b] for a, b in facts["edges"]), "explicit order violates current static dependency")
        apply_order(probe, index, order)
        selected.append((index, tuple(order)))
        simulated.append(ir_text(probe).encode())
    options = PipelineOptions(index_bits=bits, normalize_shapes=False, simplify_integers=False,
                              reuse_buffers=False, dependency_orders=tuple(selected))
    actual = apply_common_pipeline(module, options, plan_only=True)
    if len(actual.changes) != len(selected):
        raise ScheduleUnavailable("actual registered route did not retain every requested order")
    order_buffers = [ir_text(module).encode()]
    requests = []
    for step, change in enumerate(actual.changes):
        if change.kind != "sequential_dependency_schedule" or change.before_ir.encode() != order_buffers[-1]:
            raise ScheduleUnavailable("actual registered route source/current program identity differs")
        order_buffers.append(change.after_ir.encode())
        if order_buffers[-1] != simulated[step + 1]:
            raise ScheduleUnavailable("registered route differs from actual current-order simulation")
        witness = asdict(change.certificate.witnesses[0])
        requests.append(json.loads(json.dumps({"block_index": selected[step][0], "order": list(selected[step][1]), "witness": witness})))
    buffers.append(ir_text(actual.module).encode())
    for prefix, values in (("stage", buffers), ("order", order_buffers)):
        for i, data in enumerate(values):
            (destination / f"{prefix}{i}.mlir").write_bytes(data)
    (destination / "REQUESTS.json").write_text(json.dumps(requests, indent=2) + "\n")
    (destination / "ACTUAL_ROUTE.json").write_text(json.dumps({"plan_only": True, "formal_receipt_issued": False,
        "changes": [change.to_dict() for change in actual.changes], "source": file_ref(source)}, indent=2) + "\n")
    result = write_projection(reference, buffers, order_buffers, requests, destination, factor=factor, tile=tile,
                              bits=bits, mode="actual-registered-producer", diagnostics=diagnostics)
    if source.read_bytes() != original:
        raise ScheduleIdentityUnavailable("actual source bytes changed during production")
    return result


def reopen(reference, paths, orders, requests_path, destination, *, factor=2, tile=3, bits=32):
    destination.mkdir(parents=True, exist_ok=False)
    buffers, steps = [p.read_bytes() for p in paths], [p.read_bytes() for p in orders]
    request_bytes = requests_path.read_bytes()
    try:
        requests = json.loads(request_bytes)
    except json.JSONDecodeError as error:
        raise ScheduleDomainRefusal("malformed order journal JSON") from error
    for prefix, values in (("stage", buffers), ("order", steps)):
        for i, data in enumerate(values):
            (destination / f"{prefix}{i}.mlir").write_bytes(data)
    (destination / "REQUESTS.json").write_bytes(request_bytes)
    result = write_projection(reference, buffers, steps, requests, destination, factor=factor, tile=tile,
                              bits=bits, mode="independent-complete-reprojection")
    if any(p.read_bytes() != data for p, data in zip(paths, buffers, strict=True)) or any(
        p.read_bytes() != data for p, data in zip(orders, steps, strict=True)
    ) or requests_path.read_bytes() != request_bytes:
        raise ScheduleIdentityUnavailable("original/journal identities changed during independent intake")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--buffers", nargs="+", type=Path)
    parser.add_argument("--orders", nargs="+", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--chain", action="store_true")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--minimum-depth", type=int, default=1)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--tile", type=int, default=3)
    parser.add_argument("--bits", type=int, default=32)
    parser.add_argument("--explicit-orders", type=Path)
    args = vars(parser.parse_args())
    source, paths, orders, requests = [args.pop(k) for k in ("source", "buffers", "orders", "requests")]
    explicit = args.pop("explicit_orders")
    destination = args["destination"]
    try:
        if source is not None:
            require(paths is None and orders is None and requests is None, "select producer OR independent submitted originals")
            result = capture(source=source, explicit_orders=json.loads(explicit.read_text()) if explicit else None, **args)
        else:
            require(paths is not None and orders is not None and requests is not None, "complete originals/orders/journal required")
            for key in ("chain", "count", "minimum_depth"):
                args.pop(key)
            result = reopen(paths=paths, orders=orders, requests_path=requests, **args)
    except ScheduleDomainRefusal as error:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "REJECTION.json").write_text(json.dumps({
            "schema": "merlin.graph-state-schedule.explicit-domain-refusal.v1", "status": "unsupported",
            "kind": "ScheduleDomainRefusal", "reason": str(error), "qualified": False,
        }, indent=2) + "\n")
        raise SystemExit(1) from error
    except Exception as error:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "UNAVAILABLE.json").write_text(json.dumps({
            "status": "UNAVAILABLE", "kind": type(error).__name__, "reason": str(error), "cause": "UNKNOWN",
            "semantic_refusal": False, "qualified": False,
        }, indent=2) + "\n")
        traceback.print_exc()
        raise SystemExit(2) from error
    print(json.dumps({"each_stage_changed": result["each_stage_changed"], "each_order_changed": result["each_order_changed"], "Lean_checked": False}))


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
