"""Bounded compiler diagnostics from original SSA and actual compiler products.

These receipts grant no optimization permission. Source opportunities are plans;
validated instruction images are emitted compiler products, not linked execution
or measured savings. No tensor contents, reference outputs or timing coefficients
are read. The operator adapter may project the pinned scalar observations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.transforms.certificates import ir_digest


def _section(module):
    return {
        "planned": {}, "emitted": {}, "source_ir_sha256": ir_digest(module),
        "maturity": "source_analysis_only", "physical_savings_claimed": False,
    }


def packing(module, requests=()):
    """Observe explicit format requests and source-valid invocation-local reuse.

    A repeated immutable SSA read is not a physical pack call. No format or
    cross-invocation invariance is guessed. Packed byte counts require an actual
    selected representation/producer, so remain absent at this analysis boundary.
    """
    from merlin.llvmlower.prepared_operand_effects import validate_prepared_source_lifetime
    from merlin.llvmlower.tensor_preparation_identity import find_tensor_preparation_opportunities

    requests = tuple(requests)
    if len(requests) > 10000:
        raise ValueError("packing observation request budget exceeded")
    members = tuple(module.walk())
    ordinals = {operation: i for i, operation in enumerate(members)}
    if any(request.consumer not in ordinals for request in requests):
        raise ValueError("packing consumer is not in the observed module")
    result = _section(module)
    rows = []
    for opportunity in find_tensor_preparation_opportunities(requests):
        row = {"format_sha256": opportunity.format_sha256,
               "consumer_ordinals": [ordinals[op] for op in opportunity.consumers],
               "source_reads": len(opportunity.consumers),
               "lifetime_scope": "one_source_invocation", "cross_invocation_reuse": "unproved"}
        try:
            validate_prepared_source_lifetime(opportunity, module)
            row.update(status="source_eligible", reason="exact SSA views and intervening tensor effects checked",
                       candidate_prepare_calls_per_invocation=1)
        except ValueError as error:
            row.update(status="declined", reason=str(error)[:300])
        rows.append(row)
    admitted = [row for row in rows if row["status"] == "source_eligible"]
    if requests:
        result["planned"]["pack_calls"] = len(admitted)
    result.update(opportunities=rows, requested_source_reads=len(requests),
                  repeated_source_reads=sum(row["source_reads"] for row in admitted),
                  metric_definitions={"planned.pack_calls": "candidate preparation sites per invocation for selected source-valid groups; no packing emitted"},
                  missing=["physical representation byte extent", "actual preparation emission", "cross-invocation immutable ownership"])
    return result


def lifetimes(outlined, *, weight_args=(), entry="forward"):
    """Plan from the real outlined driver, retaining last-use and scope limits.

    Stored-argument identities must come from the caller's selected capture
    manifest, for example stream_plan.weight_args_beside(capture). An omitted
    binding proves no cross-invocation prepack. Arena capacity is never renamed
    to peak-live bytes, and no planned reuse is reported as emitted reuse.
    """
    from merlin.xdsl_dialects.lowering.arena_plan import _live_ranges, plan_arena
    from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program, verify_program
    from merlin.xdsl_dialects.lowering.stream_plan import plan as stream_plan

    result = _section(outlined.module)
    program = build_dispatch_program(outlined, entry=entry)
    failures = verify_program(program)
    if failures:
        raise ValueError("invalid source dispatch program: " + "; ".join(failures))
    if len(program.nodes) > 10000:
        raise ValueError("lifetime observation node budget exceeded")
    weights = tuple(sorted(set(weight_args)))
    if any(type(index) is not int or index not in program.args for index in weights):
        raise ValueError("stored argument binding is outside the actual driver ABI")
    # The current arena prototype assumes eight-byte index elements. Do not let
    # that default enter a portable diagnostic without an actual target premise.
    if any(buffer.dtype == "index" for buffer in program.buffers.values()):
        raise ValueError("index-valued arena buffers require a selected width; observation remains unknown")
    memory = plan_arena(program)
    first, last = _live_ranges(program)
    arena_ids = tuple(memory.offsets)
    live_bytes = max((sum(memory.sizes[bid] for bid in arena_ids if first.get(bid, 0) <= i <= last.get(bid, 0))
                      for i in range(len(program.nodes) + 1)), default=0)
    reused = sum(any(other != bid and memory.offsets[other] < memory.offsets[bid] + memory.sizes[bid]
                     and memory.offsets[bid] < memory.offsets[other] + memory.sizes[other]
                     for other in arena_ids) for bid in arena_ids)
    stream = stream_plan(program, outlined.dispatches, weight_args=weights)
    unknown_placement = [d.symbol for d in outlined.dispatches if not d.placement]
    if unknown_placement:
        # The prototype defaults unspecified placements to host. Preserve that
        # uncertainty instead of publishing invented zero device fences.
        stream["nodes"] = {"dispatch": program.n_dispatches, "device": None, "host": None}
        for field in ("lifetimes", "staging", "fences", "overlap"):
            stream[field] = None
        stream["placement_status"] = "unknown"
        stream["unselected_dispatches"] = unknown_placement
    else:
        stream["placement_status"] = "selected"
    result["planned"] = {"allocations": len(arena_ids), "peak_live_bytes": live_bytes,
                         "reused_buffers": reused}
    result.update(arena_plan=memory.to_dict(),
                  live_ranges=[{"buffer": bid, "first_definition": first.get(bid), "last_use": last.get(bid)} for bid in arena_ids],
                  stream_plan=stream,
                  stored_argument_indices=list(weights),
                  metric_definitions={"planned.allocations": "intermediate logical buffers, excluding caller inputs/results",
                                      "planned.peak_live_bytes": "aligned intermediate bytes simultaneously live under source last-use intervals",
                                      "planned.reused_buffers": "logical buffers whose planned byte range overlaps another nonconcurrent buffer"},
                  missing=["actual emitted allocation/free placement", "linked arena binding", "runtime peak and spill observations"])
    return result


def connected_fusion(module, *, max_candidates=1024):
    """Discover adjacent three-operation fan-outs and retain exact refusals.

    This is the existing closed multi-output pointwise reader, not a claim of
    general contraction/attention fusion legality or an instruction realization.
    """
    from merlin.compile.scheduling.multioutput_fusion import read_source_multioutput
    from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise

    members = tuple(module.walk())
    if len(members) > 10000 or type(max_candidates) is not int or not 0 < max_candidates <= 10000:
        raise ValueError("fusion observation budget exceeded")
    ordinals = {operation: i for i, operation in enumerate(members)}
    blocks = tuple(dict.fromkeys(op.parent for op in members if op.parent is not None))
    rows = []
    for block in blocks:
        operations = tuple(block.ops)
        for position, producer in enumerate(operations):
            if producer.name != "linalg.generic" or len(producer.results) != 1:
                continue
            users = {use.operation for use in producer.results[0].uses}
            if len(users) < 2:
                continue
            adjacent = operations[position:position + 3]
            if len(adjacent) != 3:
                continue
            if len(rows) == max_candidates:
                raise ValueError("fusion candidate budget exceeded; no silently truncated roster")
            row = {"operation_ordinals": [ordinals[op] for op in adjacent],
                   "producer_observers": len(users), "legal_scope": "adjacent_pointwise_two_output_fanout"}
            try:
                selected = read_source_multioutput(module, adjacent)
                row.update(status="source_eligible", reason="source dependencies, observers, typed scalar bodies and output ordering checked",
                           shape=list(selected.shape), live_outputs=len(selected.outputs))
            except (ValueError, UnsupportedElementwise) as error:
                row.update(status="declined", reason=str(error)[:300])
            rows.append(row)
    result = _section(module)
    result["planned"] = {"source_regions": sum(row["status"] == "source_eligible" for row in rows)}
    result.update(candidates=rows, metric_definitions={"planned.source_regions": "eligible adjacent fan-out compositions under the named closed source reader"},
                  missing=["selected instruction composition law", "actual emitted image", "complete numeric qualification", "transfer and intermediate-storage realization"])
    return result


def emitted_multioutput(compilation):
    """Observe an actually produced, revalidated native instruction composition.

    This is explicitly an instruction-image boundary, not the final linked ELF.
    The existing issued-object guard and independent composition checker remain
    authoritative; a dictionary or arbitrary receipt cannot stand in for them.
    """
    from merlin.compile.scheduling.multioutput_fusion import MultiOutputCompilation

    if type(compilation) is not MultiOutputCompilation:
        raise ValueError("actual issued multi-output compiler product required")
    compilation.revalidate()
    images = compilation.images()
    return {"planned": {}, "emitted": {"emitted_regions": 1},
            "maturity": "validated_instruction_image_not_linked_execution",
            "physical_savings_claimed": False,
            "source_ir_sha256": ir_digest(compilation.source),
            "instruction_images": [{"sha256": hashlib.sha256(image).hexdigest(), "bytes": len(image)} for image in images],
            "missing": ["linked executable identity", "runtime transfer counts", "measured savings"]}



def install_prepared_observed(module, plan, writer_reports, **selection):
    """Run the existing guarded preparation emitter and retain its actual sites.

    All source/effect/physical-validator requirements are delegated unchanged.
    Static preparation calls do not establish successful runtime preparation or
    bytes saved; fallback remains part of the original consumer contract.
    """
    from xdsl.dialects import func
    from merlin.llvmlower.prepared_operand_schedule import install_prepared_borrows

    before = ir_digest(module)
    report = install_prepared_borrows(module, plan, writer_reports, **selection)
    symbol = selection["prepare_symbol"]
    calls = [op for op in plan.owner.body.block.ops
             if isinstance(op, func.CallOp) and op.callee.root_reference.data == symbol]
    if len(calls) != report["epochs"]:
        raise ValueError("actual prepare sites differ from native installation receipt")
    observation = {"planned": {}, "emitted": {"pack_calls": len(calls)},
                   "source_ir_sha256": before, "emitted_ir_sha256": ir_digest(module),
                   "maturity": "installed_source_schedule_not_linked_execution",
                   "physical_savings_claimed": False, "native_installation": report,
                   "metric_definitions": {"emitted.pack_calls": "installed static preparation sites per source invocation; runtime success and physical writes unobserved"},
                   "missing": ["actual successful runtime preparations", "linked physical producer", "cross-invocation reuse"]}
    return report, observation


def bind_arena_observed(llvm_ir, *, entry="forward"):
    """Run the native guarded LLVM binder, returning actual rewritten IR + receipt.

    Byte capacity comes from the emitted arena object. It is neither runtime
    peak memory nor proof that a source-level last-use plan reached the binary.
    Native refusals and original free/CFG lifetime checks remain unchanged.
    """
    from merlin.llvmlower.arena_bind import bind_arena

    lowered, report = bind_arena(llvm_ir, entry=entry)
    observation = {"planned": {}, "emitted": {},
                   "input_llvm_sha256": hashlib.sha256(llvm_ir.encode()).hexdigest(),
                   "emitted_llvm_sha256": hashlib.sha256(lowered.encode()).hexdigest(),
                   "maturity": "rewritten_llvm_not_linked_execution",
                   "physical_savings_claimed": False, "native_binding": report.to_dict(),
                   "metric_definitions": {"emitted.arena_bytes": "actual introduced static arena global capacity",
                                          "emitted.bound_heap_allocations": "original heap sites replaced by arena offsets under native CFG/free proof"},
                   "missing": ["linked executable identity", "runtime peak memory", "spill and execution observations"]}
    if report.bound:
        observation["emitted"] = {"arena_bytes": report.arena_bytes, "bound_heap_allocations": report.bound}
    return lowered, observation


def write_receipt(path, *, compiler_sha256, capsule_sha256, scope_sha256, sections):
    """Persist exact producer outputs and return adapter-ready pinned sections.

    Caller pins select context, not correctness authority. Keep the resulting
    artifact in the same candidate snapshot as its source/compiler inputs.
    """
    allowed = {"packing_reuse", "crosslayer_lifetimes", "connected_fusion"}
    pins = {"compiler_sha256": compiler_sha256, "capsule_sha256": capsule_sha256, "scope_sha256": scope_sha256}
    if any(not is_sha256(pin) for pin in pins.values()) or set(sections) - allowed:
        raise ValueError("complete context pins and known compiler observation sections required")
    path = Path(path)
    if path.exists() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("compiler receipt must be a new direct artifact")
    document = {"schema": "merlin.compiler_source_observations.v1", "pins": pins,
                "promotion_authority": False, "sections": sections}
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    artifact = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    return {name: {"pins": pins, "artifact": artifact, "planned": section["planned"], "emitted": section["emitted"]}
            for name, section in sections.items()}
