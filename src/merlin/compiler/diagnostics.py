"""Compile debugging from a command line: list the stages, dump their IR, stop after one, keep a trace.

The same options on ``merlin-compile``, on the whole-model builder's CLI and (as ``build_options``
keys) on a launch config:

* ``--list-stages`` prints every stage the compiler names, read off the pipelines themselves: the
  stages each pipeline declares beside the code that records them
  (:func:`merlin.compiler.ir.compile_trace.declare`), and the native passes of each upstream pass pipeline
  as the lowering builds it, with this build's optional-pass selection applied;
* ``--dump-ir-after S`` / ``--dump-ir-before S`` (repeatable, comma-separated, ``all``) write the IR at
  those stages; ``S#N`` names the Nth time a stage is reached (``mlir:canonicalize#2``);
* ``--stop-after S`` writes the IR at ``S`` and ends the compile there, exit 0, with no final artifact;
* ``--trace-dir DIR`` holds the dumps, ``trace.json`` (every stage reached, in order, with its file
  and seconds), ``pipeline.txt`` and the xDSL pass log. Any of the dump/stop options without it
  writes the trace under ``out/artifacts/probes/compile-trace/<target>/``.

Nothing here compiles anything or knows a target: it builds the request the recording points read.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256
from merlin.compiler.ir import compile_trace as T
from merlin.compiler.transforms.certificates import ir_digest

# ---- inspect -----------------------------------------------------------------------------------

#: The build-option keys a launch config spells these options with.
OPTION_KEYS = ("trace_dir", "dump_ir_after", "dump_ir_before", "stop_after")


def add_arguments(ap: argparse.ArgumentParser, *, list_stages: bool = True) -> None:
    """The compile-debugging options, on any parser."""
    group = ap.add_argument_group("compile debugging (see --list-stages)")
    if list_stages:
        group.add_argument(
            "--list-stages", action="store_true", help="list every stage --stop-after/--dump-ir-* can name, and exit"
        )
    group.add_argument(
        "--dump-ir-after",
        action="append",
        default=[],
        metavar="STAGE",
        help="write the IR after STAGE (repeatable, comma-separated; 'all'; STAGE#N for its Nth occurrence)",
    )
    group.add_argument(
        "--dump-ir-before", action="append", default=[], metavar="STAGE", help="write the IR before STAGE (as above)"
    )
    group.add_argument(
        "--stop-after",
        metavar="STAGE",
        help="write the IR at STAGE and stop there: exit 0, no final artifact",
    )
    group.add_argument(
        "--trace-dir",
        type=Path,
        help="a new directory for the per-stage IR, trace.json, pipeline.txt and the pass log "
        "(default with any dump/stop option: out/artifacts/probes/compile-trace/<target>/<timestamp>)",
    )


# ------------------------------------------------------------------------------------------- stages


def _load_pipelines() -> None:
    """Import every module that declares a pipeline, so the declarations are present."""
    import merlin.analysis.whole_model.build.stages  # noqa: F401 -- whole-model builder
    import merlin.cli.compile_workload  # noqa: F401 -- frontend
    import merlin.compiler.dialects.lowering.pipeline  # noqa: F401 -- staged route
    import merlin.compiler.lower.model.codegen  # noqa: F401 -- object, link
    import merlin.compiler.lower.model.lower  # noqa: F401 -- llvm route (with its xDSL rewrites)


def native_pipelines(features: frozenset[str] | None = None) -> dict[str, list[str]]:
    """Each upstream pass pipeline the LLVM route can run, as its pass names in order: the serial one,
    the OpenMP one, and the vectorizing one, built by the same functions a lowering calls, with the
    active optional-pass selection applied. A package's own features can splice more passes in; those
    are reached as ``mlir:<pass>`` stages all the same."""
    from pathlib import PurePath

    from merlin.compiler.lower.native import pipeline as P
    from merlin.compiler.lower.options.optional_passes import selected_features
    from merlin.compiler.optimizations.registry import normalize

    feats = normalize(selected_features(features)) or frozenset()
    return {
        "serial": T.split_pipeline(P._upstream_pipeline(feats)),
        "openmp": T.split_pipeline(P._parallel_pipeline(feats)),
        # The schedule path is only an option value of the transform pass; the pass list is the same.
        "vector": T.split_pipeline(P.build_rvv_pipeline(PurePath("<schedule>"), features=feats)),
    }


#: The order the routes are SHOWN in (front door, the two lowering routes, codegen, the whole-model
#: builder); a pipeline not named here is shown after them. Presentation only: the stages are the
#: pipelines' own.
_SHOWN_FIRST = ("frontend", "staged", "llvm", "codegen", "whole-model")


def stage_rows(features: frozenset[str] | None = None) -> list[dict[str, Any]]:
    """Every nameable stage, in pipeline order: ``{stage, pipeline, entry, summary}``."""
    _load_pipelines()
    rows = []
    shown = sorted(
        T.declared(),
        key=lambda d: _SHOWN_FIRST.index(d["pipeline"]) if d["pipeline"] in _SHOWN_FIRST else len(_SHOWN_FIRST),
    )
    for decl in shown:
        for stage in decl["stages"]:
            rows.append(
                {"stage": stage, "pipeline": decl["pipeline"], "entry": decl["entry"], "summary": decl["summary"]}
            )
        if decl["pipeline"] == "llvm":
            seen: set[str] = set()
            for variant, passes in native_pipelines(features).items():
                for name in T.with_ordinals(passes):
                    if name in seen:
                        continue
                    seen.add(name)
                    rows.append(
                        {
                            "stage": T.MLIR_PREFIX + name,
                            "pipeline": f"llvm ({variant} passes)",
                            "entry": "merlin.compiler.lower.native.pipeline",
                            "summary": "a native pass; its IR is the printer's dump after (or before) it",
                        }
                    )
    return rows


def known_stages() -> set[str]:
    _load_pipelines()
    return {stage for decl in T.declared() for stage in decl["stages"]}


def list_stages(a: argparse.Namespace) -> int:
    rows = stage_rows()
    if getattr(a, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    current = None
    for row in rows:
        if row["pipeline"] != current:
            current = row["pipeline"]
            if not current.startswith("llvm ("):
                print(f"\n{current}: {row['summary']}  [{row['entry']}]")
            else:
                print(f"\n{current}:")
        print(f"  {row['stage']}")
    print(
        "\nName a stage with --stop-after / --dump-ir-after / --dump-ir-before; STAGE#N is its Nth occurrence "
        "(mlir:canonicalize#2). A route reaches only its own stages: the staged route never reaches mlir:*."
    )
    return 0


# ------------------------------------------------------------------------------------------ request


def default_trace_dir(target: str, workload: str) -> Path:
    """``out/artifacts/probes/compile-trace/<target>/<TS>_<sha7>_<workload>``: a one-off diagnostic."""
    from merlin.common.artifacts import git_sha7, utc_stamp
    from merlin.common.paths import artifacts_dir

    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in workload) or "compile"
    return artifacts_dir() / "probes" / "compile-trace" / target / f"{utc_stamp()}_{git_sha7()}_{safe}"


def make_request(
    *,
    trace_dir: str | Path | None,
    dump_after: Sequence[str] = (),
    dump_before: Sequence[str] = (),
    stop_after: str | None = None,
    default_dir: Callable[[], Path] | None = None,
) -> T.Request | None:
    """The validated request, or None when no option asks for a trace. Raises :class:`T.TraceError`.

    ``default_dir`` is called only when a trace is asked for without a directory."""
    if not (trace_dir or dump_after or dump_before or stop_after):
        return None
    known = known_stages()
    after = T.validate(dump_after, known, allow_all=True)
    before = T.validate(dump_before, known, allow_all=True)
    stop = T.validate([stop_after], known, allow_all=False) if stop_after else ()
    if len(stop) > 1:
        raise T.TraceError("--stop-after names one stage")
    directory = Path(trace_dir) if trace_dir else (default_dir() if default_dir is not None else None)
    if directory is None:
        raise T.TraceError("no trace directory; pass --trace-dir")
    return T.Request(
        directory=str(directory), dump_after=after, dump_before=before, stop_after=stop[0] if stop else None
    )


def request_from_args(
    ap: argparse.ArgumentParser, a: argparse.Namespace, *, target: str, workload: str
) -> T.Request | None:
    """The request the parsed options name (``ap.error`` on an unknown stage)."""
    try:
        return make_request(
            trace_dir=a.trace_dir,
            dump_after=a.dump_ir_after,
            dump_before=a.dump_ir_before,
            stop_after=a.stop_after,
            default_dir=lambda: default_trace_dir(target, workload),
        )
    except T.TraceError as exc:
        ap.error(str(exc))


def request_from_options(options: Mapping[str, Any], *, target: str, workload: str) -> T.Request | None:
    """The request a launch config's ``build_options`` name (:data:`OPTION_KEYS`)."""

    def listed(key: str) -> list[str]:
        value = options.get(key)
        return [] if value is None else ([value] if isinstance(value, str) else [str(v) for v in value])

    return make_request(
        trace_dir=options.get("trace_dir"),
        dump_after=listed("dump_ir_after"),
        dump_before=listed("dump_ir_before"),
        stop_after=options.get("stop_after"),
        default_dir=lambda: default_trace_dir(target, workload),
    )


def _install_pass_log(request: T.Request) -> dict[str, Any]:
    """Record xDSL pass invocations into the trace's pass log (the existing ``MERLIN_PASS_LOG`` recorder)."""
    try:
        from merlin.compiler.dialects.lowering import passes

        passes.install_pass_recorder()
    except ImportError as exc:
        return {"pass_log_recorder": f"not installed: {exc}"}
    return {"pass_log_recorder": "installed"}


@contextlib.contextmanager
def opened(request: T.Request | None, command: Sequence[str]) -> Iterator[T.Request | None]:
    """A trace session for ``request`` (with the pass log recorded), or nothing when it is None."""
    if request is None:
        yield None
        return
    with T.session(request, command=list(command), on_open=_install_pass_log) as active:
        yield active


def stopped(tool: str, stop: T.StopAfterStage, *, as_json: bool = False) -> int:
    """Report a requested stop: where the IR is, and that no artifact was produced. Exit status 0."""
    if as_json:
        print(
            json.dumps(
                {
                    "tool": tool,
                    "status": "stopped",
                    "stage": stop.stage,
                    "files": list(stop.files),
                    "trace": str(Path(stop.directory) / T.INDEX),
                    "note": stop.note,
                },
                indent=2,
            )
        )
    else:
        print(stop.message(tool))
    return 0


def not_reached(request: T.Request | None) -> str | None:
    """Why a completed compile is not what was asked: the stop stage was never reached (None otherwise)."""
    if request is None or request.stop_after is None:
        return None
    return (
        f"--stop-after {request.stop_after} was never reached: this compile's route does not pass through it "
        f"(see {Path(request.directory) / T.INDEX} for the stages it did reach)"
    )


# ---- observations ------------------------------------------------------------------------------
# Bounded compiler diagnostics from original SSA and actual compiler products.
#
# These receipts grant no optimization permission. Source opportunities are plans;
# validated instruction images are emitted compiler products, not linked execution
# or measured savings. No tensor contents, reference outputs or timing coefficients
# are read. The operator adapter may project the pinned scalar observations.


def _section(module):
    return {
        "planned": {},
        "emitted": {},
        "source_ir_sha256": ir_digest(module),
        "maturity": "source_analysis_only",
        "physical_savings_claimed": False,
    }


def packing(module, requests=()):
    """Observe explicit format requests and source-valid invocation-local reuse.

    A repeated immutable SSA read is not a physical pack call. No format or
    cross-invocation invariance is guessed. Packed byte counts require an actual
    selected representation/producer, so remain absent at this analysis boundary.
    """
    from merlin.compiler.lower.source.prepared import (
        find_tensor_preparation_opportunities,
        validate_prepared_source_lifetime,
    )

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
        row = {
            "format_sha256": opportunity.format_sha256,
            "consumer_ordinals": [ordinals[op] for op in opportunity.consumers],
            "source_reads": len(opportunity.consumers),
            "lifetime_scope": "one_source_invocation",
            "cross_invocation_reuse": "unproved",
        }
        try:
            validate_prepared_source_lifetime(opportunity, module)
            row.update(
                status="source_eligible",
                reason="exact SSA views and intervening tensor effects checked",
                candidate_prepare_calls_per_invocation=1,
            )
        except ValueError as error:
            row.update(status="declined", reason=str(error)[:300])
        rows.append(row)
    admitted = [row for row in rows if row["status"] == "source_eligible"]
    if requests:
        result["planned"]["pack_calls"] = len(admitted)
    result.update(
        opportunities=rows,
        requested_source_reads=len(requests),
        repeated_source_reads=sum(row["source_reads"] for row in admitted),
        metric_definitions={
            "planned.pack_calls": "candidate preparation sites per invocation for selected source-valid groups; no packing emitted"
        },
        missing=[
            "physical representation byte extent",
            "actual preparation emission",
            "cross-invocation immutable ownership",
        ],
    )
    return result


def lifetimes(outlined, *, weight_args=(), entry="forward"):
    """Plan from the real outlined driver, retaining last-use and scope limits.

    Stored-argument identities must come from the caller's selected capture
    manifest, for example stream_plan.weight_args_beside(capture). An omitted
    binding proves no cross-invocation prepack. Arena capacity is never renamed
    to peak-live bytes, and no planned reuse is reported as emitted reuse.
    """
    from merlin.compiler.partition.dispatch_program import build_dispatch_program, verify_program
    from merlin.compiler.partition.memory_plan import _live_ranges, plan_arena
    from merlin.compiler.partition.memory_plan import plan as stream_plan

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
    live_bytes = max(
        (
            sum(memory.sizes[bid] for bid in arena_ids if first.get(bid, 0) <= i <= last.get(bid, 0))
            for i in range(len(program.nodes) + 1)
        ),
        default=0,
    )
    reused = sum(
        any(
            other != bid
            and memory.offsets[other] < memory.offsets[bid] + memory.sizes[bid]
            and memory.offsets[bid] < memory.offsets[other] + memory.sizes[other]
            for other in arena_ids
        )
        for bid in arena_ids
    )
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
    result["planned"] = {"allocations": len(arena_ids), "peak_live_bytes": live_bytes, "reused_buffers": reused}
    result.update(
        arena_plan=memory.to_dict(),
        live_ranges=[
            {"buffer": bid, "first_definition": first.get(bid), "last_use": last.get(bid)} for bid in arena_ids
        ],
        stream_plan=stream,
        stored_argument_indices=list(weights),
        metric_definitions={
            "planned.allocations": "intermediate logical buffers, excluding caller inputs/results",
            "planned.peak_live_bytes": "aligned intermediate bytes simultaneously live under source last-use intervals",
            "planned.reused_buffers": "logical buffers whose planned byte range overlaps another nonconcurrent buffer",
        },
        missing=[
            "actual emitted allocation/free placement",
            "linked arena binding",
            "runtime peak and spill observations",
        ],
    )
    return result


def connected_fusion(module, *, max_candidates=1024):
    """Discover adjacent three-operation fan-outs and retain exact refusals.

    This is the existing closed multi-output pointwise reader, not a claim of
    general contraction/attention fusion legality or an instruction realization.
    """
    from merlin.compiler.schedule.elementwise.fusion import UnsupportedElementwise
    from merlin.compiler.transforms.source_rewrites.multioutput import read_source_multioutput

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
            adjacent = operations[position : position + 3]
            if len(adjacent) != 3:
                continue
            if len(rows) == max_candidates:
                raise ValueError("fusion candidate budget exceeded; no silently truncated roster")
            row = {
                "operation_ordinals": [ordinals[op] for op in adjacent],
                "producer_observers": len(users),
                "legal_scope": "adjacent_pointwise_two_output_fanout",
            }
            try:
                selected = read_source_multioutput(module, adjacent)
                row.update(
                    status="source_eligible",
                    reason="source dependencies, observers, typed scalar bodies and output ordering checked",
                    shape=list(selected.shape),
                    live_outputs=len(selected.outputs),
                )
            except (ValueError, UnsupportedElementwise) as error:
                row.update(status="declined", reason=str(error)[:300])
            rows.append(row)
    result = _section(module)
    result["planned"] = {"source_regions": sum(row["status"] == "source_eligible" for row in rows)}
    result.update(
        candidates=rows,
        metric_definitions={
            "planned.source_regions": "eligible adjacent fan-out compositions under the named closed source reader"
        },
        missing=[
            "selected instruction composition law",
            "actual emitted image",
            "complete numeric qualification",
            "transfer and intermediate-storage realization",
        ],
    )
    return result


def emitted_multioutput(compilation):
    """Observe an actually produced, revalidated native instruction composition.

    This is explicitly an instruction-image boundary, not the final linked ELF.
    The existing issued-object guard and independent composition checker remain
    authoritative; a dictionary or arbitrary receipt cannot stand in for them.
    """
    from merlin.compiler.transforms.source_rewrites.multioutput import MultiOutputCompilation

    if type(compilation) is not MultiOutputCompilation:
        raise ValueError("actual issued multi-output compiler product required")
    compilation.revalidate()
    images = compilation.images()
    return {
        "planned": {},
        "emitted": {"emitted_regions": 1},
        "maturity": "validated_instruction_image_not_linked_execution",
        "physical_savings_claimed": False,
        "source_ir_sha256": ir_digest(compilation.source),
        "instruction_images": [{"sha256": hashlib.sha256(image).hexdigest(), "bytes": len(image)} for image in images],
        "missing": ["linked executable identity", "runtime transfer counts", "measured savings"],
    }


def install_prepared_observed(module, plan, writer_reports, **selection):
    """Run the existing guarded preparation emitter and retain its actual sites.

    All source/effect/physical-validator requirements are delegated unchanged.
    Static preparation calls do not establish successful runtime preparation or
    bytes saved; fallback remains part of the original consumer contract.
    """
    from xdsl.dialects import func

    from merlin.compiler.lower.source.prepared import install_prepared_borrows

    before = ir_digest(module)
    report = install_prepared_borrows(module, plan, writer_reports, **selection)
    symbol = selection["prepare_symbol"]
    calls = [
        op
        for op in plan.owner.body.block.ops
        if isinstance(op, func.CallOp) and op.callee.root_reference.data == symbol
    ]
    if len(calls) != report["epochs"]:
        raise ValueError("actual prepare sites differ from native installation receipt")
    observation = {
        "planned": {},
        "emitted": {"pack_calls": len(calls)},
        "source_ir_sha256": before,
        "emitted_ir_sha256": ir_digest(module),
        "maturity": "installed_source_schedule_not_linked_execution",
        "physical_savings_claimed": False,
        "native_installation": report,
        "metric_definitions": {
            "emitted.pack_calls": "installed static preparation sites per source invocation; runtime success and physical writes unobserved"
        },
        "missing": ["actual successful runtime preparations", "linked physical producer", "cross-invocation reuse"],
    }
    return report, observation


def bind_arena_observed(llvm_ir, *, entry="forward"):
    """Run the native guarded LLVM binder, returning actual rewritten IR + receipt.

    Byte capacity comes from the emitted arena object. It is neither runtime
    peak memory nor proof that a source-level last-use plan reached the binary.
    Native refusals and original free/CFG lifetime checks remain unchanged.
    """
    from merlin.compiler.lower.host.arena import bind_arena

    lowered, report = bind_arena(llvm_ir, entry=entry)
    observation = {
        "planned": {},
        "emitted": {},
        "input_llvm_sha256": hashlib.sha256(llvm_ir.encode()).hexdigest(),
        "emitted_llvm_sha256": hashlib.sha256(lowered.encode()).hexdigest(),
        "maturity": "rewritten_llvm_not_linked_execution",
        "physical_savings_claimed": False,
        "native_binding": report.to_dict(),
        "metric_definitions": {
            "emitted.arena_bytes": "actual introduced static arena global capacity",
            "emitted.bound_heap_allocations": "original heap sites replaced by arena offsets under native CFG/free proof",
        },
        "missing": ["linked executable identity", "runtime peak memory", "spill and execution observations"],
    }
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
    document = {
        "schema": "merlin.compiler_source_observations.v1",
        "pins": pins,
        "promotion_authority": False,
        "sections": sections,
    }
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    artifact = {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    return {
        name: {"pins": pins, "artifact": artifact, "planned": section["planned"], "emitted": section["emitted"]}
        for name, section in sections.items()
    }
