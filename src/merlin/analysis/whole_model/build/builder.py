"""The measurement service's production builder, over the core whole-model build.

:class:`.service.MeasurementService` takes a builder ``build(package_dir, *, target, out_dir, **options)
-> record``.  This one calls :func:`merlin.analysis.whole_model.build.whole_model_build.build` (an OPEN model -- host regions
between its groups -- goes through :func:`merlin.analysis.whole_model.build.open_execution.service_build`) and turns
its record into the shape the service verifies and the verdict reads: the ELF and its digest, the ABI
header the program was compiled against, the oracle's expectations, the per-group route census, and
the program's own protocol spellings.

ONE THING IS ADDED THAT THE BUILD RECORD DOES NOT CARRY: which groups each group READS
(``inputs_from``).  The verdict compares an exact group only with a basis whose inputs it shares; it is
derived from the target driver's own extraction of the same capsule -- each step's buffers and which
step produced them -- never from group numbering.  Every option arrives as data from the launch
config's ``build_options``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.compiler.ir.boundary import free_values

# ---- whole_model_builder -----------------------------------------------------------------------

#: The protocol keys the verdict reads, taken from the target driver's own table at build time.
_PROTOCOL_KEYS = (
    "invocations",
    "group",
    "argmax",
    "bounded",
    "full_model",
    "bracket_sum",
    "uncounted",
    "window_end",
    "where",
    "local",
    "witness",
)

#: The build's own spelling of a tolerance comparison, and the verdict's.
_COMPARE = {"bounded": "bounded_int", "exact": "exact"}

#: Build intermediates the linked ELF already contains; the service removes them after the build.
PRUNABLE = (
    "program/*.bin",
    "program/*.o",
    "objects/*",
    "lower/*.generated",
    "lower/*.artifact.txt",
    # The prefetch pass's lowered IR (340-470 MB per candidate); the statement's replies are what it saved.
    "lower/prefetch",
)


def _reads(model: Mapping[str, Any]) -> dict[str, list[str]]:
    """``{group: [groups whose output it reads]}`` from the driver's extracted steps."""
    steps = [s for s in model.get("steps") or () if isinstance(s, Mapping)]
    producer = {str(s["out"]): str(s["group"]) for s in steps if s.get("out")}
    reads: dict[str, list[str]] = {}
    for step in steps:
        group = str(step["group"])
        # A fused region reads exactly what its members read from outside it (`reads`).
        buffers = [step.get(key) for key in ("in", "lhs", "rhs") if step.get(key)] + list(step.get("reads") or ())
        reads[group] = sorted(
            {producer[str(b)] for b in buffers if str(b) in producer and producer[str(b)] != group},
            key=lambda g: (len(g), g),
        )
    return reads


def _operands(step: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A tolerance group's operand scales as the driver's program applies them (and its saturation
    range when the driver states one; else the verdict derives it from the output width)."""
    if not isinstance(step, Mapping) or any(step.get(k) is None for k in ("lhs_load", "rhs_load", "readout")):
        return None
    operands = {k: step[k] for k in ("lhs_load", "rhs_load", "readout")} | {"relu": bool(step.get("relu"))}
    if isinstance(step.get("saturate"), (list, tuple)) and len(step["saturate"]) == 2:
        operands["saturate"] = [int(v) for v in step["saturate"]]
    return operands


def _output_widths(out_dir: str | Path) -> dict[str, int]:
    """Each group's committed output element width, from the build's own memory map."""
    path = Path(out_dir) / "memory_map.json"
    if not path.is_file():
        return {}
    layout = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["group"]): int(row["element_bytes"])
        for row in layout.get("groups") or ()
        if isinstance(row, Mapping) and row.get("element_bytes") is not None
    }


def build(
    package_dir: str | Path,
    *,
    target: str,
    out_dir: str | Path,
    model_capsule: str,
    machine: str,
    header: str,
    header_sha256: str | None = None,
    verify: str = "on_target",
    jobs: int | None = None,
    timeout: int = 600,
    harness_overrides: list[str] | tuple[str, ...] = (),
    prohibited_roles: Sequence[str] = (),
    decline: Sequence[Any] = (),
    allow_passes: bool = False,
    allow_regions: bool = False,
    phase0_recipe: str | None = None,
    descriptor: str | None = None,
    chunk_ops: int | str | None = None,
    lowering_passes: Sequence[str] = (),
    only_groups: Sequence[Any] = (),
    trace_dir: str | None = None,
    dump_ir_after: Sequence[str] | str = (),
    dump_ir_before: Sequence[str] | str = (),
    stop_after: str | None = None,
) -> dict[str, Any]:
    """Build one candidate; every option up to ``lowering_passes`` is :func:`_build`'s.

    ``lowering_passes`` selects Merlin's optional lowering passes for this build by registry name
    (:mod:`merlin.compiler.lower.options.optional_passes`; ``-name`` turns off one that is on by default). It is a
    launch config's ``build_options`` entry like the rest, empty by default -- so a config that names
    none builds exactly as before -- and the selection is recorded in the result's ``notes``.

    The compile-debugging keys (:mod:`merlin.compiler.diagnostics`) are ``build_options`` too, all empty by
    default: ``only_groups`` builds a PARTIAL program (marked on its expectations, so every verdict and
    measurement refuses it), and ``trace_dir``/``dump_ir_after``/``dump_ir_before``/``stop_after`` keep a
    trace of the build. A build stopped at a stage produces no program, so here -- where a program is
    what the caller is owed -- the stop is a :class:`WholeModelBuildError` naming where the IR is.
    """
    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.compiler import diagnostics as debug
    from merlin.compiler.ir.compile_trace import StopAfterStage
    from merlin.compiler.lower.options import optional_passes

    options = {"trace_dir": trace_dir, "dump_ir_after": dump_ir_after, "dump_ir_before": dump_ir_before}
    trace = debug.request_from_options(
        {**options, "stop_after": stop_after}, target=target, workload=Path(model_capsule).name
    )
    selection = optional_passes.Selection.parse(list(lowering_passes))
    try:
        with (
            debug.opened(trace, ["merlin.analysis.whole_model.build.builder.build"]),
            optional_passes.applied(selection) as active,
        ):
            record = _build(
                package_dir,
                target=target,
                out_dir=out_dir,
                model_capsule=model_capsule,
                machine=machine,
                header=header,
                header_sha256=header_sha256,
                verify=verify,
                jobs=jobs,
                timeout=timeout,
                harness_overrides=harness_overrides,
                prohibited_roles=prohibited_roles,
                decline=decline,
                allow_passes=allow_passes,
                allow_regions=allow_regions,
                phase0_recipe=phase0_recipe,
                descriptor=descriptor,
                chunk_ops=chunk_ops,
                only_groups=only_groups,
            )
    except StopAfterStage as stop:
        raise WMB.WholeModelBuildError(stop.message("whole-model builder")) from None
    if active:
        record.setdefault("notes", {})["lowering_passes"] = active.spell()
    if trace is not None:
        record.setdefault("notes", {})["compile_trace"] = str(Path(trace.directory).absolute() / "trace.json")
    return record


def _build(
    package_dir: str | Path,
    *,
    target: str,
    out_dir: str | Path,
    model_capsule: str,
    machine: str,
    header: str,
    header_sha256: str | None = None,
    verify: str = "on_target",
    jobs: int | None = None,
    timeout: int = 600,
    harness_overrides: list[str] | tuple[str, ...] = (),
    prohibited_roles: Sequence[str] = (),
    decline: Sequence[Any] = (),
    allow_passes: bool = False,
    allow_regions: bool = False,
    phase0_recipe: str | None = None,
    descriptor: str | None = None,
    chunk_ops: int | str | None = None,
    only_groups: Sequence[Any] = (),
) -> dict[str, Any]:
    """``decline`` (op names / group indices) is CELL MODE's own hook: naming every group outside one
    cell routes them all to the target's library, so only the cell's own groups can move whatever this
    build's objective is scored on. Empty by default, so an ordinary whole-model build is unaffected.

    ``allow_passes`` and ``allow_regions`` are both OFF BY DEFAULT and are the launch config's own
    ``build_options`` switches (see :func:`merlin.analysis.whole_model.build.whole_model_build.build`): a package that never
    declares a whole-model pass or never asks for a region sees no change either way.

    ``phase0_recipe`` / ``descriptor`` name the corpus binding every group is stated under.
    ``chunk_ops`` (an open model only) bounds the open build's lowered functions; ``None`` keeps the
    unchunked program. ``"auto"`` derives the size from an open model's forward and asks nothing of a
    closed one (it has no host forward to cut), so one build option serves both.
    """
    from merlin.analysis.whole_model.build import inputs as PARTIAL
    from merlin.analysis.whole_model.build import open_program as WO
    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.analysis.whole_model.build import whole_model_build as _merlin_whole_model_build_1
    from merlin.runtime.backends import registry as backends

    if WO.is_open_model(model_capsule, target):
        if harness_overrides or decline or allow_passes or allow_regions or only_groups:
            raise WMB.WholeModelBuildError(
                "an open-model build takes no harness overrides, declines, passes, regions or only_groups"
            )
        return WO.service_build(
            package_dir,
            target=target,
            out_dir=out_dir,
            model_capsule=model_capsule,
            machine=machine,
            header=header,
            header_sha256=header_sha256,
            verify=verify,
            jobs=jobs,
            timeout=timeout,
            prohibited_roles=prohibited_roles,
            protocol_keys=_PROTOCOL_KEYS,
            phase0_recipe=phase0_recipe,
            descriptor=descriptor,
            **({"chunk_ops": chunk_ops} if chunk_ops is not None else {}),
        )

    if chunk_ops is not None and str(chunk_ops).strip().lower() != AUTO:
        raise WMB.WholeModelBuildError("chunk_ops bounds an open model's lowered functions; this model is closed")
    record = _merlin_whole_model_build_1.build(
        package_dir,
        model_capsule,
        target=target,
        machine=machine,
        header=header,
        header_sha256=header_sha256,
        out=out_dir,
        oracle=True,
        verify=verify,
        timeout=timeout,
        decline=decline,
        jobs=jobs,
        harness_overrides=harness_overrides,
        prohibited_roles=prohibited_roles,
        allow_passes=allow_passes,
        allow_regions=allow_regions,
        phase0_recipe=phase0_recipe,
        descriptor=descriptor,
        only_groups=only_groups,
    )
    oracle = json.loads(Path(record["oracle"]["path"]).read_text(encoding="utf-8"))
    capsule = WMB.load_model_capsule(model_capsule)
    driver = backends.whole_model_driver(target)
    (value,) = capsule.inputs.values()
    (golden,) = capsule.outputs.values()
    model = driver.program.extract(
        None,
        target,
        sources={
            "linalg": capsule.interface,
            "weights_manifest": capsule.weights_manifest,
            "weights": capsule.weights,
            "input": value,
            "golden": golden,
        },
    )
    # THE PROGRAM'S OWN STEPS: every fused region the build stated as one step is one here too, by the
    # driver's own function over the same rows, so the expectations name exactly the lines it prints.
    stepped = list(((record.get("regions") or {}).get("stepped")) or ())
    if stepped:
        model, refused = driver.program.region_steps(model, stepped)
        if refused:
            raise WMB.WholeModelBuildError(f"the build's own regions do not restate over its extraction: {refused}")
    # A REGION'S INTERNAL MEMBERS PRINT NO LINE: the region is graded once, at its boundary, whose
    # oracle digest is the boundary tensor's -- a function of the model, not of how it is grouped.
    internal = {str(g) for region in stepped for g in region["members"][:-1]}
    reads = _reads(model)
    widths = _output_widths(out_dir)
    steps = {str(s["group"]): s for s in model.get("steps") or () if isinstance(s, Mapping)}
    groups: dict[str, Any] = {}
    for key, expected in (oracle.get("groups") or {}).items():
        if str(key) in internal:
            continue
        compare = _COMPARE.get(str(expected.get("compare") or "exact"))
        if compare is None:
            raise ValueError(f"group {key} declares an unknown comparison {expected.get('compare')!r}")
        row = {"compare": compare, "sum": expected.get("sum"), "fnv1a": expected.get("fnv1a")}
        if compare == "bounded_int":
            row["bound_lsb"] = expected.get("bound_lsb")
            step = steps.get(key)
            # A region ending in a sum is bounded on its boundary member's own operands.
            row["operands"] = _operands(step["members"][-1] if step and step.get("kind") == "region" else step)
        if key not in reads:
            raise ValueError(f"group {key} of the oracle is not a step of the driver's extraction")
        row["inputs_from"] = reads[key]
        if key in widths:
            row["output_element_bytes"] = widths[key]
        groups[str(key)] = row
    routes = []
    for row in (record.get("attribution") or {}).get("per_group") or ():
        entry = {
            k: row.get(k)
            for k in ("group", "op", "on", "cause", "why", "shape", "gather", "region", "graded_at")
            if k in row
        }
        entry["lowering"] = row.get("shape")
        step = steps.get(str(row.get("group")))
        if row.get("on") != "package" and step is not None:
            # The library call the group fell back to, by the driver's own spelling of it.
            entry["call"] = driver.program._call(step).split("(", 1)[0].strip()
        routes.append(entry)
    return {
        "elf": record["elf"],
        "elf_sha256": record["elf_sha256"],
        "parameter_header_sha256": record["program"]["abi_header"]["sha256"],
        "expectations": {
            "groups": groups,
            "group_count": len(groups),
            # Each fused region's internal members, graded at their boundary (no line of their own).
            **(
                {"graded_at_boundary": {g: str(r["boundary"]) for r in stepped for g in map(str, r["members"][:-1])}}
                if stepped
                else {}
            ),
            "argmax": int(oracle["argmax"]),
            "source": f"{record['oracle']['path']} (golden argmax {oracle.get('golden_argmax')})",
            # A PARTIAL build carries its marker where every verdict reads, and is refused there.
            **({PARTIAL.MARKER: record[PARTIAL.MARKER]} if record.get(PARTIAL.MARKER) else {}),
        },
        "groups": routes,
        "protocol": {key: driver.program.UART[key] for key in _PROTOCOL_KEYS if key in driver.program.UART},
        "prunable": list(PRUNABLE),
        "notes": {
            "builder": "merlin.analysis.whole_model.build.stages",
            "verify": record.get("verify"),
            "attribution_counts": (record.get("attribution") or {}).get("counts"),
            "abi_header": record["program"]["abi_header"],
            "harness_overrides": record["program"].get("harness_overrides") or {},
            "build_record": str(Path(out_dir) / "whole_model_build.json"),
        },
        "provenance": record.get("provenance"),
        **({PARTIAL.MARKER: record[PARTIAL.MARKER]} if record.get(PARTIAL.MARKER) else {}),
    }


def build_reference(
    package_dir: str | Path,
    *,
    target: str,
    out_dir: str | Path,
    model_capsule: str,
    machine: str,
    header: str,
    header_sha256: str | None = None,
    verify: str = "on_target",
    jobs: int | None = None,
    timeout: int = 600,
    harness_overrides: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """The REFERENCE arm: the same driver program with the target's library answering every group.

    Built through the same driver, header assertion and oracle as :func:`build`, so a candidate and
    the bar differ only in who answered each group.  ``package_dir`` is ignored (it only names the
    job); the record says so.
    """
    from merlin.analysis.whole_model.build import products as WMH
    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.analysis.whole_model.build.oracle import _oracle
    from merlin.analysis.whole_model.build.whole_model_memory import memory_map
    from merlin.runtime.backends import registry as backends

    capsule = WMB.load_model_capsule(model_capsule)
    abi_header = WMH.machine_header(machine, header, header_sha256)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    driver = backends.whole_model_driver(target)
    (value,) = capsule.inputs.values()
    (golden,) = capsule.outputs.values()
    model = driver.program.extract(
        None,
        target,
        sources={
            "linalg": capsule.interface,
            "weights_manifest": capsule.weights_manifest,
            "weights": capsule.weights,
            "input": value,
            "golden": golden,
        },
    )
    overrides = {str(Path(o).resolve()): WMH._sha256(o) for o in harness_overrides}
    recipe = WMH._with_header(
        backends.harness_build_recipe(target), Path(header), out / "harness", [Path(o) for o in overrides]
    )
    receipt = driver.program.build(model, None, None, out / "program", recipe=recipe, verify=verify)
    headers = WMH._headers_read(receipt, out / "program" / "group_model_program.c")
    if abi_header["sha256"] not in headers.values():
        raise WMB.WholeModelBuildError("the reference program did not read the asserted parameter header")
    if any(digest not in headers.values() for digest in overrides.values()):
        raise WMB.WholeModelBuildError("the reference program did not read every harness override")
    oracle, _entry = _oracle(capsule, target=target, digest=driver.program.group_digest)
    (out / "oracle.json").write_text(json.dumps(oracle, indent=1) + "\n", encoding="utf-8")
    layout = memory_map(receipt["elf"], model, WMB.state(capsule, target=target))
    (out / "memory_map.json").write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    widths = _output_widths(out)
    # The fields a contract measurer reads off a build record (the memory map, the oracle, the capsule),
    # in the build's own record shape, so the reference arm can be run by the same measurers.
    record_path = out / "whole_model_build.json"
    record_path.write_text(
        json.dumps(
            {
                "schema": "whole_model_reference_build_v1",
                "target": target,
                "elf": receipt["elf"],
                "elf_sha256": receipt["elf_sha256"],
                "verify": verify,
                "memory_map": str(out / "memory_map.json"),
                "oracle": {"path": str(out / "oracle.json"), "argmax": oracle.get("argmax")},
                "capsule": {"name": capsule.name, "directory": str(capsule.directory)},
                # The toolchain the program was built with, in the candidate record's shape: a batch
                # links the reference as its control variant with exactly these.
                "program": {
                    "source_sha256": receipt.get("program_sha256"),
                    "compiler": receipt.get("compiler"),
                    "flags": receipt.get("flags"),
                    "link_flags": receipt.get("link_flags"),
                    "link_script": receipt.get("link_script"),
                    "abi_header": abi_header,
                    "harness_overrides": overrides,
                },
                "linked_objects": receipt.get("linked_objects") or [],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    reads = _reads(model)
    steps = {str(s["group"]): s for s in model.get("steps") or () if isinstance(s, Mapping)}
    groups = {}
    for key, expected in (oracle.get("groups") or {}).items():
        compare = _COMPARE[str(expected.get("compare") or "exact")]
        row = {
            "compare": compare,
            "sum": expected.get("sum"),
            "fnv1a": expected.get("fnv1a"),
            "inputs_from": reads[key],
        }
        if compare == "bounded_int":
            row["bound_lsb"] = expected.get("bound_lsb")
            row["operands"] = _operands(steps.get(key))
        if key in widths:
            row["output_element_bytes"] = widths[key]
        groups[str(key)] = row
    return {
        "elf": receipt["elf"],
        "elf_sha256": receipt["elf_sha256"],
        "parameter_header_sha256": abi_header["sha256"],
        "expectations": {"groups": groups, "group_count": len(groups), "argmax": int(oracle["argmax"])},
        "groups": [
            {
                "group": s["group"],
                "op": s.get("kind"),
                "on": "vendor",
                "call": driver.program._call(s).split("(", 1)[0].strip(),
                "why": "reference arm: the target's library answers every group",
            }
            for s in model.get("steps") or ()
        ],
        "protocol": {key: driver.program.UART[key] for key in _PROTOCOL_KEYS if key in driver.program.UART},
        "prunable": list(PRUNABLE),
        "notes": {
            "builder": "reference arm over merlin.analysis.whole_model.build.stages",
            "package_dir_ignored": str(package_dir),
            "harness_overrides": overrides,
            "build_record": str(record_path),
        },
    }


# ---- whole_model_chunks ------------------------------------------------------------------------
# Cut a whole model's flat ``@forward`` into bounded, sequentially-called functions.
#
# LLVM's compile cost on one huge function is superlinear in its size, so an open-model build may
# split the host program into chunks before lowering it (``build(chunk_ops=...)``). The cut moves
# only which function's text an op's clone sits in; nothing a value computes, or its order, changes.
#
# ``chunk_ops="auto"`` derives the size from the program itself (:func:`resolve_chunk_ops`): a forward
# body larger than :data:`DEFAULT_CHUNK_OPS` is cut at that size, and one that fits is left unchunked,
# byte for byte, and recorded as unchunked so its reference arm is the unchunked one too.

#: The ``chunk_ops`` value that derives the chunk size from the program (see :func:`resolve_chunk_ops`).
AUTO = "auto"
#: The chunk size ``"auto"`` cuts a large forward at. Measured on SmolVLA's open-model build (a
#: 1,551-call ``forward``): unchunked, its host program took over two hours to compile; cut at 1,000
#: ops, the whole build took about eleven minutes.
DEFAULT_CHUNK_OPS = 1000

#: The symbol of chunk ``i`` is ``CHUNK_PREFIX + str(i)``, for ``i`` in ``range(chunk_forward(...))``.
CHUNK_PREFIX = "merlin_forward_chunk_"

#: A cheap producer (compute_groups.classify's own ``_PRODUCERS``): never left as the LAST op of a
#: chunk, because the pipeline's one-shot-bufferize does not handle one of these -- a destination a
#: later op writes into, in the out-params/destination-passing sense -- crossing a function-call
#: boundary from its own write the way a dispatch's single ``committed`` value crosses one. Measured:
#: every chunk_ops tried on a real capture (14 through 835 resulting chunks) segfaulted the upstream
#: lowering pipeline identically, which is what pointed at the CUT'S POSITION, not its size -- a
#: 53k-op quantized model is dense enough with tensor.empty()-then-write pairs that almost any cut
#: lands between one.
_CHEAP_PRODUCERS = frozenset({"tensor.empty", "arith.constant", "tensor.splat", "linalg.fill"})

#: Ops whose result is never RETURNED from a chunk: a later reader gets a copy of the op re-created in
#: the caller from the values it reads. They compute nothing (a reshape view, a constant, an empty
#: destination), and returned they can make one chunk return the SAME buffer twice once the lowering
#: canonicalizes: a row scale ``s`` and ``expand_shape(collapse_shape(s))`` both leaving a chunk fold
#: to ``s``, and buffer-results-to-out-params{modify-public-functions hoist-static-allocs} erases that
#: allocation once per returned copy -- a use-after-free that segfaulted SmolVLA int8full's lowering.
_REMATERIALIZED = frozenset(
    {"tensor.expand_shape", "tensor.collapse_shape", "arith.constant", "tensor.splat", "tensor.empty"}
)


def _dynamically_shaped(value) -> bool:
    """A value whose type has a dynamic extent (or no rank): it cannot leave a chunk as a result,
    because buffer-results-to-out-params can only turn a statically shaped result into an out-param.
    Measured: a SmolVLA int8 capture chunked at op counts reached ``memref<?xi64>`` live-outs and
    failed there, after the cheap-producer rule had cleared the earlier segfault."""
    from xdsl.dialects.builtin import UnrankedMemRefType, UnrankedTensorType

    kind = value.type
    if isinstance(kind, UnrankedTensorType | UnrankedMemRefType):
        return True
    shape = getattr(kind, "get_shape", None)
    return shape is not None and any(extent < 0 for extent in shape())


def _top_level(op, block):
    """The op of ``block`` that ``op`` is, or is nested inside (``None`` when it is in neither)."""
    while op is not None and op.parent_block() is not block:
        op = op.parent_op()
    return op


def _dynamic_reach(body_ops: Sequence[Any]) -> list[int]:
    """``reach[c]``: the furthest position (index into ``body_ops``; ``len(body_ops)`` for the
    terminator) at which a dynamically shaped value defined before position ``c`` is still read.
    A cut at ``c`` would make that value a chunk's live-out exactly when ``reach[c] >= c``."""
    position = {id(op): i for i, op in enumerate(body_ops)}
    n = len(body_ops)
    block = body_ops[0].parent_block() if body_ops else None
    last = [-1] * n
    for i, op in enumerate(body_ops):
        for result in op.results:
            if not _dynamically_shaped(result):
                continue
            for use in result.uses:
                user = _top_level(use.operation, block)
                last[i] = max(last[i], position.get(id(user), n) if user is not None else n)
    reach = [-1] * (n + 1)
    for c in range(1, n + 1):
        reach[c] = max(reach[c - 1], last[c - 1])
    return reach


def _chunk_bounds(body_ops: Sequence[Any], chunk_ops: int) -> list[int]:
    """Cut points for ``body_ops`` near every ``chunk_ops``'th op, each at a LEGAL position: not
    right after a cheap producer (see ``_CHEAP_PRODUCERS``), and not where a dynamically shaped value
    would cross (see ``_dynamically_shaped``) -- the producer and every reader of such a value stay in
    one chunk. A cut is pulled back to the nearest legal position after the previous bound, else
    pushed forward to the nearest one after its target; with neither, the rest is one chunk."""
    n = len(body_ops)
    reach = _dynamic_reach(body_ops)

    def legal(cut: int) -> bool:
        return body_ops[cut - 1].name not in _CHEAP_PRODUCERS and reach[cut] < cut

    bounds = [0]
    target = chunk_ops
    while target < n:
        cut = next((c for c in range(target, bounds[-1], -1) if legal(c)), None)
        if cut is None:
            cut = next((c for c in range(target + 1, n) if legal(c)), None)
        if cut is None:
            break
        bounds.append(cut)
        target = max(target + chunk_ops, cut + 1)
    return bounds


def resolve_chunk_ops(requested: int | str | None, *, forward_ops: int | None = None) -> int | None:
    """The chunk size a build cuts its forward at, for the requested ``chunk_ops``.

    ``None`` keeps the unchunked program. A positive integer (or its decimal spelling) is itself.
    :data:`AUTO` is :data:`DEFAULT_CHUNK_OPS` when the forward body has more ops than that, and
    ``None`` when it fits in one chunk (``forward_ops``, from :func:`forward_body_size`; unknown counts
    as large). Anything else is refused: a misspelled size must not silently build unchunked."""
    if requested is None:
        return None
    if isinstance(requested, str):
        spelled = requested.strip().lower()
        if spelled == AUTO:
            return None if forward_ops is not None and forward_ops <= DEFAULT_CHUNK_OPS else DEFAULT_CHUNK_OPS
        if not spelled.isdigit():
            raise ValueError(f"chunk_ops is a positive op count or {AUTO!r}, not {requested!r}")
        requested = int(spelled)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError(f"chunk_ops is a positive op count or {AUTO!r}, not {requested!r}")
    return requested


def _forward_block(module, function: str):
    from merlin.analysis.whole_model.build.open_program import OpenModelError

    functions = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    functions = [f for f in functions if f.sym_name.data == function] or functions[:1]
    if not functions:
        raise OpenModelError(f"the module has no function @{function} with a body")
    return functions[0].body.blocks[0]


def forward_body_size(module, *, function: str = "forward") -> int:
    """How many top-level ops ``@function``'s body holds before its terminator."""
    return max(len(list(_forward_block(module, function).ops)) - 1, 0)


def chunk_forward(module, *, chunk_ops: int, function: str = "forward") -> int:
    """Split ``@function``'s single flat block into ``chunk_ops``-bounded, sequentially-called
    functions, in place. Returns how many chunk functions were made (0 when the block already fit
    in one).

    LLVM's own compile cost on one huge function is superlinear in its size -- measured directly: a
    105,185-line, 1,551-call ``forward`` (SmolVLA's) took 76 minutes to reach one object, single-
    threaded, dwarfing every other build stage. This is the SAME cut ``externalize_dispatches`` makes
    for a device group (a bounded run of ops, its free values threaded in as arguments, replaced by
    one call at the same position), applied to plain consecutive RUNS of ops instead of to one
    group's own members, and returning every value a LATER op still reads (there can be several) in
    place of the single ``committed`` value a group cut always has exactly one of. Nothing a value
    computes, or the order it runs in, changes: only which function's text an op's clone sits in.
    """
    from xdsl.dialects import func
    from xdsl.ir import Block, Region

    from merlin.analysis.whole_model.build.open_program import OpenModelError
    from merlin.compiler.partition import outline as OL

    block = _forward_block(module, function)
    ops = list(block.ops)
    if not ops or ops[-1].name != "func.return":
        raise OpenModelError(f"@{function}'s block does not end in func.return")
    body_ops, terminator = ops[:-1], ops[-1]
    if len(body_ops) <= chunk_ops:
        return 0
    bounds = _chunk_bounds(body_ops, chunk_ops) + [len(body_ops)]
    chunks = [body_ops[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    chunks = [c for c in chunks if c]  # a boundary pulled all the way back can empty one
    for index, chunk in enumerate(chunks):
        chunk_ids = {id(op) for op in chunk}
        live_out: list = []
        seen: set[int] = set()

        # A use counts by the top-level op it sits in: a reader nested in a later op's region (a
        # tensor.extract in a generic's body) is outside the chunk, one nested in the chunk is not.
        # Judged by the nested op itself, every value a chunk read only from inside its own regions
        # became a spurious live-out -- among them SmolVLA's dynamically shaped compaction result.
        def outside(use, chunk_ids=chunk_ids) -> bool:
            return id(_top_level(use.operation, block)) not in chunk_ids

        # A value a later op reads leaves the chunk as a result, unless its op is rematerialized (see
        # ``_REMATERIALIZED``): then the op is re-created after the call from what it reads -- the
        # chunk's results, or values the caller already has -- and every value it reads from inside
        # the chunk is handled the same way, transitively.
        remat_ids: set[int] = set()

        def leave(result, chunk_ids=chunk_ids, remat_ids=remat_ids, seen=seen, live_out=live_out) -> None:
            owner = result.owner
            if id(result) in seen:
                return
            seen.add(id(result))
            if id(owner) in chunk_ids and owner.name in _REMATERIALIZED and not owner.regions:
                remat_ids.add(id(owner))
                for operand in owner.operands:
                    if id(getattr(operand, "owner", None)) in chunk_ids:
                        leave(operand)
            else:
                live_out.append(result)

        for op in chunk:
            for result in op.results:
                if any(outside(use) for use in result.uses):
                    leave(result)
        params = free_values(chunk)
        symbol = f"{CHUNK_PREFIX}{index}"

        kblock = Block(arg_types=[p.type for p in params])
        mapping = dict(zip(params, kblock.args, strict=True))
        for op in chunk:
            clone = op.clone(value_mapper=mapping)
            for old, new in zip(op.results, clone.results, strict=True):
                mapping[old] = new
            kblock.add_op(clone)
        kblock.add_op(func.ReturnOp(*[mapping[v] for v in live_out]))
        chunk_fn = func.FuncOp(symbol, ([p.type for p in params], [v.type for v in live_out]), Region([kblock]))
        module.body.block.add_op(chunk_fn)

        call = func.CallOp(symbol, list(params), [v.type for v in live_out])
        block.insert_op_before(call, chunk[0])
        outer = dict(zip(live_out, call.results, strict=True))
        for op in chunk:
            if id(op) in remat_ids:
                clone = op.clone(value_mapper=outer)
                block.insert_op_before(clone, chunk[0])
                outer.update(zip(op.results, clone.results, strict=True))
        for value, result in outer.items():
            value.replace_uses_with_if(result, outside)
        for op in reversed(chunk):
            op.detach()
            op.erase()
    del terminator  # kept in the outer block untouched; named only to state the invariant above
    return len(chunks)


def chunk_symbols(count: int) -> list[str]:
    """The symbols of the ``count`` chunk functions :func:`chunk_forward` made, in call order."""
    return [f"{CHUNK_PREFIX}{index}" for index in range(count)]


__all__ = [
    "AUTO",
    "CHUNK_PREFIX",
    "DEFAULT_CHUNK_OPS",
    "PRUNABLE",
    "build",
    "build_reference",
    "chunk_forward",
    "chunk_symbols",
    "forward_body_size",
    "resolve_chunk_ops",
]
