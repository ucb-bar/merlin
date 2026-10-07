"""The measurement service's production builder, over the core whole-model build.

:class:`.service.MeasurementService` takes a builder ``build(package_dir, *, target, out_dir, **options)
-> record``.  This one calls :func:`merlin.perf.whole_model_build.build` (an OPEN model -- host regions
between its groups -- goes through :func:`merlin.perf.whole_model_open_service.service_build`) and turns
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
    (:mod:`merlin.llvmlower.optional_passes`; ``-name`` turns off one that is on by default). It is a
    launch config's ``build_options`` entry like the rest, empty by default -- so a config that names
    none builds exactly as before -- and the selection is recorded in the result's ``notes``.

    The compile-debugging keys (:mod:`merlin.compile.debug`) are ``build_options`` too, all empty by
    default: ``only_groups`` builds a PARTIAL program (marked on its expectations, so every verdict and
    measurement refuses it), and ``trace_dir``/``dump_ir_after``/``dump_ir_before``/``stop_after`` keep a
    trace of the build. A build stopped at a stage produces no program, so here -- where a program is
    what the caller is owed -- the stop is a :class:`WholeModelBuildError` naming where the IR is.
    """
    from merlin.common.compile_trace import StopAfterStage
    from merlin.compile import debug
    from merlin.llvmlower import optional_passes
    from merlin.perf import whole_model_build as WMB

    options = {"trace_dir": trace_dir, "dump_ir_after": dump_ir_after, "dump_ir_before": dump_ir_before}
    trace = debug.request_from_options(
        {**options, "stop_after": stop_after}, target=target, workload=Path(model_capsule).name
    )
    selection = optional_passes.Selection.parse(list(lowering_passes))
    try:
        with (
            debug.opened(trace, ["merlin.perf.whole_model_builder.build"]),
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
    ``build_options`` switches (see :func:`merlin.perf.whole_model_build.build`): a package that never
    declares a whole-model pass or never asks for a region sees no change either way.

    ``phase0_recipe`` / ``descriptor`` name the corpus binding every group is stated under.
    ``chunk_ops`` (an open model only) bounds the open build's lowered functions; ``None`` keeps the
    unchunked program. ``"auto"`` derives the size from an open model's forward and asks nothing of a
    closed one (it has no host forward to cut), so one build option serves both.
    """
    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_open as WO
    from merlin.perf import whole_model_partial as PARTIAL
    from merlin.runtime.backends import base as backends

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
    from merlin.perf.whole_model_chunks import AUTO

    if chunk_ops is not None and str(chunk_ops).strip().lower() != AUTO:
        raise WMB.WholeModelBuildError("chunk_ops bounds an open model's lowered functions; this model is closed")
    record = WMB.build(
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
            "builder": "merlin.perf.whole_model_build",
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
    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_headers as WMH
    from merlin.perf.whole_model_memory import memory_map
    from merlin.perf.whole_model_oracle import _oracle
    from merlin.runtime.backends import base as backends

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
            "builder": "reference arm over merlin.perf.whole_model_build",
            "package_dir_ignored": str(package_dir),
            "harness_overrides": overrides,
            "build_record": str(record_path),
        },
    }


__all__ = ["PRUNABLE", "build", "build_reference"]
