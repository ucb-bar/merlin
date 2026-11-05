"""Host-owned WHOLE-MODEL BOUNDARY PROFILE for the measured-claims workflow.

The tuning corpus measures members one at a time. What it cannot show is where a complete program's
cycles go: how much is inside the device groups, how much is between them (host work, layout glue,
dispatch), and how each group sits against the machine's own bound. This module builds the agent's
OWN candidate (and the frozen baseline) into the operator-declared whole-model program, runs it once on
the timing engine with every routed device-group call bracketed (``compile_saved_model(...,
group_profile=True)``: ``GROUP_ID`` / ``GROUP`` / ``GAP`` lines, parsed by
:func:`merlin.runtime.whole_model_readback.parse_group_profile`), and reports per group:

* ``cycles`` and ``gap_before`` (cycles between the previous group's end and this group's start);
* ``share_of_window`` -- its cycles over the whole bracketed window (groups plus gaps);
* its derived ``roofline`` -- the compute floor of its contraction on the array and the movement floor
  of its operand/result bytes on the memory path (:mod:`merlin.perf.capsule_roofline`), with
  ``over_roofline`` = cycles / bound.

Measurement only: nothing here ranks, recommends or names a remedy. The program, the board, the host
package and the capture are the operator's deployment record (:class:`WholeModelProfileInputs`); the
extents come from the build's own device-dispatch sidecar; nothing names a target or a model.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import StageGateError

SCHEMA = "host_owned_whole_model_boundary_profile_v1"
INPUTS_SCHEMA = "merlin_whole_model_profile_inputs_v1"
_ENGINES = ("gsim",)


@dataclass(frozen=True)
class WholeModelProfileInputs:
    """One profiled whole-model program of the operator's deployment record."""

    capture: Path
    host_package: Path
    board_catalog: Path
    board: str
    dts: Path
    arena_mb: int
    datapath: str
    engine: str = "gsim"
    reference_file: str | None = None
    readback: str = "prefix"
    tolerance: Mapping[str, float] | None = None
    timeout_s: int = 14_400
    name: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WholeModelProfileSet:
    """Every program the operator declares for the whole-model boundary profile (shared host side)."""

    programs: tuple[WholeModelProfileInputs, ...]

    @property
    def engine(self) -> str:
        return self.programs[0].engine


#: Per-program keys; everything else in the record is shared by every program.
_PROGRAM_KEYS = ("name", "capture", "datapath", "reference_file", "readback", "tolerance", "arena_mb", "timeout_s")


def load_inputs(path: Path) -> WholeModelProfileSet:
    """Read and check a ``merlin_whole_model_profile_inputs_v1`` JSON record.

    Either one program at the top level (``capture``, ``datapath``, ...) or a short ``programs`` list of
    ``{name, capture, datapath, reference_file?, readback?, tolerance?, arena_mb?, timeout_s?}`` sharing
    the record's host package, board, board catalog, DTS and engine."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageGateError(f"whole-model profile inputs are unreadable: {exc}") from exc
    if not isinstance(doc, Mapping) or doc.get("schema") != INPUTS_SCHEMA:
        raise StageGateError(f"whole-model profile inputs must declare schema {INPUTS_SCHEMA}")
    base = Path(path).resolve().parent

    def where(source: Mapping[str, Any], key: str) -> Path:
        value = source.get(key)
        if not isinstance(value, str) or not value:
            raise StageGateError(f"whole-model profile inputs omit {key}")
        resolved = Path(value) if Path(value).is_absolute() else base / value
        if not resolved.exists():
            raise StageGateError(f"whole-model profile input {key} does not exist: {resolved}")
        return resolved

    engine = doc.get("engine", "gsim")
    if engine not in _ENGINES:
        raise StageGateError(f"whole-model profile engine must be one of {_ENGINES}")
    if not isinstance(doc.get("board"), str) or not doc["board"]:
        raise StageGateError("whole-model profile inputs omit board")
    rows = doc.get("programs")
    if rows is None:
        rows = [{key: doc[key] for key in _PROGRAM_KEYS if key in doc}]
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise StageGateError("whole-model profile programs must be a non-empty list of mappings")
    shared = {
        "host_package": where(doc, "host_package"),
        "board_catalog": where(doc, "board_catalog"),
        "board": str(doc["board"]),
        "dts": where(doc, "dts"),
        "engine": str(engine),
    }
    programs = []
    for row in rows:
        merged = {**{k: doc[k] for k in ("arena_mb", "timeout_s", "readback", "tolerance") if k in doc}, **row}
        arena, timeout = merged.get("arena_mb"), merged.get("timeout_s", 14_400)
        if type(arena) is not int or arena < 1 or type(timeout) is not int or timeout < 1:
            raise StageGateError("whole-model profile arena_mb and timeout_s are positive integers")
        if not isinstance(merged.get("datapath"), str) or not merged["datapath"]:
            raise StageGateError("whole-model profile program omits datapath")
        capture = where(merged, "capture")
        programs.append(
            WholeModelProfileInputs(
                capture=capture,
                arena_mb=arena,
                datapath=str(merged["datapath"]),
                reference_file=merged.get("reference_file"),
                readback=str(merged.get("readback", "prefix")),
                tolerance=merged.get("tolerance"),
                timeout_s=timeout,
                name=str(merged.get("name") or capture.name),
                **shared,
            )
        )
    names = [program.name for program in programs]
    if len(set(names)) != len(names):
        raise StageGateError("whole-model profile program names must be distinct")
    return WholeModelProfileSet(tuple(programs))


def private_model_identities(descriptor: Mapping[str, Any]) -> set[str]:
    """Every model and source-workload name the target descriptor declares PRIVATE (its
    ``phase1_gates`` private full models), lower-cased: names the boundary profile may never run."""
    found: set[str] = set()

    def visit(block: Any) -> None:
        if not isinstance(block, Mapping):
            return
        for key, value in block.items():
            if key == "private_full_models" and isinstance(value, Mapping):
                for name in value.get("models") or ():
                    found.add(str(name).lower())
                for model in value.get("programs") or {}:
                    found.add(str(model).lower())
                for workload in (value.get("source_workload_dirs") or {}).values():
                    found.add(str(workload).lower())
            visit(value)

    visit((descriptor or {}).get("phase1_gates"))
    return found


def refuse_private_programs(inputs: WholeModelProfileSet, descriptor_path: Path) -> None:
    """Operator-side guard: refuse a profile program whose capture is a declared private full model.

    A program is refused when its declared name, its capture directory name, or any directory named in
    its capture receipt's source path is one of :func:`private_model_identities`."""
    import yaml

    descriptor = yaml.safe_load(Path(descriptor_path).read_text(encoding="utf-8")) or {}
    private = private_model_identities(descriptor)
    if not private:
        return
    for program in inputs.programs:
        names = {program.name.lower(), Path(program.capture).name.lower()}
        receipt = Path(program.capture) / "capture_receipt.json"
        try:
            source = (json.loads(receipt.read_text(encoding="utf-8")).get("source") or {}).get("path")
        except (OSError, ValueError, AttributeError):
            source = None
        if isinstance(source, str):
            names.update(part.lower() for part in Path(source).parts)
        hit = sorted(names & private)
        if hit:
            raise StageGateError(
                f"whole-model profile program {program.name!r} is a declared private full model ({hit[0]}); "
                "the boundary profile runs only public programs, and private full models stay an unseen "
                "generalization test"
            )


# ------------------------------------------------------------------------------------- extents


def _tensor(text: str) -> tuple[list[int], str] | None:
    """``tensor<2x3x4xi8>`` -> ``([2, 3, 4], "i8")``, structurally; None otherwise."""
    text = str(text).strip()
    if not text.startswith("tensor<") or not text.endswith(">"):
        return None
    parts = text[len("tensor<") : -1].split("x")
    dims, dtype = parts[:-1], parts[-1]
    if not all(d.isdigit() for d in dims):
        return None
    return [int(d) for d in dims], dtype


def group_roofline_inputs(
    row: Mapping[str, Any],
) -> tuple[list[tuple[int, int, int]] | None, int | None, int | None, str]:
    """``(contractions, read bytes, write bytes, basis)`` of one routed device group.

    The routed row states the contraction's parallel and reduction extents (the linalg iteration
    space: the output's trailing parallel dimension is its column extent) and the two operand and the
    result tensor types, so the compute floor and the compulsory bytes are both derived from the
    build's own record."""
    from merlin.perf.derived_bound import _width_bits

    parallel, reduction = row.get("parallel") or [], row.get("reduction") or []
    contractions = None
    if (
        parallel
        and reduction
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in [*parallel, *reduction])
    ):
        contractions = [(math.prod(parallel[:-1]) or 1, math.prod(reduction), parallel[-1])]
    types = [_tensor(t) for t in (row.get("tensor_types") or [])]
    read = write = None
    if len(types) == 3 and all(types):
        sizes = []
        for dims, dtype in types:
            bits = _width_bits(dtype)
            sizes.append(None if bits is None else math.prod(dims) * -(-bits // 8))
        if None not in sizes:
            read, write = sizes[0] + sizes[1], sizes[2]
    basis = "routed group iteration space (M x K x N) and its operand/result tensor types"
    dtypes = row.get("dtypes") or []
    if read is None and len(parallel) == 2 and len(reduction) == 1 and len(dtypes) == 3 and contractions:
        # No tensor types recorded: a plain M x K x N contraction's operands ARE M x K and K x N and its
        # result M x N, at the row's own element types. A windowed (convolution-shaped) space is not
        # charged this way, because its input is smaller than its im2col matrix and the floor must stay one.
        widths = [_width_bits(d) for d in dtypes]
        if None not in widths:
            m, k, n = contractions[0]
            read = m * k * -(-widths[0] // 8) + k * n * -(-widths[1] // 8)
            write = m * n * -(-widths[2] // 8)
            basis = "routed group iteration space (M x K x N) and its element types"
    return contractions, read, write, basis


def _rows_by_name(routed: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    from merlin.runtime.whole_model_readback import group_names

    out: dict[str, Mapping[str, Any]] = {}
    for (name, symbol), row in zip(group_names(routed), routed, strict=True):
        out.setdefault(name, row)
        out.setdefault(symbol, row)
        out.setdefault(symbol.removeprefix("_mlir_ciface_"), row)
    return out


# ------------------------------------------------------------------------------------- document


def boundary_document(
    profile: Mapping[str, Any], routed: Sequence[Mapping[str, Any]], machine: Mapping[str, Any] | None
) -> dict[str, Any]:
    """One arm's agent-visible boundary profile from its parsed group profile and routed groups."""
    from merlin.perf import capsule_roofline as CR

    groups = list(profile.get("groups") or ())
    group_cycles = sum(int(g["cycles"]) for g in groups)
    gap_cycles = sum(int(g["gap_before"]) for g in groups) + int(profile.get("tail_gap") or 0)
    window = group_cycles + gap_cycles
    by_name = _rows_by_name(routed)
    rows = []
    for group in groups:
        source = by_name.get(str(group["name"]))
        roofline: dict[str, Any] = {"status": "unknown", "roofline_cycles": None, "limiter": None}
        if source is not None:
            contractions, read, write, _basis = group_roofline_inputs(source)
            doc = CR.roofline(contractions, read, write, machine)
            roofline = {
                "status": doc["status"],
                "roofline_cycles": doc["roofline_cycles"],
                "limiter": doc["limiter"],
                "compute_floor_cycles": doc["compute_floor_cycles"],
                "movement_floor_cycles": doc["movement_floor_cycles"],
            }
            position = CR.position(doc, int(group["cycles"]))
            if position is not None and position < 1:
                roofline.update(status="refuted", over_roofline=None)
            else:
                roofline["over_roofline"] = position
        rows.append(
            {
                "index": int(group["index"]),
                "name": str(group["name"]),
                "cycles": int(group["cycles"]),
                "gap_before": int(group["gap_before"]),
                "share_of_window": round(int(group["cycles"]) / window, 6) if window else None,
                "roofline": roofline,
            }
        )
    bounds = [r["roofline"]["roofline_cycles"] for r in rows if r["roofline"]["status"] == "derived"]
    return {
        "status": "measured",
        "window_cycles": window,
        "group_cycles": group_cycles,
        "gap_cycles": gap_cycles,
        "tail_gap": int(profile.get("tail_gap") or 0),
        "calls": int(profile.get("calls") or 0),
        "dropped_calls": int(profile.get("dropped") or 0),
        "groups_with_roofline": len(bounds),
        "roofline_cycles_of_those_groups": sum(bounds) if bounds else None,
        "groups": rows,
    }


# ------------------------------------------------------------------------------------- execution


def run_arm(
    package_dir: Path,
    inputs: WholeModelProfileInputs,
    *,
    target: str,
    rtl_facts: Path | None,
    output: Path,
    machine: Mapping[str, Any] | None,
    compile_fn: Callable[..., Mapping[str, Any]] | None = None,
    plan_fn: Callable[..., Mapping[str, Any]] | None = None,
    visible_root: Path | None = None,
) -> dict[str, Any]:
    """Route, build and run ``package_dir`` on the declared whole-model program; its boundary document,
    or ``{"status": "refused", "why": ...}`` when no profile could be measured."""
    from merlin.llvmlower.device_offload import BY_GROUP

    if compile_fn is None:
        from merlin.compile.baremetal_model import compile_saved_model as compile_fn
    if plan_fn is None:
        from merlin.compile.route_before_build import plan_before_build as plan_fn
    try:
        model = Path(inputs.capture) / "model.mlir"
        routed = plan_fn(
            target,
            model.read_text(encoding="utf-8"),
            datapath=inputs.datapath,
            device_package=str(package_dir),
            capture=model,
            granularity=BY_GROUP,
        )
        device = routed.get("device_routing")
        if device is None:
            why = f"no device route: {routed.get('device_routing_why')}"
            return {"status": "refused", "why": agent_visible_text(why, package_dir, visible_root)[:300]}
        receipt = compile_fn(
            capture=inputs.capture,
            package=inputs.host_package,
            board_catalog=inputs.board_catalog,
            board=inputs.board,
            dts=inputs.dts,
            output=output,
            target=target,
            run=inputs.engine,
            arena_mb=inputs.arena_mb,
            timeout_s=inputs.timeout_s,
            reference_file=inputs.reference_file,
            rtl_facts=rtl_facts,
            device=device,
            readback=inputs.readback,
            group_profile=True,
            tolerance=dict(inputs.tolerance) if inputs.tolerance else None,
        )
        result = receipt.get("output") or {}
        profile = json.loads(Path(result["group_profile"]["path"]).read_text(encoding="utf-8"))
        sidecar = json.loads(Path(result["device_sidecar"]["path"]).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a failed build or run is a refusal with its reason
        why = f"{type(exc).__name__}: {exc}"
        # Scrubbed BEFORE truncation, so a cut can never leave the head of a host path behind.
        return {"status": "refused", "why": agent_visible_text(why, package_dir, visible_root)[:300]}
    return boundary_document(profile, list(sidecar.get("routed") or ()), machine)


def agent_visible_text(text: str, snapshot: Path | None, visible_root: Path | None) -> str:
    """``text`` with every host path scrubbed except the agent's own work tree: the measured snapshot
    of that tree is first rewritten to the root the agent knows it by."""
    from merlin.common.path_scrub import scrub_host_paths

    if visible_root is None:
        return scrub_host_paths(text)
    rewrite = {str(snapshot): str(visible_root)} if snapshot is not None else {}
    return scrub_host_paths(text, keep=(str(visible_root),), rewrite=rewrite)


def validate_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """The agent-visible profile's closed top level, program rows and arm shapes."""
    if not isinstance(document, Mapping) or set(document) != {"schema", "engine", "purpose", "programs"}:
        raise StageGateError("whole-model boundary profile violates its schema")
    programs = document["programs"]
    if not isinstance(programs, list) or not programs:
        raise StageGateError("whole-model boundary profile carries no programs")
    for program in programs:
        if not isinstance(program, Mapping) or set(program) != {"program", "baseline", "candidate"}:
            raise StageGateError("whole-model boundary profile program row is malformed")
        for arm in ("baseline", "candidate"):
            row = program[arm]
            if not isinstance(row, Mapping) or row.get("status") not in ("measured", "refused"):
                raise StageGateError(f"whole-model boundary profile {arm} arm is malformed")
            if row["status"] == "refused" and set(row) != {"status", "why"}:
                raise StageGateError(f"whole-model boundary profile {arm} refusal is malformed")
    encoded = json.dumps(document, sort_keys=True).lower()
    for forbidden in ('"golden', '"output', '"path', '"elf', '"verilator'):
        if forbidden in encoded:
            raise StageGateError(f"whole-model boundary profile leaks forbidden field {forbidden}")
    return dict(document)


def profile_candidate(feedback: Any, candidate: Path, *, round_index: int, call_index: int) -> dict:
    """Per-group cycles, inter-group gaps and per-group rooflines of the declared whole-model
    program, built with the frozen baseline and with ``candidate`` and run on the timing engine.

    ``feedback`` is the stage's live development evaluator, which owns the workspace, the frozen baseline
    package and its per-program baseline cache."""
    if feedback.whole_model_inputs is None:
        raise StageGateError("no whole-model deployment inputs were supplied to this stage")
    candidate = Path(candidate).resolve(strict=True)
    root = feedback.work_root / f"round_{round_index:02d}" / f"whole_model_{call_index:03d}"
    if root.exists() or root.is_symlink():
        raise StageGateError(f"whole-model profile workspace is not fresh: {root}")
    root.mkdir(parents=True)
    snapshot = root / "_measured_candidate"
    shutil.copytree(candidate, snapshot, symlinks=True)
    target = str(getattr(feedback.target_experiment, "target", "") or "")

    def arm(program, name: str, package: Path, visible: Path | None) -> dict:
        return run_arm(
            package,
            program,
            target=target,
            rtl_facts=feedback.rtl_facts_path,
            output=root / program.name / name,
            machine=feedback.machine_bounds,
            visible_root=visible,
        )

    if feedback._whole_model_baseline is None:
        feedback._whole_model_baseline = {}
    rows = []
    for program in feedback.whole_model_inputs.programs:
        if program.name not in feedback._whole_model_baseline:
            # The baseline is host-owned: none of its paths is the agent's, so every one is scrubbed.
            feedback._whole_model_baseline[program.name] = arm(program, "baseline", feedback.baseline, None)
        rows.append(
            {
                "program": program.name,
                "baseline": feedback._whole_model_baseline[program.name],
                "candidate": arm(program, "candidate", snapshot, candidate),
            }
        )
    document = {
        "schema": SCHEMA,
        "engine": feedback.whole_model_inputs.engine,
        "purpose": (
            "where each declared whole-model program's cycles go: per device-group cycles, the cycles "
            "between groups, and each group against its derived bound; measurement only"
        ),
        "programs": rows,
    }
    return validate_document(document)


def load_selected_inputs(path: Path | None, target_experiment: Any = None):
    """The operator's whole-model profile programs, refused when any is a declared private full model."""
    if path is None:
        return None
    inputs = load_inputs(Path(path))
    descriptor = getattr(target_experiment, "path", None)
    if descriptor is not None and Path(descriptor).is_file():
        refuse_private_programs(inputs, Path(descriptor))
    return inputs
