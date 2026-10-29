"""Composition routes and source-composition requests a test case can use, inside its ``optimization_next``.

Measured on one run (a lead and five helpers): the author read the composition reference once, never
ran the composition catalog, and hand-wrote its convolution -> pool chain, its residual loaded sum and
its pool readouts, while per-case feedback never pointed a chain case at the library's composition
routes, and nothing said that model and program checks accept ``source_composition.json`` requests.
This module adds two advisory pointers, beside :mod:`.library_pass` and never replacing it:

* ``optimization_next["composition"]`` for a case whose public declaration (or, for a model or
  program, its interface IR) shows a producer -> consumer chain a composition route covers: product
  -> pool, product -> bias/scale/activation/requantization, residual add, sibling products sharing an
  input, per-head slices, connected stages, parameters prepared on every call. It names the route,
  one ``module:symbol``, whether ``submission/`` imports it, the catalog rows behind it with their
  roster status, the route's recipe or the symbol's contract command, and the per-case discovery
  command. A case the host-compute audit flagged (``host_compute_in_accelerator_group``, in the row
  or the response's ``offload_census``) says so first: its epilogue work runs on the host.
* ``optimization_next["source_composition_request"]`` for a model or program case whose IR or
  declaration shows what an unrequested ``source_composition.json`` option addresses: the option, the
  observation behind it, a complete file to write and the reference describing the schema.

The detection is the heavy-pass-recipes skill's ``composition_routes.py`` (loaded by path, so the
author's discovery command and this feedback read the case the same way): declared epilogue stages,
component nodes, shared inputs and the IR's provenance regions, never the case's name. Only public
files staged in the workspace and the response's own rows are read; no private reference, no
measurement. Everything is a pointer: nothing is ranked, gated or applied, no speedup is claimed,
and a failure leaves the response as it was.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

KEY = "composition"
REQUEST_KEY = "source_composition_request"
#: Cases per response that get a pointer, and routes named per line.
LIMIT = 12
PER_LINE = 2
HOST_VETO = "host_compute_in_accelerator_group"
#: Routes that move epilogue work off the host into the producer's readout or a composed stage.
EPILOGUE_ROUTES = ("producer_epilogue", "producer_pool", "residual_add")
SKILL = "heavy-pass-recipes"
SKILL_ROOTS = (Path(".agents") / "skills", Path(".claude") / "skills")
_SCRIPT = ("author_skills", SKILL, "scripts", "composition_routes.py")
_LOADED: list = []
_IR_CACHE: dict = {}


def routes_script():
    """The author skill's ``composition_routes.py``, loaded once by path."""
    if not _LOADED:
        path = Path(__file__).resolve().parents[1].joinpath(*_SCRIPT)
        spec = importlib.util.spec_from_file_location("merlin_experiments._composition_routes", path)
        module = importlib.util.module_from_spec(spec)
        before = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = before
        _LOADED.append(module)
    return _LOADED[0]


def skill_root(workspace: Path) -> str:
    for root in SKILL_ROOTS:
        if (Path(workspace) / root / SKILL).is_dir():
            return root.as_posix()
    return SKILL_ROOTS[0].as_posix()


def _roster(workspace: Path):
    """Exact module names the author can import: ``submission/compiler_library`` or the staged inventory."""
    script = routes_script()
    library = Path(workspace) / "submission" / "compiler_library"
    if (library / "merlin").is_dir():
        return script.roster(0, str(library))[1]
    try:
        manifest = json.loads((Path(workspace) / "shared-library.json").read_text(encoding="utf-8"))
        version = manifest.get("inventory_version") if isinstance(manifest, dict) else None
    except (OSError, ValueError):
        version = None
    if isinstance(version, int):
        return script.roster(version, None)[1]
    return None


def catalog_rows(workspace: Path) -> dict:
    try:
        return routes_script().catalog_rows(_roster(workspace))
    except Exception:  # noqa: BLE001 -- availability stays unknown, the pointer still names the route
        return {}


def _signals(name: str, workspace: Path):
    """``(signals, inventory, declaration)`` for a case; the IR inventory is cached by file identity."""
    from . import case_declarations

    script = routes_script()
    declaration = case_declarations.load(name, workspace)
    folder = script.case_dir(name, workspace)
    if declaration is None and folder is None:
        return {}, None, None
    key = None
    if folder is not None:
        ir = folder / str((declaration or {}).get("interface_mlir") or "capsule.interface.mlir")
        try:
            stat = ir.stat()
            key = (
                str(ir.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                json.dumps(declaration, sort_keys=True, default=str)[:4096],
            )
        except OSError:
            key = None
    if key is not None and key in _IR_CACHE:
        return _IR_CACHE[key]
    found = script.case_signals(folder, declaration=declaration, workspace=workspace)
    if key is not None:
        if len(_IR_CACHE) > 64:
            _IR_CACHE.clear()
        _IR_CACHE[key] = found
    return found


def grouped_source(name, workspace):
    """Actual public interface operand sharing; no name or technique inference."""
    from .reference_views import source_view

    folder = routes_script().case_dir(name, Path(workspace))
    if folder is None:
        return None
    from . import case_declarations

    declaration = case_declarations.load(name, workspace) or {}
    filename = declaration.get("interface_mlir") or "capsule.interface.mlir"
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None
    path = folder / filename
    root = Path(workspace).absolute()
    if (any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root))
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file() or path.stat().st_size > 4 << 20):
        return None
    try:
        observed = source_view(path.read_bytes())
    except (ValueError, KeyError, TypeError, ImportError):
        return None
    if not observed["groups"]:
        return None
    return dict(input=observed["input"], products=observed["products"],
        outputs=observed["outputs"], argument_order=observed["argument_order"],
        groups=[dict(rhs=row["rhs"], members=len(row["members"]),
            outputs=[member["output"] for member in row["members"]],
            shapes=[member["shape"] for member in row["members"]],
            readouts=[member["readout_attributes"] for member in row["members"]])
            for row in observed["groups"][:8]], omitted_groups=max(0, len(observed["groups"]) - 8),
        status="observed_source", emitted_mechanism="unknown")


def _row_status(rows: dict, names) -> str:
    parts = []
    for name in names:
        row = rows.get(name)
        if not isinstance(row, dict):
            continue
        missing = row.get("missing_modules") or []
        status = (
            f"missing {', '.join(missing[:2])}"
            if missing
            else "modules selected"
            if row.get("status") == "modules_selected"
            else "roster unknown"
        )
        parts.append(f"{name} ({status})")
    return "; ".join(parts)


def pointer(name: str, signals: dict, *, workspace: Path, imports, rows: dict, host_flag: bool) -> dict | None:
    """``optimization_next["composition"]`` for one case (module docstring), or None without a route."""
    script = routes_script()
    order = [rid for rid in script.BY_ID if rid in signals]
    if host_flag:
        order = [rid for rid in order if rid in EPILOGUE_ROUTES] + [rid for rid in order if rid not in EPILOGUE_ROUTES]
    if not order:
        return None
    skills = skill_root(workspace)
    discover = script.discovery_command(name, skills)
    fields, named = [], []
    for rid in order:
        route = script.BY_ID[rid]
        found = script.imported(route["symbol"], imports)
        field = {
            "route": rid,
            "chain": route["chain"],
            "evidence": list(signals[rid])[:3],
            "symbol": route["symbol"],
            "imported_by_submission": found,
            "catalog_rows": {row: (rows.get(row) or {}).get("status", "catalog_unavailable") for row in route["rows"]},
            "next": script.next_command(route, skills),
            "skills": list(route["skills"]),
        }
        if route.get("request"):
            field["source_composition_option"] = route["request"]
        fields.append(field)
        if len(named) < PER_LINE:
            imported_text = (
                "imported by submission/"
                if found
                else "not imported by submission/"
                if found is False
                else "imports unknown"
            )
            status = _row_status(rows, route["rows"][:2])
            named.append(
                f"{rid}: {signals[rid][0]} -> {route['symbol']} ({imported_text}"
                + (f"; rows {status}" if status else "")
                + f"); run: {field['next']}"
            )
    lead = (f"host computes this group's epilogue ({HOST_VETO}); " if host_flag else "") + "composition routes"
    more = len(order) - len(named)
    line = (
        f"{lead}: "
        + " | ".join(named)
        + (f" | {more} more route(s)" if more > 0 else "")
        + f" | all routes for this case: {discover} | read: {SKILL} reference/composition.md"
    )
    return {
        "line": line,
        "routes": fields,
        "discover": discover,
        "trigger": HOST_VETO if host_flag else "declared_or_ir_chain",
        "reference": f"{skills}/{SKILL}/reference/composition.md",
        "basis": "advisory: routes keep the original form when their premises fail; keep your own lowering "
        "where it measures better on the same cases; no speedup is implied",
    }


def _requested(row: dict, workspace: Path) -> dict:
    """Options already requested: the row's own composition report first, else the author's file."""
    features = ((row or {}).get("source_composition") or {}).get("features")
    if isinstance(features, dict):
        return {key: value.get("requested") is True for key, value in features.items() if isinstance(value, dict)}
    current = routes_script().current_request(workspace) or {}
    return {key: value is True for key, value in current.items() if key != "schema"}


def request_advice(name: str, signals: dict, inventory, declaration, *, row: dict, workspace: Path) -> dict | None:
    """``optimization_next["source_composition_request"]`` for a model or program case, or None."""
    script = routes_script()
    # The owned model and program checks read the request: a declared model, or a program case that
    # stages only its interface IR. A single declared operation or a component slice does not.
    kind = (declaration or {}).get("kind")
    op = ((declaration or {}).get("operation") or {}).get("op")
    if not (kind == "model" or op == "model" or (declaration is None and inventory)):
        return None
    requested = _requested(row, workspace)
    matches = [m for m in script.request_matches(signals, inventory, declaration) if not requested.get(m["option"])]
    if not matches:
        return None
    skills = skill_root(workspace)
    reference = f"{skills}/{SKILL}/reference/source-composition.md"
    keep = [key for key, value in requested.items() if value]
    example = script.request_example(keep + [m["option"] for m in matches if m["option"] != "op_profile"])
    shown = "; ".join(f"{m['option']} ({m['why']})" for m in matches)
    permission_note = (
        " | host layout/storage requests require original software permissions; "
        "missing permission keeps original code; "
        "layout is deferred for selected head/sibling composition"
        if any(m["option"] in {"propagate_layouts", "reuse_intermediate_storage"} for m in matches) else "")
    line = (
        f"source_composition.json options this case may use: {shown} | request: write {script.REQUEST_FILE} "
        f"{json.dumps(example, separators=(',', ':'))} | then re-run this check and read the row's "
        f"source_composition (requested, selected, applied) | schema and options: {reference} | the owner keeps "
        "the ordinary route where a request does not apply; no speedup is implied" + permission_note
    )
    return {
        "line": line,
        "options": matches,
        "file": script.REQUEST_FILE,
        "example": example,
        "already_requested": sorted(keep),
        "reference": reference,
        "discover": script.discovery_command(name, skills),
        "basis": "advisory: requests only; selection is not application and application is not a speedup",
    }


def host_flagged(document: dict, rows) -> set[str]:
    """Cases the host-compute audit flagged, from the rows and the response's offload census."""
    flagged = set()
    for row in rows or ():
        if isinstance(row, dict) and ((row.get("failure") or {}).get("category") == HOST_VETO):
            flagged.add(str(row.get("capsule") or row.get("name") or ""))
    census = document.get("offload_census") if isinstance(document, dict) else None
    if isinstance(census, dict):
        flagged.update(str(name) for name in census.get("host_compute_vetoed") or () if isinstance(name, str))
        for audit in census.get("host_compute") or ():
            if isinstance(audit, dict) and audit.get("verdict") == HOST_VETO:
                flagged.add(str(audit.get("program")))
    return flagged


def annotate(document: dict, rows, brief, *, workspace=None, context=None) -> int:
    """Add :data:`KEY` / :data:`REQUEST_KEY` to the ``optimization_next`` of each case that gets one.

    A case in the brief's ``per_case_advice`` gets them there (copied onto its row); any other case on
    its row. Returns how many cases got a pointer. Never raises.
    """
    try:
        workspace = Path(workspace) if workspace is not None else Path.cwd()
        advice = (brief or {}).get("per_case_advice") if isinstance(brief, dict) else None
        advice = advice if isinstance(advice, dict) else {}
        from ..workload_roles import PRIMARY, workspace_roles

        roles = context.roles if context is not None else (workspace_roles(workspace) or {})
        flagged = host_flagged(document, rows)
        candidates = [row for row in rows or () if isinstance(row, dict) and (row.get("capsule") or row.get("name"))]
        candidates.sort(key=lambda row: str(row.get("capsule") or row.get("name")) not in flagged)
        imports = rows_cache = None
        written = 0
        for row in candidates:
            if written >= LIMIT:
                break
            name = str(row.get("capsule") or row.get("name"))
            primary = roles.get(name) in PRIMARY
            if not primary and name not in flagged:
                continue
            signals, inventory, declaration = _signals(name, workspace)
            if not signals:
                continue
            if imports is None:
                from .library_pass import inventory_script

                imports = inventory_script().imported_modules(workspace / "submission")
                rows_cache = catalog_rows(workspace)
            found = pointer(
                name, signals, workspace=workspace, imports=imports, rows=rows_cache, host_flag=name in flagged
            )
            request = (request_advice(name, signals, inventory, declaration, row=row, workspace=workspace)
                       if primary else None)
            if found is None and request is None:
                continue
            target = advice.get(name)
            if not isinstance(target, dict):
                target = row.get("optimization_next")
                if not isinstance(target, dict):
                    target = row["optimization_next"] = {}
            if found is not None:
                target[KEY] = found
            if request is not None:
                target[REQUEST_KEY] = request
            written += 1
        return written
    except Exception:  # noqa: BLE001 -- advice only; the response stays as it was
        return 0


__all__ = ["KEY", "REQUEST_KEY", "annotate", "host_flagged", "pointer", "request_advice", "routes_script"]
