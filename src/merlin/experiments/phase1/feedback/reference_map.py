"""The operator-built reference map: expert reference regions linked to the operations they implement.

``REFERENCE_MAP.json`` (:data:`SCHEMA`) names regions of the staged expert references (a file and its
lines, function or address range) and the operation each implements -- kind, dimensions, epilogue,
dtypes -- with the techniques observed there and whether the region uses hardware loops. The launch
pins one map (:data:`ENV`, ``{"path", "sha256"}``); the session stages the same bytes read-only in the
workspace of every arm (:func:`stage`), and the self-check feedback names the matching regions for a
case at the moment the case can use them:

* only for a **performance target** of :mod:`.family_rollup` -- a bundle representative (a passing case
  that represents its family and shape class) or a large or connected passing case with headroom above
  its structural bound. A failing or unchecked case gets no reference line;
* only for **matching** entries, by operation kind and the dimensions or shape class derived from the
  case's own declaration (never its name): ``exact`` (same kind, complete dimensions),
  ``same_product`` (a matrix product and a convolution with the
  same implicit product), ``same_class`` (same
  kind, same shape class under the tile edge the target's RTL facts state) or ``technique`` (same kind);
* at most :data:`PER_CASE` compact lines per case; a region that uses hardware loops says to express the
  technique with ordinary commands. No match, no line.

Advisory only: it grades, gates and selects nothing and grants no reference access. Bounded reads, no
``re``, no target name.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SCHEMA = "merlin.reference_map.v1"
FEEDBACK_SCHEMA = "merlin.reference_map_feedback.v1"
ENV = "MERLIN_REFERENCE_MAP"
FILENAME = "REFERENCE_MAP.json"
KEY = "reference_map"
KINDS = ("matmul", "conv2d", "depthwise_conv", "pool", "residual_add", "elementwise", "requantize", "other")
CONFIDENCE = ("verified", "inferred")
GRADES = ("exact", "same_product", "same_class", "technique")
#: Kinds that are one matrix product (a window computes its implicit product), so they match across kinds.
_PRODUCT_KINDS = ("matmul", "conv2d")
#: Reference lines per case, techniques per line, characters per technique and bytes per line.
PER_CASE = 2
TECHNIQUES = 2
TECHNIQUE_CHARS = 90
LINE_BYTES = 420
_MAX_BYTES = 8 << 20
_MAX_ENTRIES = 4096
_WINDOW_KINDS = ("conv2d", "depthwise_conv", "pool")
_CACHE: dict = {}

TASK_SECTION = (
    "\n\n## Reference map\n\n"
    "`REFERENCE_MAP.json` (read-only, operator-built) links regions of the expert references in "
    "`reference-kernels/` to the operation each implements: `operation.kind`, `dims`, `epilogue` and "
    "`dtypes`, the `techniques` observed there, whether the region `uses_hardware_loops`, and the "
    "`confidence` and `evidence` of the link. Each entry names a `file` and a `locator` (lines, function or "
    "address range). Once a case passes numerically and is a performance target, its self-check feedback "
    "names at most two matching entries (exact dimensions, same shape class, or same operation kind) beside "
    "its next step. A region that uses hardware loops shows a technique to express with ordinary commands. "
    "The map grants no numerical or performance credit: check each change with ordinary selfchecks.\n"
    "A match describes operation geometry, not equal fusion, layouts, quantization or timing boundaries.\n"
)


# --- the document -----------------------------------------------------------------------------------


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _ints(value) -> bool:
    return isinstance(value, list) and bool(value) and all(type(item) is int for item in value)


def _locator(value) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    lines = value.get("lines")
    if lines is not None and not (_ints(lines) and len(lines) == 2 and 0 < lines[0] <= lines[1]):
        return False
    if "function" in value and not (isinstance(value["function"], str) and value["function"]):
        return False
    address = value.get("address_range")
    if (
        address is not None
        and not (isinstance(address, str) and address)
        and not (
            isinstance(address, list)
            and len(address) == 2
            and all(isinstance(item, (int, str)) and item != "" for item in address)
        )
    ):
        return False
    return lines is not None or "function" in value or address is not None


def _relative_file(value) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        return False
    return all(part not in ("", ".", "..") for part in value.split("/"))


def _dims(value) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and (type(item) is int or _ints(item)) for key, item in value.items()
    )


def _entry_problem(entry) -> str | None:
    if not isinstance(entry, dict):
        return "an entry is not a mapping"
    if not isinstance(entry.get("id"), str) or not entry["id"]:
        return "an entry has no id"
    name = entry["id"]
    if not _relative_file(entry.get("file")):
        return f"{name}: file must be a relative reference path"
    if not _locator(entry.get("locator")):
        return f"{name}: locator needs lines [a, b], a function or an address_range"
    operation = entry.get("operation")
    if not isinstance(operation, dict) or operation.get("kind") not in KINDS:
        return f"{name}: operation.kind must be one of {', '.join(KINDS)}"
    if "dims" in operation and not _dims(operation["dims"]):
        return f"{name}: operation.dims must map names to integers"
    if not isinstance(operation.get("epilogue", []), list) or not isinstance(operation.get("dtypes", {}), dict):
        return f"{name}: operation.epilogue is a list and operation.dtypes a mapping"
    techniques = entry.get("techniques")
    if not isinstance(techniques, list) or any(not isinstance(item, str) for item in techniques):
        return f"{name}: techniques must be a list of strings"
    if type(entry.get("uses_hardware_loops")) is not bool:
        return f"{name}: uses_hardware_loops must be true or false"
    if entry.get("confidence") not in CONFIDENCE:
        return f"{name}: confidence must be verified or inferred"
    if not isinstance(entry.get("evidence"), str):
        return f"{name}: evidence must be a string"
    return None


def validate(document) -> dict:
    """The document itself when it is a well-formed :data:`SCHEMA` map; raises ``ValueError`` otherwise."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"reference map must declare schema {SCHEMA}")
    if not _sha(document.get("reference_manifest_sha256")):
        raise ValueError("reference map must name the reference manifest it describes (reference_manifest_sha256)")
    entries = document.get("entries")
    if not isinstance(entries, list) or len(entries) > _MAX_ENTRIES:
        raise ValueError(f"reference map entries must be a list of at most {_MAX_ENTRIES}")
    for entry in entries:
        problem = _entry_problem(entry)
        if problem:
            raise ValueError(f"reference map entry invalid: {problem}")
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("reference map entry ids must be unique")
    if not isinstance(document.get("coverage", {}), dict) or not isinstance(document.get("not_covered", []), list):
        raise ValueError("reference map coverage is a mapping and not_covered a list")
    return document


def read_pin(pin) -> tuple[bytes, dict]:
    """``(bytes, document)`` of one ``{"path", "sha256"}`` pin; raises ``ValueError`` on any mismatch."""
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"} or not _sha(pin.get("sha256")):
        raise ValueError("reference map requires an exact {path, sha256} pin")
    path = Path(pin["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError(f"reference map unavailable: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError(f"reference map bytes changed: {path}")
    return raw, validate(json.loads(raw))


def selected() -> dict | None:
    """The map this process's launch pinned (:data:`ENV`), or None; cached by the file's identity."""
    raw = os.environ.get(ENV)
    if not raw:
        return None
    try:
        pin = json.loads(raw)
        stat = Path(pin["path"]).stat()
        identity = (raw, stat.st_mtime_ns, stat.st_size)
        if _CACHE.get("identity") != identity:
            _CACHE.clear()
            _CACHE.update(identity=identity, document=read_pin(pin)[1])
        return _CACHE["document"]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def stage(workspace, run_dir) -> dict | None:
    """Serve the pinned map read-only in the workspace and the run record, with its TASK.md section.

    The same bytes and the same section in every arm that selects the map; nothing without a pin. Runs
    once at fresh setup, before the treatment seal; a resumed run verifies the sealed bytes instead.
    """
    raw = os.environ.get(ENV)
    if not raw:
        return None
    pin = json.loads(raw)
    data, _ = read_pin(pin)
    roots = (Path(workspace), Path(run_dir))
    for root in roots:
        path = root / FILENAME
        if path.exists() or path.is_symlink():
            raise ValueError("reference map is already staged; resume must verify the original bytes")
    task = (roots[0] / "TASK.md").read_text()
    for root in roots:
        path = root / FILENAME
        path.write_bytes(data)
        path.chmod(0o444)
        (root / "TASK.md").write_text(task + TASK_SECTION)
    return {"path": FILENAME, "sha256": pin["sha256"], "source": pin["path"]}


# --- a case's operation, from its own declaration -------------------------------------------------------


def _pair(value, default) -> tuple:
    if type(value) is int:
        return (value, value)
    if _ints(value) and len(value) in (1, 2):
        return (value[0], value[-1])
    return default


def _padding(value) -> tuple | None:
    """``(top, bottom, left, right)`` from one number, ``[h, w]`` or the four sides; None when unstated."""
    if type(value) is int:
        return (value,) * 4
    if _ints(value) and len(value) == 2:
        return (value[0], value[0], value[1], value[1])
    if _ints(value) and len(value) == 4:
        return tuple(value)
    return None


def _listed(value):
    return list(value) if value is not None else None


def _window(declaration: dict) -> dict | None:
    """``{N, C, H, W, K, R, S, stride, pad}`` of a declared 2-D window (K absent without a weight)."""
    from . import shape_classes

    operation = declaration.get("operation") if isinstance(declaration.get("operation"), dict) else {}
    attrs = operation.get("attributes") if isinstance(operation.get("attributes"), dict) else {}
    tensors = shape_classes._tensors(declaration)
    ifm = shape_classes._shape_of(tensors, attrs.get("ifm"))
    if ifm is None:
        inputs = shape_classes._by_role(declaration, "input")
        ifm = inputs[0] if inputs else None
    layout = attrs.get("layout")
    order = layout.split("_", 1)[0].lower() if isinstance(layout, str) else ""
    if ifm is None or len(ifm) != 4 or tuple(sorted(order)) != ("c", "h", "n", "w"):
        return None
    axis = dict(zip(order, ifm))
    kernel = attrs.get("kernel")
    weight = shape_classes._shape_of(tensors, attrs.get("weight"))
    if weight is None:
        weights = shape_classes._by_role(declaration, "weight")
        weight = weights[0] if len(weights) == 1 else None
    if _ints(kernel) and len(kernel) == 4:
        r, s, _, k = kernel
    else:
        r, s = _pair(kernel, (attrs.get("kh"), attrs.get("kw")))
        if r is None:
            r, s = _pair(attrs.get("pool_size") or attrs.get("window"), (None, None))
        k = weight[-1] if weight is not None else None
    if type(r) is not int or type(s) is not int:
        return None
    dims = {
        "N": axis["n"],
        "C": axis["c"],
        "H": axis["h"],
        "W": axis["w"],
        "R": r,
        "S": s,
        # An unstated step or padding stays unknown: it never equals a stated one.
        "stride": _listed(_pair(attrs.get("stride"), None)),
        "pad": _listed(_padding(attrs.get("padding"))),
    }
    if type(k) is int:
        dims["K"] = k
    return dims


def _residual(op: str, declaration: dict) -> bool:
    from . import shape_classes

    inputs = shape_classes._by_role(declaration, "input")
    return "residual" in op or ("add" in op and len(inputs) == 2 and inputs[0] == inputs[1])


def case_operation(declaration) -> dict | None:
    """``{"kind", "dims", "shape"}`` of one declaration in the map's vocabulary, or None."""
    if not isinstance(declaration, dict):
        return None
    from merlin.targetgen import semantic_families as sf

    from . import shape_classes

    operation = declaration.get("operation") if isinstance(declaration.get("operation"), dict) else {}
    attrs = operation.get("attributes") if isinstance(operation.get("attributes"), dict) else {}
    op = str(operation.get("op") or "").lower()
    try:
        shape = shape_classes.describe(declaration)
    except Exception:  # noqa: BLE001 -- an undescribable case simply has no shape
        shape = None
    semantic = (
        (declaration.get("semantic") or {}).get("semantic_family")
        if isinstance(declaration.get("semantic"), dict)
        else None
    )
    semantic = semantic or sf.from_op(op or None)
    family = (shape or {}).get("family")
    if family in ("dense", "gemv"):
        m, k, n = (shape["extents"][axis] for axis in ("m", "k", "n"))
        return {"kind": "matmul", "dims": {"M": m, "K": k, "N": n}, "shape": shape}
    window = _window(declaration)
    if family == "conv" or (semantic == "contraction" and window is not None):
        groups = attrs.get("groups")
        depthwise = "depthwise" in op or (
            type(groups) is int and groups > 1 and window is not None and groups == window["C"]
        )
        return {"kind": "depthwise_conv" if depthwise else "conv2d", "dims": window or {}, "shape": shape}
    if semantic == "reduction" and "pool" in op:
        return {"kind": "pool", "dims": window or {}, "shape": None}
    if semantic == "elementwise_map":
        kind = "requantize" if "requant" in op else "residual_add" if _residual(op, declaration) else "elementwise"
        extents = (shape or {}).get("extents") or {}
        dims = {"M": extents["m"], "N": extents["n"]} if extents.get("m") and extents.get("n") else {}
        return {"kind": kind, "dims": dims, "shape": None}
    # A connected case declares several products; it is matched by its largest one.
    products = _component_products(op, attrs)
    if products:
        m, k, n = max(products, key=lambda product: product[0] * product[1] * product[2])
        extents = {"m": m, "k": k, "n": n}
        return {"kind": "matmul", "dims": {"M": m, "K": k, "N": n}, "shape": {"extents": extents, "work": m * k * n},
                "component_of": op}
    return None


def _component_products(op: str, attrs: dict) -> list[tuple[int, int, int]]:
    """``(M, K, N)`` of each matrix product a connected case declares: a sibling feed's attributes, or a
    component program's matmul nodes resolved from their operand shapes."""
    if all(type(attrs.get(axis)) is int for axis in ("M", "K", "H", "N")) and "sibling" in op:
        m, k, h, n = (attrs[axis] for axis in ("M", "K", "H", "N"))
        return [(m, k, h), (m, k, h), (m, h, n)]
    program = attrs.get("program") if isinstance(attrs.get("program"), dict) else None
    if program is None:
        return []
    shapes = {}
    for item in program.get("inputs") or ():
        shape = item.get("shape") if isinstance(item, dict) else None
        if isinstance(shape, list) and len(shape) == 2 and all(type(v) is int for v in shape):
            shapes[item.get("name")] = tuple(shape)
    products = []
    for node in program.get("nodes") or ():
        if not isinstance(node, dict):
            continue
        operands = [shapes.get(name) for name in node.get("inputs") or ()]
        if node.get("op") == "matmul" and len(operands) == 2 and all(operands) and operands[0][1] == operands[1][0]:
            (m, k), (_, n) = operands
            products.append((m, k, n))
            shapes[node.get("name")] = (m, n)
        elif len(operands) >= 1 and operands[0] and all(item == operands[0] for item in operands):
            shapes[node.get("name")] = operands[0]  # elementwise nodes keep their operand shape
    return products


# --- matching -----------------------------------------------------------------------------------------


def _normal(key: str, value):
    if key == "stride":
        return _pair(value, None)
    if key == "pad":
        return _padding(value)
    return tuple(value) if isinstance(value, list) else value


def _extents(kind: str, dims: dict) -> tuple | None:
    """``(M, K, N)`` of an entry's dimensions: the product, or a window's implicit product."""
    if kind == "matmul" and all(type(dims.get(axis)) is int for axis in ("M", "K", "N")):
        return dims["M"], dims["K"], dims["N"]
    if kind in ("conv2d", "depthwise_conv") and all(
        type(dims.get(axis)) is int for axis in ("N", "C", "H", "W", "K", "R", "S")
    ):
        stride, pad = _pair(dims.get("stride"), None), _padding(dims.get("pad"))
        if stride is None or pad is None:
            return None  # an unstated window step leaves the class unknown
        (sh, sw), (top, bottom, left, right) = stride, pad
        ho = (dims["H"] + top + bottom - dims["R"]) // sh + 1
        wo = (dims["W"] + left + right - dims["S"]) // sw + 1
        if ho <= 0 or wo <= 0:
            return None
        depth = dims["R"] * dims["S"] * (1 if kind == "depthwise_conv" else dims["C"])
        return dims["N"] * ho * wo, depth, dims["K"]
    return None


def _klass(extents, edge) -> str | None:
    from . import shape_classes

    if extents is None:
        return None
    m, k, n = extents
    found = shape_classes.classes({"extents": {"m": m, "k": k, "n": n}, "work": m * k * n}, edge)
    return found[0] if found else None


def grade(entry: dict, case: dict, edge) -> str | None:
    """``exact``, ``same_product`` (another product kind with the same (M, K, N)), ``same_class``,
    ``technique`` or None (another kind) for one entry and one case."""
    operation = entry.get("operation") or {}
    kind = operation.get("kind")
    dims, have = operation.get("dims") or {}, case.get("dims") or {}
    if kind != case.get("kind"):
        # A pointwise convolution and a matrix product of the same (M, K, N) are the same work.
        if kind in _PRODUCT_KINDS and case.get("kind") in _PRODUCT_KINDS:
            mine = _extents(kind, dims)
            if mine is not None and mine == _extents(case.get("kind"), have):
                return "same_product"
        return None
    if kind == "other":
        return None

    def same(key, value):
        stated = _normal(key, value)
        return stated is not None and key in have and _normal(key, have[key]) == stated

    if dims and have and set(have) <= set(dims) and all(same(key, value) for key, value in dims.items()):
        return "exact"
    shape = case.get("shape")
    if shape and isinstance(edge, int) and edge > 0:
        from . import shape_classes

        mine = shape_classes.classes(shape, edge)
        theirs = _klass(_extents(kind, dims), edge)
        if mine and theirs is not None and mine[0] == theirs:
            return "same_class"
    return "technique"


def matches(document: dict, case: dict, edge, limit: int = PER_CASE) -> list[tuple[dict, str]]:
    """The best ``(entry, grade)`` pairs for one case: exact first, then same product, same class,
    technique; verified links before inferred ones, then the entry whose implicit product is closest in
    size to the case's, then map order."""
    import math

    mine = _extents(case.get("kind"), case.get("dims") or {})
    found = []
    for index, entry in enumerate(document.get("entries") or ()):
        value = grade(entry, case, edge)
        if value is not None:
            operation = entry.get("operation") or {}
            theirs = _extents(operation.get("kind"), operation.get("dims") or {})
            distance = (
                sum(abs(math.log(a / b)) for a, b in zip(mine, theirs) if a > 0 and b > 0)
                if mine is not None and theirs is not None
                else math.inf
            )
            found.append(
                ((GRADES.index(value), entry.get("confidence") != "verified", distance, index), entry, value)
            )
    found.sort(key=lambda item: item[0])
    return [(entry, value) for _, entry, value in found[:limit]]


# --- lines --------------------------------------------------------------------------------------------


def _where(entry: dict) -> str:
    locator = entry.get("locator") or {}
    text = entry["file"]
    if locator.get("function"):
        text += f":{locator['function']}"
    if locator.get("lines"):
        text += ":{}-{}".format(*locator["lines"])
    elif locator.get("address_range") is not None:
        address = locator["address_range"]
        text += "@" + ("-".join(str(item) for item in address) if isinstance(address, list) else str(address))
    return text


def describe_operation(operation: dict) -> str:
    """``conv2d 3x3 56x56x64→64/s2``, ``matmul 64x128x256`` (M x K x N) or the bare kind."""
    kind, dims = operation.get("kind"), operation.get("dims") or {}
    if kind == "matmul" and all(type(dims.get(axis)) is int for axis in ("M", "K", "N")):
        return f"matmul {dims['M']}x{dims['K']}x{dims['N']}"
    if kind in _WINDOW_KINDS and all(type(dims.get(axis)) is int for axis in ("H", "W", "C")):
        text = f"{kind} "
        if type(dims.get("R")) is int and type(dims.get("S")) is int:
            text += f"{dims['R']}x{dims['S']} "
        text += f"{dims['H']}x{dims['W']}x{dims['C']}"
        if type(dims.get("K")) is int:
            text += f"→{dims['K']}"
        stride = _pair(dims.get("stride"), None)
        if stride is not None and stride != (1, 1):
            text += f"/s{stride[0]}" if stride[0] == stride[1] else f"/s{stride[0]}x{stride[1]}"
        return text
    if dims:
        return f"{kind} " + "x".join(str(value) for value in dims.values() if type(value) is int)
    return str(kind)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def line(entry: dict, value: str, siblings=()) -> str:
    """One compact reference line for a case."""
    text = f"reference: {_where(entry)} {describe_operation(entry.get('operation') or {})} ({value})"
    if entry.get("confidence") != "verified":
        text += " [inferred]"
    techniques = [_clip(item.strip(), TECHNIQUE_CHARS) for item in entry.get("techniques") or () if item.strip()]
    if techniques:
        text += " — " + "; ".join(techniques[:TECHNIQUES])
    if entry.get("uses_hardware_loops"):
        text += " (uses hardware loops: translate to ordinary commands)"
    names = list(siblings)
    if names:
        text += "; compare on " + ", ".join(names[:2]) + (f" (+{len(names) - 2})" if len(names) > 2 else "")
    while len(text.encode()) > LINE_BYTES:
        text = text[: len(text) - 8].rstrip() + "..."
    return text


# --- per-case delivery ----------------------------------------------------------------------------------


def targets(families) -> dict[str, list[str]]:
    """``{case: held-out siblings}`` of the roll-up's performance targets: each bundle's representative and
    its large or connected passing cases with headroom (:mod:`.family_rollup`)."""
    out: dict[str, list[str]] = {}
    for bundle in (families or {}).get("bundles") or ():
        members = [bundle["representative"], *(bundle.get("siblings") or ())]
        for name in (bundle["representative"], *(bundle.get("targets") or ())):
            out.setdefault(name, [other for other in members if other != name])
    return out


def build(families, workspace=None, *, document=None, edge=None) -> dict | None:
    """``{schema, cases: {case: {lines, matches}}}`` for the performance targets with a matching entry."""
    document = selected() if document is None else document
    if not document or not document.get("entries"):
        return None
    from . import case_declarations, family_rollup

    edge = family_rollup._edge() if edge is None else edge
    members = family_rollup.inventory(workspace) if workspace is not None else {}
    cases = {}
    for name, siblings in targets(families).items():
        declaration = case_declarations.load(name, workspace)
        if declaration is None and isinstance(members.get(name), dict):
            declaration = family_rollup._declaration_of(members[name])
        case = case_operation(declaration)
        if case is None:
            continue
        found = matches(document, case, edge)
        if found:
            cases[name] = {
                "lines": [line(entry, value, siblings) for entry, value in found],
                "matches": [{"id": entry["id"], "grade": value, "semantic_compatibility": "unknown"}
                            for entry, value in found],
            }
    if not cases:
        return None
    return {
        "schema": FEEDBACK_SCHEMA,
        "advisory_only": True,
        "scope": "performance targets (passing bundle representatives; large or connected passing cases with "
        "headroom) matched by operation kind and declared dimensions or shape class",
        "cases": cases,
    }


def attach(brief, families, *, workspace=None) -> dict | None:
    """Put :data:`KEY` on the brief and ``reference_lines`` on the roll-up; never raises."""
    try:
        found = build(families, workspace)
    except Exception:  # noqa: BLE001 -- advisory only
        return None
    if found is None:
        return None
    if isinstance(brief, dict):
        brief[KEY] = found
    if isinstance(families, dict):
        families["reference_lines"] = {name: record["lines"] for name, record in found["cases"].items()}
    return found


__all__ = [
    "ENV",
    "FILENAME",
    "KEY",
    "KINDS",
    "SCHEMA",
    "TASK_SECTION",
    "attach",
    "build",
    "case_operation",
    "describe_operation",
    "grade",
    "line",
    "matches",
    "read_pin",
    "selected",
    "stage",
    "targets",
    "validate",
]
