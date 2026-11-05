"""List the composition-catalog routes that apply to one test case, with their library symbols.

    python composition_routes.py [--case NAME] [--ws .] [--inventory N] [--library DIR]

Without ``--case`` it lists every chain route and the composition-catalog rows behind it. With
``--case`` it reads that case's public files (``<ws>/<group>/NAME/capsule.yaml`` and, for a model or
program, ``capsule.interface.mlir``) and reports the producer -> consumer chains the declaration or
the IR shows, each with:

* the route, the catalog rows behind it (``composition_capabilities``) and whether your roster holds
  their modules (``modules_selected`` / ``missing_selected_modules``);
* one library ``module:symbol`` to start from and whether a file under ``submission/`` imports it;
* a paste-ready command: the route's recipe when one exists, else the symbol's contract;
* for a model or program, the ``source_composition.json`` options that match what the IR shows
  (``reference/source-composition.md``).

The chains come from the case's own structure (declared epilogue stages, component nodes and shared
inputs, the IR's provenance regions), never from its name. Everything is a pointer: a route that
does not fit refuses or keeps the original; keep your own lowering where it measures better. No
performance claim is made here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = "merlin.composition_routes.v1"
SKILLS = ".agents/skills"
HERE = "heavy-pass-recipes"
REQUEST_FILE = "submission/source_composition.json"
REQUEST_SCHEMA = "merlin.source-composition-preferences.v1"
REQUEST_REFERENCE = f"{SKILLS}/{HERE}/reference/source-composition.md"
#: Workspace children never searched for a case (author trees, outputs, channels).
SKIP = frozenset({"submission", "variants", "selfcheck_out", "qa", "build", "__pycache__"})
MAX_BYTES = 4 << 20

#: Chain kinds -> catalog rows, a library symbol to start from, recipe and skills. ``symbol`` is one of
#: the first row's symbols (checked by the skill's test against ``composition_capabilities``).
ROUTES = (
    {
        "id": "producer_pool",
        "chain": "convolution or product -> rounded readout -> spatial pool -> rounded readout",
        "rows": (
            "bounded_spatial_packing_and_rolling",
            "bounded_spatial_max_strips",
            "ordered_quantization_boundaries",
        ),
        "symbol": "merlin.compile.scheduling.rolling_spatial:emit_rolling_spatial_producer",
        "recipe": "spatial_chain_recipe.py",
        "skills": ("heavy-pass-recipes", "fusion"),
    },
    {
        "id": "producer_epilogue",
        "chain": "product -> bias / scale / activation / requantization in the readout",
        "rows": ("ordered_quantization_boundaries", "checked_pointwise_fusion"),
        "symbol": "merlin.compile.scheduling.spatial_readout:prove_spatial_readout",
        "recipe": None,
        "skills": ("fusion", "quantization"),
    },
    {
        "id": "residual_add",
        "chain": "two quantized producers -> ordered loaded sum (residual add) -> rounded output",
        "rows": ("ordered_quantization_boundaries",),
        "symbol": "merlin.llvmlower.quantized_affine_pair:prove_ordered_loaded_sum",
        "recipe": None,
        "skills": ("quantization", "fusion"),
    },
    {
        "id": "shared_input_siblings",
        "chain": "one input -> several sibling products (shared operand, separate outputs)",
        "rows": ("member_projection_views", "source_sibling_host_preparation", "independent_shared_input_components"),
        "symbol": "merlin.compile.scheduling.shared_input_groups:find_shared_input_groups",
        "recipe": None,
        "skills": ("fusion", "compose-pass-pipeline"),
        "request": "shared_input_siblings",
    },
    {
        "id": "independent_heads",
        "chain": "slices of one tensor -> independent per-head products -> original readouts",
        "rows": ("independent_head_and_plane_products", "original_head_batch_request"),
        "symbol": "merlin.transforms.head_batching:plan_independent_head_batching",
        "recipe": None,
        "skills": ("compose-pass-pipeline",),
        "request": "independent_heads",
    },
    {
        "id": "connected_stages",
        "chain": "several stages with preserved tensor identities (whole program or model)",
        "rows": ("connected_source_execution", "source_fusion_exploration"),
        "symbol": "merlin.compile.connected_regions:plan_connected_region",
        "recipe": None,
        "skills": ("compose-pass-pipeline", "fusion"),
    },
    {
        "id": "parameter_preparation",
        "chain": "immutable parameter -> layout / quantization preparation -> product, on every call",
        "rows": ("private_intermediate_storage",),
        "symbol": "merlin.llvmlower.weight_prepack:prepacked_bundle",
        "recipe": None,
        "skills": ("compose-pass-pipeline",),
        "request": "prepack_weights",
    },
)
BY_ID = {route["id"]: route for route in ROUTES}

#: ``source_composition.json`` options: what each requests and which IR observation suggests it.
REQUESTS = (
    ("prepack_weights", "prepare immutable parameter layouts once instead of on every call"),
    ("share_exact_tensor_preparations", "merge exact duplicate view/constant/integer-to-float preparations"),
    ("dense_input_views", "pass proven dense read-only views to device groups instead of layout copies"),
    ("propagate_layouts", "request checked pre-group layouts under original software element-order permissions"),
    ("reuse_intermediate_storage", "request proven host allocation reuse under original software lifetime permissions"),
    ("shared_input_siblings", "compose sibling products that read the same input"),
    ("independent_heads", "refold independent head slices through the batch-slice interface"),
    ("op_profile", "instrumented per-operation profile (diagnostic; keep an uninstrumented control)"),
)
REQUIRED_REQUEST_FIELDS = ("shared_input_siblings", "prepack_weights")

#: Provenance families a value passes through unchanged in kind (no product, no window).
_THROUGH = frozenset({"layout", "cast", "quantize", "elementwise", "minmax"})
_PREPARATION = frozenset({"layout", "cast", "quantize"})
_PRODUCERS = frozenset({"contraction", "attention"})
_INITIALIZERS = frozenset({"arith.constant", "tensor.splat", "tensor.empty", "linalg.fill"})


# --- declarations --------------------------------------------------------------------------------------


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _words(value) -> list[str]:
    return [str(item).lower() for item in value] if isinstance(value, list | tuple) else []


def _nodes(declaration: dict) -> list[dict]:
    attrs = _mapping(_mapping(declaration.get("operation")).get("attributes"))
    for holder in (_mapping(declaration.get("component_program")), _mapping(attrs.get("program"))):
        if isinstance(holder.get("nodes"), list):
            return [node for node in holder["nodes"] if isinstance(node, dict)]
    return []


def _is_producer(declaration: dict) -> bool:
    semantic = str(_mapping(declaration.get("semantic")).get("semantic_family") or "")
    if semantic in _PRODUCERS:
        return True
    attrs = _mapping(_mapping(declaration.get("operation")).get("attributes"))
    roles = [_mapping(t).get("role") for t in declaration.get("inputs") or ()]
    return "weight" in roles or "weight" in attrs or "kh" in attrs or "kernel" in attrs


def declaration_signals(declaration) -> dict[str, list[str]]:
    """``{route id: [evidence, ...]}`` from one declaration's structure (never its name)."""
    found: dict[str, list[str]] = defaultdict(list)
    if not isinstance(declaration, dict):
        return {}
    operation = _mapping(declaration.get("operation"))
    op = str(operation.get("op") or "").lower()
    attrs = _mapping(operation.get("attributes"))
    stages = _words(attrs.get("epilogue"))
    pools = [s for s in stages if "pool" in s] + sorted(k for k in attrs if str(k).startswith("pool_"))[:1]
    if _is_producer(declaration):
        if pools:
            found["producer_pool"].append(f"declared epilogue pools the product ({', '.join(stages) or 'pool'})")
        rest = [s for s in stages if "pool" not in s]
        scaled = sorted(k for k in attrs if isinstance(k, str) and ("scale" in k or k == "bias"))
        if rest or scaled:
            found["producer_epilogue"].append(f"declared readout stages: {', '.join(dict.fromkeys(rest + scaled))}")
    if "residual" in op or any("residual" in s for s in stages):
        found["residual_add"].append(f"declared operation {op} sums two quantized operands")
    siblings = attrs.get("matmuls")
    if isinstance(siblings, list) and len(siblings) > 1:
        found["shared_input_siblings"].append(f"{len(siblings)} declared products share an operand")
    nodes = _nodes(declaration)
    readers: dict[str, list[str]] = defaultdict(list)
    names = {str(node.get("name")) for node in nodes}
    for node in nodes:
        for name in node.get("inputs") or ():
            readers[str(name)].append(str(node.get("name")))
        consumed = [str(name) for name in node.get("inputs") or () if str(name) in names]
        if consumed:
            found["connected_stages"].append(f"node {node.get('name')} consumes {', '.join(consumed)}")
            kind = str(node.get("op") or "").lower()
            if "pool" in kind:
                found["producer_pool"].append(f"node {node.get('name')} pools {', '.join(consumed)}")
            if "residual" in kind or "add" == kind:
                found["residual_add"].append(f"node {node.get('name')} adds {', '.join(consumed)}")
    shared = {name: users for name, users in readers.items() if len(users) > 1 and name not in names}
    for name, users in sorted(shared.items()):
        found["shared_input_siblings"].append(f"input {name} feeds {', '.join(users)}")
    if "attention" in op or "attention" == _mapping(declaration.get("semantic")).get("semantic_family"):
        found["independent_heads"].append(f"declared operation {op} has per-head products")
    if declaration.get("kind") == "model" or op == "model":
        found["connected_stages"].append("whole model: every device group is one stage of a connected program")
        if attrs.get("weights") or any(_mapping(t).get("role") == "weight" for t in declaration.get("inputs") or ()):
            found["parameter_preparation"].append("the model declares immutable weights")
    return {key: value[:4] for key, value in found.items()}


# --- the IR inventory ------------------------------------------------------------------------------------


def _tag(op, key):
    attribute = op.attributes.get(key)
    data = getattr(attribute, "data", None) if attribute is not None else None
    return data if isinstance(data, str) else None


def _elements(value) -> int | None:
    try:
        shape = value.type.get_shape()
    except AttributeError:
        return None
    total = 1
    for extent in shape:
        if not isinstance(extent, int) or extent < 0:
            return None
        total *= extent
    return total


def ir_inventory(source) -> dict:
    """Producer -> consumer chains in a case's typed IR, from its ``prov.region_id/family/op`` tags.

    ``source`` is IR text or a path. Needs ``merlin.common.mlir_query`` (your editable library); without
    it the inventory is ``{"status": "unavailable"}``. Counts are structural observations only.
    """
    try:
        from xdsl.dialects.builtin import TensorType
        from xdsl.ir import BlockArgument

        from merlin.common import mlir_query as mq
    except ImportError as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    try:
        module = mq.parse(str(source))
    except Exception as error:  # noqa: BLE001 -- an unparsable IR is reported, never fatal
        return {"status": "unavailable", "reason": f"IR does not parse: {type(error).__name__}"}
    region, ops_of = {}, defaultdict(list)
    for op in module.walk():
        rid = _tag(op, "prov.region_id")
        if rid is None:
            continue
        ops_of[rid].append(op)
        for result in op.results:
            region[result] = rid
    kind = {rid: (_tag(ops[0], "prov.family") or "", _tag(ops[0], "prov.op") or "") for rid, ops in ops_of.items()}

    def external(rid):
        return [v for op in ops_of[rid] for v in op.operands if isinstance(v.type, TensorType) and region.get(v) != rid]

    def constant(value):
        """A literal or an output initializer, not data another stage produced."""
        owner = None if isinstance(value, BlockArgument) else getattr(value, "op", None)
        return owner is not None and mq.op_name(owner) in _INITIALIZERS

    def reaches(values) -> set:
        seen, found, stack = set(), set(), list(values)
        while stack and len(seen) < 8192:
            value = stack.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            rid = region.get(value)
            if rid is None:
                continue
            if kind[rid][0] not in _THROUGH:
                found.add(kind[rid][0])
            else:
                stack.extend(external(rid))
        return found

    def root(value):
        """Follow layout/cast regions back to the tensor they view; ``(root, prepared_by)``."""
        steps = []
        for _ in range(64):
            rid = region.get(value)
            if rid is None or kind[rid][0] not in _PREPARATION:
                return value, steps
            inputs = [v for v in external(rid) if not constant(v)]
            if not inputs:
                return value, steps
            steps.append(rid)
            value = inputs[0]
        return value, steps

    products = [rid for rid, (family, _) in kind.items() if family in _PRODUCERS]
    users, batched, viewed, prepared_args = defaultdict(list), [], set(), defaultdict(set)
    for rid in products:
        for value in external(rid):
            if constant(value):
                continue
            base, steps = root(value)
            users[id(base)].append((rid, value, base))
            if len(value.type.get_shape() if hasattr(value.type, "get_shape") else ()) >= 3:
                batched.append(rid)
            if steps and kind[steps[0]][0] == "layout":
                viewed.add(rid)
            if steps and isinstance(base, BlockArgument):
                prepared_args[id(base)].add(steps[-1])
    siblings, heads = [], []
    for group in users.values():
        members = sorted({rid for rid, _, _ in group})
        if len(members) < 2 or members in siblings or members in heads:
            continue
        base = group[0][2]
        whole = _elements(base)
        if whole is not None and any((_elements(v) or whole) < whole for _, v, _ in group):
            heads.append(members)
        else:
            siblings.append(members)
    pools = sorted(
        rid
        for rid, (family, name) in kind.items()
        if ("pool" in family or "pool" in name) and "contraction" in reaches(external(rid))
    )
    consumers = defaultdict(set)
    for rid in ops_of:
        for value in external(rid):
            if region.get(value) in products:
                consumers[region[value]].add(kind[rid][0])
    epilogues = sorted(rid for rid in products if consumers[rid] & _THROUGH)
    chained = sorted(rid for rid in products if reaches(external(rid)) & _PRODUCERS)
    residual = []
    for rid, (family, name) in kind.items():
        if name != "add":
            continue
        inputs = [v for v in external(rid) if not constant(v)]
        if len(inputs) >= 2 and any("contraction" in reaches([v]) for v in inputs):
            residual.append(rid)
    return {
        "status": "observed",
        "products": len(products),
        "regions": len(kind),
        "shared_input_groups": siblings[:8],
        "head_slice_groups": heads[:8],
        "batched_products": sorted(set(batched))[:8],
        "pools_fed_by_products": pools[:8],
        "products_with_readout_stages": len(epilogues),
        "products_fed_by_products": len(chained),
        "residual_adds": sorted(residual)[:8],
        "products_reading_views": len(viewed),
        "layout_regions": sum(family == "layout" for family, _ in kind.values()),
        "connected_tensor_edges": len({(region[value], rid, id(value))
            for rid in ops_of for value in external(rid)
            if region.get(value) in kind and not constant(value)}),
        "parameters_prepared_per_call": sum(1 for steps in prepared_args.values() if steps),
        "parameters_prepared_more_than_once": sum(1 for steps in prepared_args.values() if len(steps) > 1),
    }


def ir_signals(inventory: dict) -> dict[str, list[str]]:
    """Route evidence from :func:`ir_inventory`."""
    if not isinstance(inventory, dict) or inventory.get("status") != "observed":
        return {}
    found: dict[str, list[str]] = {}
    if inventory["pools_fed_by_products"]:
        found["producer_pool"] = [f"IR pool regions fed by a product: {', '.join(inventory['pools_fed_by_products'])}"]
    if inventory["products_with_readout_stages"]:
        found["producer_epilogue"] = [
            f"IR: {inventory['products_with_readout_stages']} products feed scale, activation or quantization regions"
        ]
    if inventory["residual_adds"]:
        found["residual_add"] = [f"IR adds of a product and another tensor: {', '.join(inventory['residual_adds'])}"]
    if inventory["shared_input_groups"]:
        found["shared_input_siblings"] = [
            "IR products reading one input: "
            + "; ".join(", ".join(group) for group in inventory["shared_input_groups"][:3])
        ]
    if inventory["head_slice_groups"] or inventory["batched_products"]:
        found["independent_heads"] = [
            "IR products over slices of one tensor or batched head operands: "
            + "; ".join(
                ", ".join(group) for group in (inventory["head_slice_groups"] or [inventory["batched_products"]])[:3]
            )
        ]
    if inventory["products_fed_by_products"]:
        found["connected_stages"] = [
            f"IR: {inventory['products_fed_by_products']} of {inventory['products']} "
            "products consume another product's result"
        ]
    if inventory.get("layout_regions"):
        found["host_layout"] = [f"IR: {inventory['layout_regions']} explicit layout regions; "
                                "the owner must prove coordinate/effect preservation and actual savings"]
    if inventory.get("connected_tensor_edges"):
        found["intermediate_storage"] = [f"IR: {inventory['connected_tensor_edges']} tensor producer/consumer edges; "
                                         "the owner must prove final allocation lifetimes before reuse"]
    if inventory["parameters_prepared_per_call"]:
        found["parameter_preparation"] = [
            f"IR: {inventory['parameters_prepared_per_call']} function arguments reach a product only through "
            "layout, cast or quantization regions, repeated on every call"
        ]
    return found


# --- requests ------------------------------------------------------------------------------------------


def request_matches(signals: dict, inventory: dict | None, declaration: dict | None = None) -> list[dict]:
    """``source_composition.json`` options whose prerequisite the case shows, each with its reason."""
    inventory = inventory if isinstance(inventory, dict) and inventory.get("status") == "observed" else {}
    out = []

    def add(name, why, requires=None):
        out.append({"option": name, "why": why, **({"requires": requires} if requires else {})})

    if "parameter_preparation" in signals:
        add("prepack_weights", signals["parameter_preparation"][0])
    if inventory.get("parameters_prepared_more_than_once"):
        add(
            "share_exact_tensor_preparations",
            f"IR: {inventory['parameters_prepared_more_than_once']} function arguments are prepared more than once",
        )
    elif "parameter_preparation" in signals:
        add("share_exact_tensor_preparations", "argument preparations run inside the program; exact duplicates merge")
    if inventory.get("products_reading_views"):
        add("dense_input_views", f"IR: {inventory['products_reading_views']} products read reshaped or permuted views")
    if "host_layout" in signals:
        add("propagate_layouts", signals["host_layout"][0],
            "Original software logical_host/propagate_layout and independent-element-order permission; "
            "floating-environment permission when required. Selected head/sibling composition currently defers layout.")
    if "intermediate_storage" in signals:
        add("reuse_intermediate_storage", signals["intermediate_storage"][0],
            "Original software LLVM/static_arena, sequential execution, successful allocations "
            "and non-reentrant entry; "
            "proven final allocation/free lifetimes. Source edges do not prove reusable bytes or disposable KV state.")
    if "shared_input_siblings" in signals:
        add("shared_input_siblings", signals["shared_input_siblings"][0])
    if "independent_heads" in signals:
        add("independent_heads", signals["independent_heads"][0])
    screen = _mapping(_mapping(declaration).get("software_screen")).get("decisions") if declaration else None
    host = [d for d in screen or () if isinstance(d, dict) and d.get("placement") == "host"]
    if host:
        add("op_profile", f"{len(host)} declared operation kinds run on the host; profile them by category")
    return out


def request_example(options) -> dict:
    """A complete ``source_composition.json`` that turns on ``options`` (the two required fields always present)."""
    names = [name for name, _ in REQUESTS]
    chosen = {str(option) for option in options if option in names}
    body = {"schema": REQUEST_SCHEMA}
    for name in names:
        if name in REQUIRED_REQUEST_FIELDS or name in chosen:
            body[name] = name in chosen
    return body


def current_request(workspace: Path) -> dict | None:
    path = Path(workspace) / REQUEST_FILE
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


# --- the case ------------------------------------------------------------------------------------------


def case_dir(name: str, workspace: Path) -> Path | None:
    """``<ws>/<group>/NAME`` holding the case's public declaration or interface IR."""
    if not name or "/" in name or name.startswith("."):
        return None
    try:
        groups = sorted(p for p in Path(workspace).iterdir() if p.is_dir() and p.name not in SKIP)
    except OSError:
        return None
    for group in groups:
        if group.name.startswith("."):
            continue
        folder = group / name
        if (folder / "capsule.yaml").is_file() or (folder / "capsule.interface.mlir").is_file():
            return folder
    return None


def _yaml(path: Path) -> dict | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return None
        import yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- an unreadable declaration is simply not used
        return None
    return loaded if isinstance(loaded, dict) else None


def case_signals(folder: Path | None, *, declaration=None, workspace: Path | None = None, ir: bool = True):
    """``(signals, inventory, declaration)`` for one case folder; group cases of a model are included."""
    declaration = declaration if isinstance(declaration, dict) else (_yaml(folder / "capsule.yaml") if folder else None)
    signals = {key: list(value) for key, value in declaration_signals(declaration).items()}
    inventory = None
    is_program = (
        declaration is None
        or declaration.get("kind") in ("model", "model_slice")
        or _mapping(declaration.get("operation")).get("op") in ("model", "component_program")
    )
    if ir and folder is not None and is_program:
        name = str((declaration or {}).get("interface_mlir") or "capsule.interface.mlir")
        path = folder / name
        if "/" not in name and path.is_file() and path.stat().st_size <= 4 * MAX_BYTES:
            inventory = ir_inventory(path)
            for key, value in ir_signals(inventory).items():
                signals.setdefault(key, []).extend(value)
    groups = _mapping((declaration or {}).get("model_qualification")).get("group_capsules")
    root = Path(workspace) if workspace is not None else (folder.parent.parent if folder else None)
    for entry in groups or ():
        relative = str(_mapping(entry).get("capsule") or "")
        if root is None or not relative or relative.startswith(("/", ".")) or ".." in relative.split("/"):
            continue
        member = _yaml(root / relative / "capsule.yaml")
        for key, value in declaration_signals(member).items():
            if key != "connected_stages":
                signals.setdefault(key, []).append(f"device group case {relative}: {value[0]}")
    return {key: value[:4] for key, value in signals.items() if value}, inventory, declaration


# --- catalog and imports -------------------------------------------------------------------------------


def roster(inventory: int, library: str | None) -> tuple[str, tuple[str, ...]]:
    """Exact module names: the editable library's files if present, else the selected inventory."""
    root = Path(library) if library else Path("submission/compiler_library")
    if (root / "merlin").is_dir():
        names = []
        for path in sorted((root / "merlin").rglob("*.py")):
            parts = list(path.relative_to(root).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            names.append(".".join(parts))
        return str(root), tuple(dict.fromkeys(names))
    from merlin.compile.scheduling.library_sources import library_modules

    return f"library_modules({inventory})", tuple(library_modules(inventory))


def catalog_rows(modules) -> dict[str, dict]:
    from merlin.compile.composition_catalog import composition_capabilities

    catalog = composition_capabilities(selected_modules=tuple(modules) if modules is not None else None)
    return {
        row["name"]: {
            "access": row["access"],
            "status": row["source_availability"]["status"],
            "missing_modules": row["source_availability"]["missing_modules"],
            "symbols": row["symbols"],
            "requires": row["requires"],
            "limitations": row["limitations"],
        }
        for row in catalog["capabilities"]
    }


def submission_imports(workspace: Path) -> dict | None:
    """``{imported name: [files]}`` of the author's own files (the scheduling-api inventory reader)."""
    script = Path(__file__).resolve().parents[2] / "scheduling-api" / "scripts" / "library_inventory.py"
    if not script.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_composition_routes_inventory", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.imported_modules(Path(workspace) / "submission")
    except Exception:  # noqa: BLE001 -- unknown imports stay unknown
        return None


def imported(symbol: str, imports: dict | None) -> bool | None:
    if imports is None:
        return None
    module, _, name = symbol.partition(":")
    return bool(imports.get(module) or imports.get(f"{module}.{name}"))


def next_command(route: dict, skills: str = SKILLS) -> str:
    """The recipe for the route, else the contract of its symbol (both print JSON)."""
    if route.get("recipe"):
        return f"python {skills}/{HERE}/scripts/{route['recipe']}"
    return f"python {skills}/scheduling-api/scripts/dense_schedule_example.py contract {route['symbol']}"


def discovery_command(case: str | None = None, skills: str = SKILLS) -> str:
    return f"python {skills}/{HERE}/scripts/composition_routes.py" + (f" --case {case}" if case else "")


def describe(route_id: str, evidence, *, rows=None, imports=None, skills: str = SKILLS) -> dict:
    route = BY_ID[route_id]
    out = {
        "route": route_id,
        "chain": route["chain"],
        "evidence": list(evidence),
        "symbol": route["symbol"],
        "imported_by_submission": imported(route["symbol"], imports),
        "catalog_rows": {name: (rows or {}).get(name, {"status": "catalog_unavailable"}) for name in route["rows"]},
        "next": next_command(route, skills),
        "skills": list(route["skills"]),
    }
    if route.get("request"):
        out["source_composition_option"] = route["request"]
    return out


def report(case=None, *, workspace=".", inventory=22, library=None) -> dict:
    workspace = Path(workspace)
    try:
        label, modules = roster(inventory, library)
        rows = catalog_rows(modules)
    except Exception as error:  # noqa: BLE001 -- discovery without a catalog still lists routes
        label, rows = f"unavailable ({type(error).__name__}: {error})", {}
    imports = submission_imports(workspace)
    out = {
        "schema": SCHEMA,
        "roster": label,
        "advisory": (
            "Pointers only: each route keeps the original form when its premises fail; compare cycles of the "
            "same evidence kind on the same cases before keeping a change. No speedup is implied."
        ),
    }
    if case is None:
        out["routes"] = [describe(route["id"], [], rows=rows, imports=imports) for route in ROUTES]
        out["request_options"] = {name: text for name, text in REQUESTS}
        return out
    folder = case_dir(case, workspace)
    if folder is None:
        raise SystemExit(f"no public files for case {case} under {workspace}")
    signals, ir, declaration = case_signals(folder, workspace=workspace)
    out.update(
        case=case,
        folder=folder.relative_to(workspace).as_posix() if folder.is_relative_to(workspace) else folder.name,
        ir_inventory=ir,
    )
    out["routes"] = [describe(rid, signals[rid], rows=rows, imports=imports) for rid in BY_ID if rid in signals]
    matches = request_matches(signals, ir, declaration)
    if matches:
        current = current_request(workspace)
        out["source_composition"] = {
            "file": REQUEST_FILE,
            "current": current,
            "matches": matches,
            "example": request_example([m["option"] for m in matches if m["option"] != "op_profile"]),
            "reference": REQUEST_REFERENCE,
            "basis": "requests only: the original owner rederives membership and keeps the baseline when a "
            "request does not apply; self-check rows report requested, selected and applied effects",
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog="See reference/composition.md.")
    ap.add_argument("--case", default=None, help="a public test case name (its folder under the workspace)")
    ap.add_argument("--ws", default=".", help="workspace root (default: current directory)")
    ap.add_argument("--inventory", type=int, default=22, help="library inventory when no editable library exists")
    ap.add_argument("--library", default=None, help="editable compiler library (default: submission/compiler_library)")
    a = ap.parse_args(argv)
    root = Path(a.library) if a.library else Path(a.ws) / "submission" / "compiler_library"
    if root.is_dir():
        sys.path.insert(0, str(root.resolve()))
    print(json.dumps(report(a.case, workspace=a.ws, inventory=a.inventory, library=a.library), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
