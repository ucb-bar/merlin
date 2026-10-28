"""Per-operation optimization principles, selected for the operation families a run's corpus declares.

The architecture catalogue (``docs/design/joint_optimization_catalog.json``, staged as
``OPTIMIZATION_SUGGESTIONS.json``) answers "which library mechanisms does this target support". It
does not say what usually matters for a kind of operation: that a single-row product is bound by its
weight bytes, that neighbouring convolution outputs share input rows, that a contraction default
changes every shape class at once. Authors kept relearning those from cycles. The reviewed data file
``docs/design/operation_optimization_catalog.json`` states them per operation family, without naming
an architecture or a target, and links each family to the generic catalogue mechanisms, author skills
and recipes that implement them.

This module:

* derives a case's operation families from its own declaration (:func:`families_of`): operation,
  operand shapes, sliding window, epilogue and program structure, never its name;
* validates the data file against its JSON schema and its cross references (:func:`validate`);
* selects the entries of the families present in the corpus and drops every principle or mechanism
  that relies on an instruction role the run policy prohibits (:func:`select`,
  :func:`prohibited_roles` reads the ``prohibited_instruction_roles`` field wherever the policy
  carries it);
* stages the selection as ``OPTIMIZATION_BY_OPERATION.json`` with a short TASK.md section
  (:func:`stage`), and gives each self-check case a link to its entry (:func:`link`), which the
  performance brief puts in that case's ``optimization_next``.

Advice only: nothing here grades, ranks, gates or selects a schedule, and a failure to stage leaves the
launch as it was. No ``re``, no target name.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

SCHEMA = "merlin.operation_optimization_catalog.v1"
SELECTION_SCHEMA = "merlin.operation_optimization_selection.v1"
CATALOG = Path("docs") / "design" / "operation_optimization_catalog.json"
SCHEMA_FILE = CATALOG.with_name("operation_optimization_catalog.schema.json")
MECHANISM_CATALOG = CATALOG.with_name("joint_optimization_catalog.json")
FILENAME = "OPTIMIZATION_BY_OPERATION.json"
SUGGESTIONS = "OPTIMIZATION_SUGGESTIONS.json"
HEADING = "## Optimization principles by operation"
ROLES_FIELD = "prohibited_instruction_roles"
#: The operation families, in the order the data file lists them.
FAMILIES = (
    "contraction",
    "matrix_vector",
    "convolution",
    "attention",
    "elementwise_epilogue",
    "reduction_normalization",
    "data_movement",
    "connected_chain",
)
#: Canonical semantic family (:mod:`merlin.targetgen.semantic_families`) -> operation family.
_SEMANTIC = {
    "contraction": "contraction",
    "attention": "attention",
    "reduction": "reduction_normalization",
    "normalization": "reduction_normalization",
    "softmax": "reduction_normalization",
    "elementwise_map": "elementwise_epilogue",
    "movement": "data_movement",
}
#: Declared operations that stand for a whole program rather than one operation.
_PROGRAM_OPS = frozenset({"model", "component_program"})
#: Families whose producer can carry an epilogue in its readout.
_PRODUCERS = frozenset({"contraction", "matrix_vector", "convolution", "attention"})
_SKILLS = Path(__file__).with_name("author_skills")
#: The discovery command for composition routes (the heavy-pass-recipes skill's script).
COMPOSITION_COMMAND = "python .agents/skills/heavy-pass-recipes/scripts/composition_routes.py --case NAME"
#: Source-composition request options a model or program check accepts (``source_composition.json``).
SOURCE_COMPOSITION_OPTIONS = (
    "prepack_weights",
    "share_exact_tensor_preparations",
    "dense_input_views",
    "propagate_layouts",
    "reuse_intermediate_storage",
    "shared_input_siblings",
    "independent_heads",
    "op_profile",
)
COMPOSITION_TEXT = (
    "Compositions: the library's composition catalog (`merlin.compile.composition_catalog."
    "composition_capabilities`) covers producer -> consumer chains: a product feeding bias, scale, "
    "activation, requantization or a pool, residual adds, sibling products sharing an input and per-head "
    f"slices. `{COMPOSITION_COMMAND}` lists the routes that apply to one case, each with its catalog rows, "
    "a library `module:symbol`, whether `submission/` imports it and a recipe or contract command; "
    "self-check rows carry the same pointer as `optimization_next.composition`. Model and program checks "
    "also accept `submission/source_composition.json` requests ("
    + ", ".join(f"`{name}`" for name in SOURCE_COMPOSITION_OPTIONS)
    + "): the original owner applies a request only where the source proves it and keeps the ordinary "
    "route elsewhere, and each row's `source_composition` reports what was requested, selected and "
    "applied (`optimization_next.source_composition_request` names options a case may use). The "
    "heavy-pass-recipes skill's `reference/source-composition.md` gives the schema. Neither a route nor "
    "a request establishes a speedup: compare cycles on the same cases."
)
_MAX_BYTES = 8 << 20


# --- families from declarations ---------------------------------------------------------------------


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _nodes(declaration: dict) -> list[dict]:
    """The operation nodes of a declared component program, wherever the declaration spells them."""
    attrs = _mapping(_mapping(declaration.get("operation")).get("attributes"))
    for holder in (_mapping(declaration.get("component_program")), _mapping(attrs.get("program"))):
        nodes = holder.get("nodes")
        if isinstance(nodes, list):
            return [node for node in nodes if isinstance(node, dict)]
    return []


def _single_row(declaration: dict, op) -> bool:
    if isinstance(op, str) and op.startswith("gemv"):
        return True
    try:
        from .feedback import shape_classes

        described = shape_classes.describe(declaration)
    except Exception:  # noqa: BLE001 -- an undescribable case is simply not single-row
        return False
    return bool(described) and described.get("family") == "dense" and described["extents"].get("m") == 1


def _windowed(declaration: dict) -> bool:
    try:
        from merlin.targetgen.op_form_regimes import has_window

        return has_window(declaration)
    except Exception:  # noqa: BLE001 -- unknown structure is not a window
        return False


def _primary(declaration: dict) -> str | None:
    """The one operation family of a single declared operation, or None."""
    operation = _mapping(declaration.get("operation"))
    op = operation.get("op")
    semantic = _mapping(declaration.get("semantic")).get("semantic_family")
    if not isinstance(semantic, str) or not semantic:
        try:
            from merlin.targetgen import semantic_families

            semantic = semantic_families.from_op(op if isinstance(op, str) else None)
        except Exception:  # noqa: BLE001
            semantic = None
    if (isinstance(op, str) and "attention" in op) or semantic == "attention":
        return "attention"
    if _windowed(declaration):
        return "convolution"
    family = _SEMANTIC.get(semantic) if isinstance(semantic, str) else None
    if family == "contraction" and _single_row(declaration, op):
        return "matrix_vector"
    return family


def _has_epilogue(declaration: dict) -> bool:
    operation = _mapping(declaration.get("operation"))
    attrs = _mapping(operation.get("attributes"))
    op = operation.get("op")
    if isinstance(attrs.get("epilogue"), list) and attrs["epilogue"]:
        return True
    if _mapping(declaration.get("semantic")).get("composed_families"):
        return True
    if isinstance(op, str) and op.startswith("fused_"):
        return True
    return any("scale" in key and attrs[key] not in (None, False, 0) for key in attrs if isinstance(key, str))


def families_of(declaration) -> tuple[str, ...]:
    """The operation families of one case declaration, primary first; ``()`` when none is derivable."""
    if not isinstance(declaration, dict):
        return ()
    out: list[str] = []

    def add(family):
        if family in FAMILIES and family not in out:
            out.append(family)

    operation = _mapping(declaration.get("operation"))
    op = operation.get("op")
    attrs = _mapping(operation.get("attributes"))
    nodes = _nodes(declaration)
    siblings = attrs.get("matmuls")
    if isinstance(siblings, list) and len(siblings) > 1:
        # Several products sharing one operand: operand reuse first, then the chain's principles.
        add(_primary(declaration))
        add("connected_chain")
    elif declaration.get("kind") == "model" or op in _PROGRAM_OPS or len(nodes) > 1:
        add("connected_chain")
        for node in nodes:
            add(_primary({"operation": {"op": node.get("op")}}))
        semantic = _mapping(declaration.get("semantic")).get("semantic_family")
        add(_SEMANTIC.get(semantic) if isinstance(semantic, str) else None)
        if op not in _PROGRAM_OPS:
            add(_primary(declaration))
    else:
        add(_primary(declaration))
        weights = [t for t in declaration.get("inputs") or () if isinstance(t, dict) and t.get("role") == "weight"]
        if not out and len(weights) > 1:
            # An operation no vocabulary names that consumes several weights is a chain of products.
            add("connected_chain")
            add("contraction")
    if out and (out[0] in _PRODUCERS or out[0] == "connected_chain") and _has_epilogue(declaration):
        add("elementwise_epilogue")
    return tuple(out)


# --- the data file ----------------------------------------------------------------------------------


def load(repo) -> dict:
    return json.loads((Path(repo) / CATALOG).read_text(encoding="utf-8"))


def _skill_names() -> set[str]:
    try:
        return {path.parent.name for path in _SKILLS.glob("*/SKILL.md")}
    except OSError:
        return set()


def validate(catalog, *, repo=None, schema=None, mechanism_ids=None, skills=None) -> dict:
    """Check ``catalog`` against its JSON schema and its cross references; returns it or raises ValueError.

    Cross references: unique family and principle ids, families from :data:`FAMILIES`, ``related``
    naming catalogued families, catalogue mechanism ids present in the architecture catalogue, skills
    and recipes present among the author skills, and instruction roles from the closed role vocabulary.
    """
    if repo is not None:
        if schema is None:
            schema = json.loads((Path(repo) / SCHEMA_FILE).read_text(encoding="utf-8"))
        if mechanism_ids is None:
            mechanism_ids = {
                row["id"]
                for row in json.loads((Path(repo) / MECHANISM_CATALOG).read_text(encoding="utf-8"))["mechanisms"]
            }
    if schema is not None:
        import jsonschema

        try:
            jsonschema.Draft202012Validator(schema).validate(catalog)
        except jsonschema.ValidationError as error:
            raise ValueError(f"operation optimization catalog: {error.message}") from error
    if not isinstance(catalog, dict) or catalog.get("schema") != SCHEMA:
        raise ValueError(f"operation optimization catalog must declare schema {SCHEMA}")
    from merlin.compile.analysis.instruction_roles import ROLES

    skills = _skill_names() if skills is None else set(skills)
    families = catalog.get("families") or []
    ids = [entry.get("id") for entry in families]
    if len(set(ids)) != len(ids) or any(family not in FAMILIES for family in ids):
        raise ValueError(f"operation families must be unique members of {FAMILIES}")
    for entry in [catalog.get("general") or {}, *families]:
        name = entry.get("id", "general")
        principles = [p.get("id") for p in entry.get("principles") or ()]
        if len(set(principles)) != len(principles) or any(
            not isinstance(p, str) or not p.isidentifier() or p != p.lower() for p in principles
        ):
            raise ValueError(f"{name}: principle ids must be unique lower-case identifiers")
        for related in entry.get("related") or ():
            if related not in ids or related == name:
                raise ValueError(f"{name}: related family {related!r} is not another catalogued family")
        items = [*(entry.get("principles") or ()), *(entry.get("mechanisms") or ())]
        for item in items:
            unknown = set(item.get("uses_instruction_roles") or ()) - set(ROLES)
            if unknown:
                raise ValueError(f"{name}: unknown instruction roles {sorted(unknown)}")
        for mechanism in entry.get("mechanisms") or ():
            kind, ident = mechanism.get("kind"), str(mechanism.get("id"))
            if kind == "catalog" and mechanism_ids is not None and ident not in mechanism_ids:
                raise ValueError(f"{name}: catalogue mechanism {ident!r} does not exist")
            if kind == "skill" and ident not in skills:
                raise ValueError(f"{name}: skill {ident!r} is not an author skill")
            if kind == "recipe":
                skill, _, script = ident.partition("/")
                if skill not in skills or not script or not (_SKILLS / skill / "scripts" / script).is_file():
                    raise ValueError(f"{name}: recipe {ident!r} is not a staged skill script")
    return catalog


# --- policy -----------------------------------------------------------------------------------------


def prohibited_roles(*sources) -> frozenset[str]:
    """Every instruction role a policy or declaration prohibits, read from the ``prohibited_instruction_roles``
    field wherever a source carries it: top level, ``policy``, ``instruction_policy`` or the candidate arm's
    instruction policy. Malformed values are ignored rather than guessed."""
    roles: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        candidate = _mapping(_mapping(_mapping(source.get("performance")).get("arms")).get("candidate"))
        for holder in (
            source,
            source.get("policy"),
            source.get("instruction_policy"),
            candidate.get("instruction_policy"),
        ):
            value = holder.get(ROLES_FIELD) if isinstance(holder, Mapping) else None
            if isinstance(value, list | tuple):
                roles.update(role for role in value if isinstance(role, str) and role)
    return frozenset(roles)


# --- selection --------------------------------------------------------------------------------------


def _allowed(item: dict, prohibited: frozenset[str]) -> bool:
    return not set(item.get("uses_instruction_roles") or ()) & prohibited


def _mechanism(item: dict, mechanisms: Mapping[str, dict], statuses: Mapping[str, str]) -> dict:
    out = {key: item[key] for key in ("kind", "id", "use") if key in item}
    if item.get("kind") == "catalog":
        row = mechanisms.get(item.get("id")) or {}
        compiler = _mapping(row.get("compiler"))
        if row.get("mechanism"):
            out.setdefault("use", row["mechanism"])
        if compiler.get("module"):
            out["api"] = f"{compiler['module']}:{compiler.get('symbol')}"
        if item.get("id") in statuses:
            out["capability_status"] = statuses[item["id"]]
    return out


def _entry(entry: dict, prohibited, present, mechanisms, statuses, omitted: list) -> dict:
    name = entry.get("id", "general")
    out = {key: entry[key] for key in ("id", "title", "applies_when", "panel_family", "focus") if key in entry}
    if "related" in entry:
        out["related"] = [family for family in entry["related"] if family in present]
    out["principles"] = []
    for principle in entry.get("principles") or ():
        if _allowed(principle, prohibited):
            out["principles"].append({key: principle[key] for key in ("id", "kind", "text")})
        else:
            omitted.append({"entry": name, "principle": principle["id"]})
    out["mechanisms"] = []
    for mechanism in entry.get("mechanisms") or ():
        if _allowed(mechanism, prohibited):
            out["mechanisms"].append(_mechanism(mechanism, mechanisms, statuses))
        else:
            omitted.append({"entry": name, "mechanism": mechanism["id"]})
    out["evidence"] = list(entry.get("evidence") or ())
    return out


def select(
    catalog: dict,
    present: Mapping[str, int],
    *,
    prohibited: Iterable[str] = (),
    mechanism_catalog: dict | None = None,
    suggestions: dict | None = None,
) -> dict:
    """The staged document: the general entry plus the entries of the families in ``present``."""
    prohibited = frozenset(prohibited)
    mechanisms = {
        row.get("id"): row for row in (mechanism_catalog or {}).get("mechanisms") or () if isinstance(row, dict)
    }
    statuses = {
        row.get("id"): row.get("status")
        for row in (suggestions or {}).get("all_mechanisms") or ()
        if isinstance(row, dict) and isinstance(row.get("status"), str)
    }
    counts = {family: int(present[family]) for family in FAMILIES if present.get(family)}
    omitted: list[dict] = []
    general = _entry(catalog["general"], prohibited, counts, mechanisms, statuses, omitted)
    entries = [
        _entry(entry, prohibited, counts, mechanisms, statuses, omitted)
        for entry in catalog["families"]
        if entry["id"] in counts
    ]
    return {
        "schema": SELECTION_SCHEMA,
        "authority": "advisory principles only; no legality, pass order, threshold or performance credit",
        "how_to_use": (
            "Each self-check case's optimization_next.operation names its entry here. Treat principles "
            "as hypotheses: compare variants by cycles of the same evidence kind on the affected shape "
            "classes (agent_selfcheck.py --panel FAMILY) before keeping one."
        ),
        "families_from": "each case's own declaration (operation, operands, window, epilogue, program structure)",
        "families_present": counts,
        "families_not_in_corpus": [family for family in FAMILIES if family not in counts],
        "prohibited_instruction_roles": sorted(prohibited),
        "omitted_for_policy": omitted,
        "general": general,
        "entries": entries,
        "review": catalog.get("review"),
        "catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
    }


def present_families(declarations: Iterable) -> dict[str, int]:
    """Cases per operation family over a roster of declarations (a case counts in each of its families)."""
    counts: dict[str, int] = {}
    for declaration in declarations:
        for family in families_of(declaration):
            counts[family] = counts.get(family, 0) + 1
    return counts


def task_section(selection: dict) -> str:
    present = ", ".join(entry["id"] for entry in selection["entries"]) or "none derived"
    text = (
        f"\n\n{HEADING}\n\n"
        f"`{FILENAME}` holds reviewed optimization principles, pitfalls and the matching library "
        f"mechanisms, skills and recipes for the operation families of this corpus ({present}), plus "
        "principles for every operation. Families come from each case's declaration, not its name; "
        "a self-check case's `optimization_next.operation` names its entry. The principles are "
        "hypotheses: judge them by cycles of the same evidence kind on the affected shape classes "
        "(`python agent_selfcheck.py --panel FAMILY`)."
    )
    if selection["prohibited_instruction_roles"]:
        text += " Principles that need an instruction role the run policy prohibits are left out."
    return text + "\n\n" + COMPOSITION_TEXT + "\n"


def stage(workspace, run_dir, *, declarations, policy=None, repo) -> dict:
    """Write ``OPTIMIZATION_BY_OPERATION.json`` and the TASK.md section to the workspace and run directory."""
    declarations = list(declarations)
    catalog = validate(load(repo), repo=repo)
    mechanism_catalog = json.loads((Path(repo) / MECHANISM_CATALOG).read_text(encoding="utf-8"))
    suggestions = None
    try:
        suggestions = json.loads((Path(workspace) / SUGGESTIONS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        suggestions = None
    roles = prohibited_roles(policy, *declarations)
    selection = select(
        catalog,
        present_families(declarations),
        prohibited=roles,
        mechanism_catalog=mechanism_catalog,
        suggestions=suggestions,
    )
    data = json.dumps(selection, indent=2) + "\n"
    section = task_section(selection)
    for root in (Path(workspace), Path(run_dir)):
        (root / FILENAME).write_text(data)
    task = Path(workspace) / "TASK.md"
    text = task.read_text() + section
    task.write_text(text)
    (Path(run_dir) / "TASK.md").write_text(text)
    return {
        "schema": SELECTION_SCHEMA,
        "sha256": hashlib.sha256(data.encode()).hexdigest(),
        "families_present": selection["families_present"],
        "prohibited_instruction_roles": selection["prohibited_instruction_roles"],
    }


def stage_selected(workspace, run_dir, *, declarations, policy=None, repo=None) -> dict | None:
    """:func:`stage` for a launch; advice only, so a failure is reported and the launch continues."""
    if repo is None or not (Path(repo) / CATALOG).is_file():
        print("[operation-optimizations] skipped: no catalogue in the repository", flush=True)
        return None
    try:
        return stage(workspace, run_dir, declarations=declarations, policy=policy, repo=repo)
    except Exception as error:  # noqa: BLE001 -- advice only; never fails the launch
        print(f"[operation-optimizations] unavailable: {type(error).__name__}: {error}", flush=True)
        return None


# --- per-case link ----------------------------------------------------------------------------------


def load_selection(workspace) -> dict | None:
    """The staged selection, or None when it was not staged."""
    path = Path(workspace) / FILENAME
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) and document.get("schema") == SELECTION_SCHEMA else None


def link(declaration, selection) -> dict | None:
    """``optimization_next["operation"]`` for one case: its families' entry ids in the staged file."""
    if not isinstance(selection, dict):
        return None
    staged = {entry.get("id"): entry for entry in selection.get("entries") or () if isinstance(entry, dict)}
    families = [family for family in families_of(declaration) if family in staged]
    if not families:
        return None
    primary = staged[families[0]]
    out = {"family": families[0], "entry": families[0], "file": FILENAME, "focus": primary.get("focus")}
    if len(families) > 1:
        out["also"] = families[1:]
    if primary.get("panel_family"):
        out["panel"] = f"python agent_selfcheck.py --panel {primary['panel_family']}"
    return out


__all__ = [
    "FAMILIES",
    "FILENAME",
    "SELECTION_SCHEMA",
    "families_of",
    "link",
    "load",
    "load_selection",
    "present_families",
    "prohibited_roles",
    "select",
    "stage",
    "stage_selected",
    "validate",
]
