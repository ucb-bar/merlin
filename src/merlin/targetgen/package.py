"""What a target package can DO — one classifier, asked by every loader instead of each guessing.

THE PROBLEM THIS IS FOR. Merlin has three loaders that each take a directory and hope:

* :func:`merlin.runtime.backends.base._load_oot_backend` wants ``contracts/target_contract.yaml`` with
  a ``plugin.backend`` naming a module that self-registers;
* :func:`merlin.targetgen.oot_runner.load_package` wants ``manifest.yaml`` (experiment ABI v0.1), an
  executable tool, and four subprocess ``commands``;
* :func:`merlin.targetgen.registry.load_target` wants ``manifest.yaml``, ``dialect.py`` exposing
  ``SPEC_OPS``, and ``lowering.yaml``.

The second and third **collide on one filename with disjoint schemas in sibling directories**: given
``<dir>/manifest.yaml`` alone, no code can say which contract it claims without trial-parsing against
two schemas, and each loader discovers the mismatch by failing somewhere inside itself — a ``KeyError``,
a ``ContractViolation``, a missing attribute — long after the point where the honest answer was simply
"this package does not do that".

WHAT CONVERGES, AND WHAT DELIBERATELY DOES NOT. Not the file layout. ``oot_runner``'s ``manifest.yaml``
is a FROZEN experiment ABI (``merlin/contract/mlir_oot_backend_contract.yaml``, ``version: "0.1"``),
schema-validated, mirrored, and depended on by certified packages and in-flight agent sessions; changing
its required keys would retro-invalidate artifacts that already carry verdicts. What converges is the
VOCABULARY: one closed set of capability names, one answer shape, inferred from what packages already
have on disk. Nothing migrates and nothing is invalidated.

THE ANSWER SHAPE IS THE POINT. A capability is either provided WITH EVIDENCE (which file, which key) or
absent WITH THE EXACT MISSING FILE OR KEY as its reason. There is no third state and no boolean: a bare
``False`` is how "this package has no compiler" and "we could not tell" became the same value, and that
is the shape every one of the failures above already has.

EXPLICIT BEATS INFERRED. A package may DECLARE ``package_capabilities:`` and that wins. The key is
deliberately NOT ``capabilities:`` — that name is already taken in ``target_contract.yaml``, where it
holds the hardware's op/layout/capacity surface, and reusing it would make one key mean two unrelated
things depending on which reader got there first.

NO CALLERS, ON PURPOSE (at first). This lands as a pure classifier — it reads files, it answers
questions, it changes nobody's behaviour — so that the classification can be reviewed and tested on its
own before three live loaders start refusing things because of it. That is the same discipline
:mod:`merlin.runtime.route_partition` documents for itself.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: The closed capability vocabulary. Closed because an open one cannot be reasoned about: a caller
#: asking for a capability nobody produces would get a plausible-looking absence instead of an error.
CAPABILITIES: tuple[str, ...] = ("backend", "compiler", "dialect")

#: What each capability MEANS, stated once so a reader never has to infer it from the inference rules.
CAPABILITY_MEANING: dict[str, str] = {
    "backend": (
        "the package ships a runtime backend module that self-registers with "
        "merlin.runtime.backends.base.register(), so get_backend(<target>) returns it"
    ),
    "compiler": (
        "the package ships an out-of-tree compiler satisfying the experiment ABI v0.1 — a manifest.yaml "
        "with a resolvable tool and the four certification commands, invoked only by subprocess"
    ),
    "dialect": (
        "the package ships a target dialect the staged pipeline can load — dialect.py exposing SPEC_OPS "
        "plus a lowering.yaml, or a plugin.dialect module that self-registers a TargetSpec"
    ),
}

#: Where an explicit declaration may live, in precedence order. Both files are read because the two
#: package shapes keep their metadata in different places and neither is going to move.
_DECLARATION_FILES: tuple[str, ...] = ("contracts/target_contract.yaml", "manifest.yaml")

#: The key an explicit declaration uses. NOT ``capabilities`` — see the module docstring.
DECLARATION_KEY = "package_capabilities"

#: The contract file the backend capability is declared in.
_CONTRACT = "contracts/target_contract.yaml"
#: The experiment-ABI manifest the compiler capability is declared in.
_MANIFEST = "manifest.yaml"
#: The four certification commands the ABI requires. Read from the frozen contract rather than
#: restated, so this module cannot drift from the schema the runner actually validates against.
_REQUIRED_COMMANDS: tuple[str, ...] = (
    "parse",
    "lower_interface_to_target",
    "emit_command_buffer",
    "lower_target_to_llvm",
)


class PackageCapabilityMissing(LookupError):
    """A caller asked a package for a capability it does not provide, and here is exactly what is missing."""

    def __init__(self, root: Path, capability: Capability):
        self.root = root
        self.capability = capability
        super().__init__(capability.explain(root))


@dataclass(frozen=True)
class Capability:
    """One capability of one package: provided with evidence, or absent with a named cause.

    ``missing`` names FILES AND KEYS, not prose — ``manifest.yaml:commands.emit_command_buffer``,
    ``contracts/target_contract.yaml:plugin.backend`` — because the caller's next action is to go look
    at that exact thing, and a sentence it has to parse first is a sentence it will print instead.
    """

    name: str
    provided: bool
    #: ``declared`` when the package stated it, ``inferred`` when it was read off the files present.
    source: str
    #: Why it IS provided (which file, which key, which resolved path). Empty when absent.
    evidence: str = ""
    #: The exact files/keys whose absence is the reason. Empty when provided.
    missing: tuple[str, ...] = ()

    def explain(self, root: Path | None = None) -> str:
        where = f" at {root}" if root is not None else ""
        if self.provided:
            return f"package{where} provides {self.name!r} ({self.source}): {self.evidence}"
        return (
            f"package{where} does not provide {self.name!r} ({self.source}): "
            f"{CAPABILITY_MEANING[self.name]}. Missing: {', '.join(self.missing) or 'unknown'}"
        )


def _provided(name: str, evidence: str, *, source: str = "inferred") -> Capability:
    return Capability(name=name, provided=True, source=source, evidence=evidence)


def _absent(name: str, *missing: str, source: str = "inferred") -> Capability:
    return Capability(name=name, provided=False, source=source, missing=tuple(missing))


def _load_yaml(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """``(mapping, problem)`` — a document that is absent, unparseable, or not a mapping yields None."""
    if not path.is_file():
        return None, "absent"
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(doc, dict):
        return None, f"not a mapping (got {type(doc).__name__})"
    return doc, None


# --------------------------------------------------------------------------- explicit declaration
def _declared(root: Path) -> dict[str, Capability]:
    """Capabilities the package DECLARES, from the first declaration file that carries the key.

    A declaration is either a bool or a mapping with ``provided`` plus ``evidence`` / ``missing``. An
    unknown capability name RAISES: the vocabulary is closed, and a package inventing a name would
    otherwise be silently ignored — the same failure mode ``plugins.validate`` exists to stop.
    """
    for rel in _DECLARATION_FILES:
        doc, _ = _load_yaml(root / rel)
        if doc is None:
            continue
        block = doc.get(DECLARATION_KEY)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ValueError(f"{root / rel}: {DECLARATION_KEY} must be a mapping, got {type(block).__name__}")
        unknown = sorted(set(block) - set(CAPABILITIES))
        if unknown:
            raise ValueError(
                f"{root / rel}: {DECLARATION_KEY} declares unknown capabilities {unknown}; "
                f"the vocabulary is {list(CAPABILITIES)}"
            )
        out: dict[str, Capability] = {}
        for name, value in block.items():
            if isinstance(value, bool):
                out[name] = (
                    _provided(name, f"declared in {rel}", source="declared")
                    if value
                    else _absent(name, f"{rel}:{DECLARATION_KEY}.{name} = false", source="declared")
                )
                continue
            if not isinstance(value, dict) or "provided" not in value:
                raise ValueError(f"{root / rel}: {DECLARATION_KEY}.{name} must be a bool or a mapping with 'provided'")
            if value["provided"]:
                out[name] = _provided(name, str(value.get("evidence") or f"declared in {rel}"), source="declared")
            else:
                miss = value.get("missing") or [f"{rel}:{DECLARATION_KEY}.{name}.provided = false"]
                out[name] = _absent(name, *(str(m) for m in miss), source="declared")
        return out
    return {}


# --------------------------------------------------------------------------- inference: backend
def _infer_backend(root: Path) -> Capability:
    doc, problem = _load_yaml(root / _CONTRACT)
    if doc is None:
        return _absent("backend", f"{_CONTRACT} ({problem})")
    plugin = doc.get("plugin")
    if not isinstance(plugin, dict) or not plugin.get("backend"):
        return _absent("backend", f"{_CONTRACT}:plugin.backend")
    rel = str(plugin["backend"])
    from .plugins import _reference_path  # local: avoid a cycle at module import

    resolved = _reference_path(root, rel, "path")
    if resolved is None:
        return _absent("backend", f"{_CONTRACT}:plugin.backend -> {rel} (resolves to nothing under {root})")
    return _provided("backend", f"{_CONTRACT}:plugin.backend -> {resolved.relative_to(root)}")


# --------------------------------------------------------------------------- inference: compiler
def _infer_compiler(root: Path) -> Capability:
    doc, problem = _load_yaml(root / _MANIFEST)
    if doc is None:
        return _absent("compiler", f"{_MANIFEST} ({problem})")
    # A registry-format manifest and an ABI manifest share this filename with disjoint schemas, so the
    # discriminator is read as DATA: only the ABI manifest carries the artifact_type the contract names.
    from .contract import schemas

    try:
        schemas.validate_manifest(doc)
    except schemas.ContractViolation as exc:
        return _absent("compiler", f"{_MANIFEST}: {exc}")
    except Exception as exc:  # noqa: BLE001 — a schema machinery fault is UNKNOWN, and says so
        return _absent("compiler", f"{_MANIFEST}: schema check unavailable ({type(exc).__name__}: {exc})")
    build = doc.get("build")
    tool_key = "build.tool_output" if build else "entrypoints.tool"
    tool_rel = build.get("tool_output") if build else (doc.get("entrypoints") or {}).get("tool")
    if not tool_rel:
        return _absent("compiler", f"{_MANIFEST}:{tool_key}")
    tool = root / str(tool_rel)
    # A `build` block means the tool is PRODUCED by the build, so its absence before the build is not
    # a missing capability. Without a build block the tool must already be there.
    if not build and not tool.exists():
        return _absent("compiler", f"{_MANIFEST}:{tool_key} -> {tool_rel} (no such file under {root})")
    commands = doc.get("commands") or {}
    absent_cmds = [f"{_MANIFEST}:commands.{c}" for c in _REQUIRED_COMMANDS if c not in commands]
    if absent_cmds:
        return _absent("compiler", *absent_cmds)
    return _provided("compiler", f"{_MANIFEST}:{tool_key} -> {tool_rel}, all {len(_REQUIRED_COMMANDS)} commands")


# --------------------------------------------------------------------------- inference: dialect
def _defines_module_level(path: Path, symbol: str) -> bool:
    """Does ``path`` bind ``symbol`` at module level? Structural (AST), never a text match.

    Importing would be the other way to ask, and it is the wrong one here: importing a generated
    dialect executes it, and a classifier that has side effects cannot be asked speculatively.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    return True
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            return True
    return False


def _infer_dialect(root: Path) -> Capability:
    missing: list[str] = []
    dialect_py, lowering = root / "dialect.py", root / "lowering.yaml"
    if dialect_py.is_file() and lowering.is_file() and _defines_module_level(dialect_py, "SPEC_OPS"):
        return _provided("dialect", "dialect.py defines SPEC_OPS, beside lowering.yaml")
    if not dialect_py.is_file():
        missing.append("dialect.py")
    elif not _defines_module_level(dialect_py, "SPEC_OPS"):
        missing.append("dialect.py:SPEC_OPS")
    if not lowering.is_file():
        missing.append("lowering.yaml")

    doc, problem = _load_yaml(root / _CONTRACT)
    plugin = doc.get("plugin") if isinstance(doc, dict) else None
    if not isinstance(plugin, dict) or not plugin.get("dialect"):
        missing.append(f"{_CONTRACT}:plugin.dialect" if doc is not None else f"{_CONTRACT} ({problem})")
        return _absent("dialect", *missing)
    rel = str(plugin["dialect"])
    from .plugins import _reference_path

    resolved = _reference_path(root, rel, "path")
    if resolved is None:
        missing.append(f"{_CONTRACT}:plugin.dialect -> {rel} (resolves to nothing under {root})")
        return _absent("dialect", *missing)
    return _provided("dialect", f"{_CONTRACT}:plugin.dialect -> {resolved.relative_to(root)}")


_INFER = {"backend": _infer_backend, "compiler": _infer_compiler, "dialect": _infer_dialect}


# --------------------------------------------------------------------------- the classifier
def capabilities_of(root: str | Path) -> dict[str, Capability]:
    """Every capability in :data:`CAPABILITIES`, settled for the package at ``root``.

    Every name is settled — a capability is never simply absent from the result, because "not in the
    dict" is the shape that let a missing answer read as a negative one. An explicit
    ``package_capabilities`` declaration wins over inference, and says so in ``source``.
    """
    path = Path(root)
    declared = _declared(path)
    return {name: declared.get(name) or _INFER[name](path) for name in CAPABILITIES}


def capability(root: str | Path, name: str) -> Capability:
    """One capability. Raises ``KeyError`` for a name outside the closed vocabulary."""
    if name not in CAPABILITIES:
        raise KeyError(f"unknown capability {name!r}; the vocabulary is {list(CAPABILITIES)}")
    path = Path(root)
    return _declared(path).get(name) or _INFER[name](path)


def require(root: str | Path, name: str) -> Capability:
    """The capability, or raise :class:`PackageCapabilityMissing` naming the exact missing file/key."""
    cap = capability(root, name)
    if not cap.provided:
        raise PackageCapabilityMissing(Path(root), cap)
    return cap


def describe(root: str | Path) -> str:
    """One line per capability — what a loader would print when it refuses."""
    path = Path(root)
    return "\n".join(cap.explain(path) for _, cap in sorted(capabilities_of(path).items()))
