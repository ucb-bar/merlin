"""Exact semantic interface readers for raw-RTL EL2, separate from target authorship.

The existing harness frontend provides the already-lowered test inputs. These
sources parse/serialize that input grammar; they do not perform upstream lowering
and supply no target dialect, code generator, scheduling, pass or extraction API.
"""

from __future__ import annotations

import ast
import hashlib
import json
from importlib.util import resolve_name
from pathlib import Path

from merlin.development.backend import full_compiler_workspace as full

SCHEMA = "merlin.phase1.raw_interface_sources.v1"
ENTRIES = (
    "merlin.targetgen.contract.interface_emit",
    "merlin.targetgen.contract.linalg_iface",
    "merlin.targetgen.oot_starterkit.iface",
)
SUPPORT = (
    "merlin.compiler.ir.logical_abi",
    "merlin.xdsl_dialects.fp8",
    "merlin.xdsl_dialects._common",
    "merlin.common.ir_lock",
    "merlin.common.dense_float_ir",
)
PACKAGES = (
    "merlin",
    "merlin.targetgen",
    "merlin.targetgen.contract",
    "merlin.targetgen.oot_starterkit",
    "merlin.xdsl_dialects",
    "merlin.common",
)
MODULES = frozenset((*ENTRIES, *SUPPORT, *PACKAGES))
# These lazy branches are deliberately not a dependency grant. Current selected
# inputs contain no storage encoding. Default ABI path lookup is not a reader API.
UNAVAILABLE = (
    "merlin.common.paths",
    "merlin.perf.storage_encoding",
)
MUTABLE_NAMESPACES = ("merlin.targetgen",)
SCOPE = (
    "semantic interface input readers only; upstream frontend is harness-owned and provided as prelowered inputs; "
    "no author-callable frontend lowering, target dialect builders, C++ generators, scheduling, passes or extraction; "
    "Python/xDSL parser support does not change the C++ target-backend requirement"
)


def _path(module):
    return module.replace(".", "/") + ("/__init__.py" if module in PACKAGES else ".py")


def _check_imports(payload, module):
    package = module if module in PACKAGES else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(payload, filename=_path(module))):
        if isinstance(node, ast.Import):
            dependencies = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module or ""
            dependencies = [base]
            # "from . import reader" names a module rather than an attribute.
            if base in PACKAGES:
                dependencies.extend(base + "." + alias.name for alias in node.names if alias.name != "*")
        else:
            continue
        for dependency in dependencies:
            if (dependency == "merlin" or dependency.startswith("merlin.")) and dependency not in MODULES:
                if dependency not in UNAVAILABLE:
                    raise ValueError(f"semantic reader imports an unreviewed dependency: {module} -> {dependency}")


def verify_sources(selection):
    """Verify the exact supplied inventory, not an unconstrained transitive closure."""
    if type(selection) is not full.FullCompilerSelection:
        raise ValueError("semantic readers require an original FullCompilerSelection")
    selection.verify()
    document = json.loads(selection.manifest_path.read_bytes())
    expected = {_path(module): module for module in MODULES}
    actual = {path: module for path, _digest, module in selection.members}
    if (
        actual != expected
        or selection.mutable_namespaces != MUTABLE_NAMESPACES
        or selection.diagnostics != UNAVAILABLE
        or document.get("entry_modules") != list(ENTRIES)
        or document.get("raw_interface_sources_schema") != SCHEMA
        or document.get("scope") != SCOPE
        or document.get("source_roots") != []
    ):
        raise ValueError("semantic reader source inventory differs from the reviewed boundary")
    for path, _digest, module in selection.members:
        _check_imports(full._read(selection.root, path), module)
    return selection


def select(root, manifest_path, manifest_sha256):
    return verify_sources(full.select(root, manifest_path, manifest_sha256))


def prepare(repo, destination, *, review_id):
    """Copy exact existing reader/support source bytes, with no recursive export."""
    repo, destination = Path(repo).resolve(), Path(destination).absolute()
    if not review_id or destination.exists() or destination.is_symlink():
        raise ValueError("semantic reader preparation requires a fresh destination and review identity")
    if destination.parent.resolve() != destination.parent:
        raise ValueError("semantic reader destination must have canonical parents")
    payloads = {}
    for module in sorted(MODULES):
        relative = _path(module)
        raw = full._read(repo / "src", relative)
        _check_imports(raw, module)
        payloads[relative] = (raw, module)
    root = destination / "python"
    root.mkdir(parents=True)
    rows = []
    for relative, (raw, module) in sorted(payloads.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o444)
        rows.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "module": module})
    document = {
        "schema": full.SCHEMA,
        "raw_interface_sources_schema": SCHEMA,
        "review_id": review_id,
        "members": rows,
        "entry_modules": list(ENTRIES),
        "source_roots": [],
        "mutable_namespaces": list(MUTABLE_NAMESPACES),
        "unavailable_dependencies": list(UNAVAILABLE),
        "scope": SCOPE,
    }
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    manifest.chmod(0o444)
    return select(root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
