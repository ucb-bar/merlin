"""Static source-use edges for closed workers and pinned backend consumers.

These edges establish source dependencies, never execution or evaluator access.
Only the existing source loaders' actual read/execute paths select their finite
rosters. A bare export list, documentation string or dependency label is inert.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _calls(tree: ast.AST, spelling: str) -> bool:
    return any(isinstance(n, ast.Call) and ast.unparse(n.func) == spelling for n in ast.walk(tree))


def _strings(tree: ast.AST, name: str) -> tuple[str, ...]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return ()
            if isinstance(values, (list, tuple)) and all(
                isinstance(v, str) and all(p.isidentifier() for p in v.split(".")) for v in values
            ):
                return tuple(values)
    return ()


def _staged_rtl_sources(root: Path, tree: ast.AST) -> set[str]:
    """Source use through the existing arm-selected RTL tool staging flow.

    A registry path alone is inert. Require its selection by the live producer,
    the original source enumeration and the actual byte-copy consumer. Frozen
    fresh10 payload pins retain their separate exact-byte interpretation.
    """
    loader = _function(tree, "_tool_sources")
    prepare = _function(tree, "prepare_shared_source_release")
    if loader is None or prepare is None:
        return set()
    loader_forms = {ast.unparse(node) for node in ast.walk(loader)}
    prepare_forms = {ast.unparse(node) for node in ast.walk(prepare)}
    if (
        not {
            "prefix = 'merlin/python/'",
            "spec(name).bundle_paths",
            "logical.removeprefix(prefix).rstrip('/')",
            "path = root / selected_source",
            "path.rglob('*') if path.is_dir() else (path,)",
            "original = source.relative_to(root).as_posix()",
            "staged = (Path(destination) / source.relative_to(path)).as_posix() if path.is_dir() else destination",
            "selected[staged] = original",
            "author_inaccessible(staged_module(p))",
        }
        <= loader_forms
        or not {
            "tool_sources = _tool_sources(root, tools, withhold=withhold)",
            "sources.update(tool_sources)",
            "before = {relative: _read(root, relative) for relative in dict.fromkeys(sources.values())}",
            "payload = before[selected_source]",
            "destination.write_bytes(payload)",
        }
        <= prepare_forms
    ):
        return set()
    imports = {
        (node.level, node.module, alias.name, alias.asname)
        for node in ast.walk(loader)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    if not {(1, "tool_registry", "spec", None), (0, "merlin.common.access", "author_inaccessible", None)} <= imports:
        return set()
    producer = root / "src/merlin/experiments/corpus/preparation.py"
    registry = root / "src/merlin/targetgen/tool_registry.py"
    policy = root / "src/merlin/common/access.py"
    if not all(path.is_file() for path in (producer, registry, policy)):
        return set()
    scaffold = _function(ast.parse(producer.read_text(encoding="utf-8")), "scaffold")
    if scaffold is None or not _calls(scaffold, "prepare_shared_source_release"):
        return set()
    forms = {ast.unparse(node) for node in ast.walk(scaffold)}
    if not {"tool_registry.ARM_TOOLS[arm]", "tools=tools"} <= forms:
        # Keywords are AST nodes too; the selected producer must pass its tools.
        return set()
    declared = ast.parse(registry.read_text(encoding="utf-8"))
    assignments = {
        node.target.id: node.value
        for node in declared.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    tools, arms = assignments.get("TOOLS"), assignments.get("ARM_TOOLS")
    if not isinstance(tools, ast.Dict) or not isinstance(arms, ast.Dict):
        return set()
    if not any(
        isinstance(node, ast.Constant) and node.value == "rtl_generators"
        for value in arms.values
        for node in ast.walk(value)
    ):
        return set()
    tool = next(
        (
            value
            for key, value in zip(tools.keys, tools.values)
            if isinstance(key, ast.Constant) and key.value == "rtl_generators"
        ),
        None,
    )
    if not isinstance(tool, ast.Call) or ast.unparse(tool.func) != "ToolSpec":
        return set()
    paths = next((keyword.value for keyword in tool.keywords if keyword.arg == "bundle_paths"), None)
    prefix = next(
        (
            node.value.value
            for node in declared.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and any(isinstance(target, ast.Name) and target.id == "_PY" for target in node.targets)
        ),
        None,
    )
    if prefix != "merlin/python/merlin/" or not isinstance(paths, ast.Tuple):
        return set()
    directories = []
    for value in paths.elts:
        if not (
            isinstance(value, ast.JoinedStr)
            and len(value.values) == 2
            and isinstance(value.values[0], ast.FormattedValue)
            and isinstance(value.values[0].value, ast.Name)
            and value.values[0].value.id == "_PY"
            and value.values[0].conversion == -1
            and value.values[0].format_spec is None
            and isinstance(value.values[1], ast.Constant)
            and isinstance(value.values[1].value, str)
        ):
            return set()
        relative = Path("merlin") / value.values[1].value
        if relative.is_absolute() or ".." in relative.parts:
            return set()
        directories.append(root / "src" / relative)
    # Interpret only the existing literal access declarations, without importing
    # compiler/evaluator code. These source edges cannot expose withheld bodies.
    access = ast.parse(policy.read_text(encoding="utf-8"))
    withheld = _strings(access, "AUTHOR_WITHHELD_PREFIXES")
    records = next(
        (
            node.value
            for node in access.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "MODULE_ACCESS" for target in node.targets)
        ),
        None,
    )
    if not withheld or not isinstance(records, ast.Tuple):
        return set()
    denied = []
    for item in records.elts:
        if not isinstance(item, ast.Call) or ast.unparse(item.func) != "_module":
            return set()
        try:
            module = ast.literal_eval(item.args[0])
            options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in item.keywords}
        except (ValueError, TypeError):
            return set()
        # _module's declared optional directory flag is false when omitted.
        recursive = "directory" in options and options["directory"]
        denied.extend((name, recursive) for name in (module, *options.get("aliases", ())))
    selected = set()
    for directory in directories:
        if not directory.is_dir() or any(path.is_symlink() for path in (directory, *directory.parents)):
            return set()
        for source in directory.rglob("*.py"):
            if "__pycache__" in source.parts or source.is_symlink() or not source.is_file():
                continue
            parts = list(source.relative_to(root / "src").with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join(parts)
            if any(module == prefix or module.startswith(prefix + ".") for prefix in withheld):
                continue
            if any(module == name or (recursive and module.startswith(name + ".")) for name, recursive in denied):
                continue
            selected.add(module)
    return selected


def closed_source_imports(path: Path, root: Path, tree: ast.AST) -> set[str]:
    """Follow three existing finite source loaders, without importing them."""
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return set()
    if relative == "src/merlin/targetgen/tool_profile_shared.py":
        return _staged_rtl_sources(root, tree)
    if relative == "src/merlin/compiler/quantization/torchao/portable.py":
        helper, prepare = _function(tree, "helper"), _function(tree, "prepare_helpers")
        if helper is None or prepare is None:
            return set()
        required = (
            "name not in DEPENDENCIES",
            "_TARGETGEN / (name + '.py')",
            "'merlin.targetgen.' + name",
        )
        forms = {ast.unparse(n) for n in ast.walk(helper)}
        invoked = any(
            isinstance(n, ast.For) and ast.unparse(n.iter) == "DEPENDENCIES" and _calls(n, "helper")
            for n in ast.walk(prepare)
        )
        if not invoked or not all(value in forms for value in required):
            return set()
        if not _calls(helper, "importlib.util.spec_from_file_location") or not _calls(
            helper, "spec.loader.exec_module"
        ):
            return set()
        initializer = path.parent / "__init__.py"
        if not initializer.is_file():
            return set()
        init = ast.parse(initializer.read_text(encoding="utf-8"))
        if not _calls(init, "prepare_helpers") or not any(
            isinstance(n, ast.ImportFrom)
            and n.level == 1
            and n.module == "portable"
            and any(a.name == "prepare_helpers" and a.asname is None for a in n.names)
            for n in init.body
        ):
            return set()
        return {"merlin.targetgen." + name for name in _strings(tree, "DEPENDENCIES")}
    if relative not in {
        "src/merlin/development/backend/full_compiler_workspace.py",
        "src/merlin/development/backend/raw_interface_sources.py",
    }:
        return set()
    prepare = _function(tree, "prepare")
    if prepare is None:
        return set()
    names = {n.id for n in ast.walk(prepare) if isinstance(n, ast.Name)}
    if relative.endswith("/full_compiler_workspace.py"):
        # PUBLIC_HELPERS feeds pending, which the existing _read consumes.
        if "PUBLIC_HELPERS" not in names or not _calls(prepare, "_read"):
            return set()
        return set(_strings(tree, "PUBLIC_HELPERS"))
    if "MODULES" not in names or not _calls(prepare, "full._read"):
        return set()
    return set().union(*(_strings(tree, name) for name in ("ENTRIES", "SUPPORT", "PACKAGES")))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def published_source_modules(root: Path) -> set[str]:
    """Bodies actually handed to compiler authors are inputs, not host calls.

    The pinned fresh10 producer verifies and copies every full member into a
    selected compiler workspace. Counting this source use preserves the same
    boundary as literal module_source_path reads. It grants no new source and
    does not claim any exported function executed or any result was verified.
    """
    base = root / "src/merlin/development/backend"
    producer, staging = base / "fresh10_sources.py", base / "full_compiler_workspace.py"
    inventory = base / "fresh10_profile_data.json"
    if not all(p.is_file() for p in (producer, staging, inventory)):
        return set()
    tree = ast.parse(producer.read_text(encoding="utf-8"))
    prepare = _function(tree, "prepare_sources")
    stage = _function(ast.parse(staging.read_text(encoding="utf-8")), "stage")
    if (
        prepare is None
        or stage is None
        or not all(_calls(prepare, name) for name in ("_rows", "_payload", "F.select", "verify_sources"))
        or not all(_calls(stage, name) for name in ("path.open", "stream.write", "_read"))
    ):
        return set()
    stage_forms = {ast.unparse(n) for n in ast.walk(stage)}
    if not {"payload = _read(selection.root, relative)", "path.open('xb')", "stream.write(payload)"} <= stage_forms:
        return set()
    pinned = next(
        (
            n.value.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DATA_SHA256" for t in n.targets)
            and isinstance(n.value, ast.Constant)
        ),
        None,
    )
    raw = inventory.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pinned:
        raise ValueError("published compiler source inventory differs from its producer pin")
    data = json.loads(raw)
    if data.get("schema") != "merlin.fresh10_tooling_projection.v1":
        raise ValueError("published compiler source inventory has an unsupported schema")
    admitted = set()
    for profile in data["profiles"].values():
        for row in profile["full_members"]:
            module = row.get("module")
            if type(module) is not str or not module.startswith("merlin."):
                continue
            source = root / row.get("source_relative", "src/" + row["path"])
            if (
                source.is_file()
                and not any(p.is_symlink() for p in (source, *source.parents))
                and source.resolve().is_relative_to(root.resolve())
            ):
                if hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]:
                    admitted.add(module)
    return admitted


def compiler_source_inputs(root: Path) -> set[str]:
    """Current closed source-copy rosters plus admitted versioned sources."""
    names = published_source_modules(root)
    for relative in (
        "src/merlin/targetgen/tool_profile_shared.py",
        "src/merlin/development/backend/full_compiler_workspace.py",
        "src/merlin/development/backend/raw_interface_sources.py",
    ):
        path = root / relative
        if path.is_file():
            names.update(closed_source_imports(path, root, ast.parse(path.read_text(encoding="utf-8"))))
    return names


def _package_files(source: Path) -> set[Path]:
    """Actual packaged Python owners plus standalone backend operator tools."""
    data = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    config = data["tool"]["setuptools"]
    mapping = config.get("package-dir", {})
    packages = config["packages"]
    files = set()
    if isinstance(packages, list):
        for package in packages:
            prefixes = [p for p in mapping if p and (package == p or package.startswith(p + "."))]
            prefix = max(prefixes, key=len) if prefixes else ""
            suffix = package[len(prefix) :].lstrip(".").replace(".", "/")
            default_root = mapping.get("")
            if default_root is None:
                default_root = "."  # setuptools package-dir policy
            directory = source / mapping.get(prefix, default_root) / suffix
            files.update(directory.glob("*.py"))
    else:
        find = packages.get("find", {})
        # An omitted setuptools search root means the project root: authored
        # packaging policy, not a fabricated observed consumer datum.
        locations = find.get("where")
        if locations is None:
            locations = ["."]
        for where in locations:
            base = source / where
            includes = find.get("include")
            directories = [base / p.rstrip("*").replace(".", "/") for p in includes] if includes else [base]
            for directory in directories:
                files.update(directory.rglob("*.py"))
    files.update((source / "tools").glob("*.py"))
    return {p for p in files if "__pycache__" not in p.parts and p.is_file()}


def tracked_consumer_files(source: Path, revision: str, paths: set[Path]) -> tuple[Path, ...]:
    """Reopen selected original Git blobs without importing or executing them."""
    source = source.resolve(strict=True)
    if _git(source, "rev-parse", "HEAD") != revision or _git(
        source, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"
    ):
        raise ValueError("backend consumer source is changed or at another revision: " + str(source))
    tracked = {}
    listing = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "-z", "HEAD"])
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, digest = header.split()
        if kind == b"blob" and mode in (b"100644", b"100755"):
            tracked[name.decode()] = digest.decode()
    files = []
    for path in sorted(paths):
        relative = path.relative_to(source).as_posix()
        if relative not in tracked:
            raise ValueError("backend consumer is not an exact tracked regular source: " + str(path))
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(source):
            raise ValueError("backend consumer source escapes its checkout or is unavailable: " + str(path))
        raw = path.read_bytes()
        # Bind the actual parsed payload too: skip-worktree cannot hide edits.
        digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if digest != tracked[relative]:
            raise ValueError("backend consumer bytes differ from pinned HEAD: " + str(path))
        files.append(path)
    return tuple(files)


def selected_backend_sources(selected: dict[str, str], root: Path) -> tuple[Path, ...]:
    """Read every reviewed provider at its actual clean Git identity."""
    pins_path = root / "build_tools/dependencies/backend-boundaries.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8"))["backends"]
    if type(selected) is not dict or set(selected) != {r["id"] for r in pins}:
        raise ValueError("backend consumer report must name every exact source checkout")
    files = set()
    for row in pins:
        source = Path(selected[row["id"]]).resolve(strict=True)
        # Ignored build outputs are not admitted consumer inputs.
        names = set(_git(source, "ls-files", "-z").split("\0"))
        paths = {path for path in _package_files(source) if path.relative_to(source).as_posix() in names}
        admitted = tracked_consumer_files(source, row["revision"], paths | {source / "pyproject.toml"})
        files.update(path for path in admitted if path.suffix == ".py")
    return tuple(sorted(files))


def selected_static_sources(selected: dict[str, str], root: Path) -> tuple[Path, ...]:
    """Exact pinned support call chains are source edges, not installed providers."""
    manifest = json.loads((root / "build_tools/dependencies/backend-boundaries.json").read_text(encoding="utf-8"))
    pins = manifest.get("source_consumers")
    if pins is None:
        pins = []  # Manifests without separate static consumers select none.
    if type(selected) is not dict or set(selected) != {row["id"] for row in pins}:
        raise ValueError("static consumer report must name every exact source checkout")
    files = set()
    for row in pins:
        source = Path(selected[row["id"]]).resolve(strict=True)
        paths = set()
        for name in row["files"]:
            if not isinstance(name, str) or not name or any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError("static consumer pathname escapes its checkout")
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("static consumer pathname escapes its checkout")
            paths.add(source / path)
        actual = tracked_consumer_files(source, row["revision"], paths)
        files.update(path for path in actual if path.suffix == ".py")
    return tuple(sorted(files))


def backend_sources(report: Path, root: Path) -> tuple[Path, ...]:
    """Reopen the runner's exact sources; a saved success cannot forge imports."""
    pins_path = root / "build_tools/dependencies/backend-boundaries.json"
    result = json.loads(report.read_text(encoding="utf-8"))
    if (
        not isinstance(result, dict)
        or result.get("schema") != "merlin.backend-boundaries.v1"
        or result.get("passed") is not True
        or result.get("pins_sha256") != hashlib.sha256(pins_path.read_bytes()).hexdigest()
    ):
        raise ValueError("backend consumer report must pass against the current reviewed pins")
    sources = selected_backend_sources(result.get("sources"), root)
    static = result.get("static_sources")
    if static is None:
        static = {}  # A report without this section cannot satisfy any selected static consumer.
    return tuple(sorted(set(sources) | set(selected_static_sources(static, root))))
