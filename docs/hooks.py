from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

# Large generated/derived directories that should never feed API docs.
_EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "compiled_models",
    "compilation_phases_fc",
    "compilation_phases_resnet",
    "docs",
    "mlirEvolve",
    "report_pt_export",
    "third_party",
    "tmp",
}

_PYTHON_SCAN_ROOTS = ("tools", "models", "benchmarks", "samples")
_EXCLUDED_PATH_SNIPPETS = ("samples/SaturnOPU/custom_dispatch_ukernels/",)


def _config_file_path(config) -> Path:
    config_file = getattr(config, "config_file_path", None)
    if not config_file and isinstance(config, dict):
        config_file = config.get("config_file_path")
    if not config_file:
        raise RuntimeError("MkDocs config file path could not be resolved.")
    return Path(config_file).resolve()


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _is_excluded_python(path: Path) -> bool:
    if path.suffix != ".py":
        return True
    if path.name == "__init__.py":
        return True
    if any(part in _EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    if path.name.startswith("."):
        return True
    path_posix = path.as_posix()
    if any(snippet in path_posix for snippet in _EXCLUDED_PATH_SNIPPETS):
        return True
    return False


def _collect_python_modules(repo_root: Path) -> list[tuple[str, Path]]:
    modules: list[tuple[str, Path]] = []
    for root_name in _PYTHON_SCAN_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            if _is_excluded_python(py_file):
                continue
            rel_no_suffix = py_file.relative_to(repo_root).with_suffix("")
            module_name = ".".join(rel_no_suffix.parts)
            modules.append((module_name, py_file.relative_to(repo_root)))
    return modules


def _generate_python_api_docs(repo_root: Path, docs_root: Path) -> None:
    modules = _collect_python_modules(repo_root)
    generated_root = docs_root / "reference" / "generated" / "python"

    index_lines = [
        "# Python API",
        "",
        "The pages below are generated from Python source files at docs build time.",
        "",
    ]

    current_group = None
    for module_name, rel_file in modules:
        group = module_name.split(".", 1)[0]
        if group != current_group:
            if current_group is not None:
                index_lines.append("")
            index_lines.append(f"## `{group}/`")
            index_lines.append("")
            current_group = group

        doc_path = generated_root / ("/".join(module_name.split(".")) + ".md")
        rel_link = Path(os.path.relpath(doc_path, docs_root / "reference" / "python"))
        page = "\n".join(
            [
                f"# `{module_name}`",
                "",
                f"Source: `{rel_file}`",
                "",
                f"::: {module_name}",
                "",
            ]
        )
        _write_if_changed(doc_path, page)
        index_lines.append(f"- [`{module_name}`]({rel_link.as_posix()})")

    _write_if_changed(docs_root / "reference" / "python" / "index.md", "\n".join(index_lines) + "\n")


def _resolve_mlir_tblgen(repo_root: Path) -> Path:
    env_path = os.environ.get("MLIR_TBLGEN")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise RuntimeError(f"MLIR_TBLGEN is set but not found: {candidate}")

    candidates = [
        repo_root / "build" / "host-vanilla-release" / "llvm-project" / "bin" / "mlir-tblgen",
        repo_root / "build" / "host-merlin-release" / "llvm-project" / "bin" / "mlir-tblgen",
        Path("/usr/lib/llvm-20/bin/mlir-tblgen"),
        Path("/usr/lib/llvm-19/bin/mlir-tblgen"),
        Path("/usr/lib/llvm-18/bin/mlir-tblgen"),
        Path("/usr/lib/llvm-17/bin/mlir-tblgen"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    which = shutil.which("mlir-tblgen")
    if which:
        return Path(which).resolve()

    raise RuntimeError(
        "mlir-tblgen was not found. Set MLIR_TBLGEN or provide a host build with "
        "build/*/llvm-project/bin/mlir-tblgen."
    )


def _run_tblgen(tblgen: Path, include_dirs: list[Path], mode: str, td_file: Path, output: Path) -> None:
    cmd = [str(tblgen)]
    for include_dir in include_dirs:
        if include_dir.exists():
            cmd.extend(["-I", str(include_dir)])
    cmd.extend([mode, str(td_file), "-o", str(output)])
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=td_file.parent)


def _generate_mlir_docs(repo_root: Path, docs_root: Path) -> None:
    tblgen = _resolve_mlir_tblgen(repo_root)

    include_dirs = [
        repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "IR",
        repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "Transforms",
        repo_root / "compiler" / "src" / "merlin" / "Codegen" / "Dialect" / "Stream" / "Transforms",
        repo_root / "third_party" / "iree_bar" / "compiler" / "src",
        repo_root / "third_party" / "iree_bar" / "third_party" / "llvm-project" / "mlir" / "include",
        Path("/usr/lib/llvm-20/include"),
        Path("/usr/lib/llvm-19/include"),
        Path("/usr/lib/llvm-18/include"),
        Path("/usr/lib/llvm-17/include"),
    ]

    mlir_generated = docs_root / "reference" / "generated" / "mlir"
    ops_td = repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "IR" / "GemminiOps.td"
    attrs_td = repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "IR" / "GemminiAttrs.td"
    gemmini_passes_td = repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "Transforms" / "Passes.td"
    stream_passes_td = (
        repo_root / "compiler" / "src" / "merlin" / "Codegen" / "Dialect" / "Stream" / "Transforms" / "Passes.td"
    )

    _run_tblgen(tblgen, include_dirs, "-gen-dialect-doc", ops_td, mlir_generated / "gemmini_dialect.md")
    _run_tblgen(tblgen, include_dirs, "-gen-op-doc", ops_td, mlir_generated / "gemmini_ops.md")
    _run_tblgen(tblgen, include_dirs, "-gen-attrdef-doc", attrs_td, mlir_generated / "gemmini_attrs.md")
    _run_tblgen(tblgen, include_dirs, "-gen-pass-doc", gemmini_passes_td, mlir_generated / "gemmini_passes.md")
    _run_tblgen(tblgen, include_dirs, "-gen-pass-doc", stream_passes_td, mlir_generated / "stream_passes.md")

    page_lines = [
        "# MLIR Dialects & Passes",
        "",
        "The files linked below are generated from TableGen (`.td`) definitions during docs build.",
        "",
        "- [Gemmini dialect](generated/mlir/gemmini_dialect.md)",
        "- [Gemmini operations](generated/mlir/gemmini_ops.md)",
        "- [Gemmini attributes](generated/mlir/gemmini_attrs.md)",
        "- [Gemmini passes](generated/mlir/gemmini_passes.md)",
        "- [Stream codegen passes](generated/mlir/stream_passes.md)",
        "",
    ]
    _write_if_changed(docs_root / "reference" / "mlir.md", "\n".join(page_lines))


def _cmake_file_is_excluded(path: Path) -> bool:
    return any(part in {"third_party", "build", "mlirEvolve", "tmp", ".venv", ".git"} for part in path.parts)


def _extract_iree_name(block: str) -> str | None:
    match = re.search(r"\bNAME\s+([^\s\)]+)", block, flags=re.MULTILINE)
    return match.group(1) if match else None


def _generate_cmake_targets_page(repo_root: Path, docs_root: Path) -> None:
    rows: list[tuple[str, str, str]] = []
    for cmake_file in sorted(repo_root.rglob("CMakeLists.txt")):
        rel = cmake_file.relative_to(repo_root)
        if _cmake_file_is_excluded(rel):
            continue

        text = cmake_file.read_text(encoding="utf-8")

        for kind, name in re.findall(r"\badd_(library|executable|custom_target)\s*\(\s*([^\s\)]+)", text):
            rows.append((f"add_{kind}", name, rel.as_posix()))

        for macro in ("iree_cc_library", "iree_tablegen_library"):
            for block in re.findall(rf"\b{macro}\s*\((.*?)\)\s*", text, flags=re.DOTALL):
                name = _extract_iree_name(block)
                if name:
                    rows.append((macro, name, rel.as_posix()))

        for plugin_id, target in re.findall(
            r"\biree_compiler_register_plugin\s*\(\s*PLUGIN_ID\s+([^\s\)]+)\s+TARGET\s+([^\s\)]+)",
            text,
            flags=re.MULTILINE,
        ):
            rows.append(("iree_compiler_register_plugin", f"{plugin_id} -> {target}", rel.as_posix()))

    rows.sort(key=lambda x: (x[2], x[0], x[1]))

    lines = [
        "# CMake Targets",
        "",
        "This inventory is generated from repository `CMakeLists.txt` files.",
        "",
        "| Kind | Name | Declared In |",
        "| --- | --- | --- |",
    ]
    for kind, name, declared_in in rows:
        lines.append(f"| `{kind}` | `{name}` | `{declared_in}` |")
    lines.append("")
    _write_if_changed(docs_root / "reference" / "cmake_targets.md", "\n".join(lines))


def _build_tree(lines: list[str], node: dict[str, dict], prefix: str = "") -> None:
    keys = sorted(node.keys())
    for idx, key in enumerate(keys):
        branch = "└── " if idx == len(keys) - 1 else "├── "
        lines.append(f"{prefix}{branch}{key}")
        child = node[key]
        if child:
            extension = "    " if idx == len(keys) - 1 else "│   "
            _build_tree(lines, child, prefix + extension)


def _generate_repository_guide(repo_root: Path, docs_root: Path) -> None:
    git_ls = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    tracked_paths = [Path(p) for p in git_ls.stdout.splitlines() if p]

    max_depth = 3
    tree: dict[str, dict] = {}
    for path in tracked_paths:
        if any(part in {".github", ".codex"} for part in path.parts):
            continue
        parts = path.parts[:max_depth]
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    tree_lines = ["```text", "merlin/"]
    _build_tree(tree_lines, tree)
    tree_lines.append("```")

    guide = [
        "# Repository Guide",
        "",
        "Merlin is organized to separate model frontends, compiler internals, and hardware-targeted runtimes.",
        "",
        "## Core Directories",
        "",
        "- `compiler/`: C++ and MLIR compiler code (dialects, passes, plugins).",
        "- `tools/`: Python developer entrypoints (`build.py`, `compile.py`, `setup.py`, `ci.py`, etc.).",
        "- `models/`: Model definitions, exports, and quantization helpers.",
        "- `samples/`: C/C++ runtime examples and hardware-facing sample flows.",
        "- `benchmarks/`: Benchmark scripts and board-specific profiling helpers.",
        "- `docs/`: Documentation source consumed by MkDocs.",
        "",
        "## Placement Conventions (Where New Code Should Go)",
        "",
        "- New compiler dialects/passes/transforms: `compiler/src/merlin/`.",
        "- New plugin/target registration glue: `compiler/plugins/`.",
        "- New model exports or conversion flows: `models/<model_name>/`.",
        "- New target flag bundles for `tools/compile.py`: `models/<target>.yaml`.",
        "- New board/runtime sample executables: `samples/<platform>/`.",
        "- New benchmark flows and parsers: `benchmarks/<target>/`.",
        "- New end-user docs and guides: `docs/`.",
        "",
        "## Tracked Tree Snapshot (Depth 3)",
        "",
        *tree_lines,
        "",
        "This tree is generated from `git ls-files` so it reflects tracked repository state.",
        "",
    ]
    _write_if_changed(docs_root / "repository_guide.md", "\n".join(guide))


def on_pre_build(config, **kwargs) -> None:
    config_path = _config_file_path(config)
    repo_root = config_path.parent
    docs_root = repo_root / "docs"

    _generate_python_api_docs(repo_root, docs_root)
    _generate_mlir_docs(repo_root, docs_root)
    _generate_cmake_targets_page(repo_root, docs_root)
    _generate_repository_guide(repo_root, docs_root)
