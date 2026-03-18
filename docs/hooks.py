from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
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
_CLI_MODULES = (
    ("merlin.py", "merlin", "Unified Merlin developer command reference parser."),
    ("build.py", "build", "Configure and build Merlin and target runtimes"),
    ("compile.py", "compile", "Compile MLIR/ONNX models to target artifacts"),
    ("setup.py", "setup", "Bootstrap developer environment and toolchains"),
    ("ci.py", "ci", "Run repository CI/lint/patch workflows"),
    ("patches.py", "patches", "Apply/verify/refresh/drift patch stack"),
    ("benchmark.py", "benchmark", "Run benchmark helper scripts"),
)


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
    expected_docs: set[Path] = set()

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
        expected_docs.add(doc_path.resolve())
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

    if generated_root.exists():
        for existing in generated_root.rglob("*.md"):
            if existing.resolve() not in expected_docs:
                existing.unlink()

    _write_if_changed(docs_root / "reference" / "python" / "index.md", "\n".join(index_lines) + "\n")


def _format_option_name(action: argparse.Action) -> str:
    if action.option_strings:
        return ", ".join(f"`{opt}`" for opt in action.option_strings)
    return f"`{action.dest}`"


def _format_option_default(action: argparse.Action) -> str:
    if action.default in (None, argparse.SUPPRESS):
        return "-"
    return f"`{action.default}`"


def _format_option_choices(action: argparse.Action) -> str:
    if not action.choices:
        return "-"
    values = ", ".join(str(choice) for choice in action.choices)
    return f"`{values}`"


def _render_option_table(parser: argparse.ArgumentParser) -> list[str]:
    lines = [
        "| Argument | Required | Default | Choices | Help |",
        "| --- | --- | --- | --- | --- |",
    ]
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or isinstance(action, argparse._SubParsersAction):
            continue
        help_text = (action.help or "").replace("\n", " ").strip()
        lines.append(
            f"| {_format_option_name(action)} | "
            f"{'yes' if getattr(action, 'required', False) else 'no'} | "
            f"{_format_option_default(action)} | "
            f"{_format_option_choices(action)} | "
            f"{help_text} |"
        )
    return lines


def _render_subcommand_tables(parser: argparse.ArgumentParser) -> list[str]:
    lines: list[str] = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for sub_name in sorted(action.choices):
            sub_parser = action.choices[sub_name]
            lines.extend(
                [
                    f"#### Subcommand `{sub_name}`",
                    "",
                    "```text",
                    sub_parser.format_usage().strip(),
                    "```",
                    "",
                    *_render_option_table(sub_parser),
                    "",
                ]
            )
    return lines


def _build_cli_reference(repo_root: Path, docs_root: Path) -> None:
    tools_dir = repo_root / "tools"
    os.sys.path.insert(0, str(tools_dir))
    try:
        lines = [
            "# CLI Reference",
            "",
            "This page is generated from real argparse parsers in `tools/*.py`.",
            "",
            "Each command is shown with argument introspection and raw `--help` output.",
            "",
        ]

        for script_name, module_name, summary in _CLI_MODULES:
            module = importlib.import_module(module_name)
            parser = argparse.ArgumentParser(
                prog=f"uv run tools/{script_name}",
                description=summary,
            )
            module.setup_parser(parser)

            lines.extend(
                [
                    f"## `tools/{script_name}`",
                    "",
                    summary,
                    "",
                    "### Usage",
                    "",
                    "```text",
                    parser.format_usage().strip(),
                    "```",
                    "",
                    "### Arguments",
                    "",
                    *_render_option_table(parser),
                    "",
                    *_render_subcommand_tables(parser),
                    "### `--help` Output",
                    "",
                    "```text",
                    parser.format_help().rstrip(),
                    "```",
                    "",
                ]
            )
    finally:
        try:
            os.sys.path.remove(str(tools_dir))
        except ValueError:
            pass

    _write_if_changed(docs_root / "reference" / "cli.md", "\n".join(lines))


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


def _find_include_file(include_dirs: list[Path], relative_path: Path) -> Path | None:
    for include_dir in include_dirs:
        candidate = include_dir / relative_path
        if candidate.exists():
            return candidate
    return None


def _generate_mlir_docs(repo_root: Path, docs_root: Path) -> None:
    tblgen = _resolve_mlir_tblgen(repo_root)

    include_dirs = [
        repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "IR",
        repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "Transforms",
        repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "IR",
        repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "Transforms",
        repo_root / "compiler" / "src" / "merlin" / "Codegen" / "Dialect" / "Stream" / "Transforms",
        repo_root / "third_party" / "iree_bar" / "compiler" / "src",
        repo_root / "third_party" / "iree_bar" / "third_party" / "llvm-project" / "mlir" / "include",
        Path("/usr/lib/llvm-20/include"),
        Path("/usr/lib/llvm-19/include"),
        Path("/usr/lib/llvm-18/include"),
        Path("/usr/lib/llvm-17/include"),
    ]

    required_td = Path("mlir/IR/DialectBase.td")
    include_hit = _find_include_file(include_dirs, required_td)
    if not include_hit:
        print(
            "WARNING: skipping MLIR TableGen docs generation because "
            f"'{required_td.as_posix()}' was not found in include dirs."
        )
        return

    mlir_generated = docs_root / "reference" / "generated" / "mlir"
    jobs = [
        (
            "Gemmini dialect",
            "-gen-dialect-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "IR" / "GemminiOps.td",
            "gemmini_dialect.md",
        ),
        (
            "Gemmini operations",
            "-gen-op-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "IR" / "GemminiOps.td",
            "gemmini_ops.md",
        ),
        (
            "Gemmini attributes",
            "-gen-attrdef-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "IR" / "GemminiAttrs.td",
            "gemmini_attrs.md",
        ),
        (
            "Gemmini passes",
            "-gen-pass-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "Gemmini" / "Transforms" / "Passes.td",
            "gemmini_passes.md",
        ),
        (
            "NPU kernel dialect",
            "-gen-dialect-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "IR" / "NPUKernelOps.td",
            "npu_kernel_dialect.md",
        ),
        (
            "NPU kernel operations",
            "-gen-op-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "IR" / "NPUKernelOps.td",
            "npu_kernel_ops.md",
        ),
        (
            "NPU schedule dialect",
            "-gen-dialect-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "IR" / "NPUScheduleOps.td",
            "npu_schedule_dialect.md",
        ),
        (
            "NPU schedule operations",
            "-gen-op-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "IR" / "NPUScheduleOps.td",
            "npu_schedule_ops.md",
        ),
        (
            "NPU ISA dialect",
            "-gen-dialect-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "IR" / "NPUISAOps.td",
            "npu_isa_dialect.md",
        ),
        (
            "NPU ISA operations",
            "-gen-op-doc",
            repo_root / "compiler" / "src" / "merlin" / "Dialect" / "NPU" / "IR" / "NPUISAOps.td",
            "npu_isa_ops.md",
        ),
        (
            "Stream codegen passes",
            "-gen-pass-doc",
            repo_root / "compiler" / "src" / "merlin" / "Codegen" / "Dialect" / "Stream" / "Transforms" / "Passes.td",
            "stream_passes.md",
        ),
    ]

    generated_links: list[str] = []
    expected_docs: set[Path] = set()
    for label, mode, td_file, output_name in jobs:
        if not td_file.exists():
            continue
        output = mlir_generated / output_name
        _run_tblgen(tblgen, include_dirs, mode, td_file, output)
        expected_docs.add(output.resolve())
        generated_links.append(f"- [{label}](generated/mlir/{output_name})")

    if mlir_generated.exists():
        for existing in mlir_generated.glob("*.md"):
            if existing.resolve() not in expected_docs:
                existing.unlink()

    page_lines = [
        "# MLIR Dialects & Passes",
        "",
        "The files linked below are generated from TableGen (`.td`) definitions during docs build.",
        "",
    ]
    if generated_links:
        page_lines.extend(generated_links)
    else:
        page_lines.extend(["- No TableGen docs were generated (no matching `.td` inputs found)."])
    page_lines.append("")
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


## ---------------------------------------------------------------------------
# C/C++ API docs: Doxygen XML → Markdown
# ---------------------------------------------------------------------------

_CPP_HEADER_DIRS = ("samples/common/core", "samples/common/runtime", "samples/common/dispatch")

_CPP_SUBDIR_LABELS = {
    "core": ("Core Utilities", "`samples/common/core/` — Generic utilities (no IREE dependency)"),
    "runtime": ("Runtime Utilities", "`samples/common/runtime/` — IREE runtime helpers"),
    "dispatch": ("Dispatch Scheduling", "`samples/common/dispatch/` — Dispatch graph types, parsing, and output"),
}


def _xml_text(node) -> str:
    """Recursively extract plain text from a Doxygen XML element."""
    if node is None:
        return ""
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        tag = child.tag
        if tag == "computeroutput":
            parts.append(f"`{_xml_text(child)}`")
        elif tag == "ref":
            parts.append(f"`{_xml_text(child)}`")
        elif tag == "parameterlist":
            pass  # handled separately
        elif tag == "simplesect":
            pass  # handled separately
        else:
            parts.append(_xml_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _xml_paras(node) -> str:
    """Extract text from all <para> children, joined by double newlines."""
    if node is None:
        return ""
    paras = []
    for para in node.findall("para"):
        text = _xml_text(para)
        if text:
            paras.append(text)
    return "\n\n".join(paras)


def _extract_param_docs(detail_node) -> dict[str, str]:
    """Extract @param name → description from a detaileddescription element."""
    docs: dict[str, str] = {}
    if detail_node is None:
        return docs
    for para in detail_node.findall("para"):
        for plist in para.findall("parameterlist"):
            if plist.get("kind") != "param":
                continue
            for item in plist.findall("parameteritem"):
                names = item.findall("parameternamelist/parametername")
                desc = item.find("parameterdescription")
                if names and desc:
                    name = names[0].text or ""
                    docs[name] = _xml_paras(desc)
    return docs


def _extract_return_doc(detail_node) -> str:
    """Extract @return description from a detaileddescription element."""
    if detail_node is None:
        return ""
    for para in detail_node.findall("para"):
        for sect in para.findall("simplesect"):
            if sect.get("kind") == "return":
                return _xml_paras(sect)
    return ""


def _extract_note_doc(detail_node) -> str:
    """Extract @note from a detaileddescription element."""
    if detail_node is None:
        return ""
    for para in detail_node.findall("para"):
        for sect in para.findall("simplesect"):
            if sect.get("kind") == "note":
                return _xml_paras(sect)
    return ""


def _run_doxygen(repo_root: Path, output_dir: Path) -> Path | None:
    """Run Doxygen on samples/common/ headers. Returns XML dir or None."""
    doxygen_bin = shutil.which("doxygen")
    if not doxygen_bin:
        print("WARNING: doxygen not found, skipping C++ API docs generation.")
        return None

    doxyfile_template = repo_root / "docs" / "Doxyfile.in"
    if not doxyfile_template.exists():
        print("WARNING: docs/Doxyfile.in not found, skipping C++ API docs.")
        return None

    input_dirs = " ".join(str(repo_root / d) for d in _CPP_HEADER_DIRS if (repo_root / d).exists())
    if not input_dirs:
        print("WARNING: no C++ header directories found, skipping C++ API docs.")
        return None

    template = doxyfile_template.read_text(encoding="utf-8")
    doxyfile_content = template.replace("@INPUT_DIRS@", input_dirs).replace("@OUTPUT_DIR@", str(output_dir))

    doxyfile_path = output_dir / "Doxyfile"
    doxyfile_path.write_text(doxyfile_content, encoding="utf-8")

    result = subprocess.run(
        [doxygen_bin, str(doxyfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: Doxygen failed (exit {result.returncode}): {result.stderr[:500]}")
        return None

    xml_dir = output_dir / "xml"
    if not (xml_dir / "index.xml").exists():
        print("WARNING: Doxygen produced no index.xml, skipping C++ API docs.")
        return None

    return xml_dir


def _parse_compound_xml(xml_dir: Path, refid: str):
    """Parse a single Doxygen compound XML file."""
    xml_path = xml_dir / f"{refid}.xml"
    if not xml_path.exists():
        return None
    return ET.parse(xml_path).getroot()


def _parse_members(sectiondef, param_style: str = "table") -> list[dict]:
    """Parse memberdef elements from a sectiondef."""
    members = []
    for mdef in sectiondef.findall("memberdef"):
        kind = mdef.get("kind", "")
        name_el = mdef.find("name")
        name = name_el.text if name_el is not None else ""
        type_el = mdef.find("type")
        type_str = _xml_text(type_el) if type_el is not None else ""
        brief = _xml_paras(mdef.find("briefdescription"))
        detail = mdef.find("detaileddescription")
        detail_text = _xml_paras(detail)
        param_docs = _extract_param_docs(detail)
        return_doc = _extract_return_doc(detail)

        # Build argsstring for functions
        argsstring_el = mdef.find("argsstring")
        argsstring = argsstring_el.text if argsstring_el is not None else ""

        # Parse params
        params = []
        for p in mdef.findall("param"):
            ptype = _xml_text(p.find("type")) if p.find("type") is not None else ""
            pname_el = p.find("declname")
            pname = pname_el.text if pname_el is not None else ""
            pdesc = param_docs.get(pname, "")
            params.append({"name": pname, "type": ptype, "description": pdesc})

        members.append(
            {
                "kind": kind,
                "name": name,
                "type": type_str,
                "brief": brief,
                "detail": detail_text,
                "return_doc": return_doc,
                "argsstring": argsstring,
                "params": params,
                "static": mdef.get("static", "no") == "yes",
            }
        )
    return members


def _parse_struct(xml_dir: Path, refid: str, qualified_name: str) -> dict | None:
    """Parse a struct/class compound into a descriptor."""
    root = _parse_compound_xml(xml_dir, refid)
    if root is None:
        return None
    cdef = root.find(".//compounddef")
    if cdef is None:
        return None

    brief = _xml_paras(cdef.find("briefdescription"))
    detail = _xml_paras(cdef.find("detaileddescription"))

    variables = []
    methods = []
    for sdef in cdef.findall("sectiondef"):
        mems = _parse_members(sdef)
        for m in mems:
            if m["kind"] == "variable":
                variables.append(m)
            elif m["kind"] == "function":
                methods.append(m)
            elif m["kind"] == "enum":
                methods.append(m)  # enums inside structs

    return {
        "name": qualified_name.split("::")[-1],
        "qualified": qualified_name,
        "brief": brief,
        "detail": detail,
        "variables": variables,
        "methods": methods,
    }


def _parse_doxygen_xml(xml_dir: Path, repo_root: Path) -> list[dict]:
    """Parse Doxygen XML into a list of file descriptors."""
    index = ET.parse(xml_dir / "index.xml").getroot()

    file_descs = []
    for compound in index.findall("compound"):
        if compound.get("kind") != "file":
            continue
        refid = compound.get("refid", "")
        root = _parse_compound_xml(xml_dir, refid)
        if root is None:
            continue
        cdef = root.find(".//compounddef")
        if cdef is None:
            continue

        # Get source location to determine subdir
        location = cdef.find("location")
        if location is None:
            continue
        file_path = location.get("file", "")
        try:
            rel_path = Path(file_path).relative_to(repo_root).as_posix()
        except ValueError:
            rel_path = file_path

        # Determine subdir (core/runtime/dispatch)
        subdir = ""
        for d in _CPP_HEADER_DIRS:
            if rel_path.startswith(d + "/") or rel_path.startswith(d.replace("/", os.sep) + os.sep):
                subdir = d.split("/")[-1]
                break
        if not subdir:
            continue

        filename = cdef.findtext("compoundname", "")
        brief = _xml_paras(cdef.find("briefdescription"))
        detail = _xml_paras(cdef.find("detaileddescription"))

        # Collect structs/classes
        structs = []
        for inner in cdef.findall("innerclass"):
            inner_refid = inner.get("refid", "")
            inner_name = inner.text or ""
            s = _parse_struct(xml_dir, inner_refid, inner_name)
            if s:
                structs.append(s)

        # Collect enums and free functions
        enums = []
        functions = []
        for sdef in cdef.findall("sectiondef"):
            skind = sdef.get("kind", "")
            if "enum" in skind:
                for mdef in sdef.findall("memberdef"):
                    if mdef.get("kind") != "enum":
                        continue
                    name_el = mdef.find("name")
                    name = name_el.text if name_el is not None else ""
                    ebrief = _xml_paras(mdef.find("briefdescription"))
                    values = []
                    for ev in mdef.findall("enumvalue"):
                        ev_name = ev.findtext("name", "")
                        ev_brief = _xml_paras(ev.find("briefdescription"))
                        ev_init = ev.findtext("initializer", "")
                        values.append({"name": ev_name, "brief": ev_brief, "initializer": ev_init})
                    enums.append({"name": name, "brief": ebrief, "values": values})
            elif "func" in skind:
                functions.extend(_parse_members(sdef))

        file_descs.append(
            {
                "filename": filename,
                "location": rel_path,
                "subdir": subdir,
                "brief": brief,
                "detail": detail,
                "structs": structs,
                "enums": enums,
                "functions": functions,
            }
        )

    file_descs.sort(key=lambda d: (d["subdir"], d["filename"]))
    return file_descs


def _render_cpp_file_page(desc: dict) -> str:
    """Render a file descriptor as a markdown page."""
    lines = [f"# {desc['filename']}", "", f"**Source:** `{desc['location']}`", ""]
    if desc["brief"]:
        lines.extend([desc["brief"], ""])
    if desc["detail"]:
        lines.extend([desc["detail"], ""])

    # Enums
    if desc["enums"]:
        lines.append("---")
        lines.append("")
        lines.append("## Enums")
        lines.append("")
        for enum in desc["enums"]:
            lines.append(f"### `{enum['name']}`")
            lines.append("")
            if enum["brief"]:
                lines.extend([enum["brief"], ""])
            if enum["values"]:
                lines.append("| Value | Description |")
                lines.append("| --- | --- |")
                for v in enum["values"]:
                    desc_text = v["brief"] or ""
                    lines.append(f"| `{v['name']}` | {desc_text} |")
                lines.append("")

    # Structs
    if desc["structs"]:
        lines.append("---")
        lines.append("")
        lines.append("## Structs")
        lines.append("")
        for s in desc["structs"]:
            lines.append(f"### `{s['name']}`")
            lines.append("")
            if s["brief"]:
                lines.extend([s["brief"], ""])
            if s["detail"]:
                lines.extend([s["detail"], ""])

            if s["variables"]:
                lines.append("#### Members")
                lines.append("")
                lines.append("| Name | Type | Description |")
                lines.append("| --- | --- | --- |")
                for v in s["variables"]:
                    lines.append(f"| `{v['name']}` | `{v['type']}` | {v['brief']} |")
                lines.append("")

            if s["methods"]:
                lines.append("#### Methods")
                lines.append("")
                for m in s["methods"]:
                    lines.append(f"##### `{m['name']}{m['argsstring']}`")
                    lines.append("")
                    if m["brief"]:
                        lines.extend([m["brief"], ""])
                    if m["params"]:
                        lines.append("**Parameters:**")
                        lines.append("")
                        lines.append("| Name | Type | Description |")
                        lines.append("| --- | --- | --- |")
                        for p in m["params"]:
                            lines.append(f"| `{p['name']}` | `{p['type']}` | {p['description']} |")
                        lines.append("")
                    if m["return_doc"]:
                        lines.append(f"**Returns:** {m['return_doc']}")
                        lines.append("")

    # Free functions
    if desc["functions"]:
        lines.append("---")
        lines.append("")
        lines.append("## Functions")
        lines.append("")
        for f in desc["functions"]:
            sig_prefix = "static " if f["static"] else ""
            lines.append(f"### `{f['name']}`")
            lines.append("")
            lines.append("```cpp")
            lines.append(f"{sig_prefix}{f['type']} {f['name']}{f['argsstring']}")
            lines.append("```")
            lines.append("")
            if f["brief"]:
                lines.extend([f["brief"], ""])
            if f["params"]:
                lines.append("**Parameters:**")
                lines.append("")
                lines.append("| Name | Type | Description |")
                lines.append("| --- | --- | --- |")
                for p in f["params"]:
                    lines.append(f"| `{p['name']}` | `{p['type']}` | {p['description']} |")
                lines.append("")
            if f["return_doc"]:
                lines.append(f"**Returns:** {f['return_doc']}")
                lines.append("")

    return "\n".join(lines) + "\n"


def _render_cpp_index_page(file_descs: list[dict]) -> str:
    """Render the C/C++ API index page."""
    lines = [
        "# C/C++ API",
        "",
        "Auto-generated from Doxygen comments in `samples/common/` headers.",
        "",
    ]
    current_subdir = None
    for desc in file_descs:
        if desc["subdir"] != current_subdir:
            current_subdir = desc["subdir"]
            label, subtitle = _CPP_SUBDIR_LABELS.get(current_subdir, (current_subdir, ""))
            lines.extend(["", f"## {label}", "", subtitle, ""])
        rel_link = f"generated/cpp/{desc['subdir']}/{Path(desc['filename']).stem}.md"
        brief = f" — {desc['brief']}" if desc["brief"] else ""
        lines.append(f"- [`{desc['filename']}`]({rel_link}){brief}")

    lines.append("")
    return "\n".join(lines)


def _generate_cpp_api_docs(repo_root: Path, docs_root: Path) -> None:
    """Generate C/C++ API reference docs from Doxygen XML."""
    tmp_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="merlin_doxygen_"))
        xml_dir = _run_doxygen(repo_root, tmp_dir)
        if xml_dir is None:
            return

        file_descs = _parse_doxygen_xml(xml_dir, repo_root)
        if not file_descs:
            print("WARNING: Doxygen XML produced no file descriptors.")
            return

        generated_root = docs_root / "reference" / "generated" / "cpp"
        expected_docs: set[Path] = set()

        for desc in file_descs:
            stem = Path(desc["filename"]).stem
            doc_path = generated_root / desc["subdir"] / f"{stem}.md"
            expected_docs.add(doc_path.resolve())
            _write_if_changed(doc_path, _render_cpp_file_page(desc))

        # Clean stale files
        if generated_root.exists():
            for existing in generated_root.rglob("*.md"):
                if existing.resolve() not in expected_docs:
                    existing.unlink()

        # Write index page
        _write_if_changed(docs_root / "reference" / "cpp.md", _render_cpp_index_page(file_descs))

        print(f"C++ API docs: generated {len(file_descs)} pages from Doxygen XML.")
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def generate_reference_docs(repo_root: Path, docs_root: Path) -> None:
    _build_cli_reference(repo_root, docs_root)
    _generate_python_api_docs(repo_root, docs_root)
    _generate_mlir_docs(repo_root, docs_root)
    _generate_cmake_targets_page(repo_root, docs_root)
    _generate_repository_guide(repo_root, docs_root)
    _generate_cpp_api_docs(repo_root, docs_root)


def on_pre_build(config, **kwargs) -> None:
    config_path = _config_file_path(config)
    repo_root = config_path.parent
    docs_root = repo_root / "docs"
    generate_reference_docs(repo_root, docs_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Merlin docs reference pages.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root path (default: inferred from docs/hooks.py location).",
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=None,
        help="Docs source root path (default: <repo-root>/docs).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    docs_root = args.docs_root.resolve() if args.docs_root else (repo_root / "docs").resolve()
    generate_reference_docs(repo_root, docs_root)


if __name__ == "__main__":
    main()
