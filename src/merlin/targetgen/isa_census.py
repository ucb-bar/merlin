"""Source-linked ISA census from a selected decoder, patterns, and model ISA.

This is an audit reader, not an instruction legality oracle. A BitPat and a
decoder-table row show a decode route; execution conditions, reserved fields,
semantics and selected-configuration admission need separate evidence.

Where a model class's keyword fields (``opcode=``, ``funct3=`` ...) sit in the
instruction word is NOT assumed. It is read from the model's own instruction
FORMAT definitions -- the encoder each format class uses to build its word
(:func:`derive_format_layouts`) -- because that encoder is what gives the
keyword its meaning. A class whose format cannot be found or parsed has an
``UNKNOWN`` layout: it is reported and never matched against a pattern.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import subprocess
from pathlib import Path
from typing import Any

UNKNOWN = "UNKNOWN"


def _git_source_at_revision(path: Path, revision: str) -> dict[str, str]:
    """Bind an observed source file to bytes in one exact local Git commit."""
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("source revision must be an exact 40-character commit")
    source = path.resolve(strict=True)

    def git(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(source.parent), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                f"source revision cannot be verified for {source}: {result.stderr.decode(errors='replace').strip()}"
            )
        return result.stdout

    root = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    relative = source.relative_to(root).as_posix()
    head = git("rev-parse", "HEAD").decode().strip()
    if head != revision:
        raise ValueError(f"selected source checkout HEAD differs from declared revision: {source}")
    selected = git("show", f"{revision}:{relative}")
    observed = source.read_bytes()
    if selected != observed:
        raise ValueError(f"selected source bytes differ from declared Git revision: {source}")
    return {"revision": revision, "path_at_revision": relative, "sha256": hashlib.sha256(observed).hexdigest()}


def _source(path: Path) -> tuple[str, dict[str, str]]:
    data = path.read_bytes()
    return data.decode("utf-8"), {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def _patterns(source: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, source_line in enumerate(source.splitlines(), 1):
        line = source_line.split("//", 1)[0].strip()
        declaration, separator, expression = line.partition("=")
        words = declaration.split()
        if not separator or len(words) != 2 or words[0] != "def" or not expression.strip().startswith("BitPat"):
            continue
        name = words[1]
        function, opening, argument = expression.strip().partition("(")
        if (
            not name.isidentifier()
            or function.strip() != "BitPat"
            or not opening
            or not argument.startswith('"')
            or not argument.endswith('")')
        ):
            raise ValueError("instruction source has an unparsed BitPat definition")
        encoded = argument[1:-2]
        bits = encoded.removeprefix("b").replace("_", "")
        if not encoded.startswith("b") or any(bit not in "01?" for bit in bits) or len(bits) != 32 or name in rows:
            raise ValueError(f"invalid or duplicate 32-bit instruction pattern {name}")
        rows[name] = {
            "name": name,
            "bits": bits,
            "mask": int("".join("0" if bit == "?" else "1" for bit in bits), 2),
            "value": int("".join("0" if bit == "?" else bit for bit in bits), 2),
            "line": line_number,
        }
    if not rows:
        raise ValueError("no BitPat instructions found")
    return rows


def _decode_rows(source: str) -> dict[str, dict[str, Any]]:
    lines = source.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip().startswith("val table:")]
    if len(starts) != 1:
        raise ValueError("decoder has no table declaration")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, source_line in enumerate(lines[starts[0] + 1 :], starts[0] + 2):
        line = source_line.split("//", 1)[0].strip()
        name, separator, expression = line.partition("->")
        if not separator or not expression.strip().startswith("List"):
            continue
        name = name.strip()
        function, opening, argument = expression.strip().partition("(")
        argument = argument.removesuffix(",").strip()
        if not name.isidentifier() or function.strip() != "List" or not opening or not argument.endswith(")"):
            raise ValueError("decoder table has an unparsed or duplicate row")
        controls = tuple(field.strip() for field in argument[:-1].split(","))
        if len(controls) != 17 or name in rows:
            raise ValueError(f"invalid or duplicate 17-column decode row {name}")
        rows[name] = {"name": name, "controls": controls, "line": line_number}
    if not rows:
        raise ValueError("decoder table has no rows")
    return rows


def _encoder_terms(node: ast.AST) -> list[tuple[str, int]] | None:
    """``[(name, shift)]`` for an expression ``a << k | b << m | c`` (``c`` at shift 0), or None."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _encoder_terms(node.left), _encoder_terms(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Name):
        return [(node.id, 0)]
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.LShift)
        and isinstance(node.left, ast.Name)
        and isinstance(node.right, ast.Constant)
        and type(node.right.value) is int
    ):
        return [(node.left.id, node.right.value)]
    return None


def _format_layout(node: ast.ClassDef) -> dict[str, tuple[int, int]]:
    """``{field: (shift, width)}`` read from one format class's encoder.

    A field is an attribute of the instruction (``self.<field>``) narrowed to a literal width by a call
    (``x = <helper>(self.<field>, <width>)``) and placed by the encoder's return expression, a ``|`` of
    ``x << <shift>`` terms. Only that shape is read; anything else in the method places no field, and a
    field placed at two different positions is dropped rather than guessed between."""
    widths: dict[str, tuple[str, int]] = {}
    placed: dict[str, set[int]] = {}
    for method in node.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for statement in ast.walk(method):
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Call)
                and len(statement.value.args) == 2
                and isinstance(statement.value.args[0], ast.Attribute)
                and isinstance(statement.value.args[0].value, ast.Name)
                and statement.value.args[0].value.id == "self"
                and isinstance(statement.value.args[1], ast.Constant)
                and type(statement.value.args[1].value) is int
            ):
                widths[statement.targets[0].id] = (statement.value.args[0].attr, statement.value.args[1].value)
            if isinstance(statement, ast.Return) and statement.value is not None:
                for name, shift in _encoder_terms(statement.value) or ():
                    placed.setdefault(name, set()).add(shift)
    layout: dict[str, tuple[int, int]] = {}
    for local, (field, width) in widths.items():
        shifts = placed.get(local) or set()
        if len(shifts) == 1 and width > 0:
            layout[field] = (next(iter(shifts)), width)
    return layout


def derive_format_layouts(source: str) -> dict[str, dict[str, tuple[int, int]]]:
    """``{format class: {field: (shift, width)}}`` for every class in ``source`` whose encoder places at
    least one field (see :func:`_format_layout`), or that inherits exactly one such layout from a base
    defined in the same module (a format that only renames another encodes the same way)."""
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    layouts = {node.name: layout for node in nodes if (layout := _format_layout(node))}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node.name in layouts:
                continue
            inherited = [layouts[name] for name in _base_names(node) if name in layouts]
            if len(inherited) == 1:
                layouts[node.name] = inherited[0]
                changed = True
    return layouts


def _base_names(node: ast.ClassDef) -> tuple[str, ...]:
    """Base class names, including a parameterised base (``Base[T]`` is ``Base``)."""
    names = []
    for base in node.bases:
        if isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            names.append(base.id)
    return tuple(names)


def _integer_keywords(node: ast.ClassDef) -> dict[str, int]:
    """The class keywords whose value is an integer literal (a ``bool`` is not one)."""
    out: dict[str, int] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            continue
        if isinstance(keyword.value, ast.Constant) and type(keyword.value.value) is int:
            if keyword.arg in out:
                raise ValueError(f"model class {node.name} repeats {keyword.arg}")
            out[keyword.arg] = keyword.value.value
    return out


def _model_classes(source: str, layouts: dict[str, dict[str, tuple[int, int]]]) -> dict[str, dict[str, Any]]:
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    known_fields = {field for layout in layouts.values() for field in layout}
    rows: dict[str, dict[str, Any]] = {}

    def implementation(node: ast.ClassDef) -> str:
        if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "exec" for item in node.body):
            return f"{node.name}.exec:{node.lineno}"
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in classes:
                inherited = implementation(classes[base.id])
                if inherited:
                    return inherited
        return ""

    def formats(node: ast.ClassDef, seen: frozenset[str] = frozenset()) -> set[str]:
        found: set[str] = set()
        for name in _base_names(node):
            if name in layouts:
                found.add(name)
            elif name in classes and name not in seen:
                found |= formats(classes[name], seen | {name})
        return found

    for node in classes.values():
        declared = _integer_keywords(node)
        for keyword in node.keywords:
            if keyword.arg in known_fields and keyword.arg not in declared:
                raise ValueError(f"model class {node.name} has a nonliteral {keyword.arg}")
        if not declared:
            continue
        base_names = _base_names(node)
        selected = formats(node)
        layout = layouts[next(iter(selected))] if len(selected) == 1 else None
        unresolved = sorted(declared) if layout is None else sorted(set(declared) - set(layout))
        mask = value = 0
        fixed: dict[str, int] = {}
        for field, number in declared.items():
            if layout is None or field not in layout:
                continue
            shift, width = layout[field]
            if not 0 <= number < (1 << width):
                raise ValueError(f"model class {node.name} has out-of-range {field}")
            fixed[field] = number
            mask |= ((1 << width) - 1) << shift
            value |= number << shift
        if node.name in rows:
            raise ValueError(f"duplicate model class {node.name}")
        derived = layout is not None and not unresolved
        rows[node.name] = {
            "name": node.name,
            "line": node.lineno,
            "fixed_fields": fixed,
            "mask": mask if derived else None,
            "value": value if derived else None,
            "format_bases": base_names,
            "format": next(iter(selected)) if len(selected) == 1 else (sorted(selected) or UNKNOWN),
            "layout": "derived" if derived else UNKNOWN,
            **({"unresolved_fields": unresolved} if unresolved else {}),
            "exec_source": implementation(node) or "external_or_absent",
        }
    if not rows:
        raise ValueError("model ISA has no literal-keyword instruction classes")
    return rows


def _format_sources(model_isa_file: Path, model_text: str) -> list[Path]:
    """The modules the model ISA imports its names from, found beside it (nearest ancestor first).

    ``from pkg.mod import X`` resolves to ``<ancestor>/pkg/mod.py`` for the closest ancestor of the
    model file that holds it; a module that cannot be found is simply not a source."""
    found: list[Path] = []
    for node in ast.parse(model_text).body:
        if not isinstance(node, ast.ImportFrom) or not node.module or node.level:
            continue
        relative = Path(*node.module.split(".")).with_suffix(".py")
        for ancestor in Path(model_isa_file).resolve().parents:
            candidate = ancestor / relative
            if candidate.is_file():
                if candidate not in found:
                    found.append(candidate)
                break
    return found


def _layouts_from(paths: list[Path]) -> tuple[dict[str, dict[str, tuple[int, int]]], list[dict[str, str]]]:
    layouts: dict[str, dict[str, tuple[int, int]]] = {}
    conflicting: set[str] = set()
    sources = []
    for path in paths:
        text, source = _source(path)
        for name, layout in derive_format_layouts(text).items():
            if name in layouts and layouts[name] != layout:
                conflicting.add(name)
            layouts.setdefault(name, layout)
        sources.append(source)
    for name in conflicting:
        layouts.pop(name)  # two sources disagree on one format: neither is taken
    return layouts, sources


def _family(controls: tuple[str, ...]) -> str:
    if controls[11] != "MXU_X":
        return "mxu"
    if controls[10] != "DMA_X":
        return "dma"
    if controls[14] != "XLU_X":
        return "xlu"
    if controls[13] in {"VPU_FP8PACK", "VPU_FP8UNPACK"}:
        return "scale_pack"
    if controls[13] != "VPU_X":
        return "vpu"
    if controls[16] != "LSU_X":
        return "tensor_transfer"
    if controls[9] != "CSR_X":
        return "csr"
    if controls[15] != "MEM_X":
        return "scalar_memory"
    return "scalar_control"


def derive_source_census(
    *,
    pattern_file: Path,
    decoder_file: Path,
    model_isa_file: Path,
    rtl_revision: str,
    format_file: Path | None = None,
    model_revision: str | None = None,
    verify_revisions: bool = False,
) -> dict[str, Any]:
    """Cross-link selected source bytes; retain all unqualified obligations.

    ``format_file`` is the model's instruction-format module; when omitted, the modules the model ISA
    imports from are looked up beside it. Keyword positions come only from those encoders.

    With ``verify_revisions`` every source read is bound to its declared commit: the patterns and the
    decoder to ``rtl_revision``, the model ISA and each format module it is encoded with to
    ``model_revision``."""
    if not rtl_revision or len(rtl_revision) != 40 or any(c not in "0123456789abcdef" for c in rtl_revision):
        raise ValueError("exact selected RTL commit is required")
    revision_verification: dict[str, Any] = {"status": "unverified"}
    if verify_revisions:
        if model_revision is None:
            raise ValueError("model revision is required for source revision verification")
        rtl_patterns = _git_source_at_revision(pattern_file, rtl_revision)
        rtl_decoder = _git_source_at_revision(decoder_file, rtl_revision)
        model_source = _git_source_at_revision(model_isa_file, model_revision)
        revision_verification = {
            "status": "verified",
            "rtl_revision": rtl_revision,
            "model_revision": model_revision,
            "patterns": rtl_patterns,
            "decoder": rtl_decoder,
            "model_isa": model_source,
        }
    pattern_text, pattern_source = _source(pattern_file)
    decoder_text, decoder_source = _source(decoder_file)
    model_text, model_source = _source(model_isa_file)
    patterns = _patterns(pattern_text)
    decoded = _decode_rows(decoder_text)
    format_paths = [Path(format_file)] if format_file is not None else _format_sources(model_isa_file, model_text)
    if verify_revisions:
        # The format modules place every keyword field, so their bytes are bound like the model's own.
        revision_verification["formats"] = [_git_source_at_revision(path, str(model_revision)) for path in format_paths]
    layouts, format_sources = _layouts_from(format_paths)
    models = _model_classes(model_text, layouts)

    overlaps = [
        [left["name"], right["name"]]
        for left, right in itertools.combinations(patterns.values(), 2)
        if ((left["value"] ^ right["value"]) & left["mask"] & right["mask"]) == 0
    ]
    rows: list[dict[str, Any]] = []
    matched_models: set[str] = set()
    dma_kind_conflicts: list[list[str]] = []
    for name, pattern in patterns.items():
        decode = decoded.get(name)
        candidates = []
        for model in models.values():
            if model["mask"] is None:
                continue  # an underived layout is matched against nothing (reported below)
            shared_mask = pattern["mask"] & model["mask"]
            if ((pattern["value"] ^ model["value"]) & shared_mask) != 0:
                continue
            relation = (
                "keyword_encoding_implies_pattern"
                if pattern["mask"] & ~model["mask"] == 0
                else ("partial_keyword_compatibility")
            )
            if name.startswith("DMA_") and model["name"].startswith("DMA_"):
                pattern_kind = name.split("_")[1]
                model_kind = model["name"].split("_")[1]
                if pattern_kind != model_kind:
                    dma_kind_conflicts.append([name, model["name"]])
            candidates.append(
                {
                    "name": model["name"],
                    "line": model["line"],
                    "relation": relation,
                    "fixed_fields": model["fixed_fields"],
                    "format_bases": model["format_bases"],
                    "exec_source": model["exec_source"],
                }
            )
            matched_models.add(model["name"])
        rows.append(
            {
                "name": name,
                "pattern_bits": pattern["bits"],
                "pattern_line": pattern["line"],
                "decode_line": decode["line"] if decode else None,
                "decode_controls": list(decode["controls"]) if decode else None,
                "family": _family(decode["controls"]) if decode else "unmapped",
                "model_candidates": candidates,
                "reachability": "decoder_table_entry_only" if decode else "not_in_decoder_table",
                "required_status": "unfrozen",
                "qualification": {
                    "legality": "unreviewed",
                    "computation": "unreviewed",
                    "storage": "unreviewed",
                    "timing": "unreviewed",
                    "software_admission": "unreviewed",
                    "emission": "unreviewed",
                    "execution": "unreviewed",
                },
            }
        )
    return {
        "schema": "merlin.isa_source_census.v1",
        "scope": "source_crosswalk_only_no_legal_or_executable_variant_denominator",
        "rtl_revision": rtl_revision,
        "source_revision_verification": revision_verification,
        "sources": {
            "patterns": pattern_source,
            "decoder": decoder_source,
            "model_isa": model_source,
            "formats": format_sources or {"status": UNKNOWN, "reason": "no instruction-format module was found"},
        },
        "format_layouts": {name: {f: list(span) for f, span in layout.items()} for name, layout in layouts.items()},
        "summary": {
            "patterns": len(patterns),
            "decoder_rows": len(decoded),
            "model_opcode_classes": len(models),
            "patterns_not_decoded": sorted(set(patterns) - set(decoded)),
            "decoder_rows_without_pattern": sorted(set(decoded) - set(patterns)),
            "model_classes_without_compatible_pattern": sorted(
                name for name in set(models) - matched_models if models[name]["mask"] is not None
            ),
            "model_classes_without_derived_layout": sorted(n for n, m in models.items() if m["mask"] is None),
            "overlapping_patterns": overlaps,
            "dma_kind_conflicts": dma_kind_conflicts,
        },
        "rows": rows,
    }
