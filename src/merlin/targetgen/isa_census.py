"""Source-linked ISA census from a selected decoder, patterns, and model ISA.

This is an audit reader, not an instruction legality oracle. A BitPat and a
decoder-table row show a decode route; execution conditions, reserved fields,
semantics and selected-configuration admission need separate evidence.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import subprocess
from pathlib import Path
from typing import Any

_FIELDS = {"opcode": (0, 7), "funct3": (12, 3), "funct2": (13, 2), "funct7": (25, 7)}


def _git_source_at_revision(path: Path, revision: str) -> dict[str, str]:
    """Bind an observed source file to bytes in one exact local Git commit."""
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("source revision must be an exact 40-character commit")
    source = path.resolve(strict=True)

    def git(*arguments: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(source.parent), *arguments],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if result.returncode:
            raise ValueError(f"source revision cannot be verified for {source}: {result.stderr.decode(errors='replace').strip()}")
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
        if not name.isidentifier() or function.strip() != "BitPat" or not opening or not argument.startswith('"') or not argument.endswith('")'):
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


def _integer_keyword(node: ast.ClassDef, name: str) -> int | None:
    found = [keyword.value for keyword in node.keywords if keyword.arg == name]
    if not found:
        return None
    if len(found) != 1 or not isinstance(found[0], ast.Constant) or type(found[0].value) is not int:
        raise ValueError(f"model class {node.name} has a nonliteral or repeated {name}")
    value = found[0].value
    _, width = _FIELDS[name]
    if not 0 <= value < (1 << width):
        raise ValueError(f"model class {node.name} has out-of-range {name}")
    return value


def _model_classes(source: str) -> dict[str, dict[str, Any]]:
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
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

    for node in classes.values():
        opcode = _integer_keyword(node, "opcode")
        if opcode is None:
            continue
        base_names = tuple(base.id for base in node.bases if isinstance(base, ast.Name))
        mask = value = 0
        fixed: dict[str, int] = {}
        for field, (shift, width) in _FIELDS.items():
            number = _integer_keyword(node, field)
            if number is None:
                continue
            if field == "funct3" and "VIType" in base_names:
                # VI has six destination-register bits at [12:7], so its
                # three-bit mode occupies [15:13], unlike RV32I funct3.
                shift = 13
            fixed[field] = number
            mask |= ((1 << width) - 1) << shift
            value |= number << shift
        if node.name in rows:
            raise ValueError(f"duplicate model class {node.name}")
        rows[node.name] = {
            "name": node.name,
            "line": node.lineno,
            "fixed_fields": fixed,
            "mask": mask,
            "value": value,
            "format_bases": base_names,
            "exec_source": implementation(node) or "external_or_absent",
        }
    if not rows:
        raise ValueError("model ISA has no literal opcode classes")
    return rows


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
    model_revision: str | None = None,
    verify_revisions: bool = False,
) -> dict[str, Any]:
    """Cross-link selected source bytes; retain all unqualified obligations."""
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
            "status": "verified", "rtl_revision": rtl_revision,
            "model_revision": model_revision,
            "patterns": rtl_patterns, "decoder": rtl_decoder, "model_isa": model_source,
        }
    pattern_text, pattern_source = _source(pattern_file)
    decoder_text, decoder_source = _source(decoder_file)
    model_text, model_source = _source(model_isa_file)
    patterns = _patterns(pattern_text)
    decoded = _decode_rows(decoder_text)
    models = _model_classes(model_text)

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
        "sources": {"patterns": pattern_source, "decoder": decoder_source, "model_isa": model_source},
        "summary": {
            "patterns": len(patterns),
            "decoder_rows": len(decoded),
            "model_opcode_classes": len(models),
            "patterns_not_decoded": sorted(set(patterns) - set(decoded)),
            "decoder_rows_without_pattern": sorted(set(decoded) - set(patterns)),
            "model_classes_without_compatible_pattern": sorted(set(models) - matched_models),
            "overlapping_patterns": overlaps,
            "dma_kind_conflicts": dma_kind_conflicts,
        },
        "rows": rows,
    }
