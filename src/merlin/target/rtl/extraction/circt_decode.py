"""Deterministic, target-agnostic CIRCT/FIRRTL structural fact extraction.

Prefer the version-matched ModelIR HW discovery for decoded opcodes, geometry,
memory and timing. Use declared FIRRTL for census and arithmetic-cell evidence;
legacy HW-port/ISA parsing is available only when its source is selected.
Instruction codes come from RTL. Their names are declared ISA vocabulary, not
additional hardware observations.

Record exact source-byte commitments and report missing evidence as unknown.
Source consistency requires a separate production/equivalence receipt; matching
configuration names alone cannot establish it. ``validate`` reports contradictions
against software declarations instead of reconciling them silently.

CLI: ``python -m merlin.target.rtl.extraction.circt_decode --target <t> --out facts.json``.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.common.target_data import interface as _facts_interface
from merlin.target.rtl.extraction import census as V1
from merlin.target.rtl.extraction import declared_readers as extraction_contract
from merlin.target.rtl.extraction.declared_readers import accumulator_layout as _accumulator_layout
from merlin.target.rtl.extraction.declared_readers import boolean_feature_gate as _boolean_feature_gate
from merlin.target.rtl.extraction.declared_readers import extract_boolean_feature_gate
from merlin.target.rtl.extraction.declared_readers import scala_funct_block as _scala_funct_block
from merlin.target.rtl.facts import rtl_cache_dir
from merlin.target.rtl.firrtl.ports import _module_port_sig, extract_accumulator, memories_from_port_geometry

# ---- circt_introspect --------------------------------------------------------------------------

_REPO = repo_root()


GENERATOR_VERSION = "rtl-introspect-v12-inherent-attribute-graph"


_RISCV_CUSTOM_OPCODES = {0: 0x0B, 1: 0x2B, 2: 0x5B, 3: 0x7B}  # derived-ok: RISC-V standard custom-0/1/2/3 encodings


_ROCC_FUNCT3_DEFAULT = 0x3  # derived-ok: RoCC xd/xs1/xs2 reg-usage field (standard, not identity/per-target)


def _contract_rocc_slot(target: str | None) -> int | None:
    """The RISC-V custom SLOT (0..3) this target's RoCC is wired to, read from its reviewed contract
    (``encoding.rocc_custom_slot``). Reads the raw contract yaml (NOT the manifest — loading the manifest
    would re-enter facts regeneration). None when the target declares no slot (a non-RoCC target, or one
    that has not declared it) -> the opcode fails closed rather than defaulting to custom-3."""
    if not target:
        return None
    try:
        import yaml  # function-local: off the hot path, avoids an import cycle

        from merlin.target.rtl.facts import target_contract_path

        p = target_contract_path(target)
        if not p.is_file():
            return None
        enc = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("encoding") or {}
        slot = enc.get("rocc_custom_slot")
        return int(slot) if slot is not None else None
    except Exception:  # noqa: BLE001 — a missing/broken contract is simply "no declared slot"
        return None


def _rocc_custom_opcode(target: str | None) -> int | None:
    """The RoCC MAJOR opcode for ``target`` = the RISC-V-standard encoding of the custom SLOT it uses.
    The slot is the per-target fact (from the reviewed contract encoding); slot->opcode is a RISC-V
    standard. Returns None (UNKNOWN — fail closed) when the slot is undeclared, never a baked default."""
    slot = _contract_rocc_slot(target)
    return _RISCV_CUSTOM_OPCODES.get(slot) if slot is not None else None


def _soc_hw_path(target: str) -> Path:
    """The per-target SoC HW-dialect cache (``firtool --ir-hw`` output) the accumulator port-parser
    reads, under the PURGEABLE rtl cache — never inside merlin/, never a baked target path. This is
    the SoC dialect the ``@AccumulatorMem`` port-parse needs (mlc's core dialect drives the decoder)."""
    return rtl_cache_dir(target) / f"{target}_soc.hw.mlir"


def isa_scala_path(target: str, chipyard_root: str | Path | None = None) -> Path:
    """The target's Chisel ISA source, by the chipyard generator convention
    ``generators/<t>/src/main/scala/<t>/<T>ISA.scala`` — DERIVED from the target name, not a hardcoded
    ISA file name (the first target resolves the identical file it always did). Returns the path whether
    or not it exists; callers gate on ``.is_file()``."""
    root = Path(chipyard_root) if chipyard_root is not None else Path(V1.default_chipyard())
    cap = target[:1].upper() + target[1:]
    return root / "generators" / target / "src" / "main" / "scala" / target / f"{cap}ISA.scala"


def extract_funct_table(isa_src: str, *, start_comment: str, stop_declaration: str) -> dict[str, Any]:
    """Read a target-declared Scala funct span, never infer its endpoints from one ISA.

    The target contract owns the comment beginning the funct block and the first declaration after
    it. Width-annotated bindings outside that span are still available to the weaker, ambiguity-aware
    naming recovery, but cannot become legal functs in the header fallback.
    """
    if not start_comment or not stop_declaration:
        raise ValueError("Scala funct block needs a start comment and stop declaration")
    lines = isa_src.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(start_comment)), None)
    if start is None:
        raise ValueError(f"Scala funct block start comment {start_comment!r} is absent")
    table: dict[int, str] = {}
    # Structural parse of ``val NAME = <n>.U`` (no bit-width annotation) — split the line, not a
    # regex. Reject ``<n>.U(<w>.W)`` (bit-width-annotated) by requiring only whitespace/comment after
    # ``.U``, matching the declared funct block (annotated values are excluded there).
    stopped = False
    for ln in lines[start + 1 :]:
        declaration = ln.split("//", 1)[0].strip()
        if declaration.startswith("val ") and declaration[4:].partition("=")[0].strip() == stop_declaration:
            stopped = True
            break
        s = ln.strip()
        if not s.startswith("val "):
            continue
        lhs, sep, rhs = s.partition("=")
        if not sep or ".U" not in rhs:
            continue
        name = lhs[4:].strip()
        num, after = rhs.strip().split(".U", 1)
        num = num.strip()
        if not (num.isdigit() and name and all(c.isupper() or c.isdigit() or c == "_" for c in name)):
            continue
        if after.strip() and not after.strip().startswith("//"):
            continue  # e.g. ``.U(3.W)`` -> not a funct code
        selector = int(num)
        if selector in table and table[selector] != name:
            raise ValueError("Scala funct block has conflicting selector/name bindings")
        if any(bound == name and code != selector for code, bound in table.items()):
            raise ValueError("Scala funct block assigns one name to different selectors")
        table[selector] = name
    if not stopped:
        raise ValueError(f"Scala funct block stop declaration {stop_declaration!r} is absent")
    legal = sorted(table)
    if not legal:
        raise ValueError("Scala funct block contains no unannotated integer funct bindings")
    return {
        "name": "funct_decode_table",
        "legal_funct": legal,
        "names": {str(k): table[k] for k in legal},
        "source_declared_names": {str(k): table[k] for k in legal},
        "source_declared_names_provenance": {
            "kind": "scala_funct_block",
            "scope": "selected_source_declaration",
            "source_sha256": hashlib.sha256(isa_src.encode("utf-8")).hexdigest(),
            "start_comment": start_comment,
            "stop_declaration": stop_declaration,
        },
        # Bindings OUTSIDE the funct block, for naming codes the SILICON decodes that the block never
        # mentions. Kept separate because they are not authoritative: see :func:`outside_block_names`.
        "outside_block_names": outside_block_names(isa_src, start_at=start),
        "evidence": f"Scala funct block {start_comment!r} to {stop_declaration!r}: "
        f"{len(legal)} codes" + (f" [{legal[0]}..{legal[-1]}]" if legal else ""),
    }


def _scala_int_expr(node: ast.AST, values: dict[str, int]) -> int:
    """Evaluate the small integer-expression subset used by Chisel bundle width declarations.

    This is deliberately not Python ``eval``: only integer literals, previously-derived constant names,
    and the four arithmetic operators needed by width declarations are accepted. Anything else fails
    closed and the containing bundle is omitted from the fact record.
    """
    if isinstance(node, ast.Expression):
        return _scala_int_expr(node.body, values)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
        lhs, rhs = _scala_int_expr(node.left, values), _scala_int_expr(node.right, values)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        return lhs // rhs
    raise ValueError(f"unsupported Scala integer-width expression {ast.dump(node)}")


def _scala_int_constants(text: str) -> dict[str, int]:
    """Resolve ``val UPPER_CASE = <integer expression>`` declarations structurally."""
    pending: dict[str, str] = {}
    for line in text.splitlines():
        source = line.split("//", 1)[0].strip()
        if not source.startswith("val "):
            continue
        lhs, sep, rhs = source.partition("=")
        name = lhs[4:].strip()
        if not sep or not name or not all(c.isupper() or c.isdigit() or c == "_" for c in name):
            continue
        pending[name] = rhs.strip()
    values: dict[str, int] = {}
    progress = True
    while pending and progress:
        progress = False
        for name, expr in list(pending.items()):
            try:
                value = _scala_int_expr(ast.parse(expr, mode="eval"), values)
            except (SyntaxError, ValueError, ZeroDivisionError):
                continue
            if value <= 0:
                continue
            values[name] = value
            del pending[name]
            progress = True
    return values


_Linear = tuple[int, dict[str, int]]


def _linear(constant: int = 0, **symbols: int) -> _Linear:
    return constant, {name: factor for name, factor in symbols.items() if factor}


def _linear_add(lhs: _Linear, rhs: _Linear, sign: int = 1) -> _Linear:
    symbols = dict(lhs[1])
    for name, factor in rhs[1].items():
        symbols[name] = symbols.get(name, 0) + sign * factor
    return lhs[0] + sign * rhs[0], {name: factor for name, factor in symbols.items() if factor}


def _scala_linear_expr(node: ast.AST, values: dict[str, int], symbols: frozenset[str]) -> _Linear:
    """A width expression as a linear form over ``symbols``; constants fold as in ``_scala_int_expr``.

    ``p.getWidth`` of a constructor parameter ``p`` is a symbol too: a Bundle that embeds a typed
    sub-field states that field's width this way, and the spacer before it subtracts the same term.
    """
    if isinstance(node, ast.Expression):
        return _scala_linear_expr(node.body, values, symbols)
    if isinstance(node, ast.Name) and node.id in symbols:
        return _linear(**{node.id: 1})
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "getWidth"
        and isinstance(node.value, ast.Name)
        and node.value.id in symbols
    ):
        return _linear(**{f"{node.value.id}.getWidth": 1})
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        lhs = _scala_linear_expr(node.left, values, symbols)
        rhs = _scala_linear_expr(node.right, values, symbols)
        return _linear_add(lhs, rhs, 1 if isinstance(node.op, ast.Add) else -1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        lhs = _scala_linear_expr(node.left, values, symbols)
        rhs = _scala_linear_expr(node.right, values, symbols)
        if lhs[1] and rhs[1]:
            raise ValueError("a product of two parameters is not a linear width")
        scalar, form = (lhs[0], rhs) if not lhs[1] else (rhs[0], lhs)
        return form[0] * scalar, {name: factor * scalar for name, factor in form[1].items()}
    return _linear(_scala_int_expr(node, values))


def _bundle_parameters(head: str) -> dict[str, int | None]:
    """``Name(a: Int, b: Int = 32, t: Some)`` -> every parameter, with its integer default if any."""
    if "(" not in head:
        return {}
    inner = head.split("(", 1)[1].rpartition(")")[0]
    parameters: dict[str, int | None] = {}
    for part in inner.split(","):
        name, _, rest = part.partition(":")
        name = name.strip().removeprefix("val ").strip()
        if not name:
            continue
        default = rest.partition("=")[2].strip()
        parameters[name] = int(default) if default.isdigit() else None
    return parameters


def _bundle_layouts(isa_src: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Every packed Bundle layout the source closes, and why each other Bundle was left out."""
    values = _scala_int_constants(isa_src)
    lines = isa_src.splitlines()
    out: dict[str, dict[str, Any]] = {}
    unresolved: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i].split("//", 1)[0].strip()
        if not line.startswith("class ") or "extends Bundle" not in line:
            i += 1
            continue
        head = line[6:].split("extends Bundle", 1)[0].strip()
        name = head.split("(", 1)[0].strip().split()[0]
        parameters = _bundle_parameters(head)
        symbols = frozenset(parameters)
        depth = line.count("{") - line.count("}")
        body: list[str] = []
        i += 1
        while i < len(lines) and depth > 0:
            current = lines[i].split("//", 1)[0].strip()
            depth += current.count("{") - current.count("}")
            if depth >= 0:
                body.append(current)
            i += 1
        fields: list[tuple[str, _Linear]] = []
        problem = None if body else "empty body"
        for current in body:
            if not current.startswith("val "):
                continue
            lhs, _, rhs = current.partition("=")
            field, rhs = lhs[4:].strip(), rhs.strip()
            try:
                if rhs.startswith("UInt(") and rhs.endswith(".W)"):
                    width = _scala_linear_expr(
                        ast.parse(rhs[len("UInt(") : -len(".W)")].strip(), mode="eval"), values, symbols
                    )
                elif rhs.endswith(".cloneType") and rhs[: -len(".cloneType")] in symbols:
                    width = _linear(**{f"{rhs[: -len('.cloneType')]}.getWidth": 1})
                else:
                    # A field this reader cannot size still occupies bits. Skipping it would shift
                    # every field above it and report the shifted offsets as derived.
                    raise ValueError(f"field {field!r} has no width this reader can derive")
            except (SyntaxError, ValueError, ZeroDivisionError) as error:
                problem = str(error)
                break
            if not width[1] and width[0] <= 0:
                problem = f"field {field!r} has non-positive width {width[0]}"
                break
            fields.append((field, width))
        if problem is None and not fields:
            problem = "no sized fields"
        if problem is not None:
            unresolved[name] = problem
            continue

        # A parametric field is placed when the bits BELOW it are a constant, and bounded when the
        # spacer above it cancels its parameter: `_spacer = UInt((W - p).W)` over `f = UInt(p.W)` is a
        # W-bit slot holding f at its low end, whatever p elaborates to.
        offset: _Linear = _linear()
        packed: dict[str, dict[str, Any]] = {}
        open_slots: list[tuple[str, int]] = []
        for field, width in reversed(fields):
            if offset[1]:
                if not field.startswith("_"):
                    problem = f"field {field!r} sits above bits whose width is not closed"
                    break
            elif not field.startswith("_"):
                if width[1]:
                    ((symbol, factor),) = width[1].items() if len(width[1]) == 1 else ((None, 0),)
                    if symbol is None or factor != 1 or width[0]:
                        problem = f"field {field!r} has a width that is not a single parameter"
                        break
                    record: dict[str, Any] = {"offset": offset[0], "width": None, "width_param": symbol}
                    default = parameters.get(symbol)
                    if default is not None:
                        record["default_width"] = default
                    packed[field] = record
                    open_slots.append((field, offset[0]))
                else:
                    packed[field] = {"offset": offset[0], "width": width[0]}
            offset = _linear_add(offset, width)
            if not offset[1]:
                for slot_field, slot_offset in open_slots:
                    packed[slot_field]["slot_width"] = offset[0] - slot_offset
                open_slots = []
        if problem is not None:
            unresolved[name] = problem
            continue
        # Only the TOP of the Bundle may stay open: every recorded field below it is already placed.
        out[name] = {"width": None if offset[1] else offset[0], "fields": packed}
        if any(record.get("width") is None for record in packed.values()):
            out[name]["parameters"] = sorted(parameters)
    return out, unresolved


def extract_register_bundle_layouts(isa_src: str) -> dict[str, dict[str, Any]]:
    """Derive packed register fields from Chisel ``Bundle`` declarations in an ISA source.

    Chisel packs a Bundle in reverse declaration order: the last field occupies the least-significant
    bits. Field widths resolve from the source's own ``val *_WIDTH`` declarations; spacer fields count
    toward offsets and are omitted from the consumer-facing map.

    A Bundle parameterised by its field widths is still placed. Its parametric field records
    ``width: None`` with the parameter that sets it and the ``slot_width`` that bounds it, so a
    consumer learns WHERE a register field lives and how wide it can be without this reader guessing
    an elaboration. Such Bundles used to be dropped whole, which removed exactly the fields that carry
    a design's numeric configuration. :func:`unresolved_register_bundles` names what is still left out.
    """
    return _bundle_layouts(isa_src)[0]


def unresolved_register_bundles(isa_src: str) -> dict[str, str]:
    """Bundle name -> why no layout was derived for it. Absence from the layouts is never silent."""
    return _bundle_layouts(isa_src)[1]


def outside_block_names(isa_src: str, start_at: int = 0) -> dict[str, list[str]]:
    """Every ``val NAME = <n>.U`` binding in the file, code -> the names bound to it.

    WHY THIS IS SEPARATE FROM THE FUNCT BLOCK. The funct block is authoritative and this is not. Two
    measured reasons, both from GemminiISA.scala itself:

    * The "cisc-gemmini opcodes" group binds 10..17 and the file SAYS SO in a comment — ``// TODO the
      numbers here overlap with the LOOP_WS commands``, ``// same as COMPUTE_AND_FLIP``. Naming a
      funct from a whole-file scan would rename LOOP_WS_CONFIG_ADDRS_AB to ADDR_AB.
    * The rs1-subfield group rebinds small numbers (``CONFIG_EX = 0.U``) that collide with real
      functs (``CONFIG_CMD = 0.U``).

    So this returns the FULL list of names per code, never one name, and the caller may only use a
    code whose list has exactly one entry. An ambiguous code stays unnamed — which is the honest
    result, because guessing between two bindings is how a decode table starts lying.

    Bit-width annotation is NOT a rejection criterion here. The funct block excludes ``<n>.U(<w>.W)``
    because its rs1-subfields are annotated; but ``COUNTER_OP = 126.U(7.W)`` is a real funct the
    silicon decodes, written with a width. A pattern narrow enough to exclude the subfields also
    excluded a live instruction, which is exactly the too-narrow-pattern failure the repo forbids.
    """
    out: dict[str, list[str]] = {}
    for ln in isa_src.splitlines()[start_at:]:
        s = ln.strip()
        if not s.startswith("val "):
            continue
        lhs, sep, rhs = s.partition("=")
        if not sep or ".U" not in rhs:
            continue
        name = lhs[4:].strip()
        num, _after = rhs.strip().split(".U", 1)
        num = num.strip()
        if not (num.isdigit() and name and all(c.isupper() or c.isdigit() or c == "_" for c in name)):
            continue
        names = out.setdefault(num, [])
        if name not in names:
            names.append(name)
    return out


def extract_funct_table_via_decoder(target: str) -> dict[str, Any] | None:
    """Observe one HW-dialect decoder comparison field via mlc's equality fan-out.

    The legacy table shape records values and provenance, but is not a complete
    ISA or endpoint proof. Returns None when that observation is unavailable.
    """
    try:
        from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

        if not mlc_bridge.mlc_available()[0]:
            return None
        # Agnostic extraction: mlc resolves the target's version-matched core HW dialect (the passed
        # hw_path is only a legacy fallback for the accumulator/mesh path). ``target`` is a parameter.
        res = mlc_bridge.discover_legal_opcodes(target)
    except Exception:  # noqa: BLE001 — a parse/version skew means "no decoder facts", fall back cleanly
        return None
    legal = res.get("legal_opcodes")
    if not legal:
        return None
    return {
        "name": "funct_decode_table",
        "legal_funct": legal,
        "width": res.get("width"),
        "scope": res.get("scope", "observed_decode_field"),
        "complete_isa": res.get("complete_isa") is True,
        "names": {},  # the decoder yields numeric codes; names are cross-referenced from the header below
        "hw_source": res.get("hw_source"),
        "module": res.get("module"),
        "fanout": res.get("fanout"),
        "method": res.get("method", "decoder_icmp_fanout(mlc)"),
        "evidence": res.get("evidence", "mlc decoder comb.icmp-eq fan-out"),
    }


def _reconcile_funct(decoder: dict[str, Any] | None, header: dict[str, Any] | None) -> dict[str, Any] | None:
    """Preserve observed decoder values and compare them with header declarations.

    The two sets can differ without proving either complete; names remain useful
    diagnostics, not an instruction-legality certificate.
    """
    if decoder is None:
        if header is not None:
            header.setdefault("method", "scala_header_parse")
        return header
    if header is not None:
        hs, ds = set(header.get("legal_funct", [])), set(decoder["legal_funct"])
        hnames = header.get("names", {})
        for selector, observed_name in (decoder.get("names") or {}).items():
            declared_name = hnames.get(str(selector))
            if declared_name is not None and observed_name != declared_name:
                raise ValueError("decoder and source have conflicting selector/name bindings")
        if "source_declared_names" in header:
            decoder["source_declared_names"] = dict(header["source_declared_names"])
            decoder["source_declared_names_provenance"] = dict(header["source_declared_names_provenance"])
        decoder["names"] = {str(k): hnames.get(str(k), "?") for k in decoder["legal_funct"]}
        # A code the SILICON decodes that the funct block never named reads as "?" -- and an unnamed
        # instruction cannot be given a role, so it lands in `claimed_no_role` forever. Measured on
        # gemmini: funct 126 is decoded by the silicon, is bound exactly once in the ISA source as
        # COUNTER_OP, and sits after the block's CONFIG_EX stop, so the block never reaches it.
        # Name it from the unambiguous binding; leave it "?" when the file binds the code more than
        # once, and RECORD which codes were named this way so the weaker provenance stays visible.
        outside = header.get("outside_block_names") or {}
        recovered: dict[str, str] = {}
        for code in decoder["legal_funct"]:
            if decoder["names"].get(str(code)) != "?":
                continue
            cands = outside.get(str(code)) or []
            if len(cands) == 1:
                decoder["names"][str(code)] = cands[0]
                recovered[str(code)] = cands[0]
        if recovered:
            decoder["names_recovered_from_outside_block"] = recovered
        decoder["header_only_functs"] = sorted(hs - ds)  # declared but not observed in this comparison family
        decoder["decoder_only_functs"] = sorted(ds - hs)  # observed but not declared in the selected header span
        decoder["evidence"] += f"; vs header: header_only={sorted(hs - ds)} observed_only={sorted(ds - hs)}"
    else:
        # No name vocabulary at all (target declares no ISA source) -> generic ``funct_<code>`` labels.
        # The CODES are the derived fact; the names are a convenience the target can supply later.
        decoder["names"] = {str(k): f"funct_{k}" for k in decoder["legal_funct"]}
    return decoder


def _functs_from_headers(headers: Iterable[str | Path]) -> dict[int, str]:
    """Map funct code -> name from a target's C ISA headers by the generic ``#define k_<NAME> <N>``
    convention — a structured split (no regex), mirroring
    :func:`merlin.development.targetgen.oot_starterkit.verify.legal_functs`. Earlier headers win a code."""
    out: dict[int, str] = {}
    for h in headers:
        try:
            txt = Path(h).read_text(errors="ignore")
        except OSError:
            continue
        for line in txt.splitlines():
            parts = line.split()  # ``#define  k_<NAME>  <N>`` — any whitespace
            if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("k_") and parts[2].isdigit():
                name = parts[1][2:]
                if name and all(c.isupper() or c.isdigit() or c == "_" for c in name):
                    out.setdefault(int(parts[2]), name)
    return out


def _defines_from_headers(headers: Iterable[str | Path], prefix: str) -> dict[str, int]:
    """Map NAME -> value from a target's C ISA headers by the generic ``#define <prefix><NAME> <N>``
    convention (structured split, no regex). Used for the ``k_`` funct defines and the un-prefixed
    ``CONFIG_*`` subtype defines. Earlier headers win a name."""
    out: dict[str, int] = {}
    for h in headers:
        try:
            txt = Path(h).read_text(errors="ignore")
        except OSError:
            continue
        for line in txt.splitlines():
            parts = line.split()  # ``#define  <prefix><NAME>  <N>`` — any whitespace
            if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith(prefix) and parts[2].isdigit():
                name = parts[1][len(prefix) :]
                if name and all(c.isupper() or c.isdigit() or c == "_" for c in name):
                    out.setdefault(name, int(parts[2]))
    return out


def crosscheck_semantic_class_names(target: str, semantic_class: dict | None = None) -> dict[str, list]:
    """Prove the manifest ``semantic_class`` NAMES are HEADER-DERIVED: for each ``code -> name`` the manifest
    declares, compare the name to the target's ISA-header ``#define k_<NAME> <code>`` at the SAME code.
    Returns ``{exact, alias, missing, mismatch}``:

      * ``exact``    — code present in the headers with the identical name (the derivation is byte-exact);
      * ``alias``    — the manifest name is a PREFIX of the header name (a compiler alias that drops a
                       dataflow suffix, e.g. ``LOOP_CONV`` for the header's ``LOOP_CONV_WS``) — reviewed, benign;
      * ``missing``  — code the headers do not define (an ungrounded manifest name — a real alarm);
      * ``mismatch`` — code present but the name neither matches nor prefixes the header (a real alarm).

    The gate is ``missing == mismatch == []`` (every declared class is a real header funct, named by the
    header modulo a reviewed prefix-alias). Skips entirely (empty lists) when the target declares no ISA
    headers. No mlc needed — a pure header parse."""
    from merlin.target.registry.experiment_descriptor import load_capability_manifest

    if semantic_class is None:
        semantic_class = (load_capability_manifest(target).encoding or {}).get("semantic_class") or {}
    hdr_by_code = _functs_from_headers(_declared_isa_headers(target))  # already {code: name}
    out: dict[str, list] = {"exact": [], "alias": [], "missing": [], "mismatch": []}
    if not hdr_by_code:
        return out
    for code, name in semantic_class.items():
        code, name = int(code), str(name)
        hname = hdr_by_code.get(code)
        if hname is None:
            out["missing"].append((code, name))
        elif hname == name:
            out["exact"].append((code, name))
        elif hname.startswith(name):
            out["alias"].append((code, name, hname))
        else:
            out["mismatch"].append((code, name, hname))
    return out


def crosscheck_config_subtype_names(target: str, config_subtype: dict | None = None) -> dict[str, list]:
    """Prove the manifest ``config_subtype`` NAMES are HEADER-DERIVED: compare each ``code -> name`` to the
    ISA-header ``#define <NAME> <code>`` (the un-prefixed ``CONFIG_EX``/``CONFIG_LD``/``CONFIG_ST`` defines).
    Returns ``{exact, missing, mismatch}``; the gate is ``missing == mismatch == []``. Pure header parse."""
    from merlin.target.registry.experiment_descriptor import load_capability_manifest

    if config_subtype is None:
        config_subtype = (load_capability_manifest(target).encoding or {}).get("config_subtype") or {}
    hdr = _defines_from_headers(_declared_isa_headers(target), "")  # un-prefixed ``#define NAME N``
    out: dict[str, list] = {"exact": [], "missing": [], "mismatch": []}
    if not hdr:
        return out
    for code, name in config_subtype.items():
        code, name = int(code), str(name)
        hcode = hdr.get(name)
        if hcode is None:
            out["missing"].append((code, name))
        elif hcode == code:
            out["exact"].append((code, name))
        else:
            out["mismatch"].append((code, name, hcode))
    return out


def _declared_isa_headers(target: str) -> list[Path]:
    """The ISA header files the target's ``target_experiment.yaml`` declares (bundle-convention,
    repo-root-relative strings), resolved to absolute paths. Empty when the target has no descriptor
    or declares no headers (e.g. arc/cyclotron targets) — the caller then falls back to generic names.

    The descriptor is resolved through the corpus locator, the one module that owns the
    ``experiments/`` layout. This used to glob ``experiments/studies/*/target_experiment.yaml``, which
    the target-neutral layout (``capsule_bench/targets/<target>/``) stopped matching, so every target
    silently reported "declares no headers" and fell through to generic ``funct_<code>`` names."""
    from merlin.common.paths import resolve_grant
    from merlin.target.registry import descriptor_loading as corpora

    te = corpora.experiment_for(target)
    if te is None:
        return []
    # Preserve logical descriptor identities within the selected source owner.
    out: list[Path] = []
    for h in te.isa_headers:
        p = resolve_grant(str(h), root=_REPO)
        if p.is_file():
            out.append(p)
    return out


def _funct_name_table(target: str, isa_path: Path | None) -> dict[str, Any] | None:
    """The funct-name vocabulary for ``target``, sourced agnostically: the Chisel ISA source
    (``<T>ISA.scala``) when the target declares its funct span, else its DECLARED ISA headers
    (``#define k_*``), else None (codes-only, generic ``funct_<code>`` names downstream). Returns
    a funct-table dict (``legal_funct``/``names``), whose names the decoder-derived set borrows."""
    if isa_path is not None and Path(isa_path).is_file():
        block = _scala_funct_block(target)
        if block is not None:
            table = extract_funct_table(
                Path(isa_path).read_bytes().decode("utf-8"),
                start_comment=block[0],
                stop_declaration=block[1],
            )
            table["source_declared_names_provenance"]["source_path"] = str(Path(isa_path).absolute())
            return table
    headers = _declared_isa_headers(target)
    code_name = _functs_from_headers(headers)
    if not code_name:
        return None
    legal = sorted(code_name)
    return {
        "name": "funct_decode_table",
        "legal_funct": legal,
        "names": {str(k): code_name[k] for k in legal},
        "evidence": f"ISA headers {[Path(h).name for h in headers]} '#define k_* <N>': "
        f"{len(legal)} names [{legal[0]}..{legal[-1]}]",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "missing"


def _sha256(path: Path) -> str:
    """Full content digest for a consumer which must bind an observation to CIRCT bytes.

    ``_sha`` predates evidence-carrying performance receipts and remains a short, human-facing
    cache identity.  It is not a safe equality witness: a receipt must never accept unrelated
    elaborated RTL merely because it shares a 64-bit prefix.  Keep the old field for compatibility
    and publish a separately named full digest for fail-closed consumers.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def _core_hw_input(target: str) -> dict[str, str]:
    """The CORE HW dialect that mlc discovery and the pipeline-depth walk read, as provenance fields.

    Distinct from the SoC dialect ``_soc_hw_path`` resolves: that one feeds the legacy accumulator
    port-parse and is absent for most targets. Target-agnostic -- the path comes from mlc, never from a
    per-target literal. Three outcomes are kept apart rather than collapsed onto one sentinel, because
    "mlc could not resolve a dialect" and "the dialect is resolved but not on disk" are different facts
    and only the second is fixable by rebuilding."""
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    try:
        path = mlc_bridge.core_hw_mlir(target)
    except Exception:
        path = None
    if path is None:
        return {"core_hw_mlir": "unresolved", "core_hw_sha": "unresolved", "core_hw_sha256": "unresolved"}
    core = Path(path)
    return {
        "core_hw_mlir": core.name,
        "core_hw_path": str(core.resolve()),
        "core_hw_sha": _sha(core),
        "core_hw_sha256": _sha256(core),
    }


def _facts_from_discovery(target: str, facts: dict) -> list[str]:
    """Override mesh + operand-scratchpad + accumulator facts with mlc RTL discovery (target-agnostic:
    DIM from the discovered mesh, capacities summed from the discovered banks). Mutates ``facts`` in
    place; returns the list of fact names that came from discovery (for provenance)."""
    from merlin.target.rtl.extraction import mlc_discovery as _merlin_mlc_discovery_1
    from merlin.target.rtl.extraction import mlc_roles as _merlin_mlc_roles_1
    from merlin.target.rtl.extraction import mlc_toolchain as mlc_bridge

    sourced: list[str] = []
    dim = _merlin_mlc_roles_1.discovered_dim(target)
    if dim:
        # Carry the OBSERVATION, not just the conclusion. "a 17x17 mesh" is unfalsifiable by anyone
        # downstream; "289 instances of a D-flip-flop inside a divide/sqrt unit" is refuted on sight,
        # and that is the difference between a fact and a ranking. `mesh_corroborated` is what the
        # engine audit gates on -- an uncorroborated geometry evidences nothing.
        mesh = _merlin_mlc_roles_1.discovered_mesh(target) or {}
        rec = {"name": "mesh", "rows": dim, "cols": dim, "source": "mlc_discovery"}
        if mesh:
            rec.update(
                {
                    "container": mesh.get("parent"),  # the module holding the grid
                    "element": mesh.get("child"),  # the replicated cell
                    "instances": mesh.get("count"),  # summed across CIRCT structural variants
                    "element_variants": mesh.get("elements") or [],
                    "mac_idiom": {k: mesh.get(k) for k in ("muls", "adds", "regs") if mesh.get(k)},
                    "corroborated": bool(mesh.get("corroborated")),
                }
            )
        facts["arrays"] = [a for a in facts.get("arrays", []) if a.get("name") != "mesh"]
        facts["arrays"].append(rec)
        sourced.append("mesh")
    caps = _merlin_mlc_discovery_1.discovered_capacities(target)
    if caps:
        keep = [m for m in facts.get("memories", []) if m.get("name") not in ("scratchpad", "accumulator")]
        if caps.get("operand_bytes"):
            keep.append(
                {
                    "name": "scratchpad",
                    "bytes": caps["operand_bytes"],
                    "depth": caps.get("operand_depth"),
                    "source": "mlc_discovery",
                }
            )
            sourced.append("scratchpad")
        if caps.get("accumulator_bytes"):
            keep.append(
                {
                    "name": "accumulator",
                    "bytes": caps["accumulator_bytes"],
                    "depth": caps.get("accumulator_depth"),
                    "source": "mlc_discovery",
                }
            )
            sourced.append("accumulator")
        facts["memories"] = keep
    return sourced


_FIRRTL_SUBDIR = "firrtl"
