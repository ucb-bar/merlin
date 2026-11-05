"""A MINIMAL C ISA header for a RoCC target, generated deterministically from RTL facts + contract.

A fresh compiler experiment must not find a vendor kernel library on its include path: an upstream
accelerator header can ship complete kernels next to its instruction macros, and a harness that
can ``#include`` it makes them callable by every candidate. So the
runner-owned build places only the bare-metal runtime (CRT, link script, printf/util basics) and this
generated header on the include path.

The header states hardware FACTS and nothing else:

* the RoCC custom opcode and every funct the elaborated decoder accepts, by the RTL source's own
  names (``interfaces.funct_decode_table``: header-only codes the decoder lacks are NOT emitted);
* array, memory and datapath geometry/dtypes from the fact bundle;
* the contract's reviewed encoding anchors (``encoding.addr_len``, ``encoding.rocc_custom_slot``) and
  the DMA payload bound (``memory_model.dma.max_transfer_bytes``);
* one ``.insn r`` macro per RoCC operand shape (standard RoCC xd/xs1/xs2 funct3 bits) and a fence.

No routine, loop, tiling or layout convention is generated. Output is a pure function of the
fact bytes and the contract's values, so its SHA-256 identifies it; :func:`materialize` writes it
under ``out/build/isa_headers/<target>/<sha16>/`` and never rewrites a different header in place.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GENERATOR = "merlin.targetgen.isa_header_gen.v1"

__all__ = ["GENERATOR", "IsaHeaderError", "materialize", "render"]


class IsaHeaderError(RuntimeError):
    """The facts or the contract cannot ground a header; nothing is guessed in its place."""


#: C fixed-width spelling of a fact-bundle dtype token.
_C_TYPES = {
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "f32": "float",
    "f64": "double",
}


def _ident(text: str) -> str:
    out = "".join(ch.upper() if ch.isalnum() else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    if not out or out[0].isdigit():
        raise IsaHeaderError(f"cannot form a C identifier from {text!r}")
    return out


def _decode_table(facts: Mapping[str, Any]) -> Mapping[str, Any]:
    for row in facts.get("interfaces") or ():
        if isinstance(row, Mapping) and row.get("name") == "funct_decode_table":
            return row
    raise IsaHeaderError("the RTL facts carry no interfaces.funct_decode_table")


def _int(value: Any, what: str) -> int:
    if type(value) is not int or value < 0:
        raise IsaHeaderError(f"{what} must be a non-negative integer fact, got {value!r}")
    return value


def _register_layouts(prefix: str, body: Mapping[str, Any]) -> list[str]:
    """Each command-register field's bit offset and width, from the RTL's own Bundle declarations
    (``interfaces.register_bundle_layouts``). A field whose width is a parameter the facts did not
    resolve gets its packed slot width, which is the width the field occupies in the register."""
    row = next(
        (
            r
            for r in body.get("interfaces") or ()
            if isinstance(r, Mapping) and r.get("name") == "register_bundle_layouts"
        ),
        None,
    )
    if row is None:
        return []
    lines = ["", "/* command-register field layouts (RTL Bundle declarations): bit offset, bit width */"]
    for bundle, layout in sorted((row.get("bundles") or {}).items()):
        for field, spec in (layout.get("fields") or {}).items():
            if not isinstance(spec, Mapping) or spec.get("offset") is None:
                continue
            width = spec.get("width") if spec.get("width") is not None else spec.get("slot_width")
            name = f"{prefix}_{_ident(bundle)}_{_ident(field)}"
            lines.append(f"#define {name}_OFFSET {_int(spec['offset'], f'{bundle}.{field}.offset')}")
            if width is not None:
                lines.append(f"#define {name}_WIDTH {_int(width, f'{bundle}.{field}.width')}")
    return lines


def _contract_encodings(prefix: str, contract: Mapping[str, Any]) -> list[str]:
    """Reviewed encoding facts the RTL cannot ground: the CONFIG selector codes and the local-address
    flag bits (``rocc_operand_roles.local_address_flags``, counted down from the address MSB)."""
    encoding = contract.get("encoding") or {}
    lines: list[str] = []
    subtypes = encoding.get("config_subtype") or {}
    if isinstance(subtypes, Mapping) and subtypes:
        lines += ["", "/* CONFIG selector codes (contract encoding.config_subtype) */"]
        for code, name in sorted(subtypes.items(), key=lambda kv: int(kv[0])):
            lines.append(f"#define {prefix}_{_ident(name)} {_int(int(code), 'config_subtype code')}")
    flags = ((contract.get("rocc_operand_roles") or {}).get("local_address_flags")) or {}
    addr_len = encoding.get("addr_len")
    if isinstance(flags, Mapping) and flags and addr_len is not None:
        top = _int(addr_len, "encoding.addr_len") - 1
        lines += ["", "/* local-address flag bits (contract rocc_operand_roles.local_address_flags) */"]
        for name, spec in flags.items():
            offset = _int((spec or {}).get("msb_offset"), f"local_address_flags.{name}.msb_offset")
            lines.append(f"#define {prefix}_LOCAL_ADDR_{_ident(name)}_BIT {top - offset}")
    return lines


def render(target: str, *, facts: Mapping[str, Any], contract: Mapping[str, Any], facts_sha256: str) -> str:
    """The header text. Raises :class:`IsaHeaderError` when a required fact is absent."""
    body = facts.get("facts", facts)
    prefix = _ident(target)
    table = _decode_table(body)
    opcode = _int(table.get("custom_opcode"), "funct_decode_table.custom_opcode")
    legal = table.get("legal_funct")
    names = table.get("names") or {}
    if not isinstance(legal, list) or not legal or not isinstance(names, Mapping):
        raise IsaHeaderError("funct_decode_table needs legal_funct and names")
    encoding = contract.get("encoding") or {}
    lines = [
        f"/* GENERATED by {GENERATOR} for target {target!r}. DO NOT EDIT.",
        f" * RTL facts sha256 {facts_sha256}; contract encoding anchors and DMA bound.",
        " * Hardware facts only: no kernel routine, tiling or layout convention is provided here. */",
        f"#ifndef MERLIN_{prefix}_ISA_H",
        f"#define MERLIN_{prefix}_ISA_H",
        "",
        "#include <stdint.h>",
        "",
        "/* RoCC encoding (elaborated decoder; the contract's reviewed anchors) */",
        f"#define {prefix}_ROCC_OPCODE {opcode}",
    ]
    if encoding.get("rocc_custom_slot") is not None:
        lines.append(
            f"#define {prefix}_ROCC_CUSTOM_SLOT {_int(encoding['rocc_custom_slot'], 'encoding.rocc_custom_slot')}"
        )
    if encoding.get("addr_len") is not None:
        lines.append(f"#define {prefix}_ADDR_LEN {_int(encoding['addr_len'], 'encoding.addr_len')}")
    complete = table.get("complete_isa")
    lines += [
        "",
        f"/* funct values the RTL decoder accepts (complete_isa={str(complete).lower()}); a code the decoder",
        " * does not accept is not emitted even when a software header names it. */",
    ]
    seen: set[str] = set()
    for code in sorted(_int(value, "legal_funct entry") for value in legal):
        name = names.get(str(code))
        if not isinstance(name, str) or not name:
            lines.append(f"#define {prefix}_FUNCT_{code} {code}")
            continue
        ident = _ident(name)
        if ident in seen:
            raise IsaHeaderError(f"funct name {name!r} maps to a duplicate identifier")
        seen.add(ident)
        lines.append(f"#define {prefix}_FUNCT_{ident} {code}")
    lines += ["", "/* geometry and storage (RTL facts) */"]
    for array in body.get("arrays") or ():
        if not isinstance(array, Mapping) or not array.get("name"):
            continue
        name = _ident(array["name"])
        for key in ("rows", "cols"):
            if array.get(key) is not None:
                lines.append(f"#define {prefix}_{name}_{key.upper()} {_int(array[key], f'arrays.{key}')}")
    for memory in body.get("memories") or ():
        if not isinstance(memory, Mapping) or not memory.get("name"):
            continue
        name = _ident(memory["name"])
        for key in ("banks", "depth", "row_elems", "lanes", "elem_bits", "bytes"):
            if memory.get(key) is not None:
                lines.append(f"#define {prefix}_{name}_{key.upper()} {_int(memory[key], f'memories.{key}')}")
    dma = ((contract.get("memory_model") or {}).get("dma") or {}).get("max_transfer_bytes")
    if dma is not None:
        lines.append(f"#define {prefix}_DMA_MAX_TRANSFER_BYTES {_int(dma, 'memory_model.dma.max_transfer_bytes')}")
    lines += _register_layouts(prefix, body)
    lines += _contract_encodings(prefix, contract)
    lines += ["", "/* datapath element types (RTL facts) */"]
    for datapath in body.get("datapaths") or ():
        if not isinstance(datapath, Mapping) or not datapath.get("name"):
            continue
        ctype = _C_TYPES.get(str(datapath.get("dtype")))
        if ctype is None:
            raise IsaHeaderError(f"datapath {datapath.get('name')!r} has no C type for {datapath.get('dtype')!r}")
        lines.append(f"typedef {ctype} {prefix.lower()}_{_ident(datapath['name']).lower()}_t;")
    lines += [
        "",
        "/* one RoCC instruction: .insn r opcode, funct3={xd,xs1,xs2}, funct7, rd, rs1, rs2 */",
        f"#define {prefix}_ROCC(funct, rs1, rs2) \\",
        f'  __asm__ volatile(".insn r %0, 3, %1, x0, %2, %3" : : "i"({prefix}_ROCC_OPCODE), "i"(funct), '
        '"r"((uint64_t)(rs1)), "r"((uint64_t)(rs2)) : "memory")',
        f"#define {prefix}_ROCC_RD(rd, funct, rs1, rs2) \\",
        f'  __asm__ volatile(".insn r %1, 7, %2, %0, %3, %4" : "=r"(rd) : "i"({prefix}_ROCC_OPCODE), "i"(funct), '
        '"r"((uint64_t)(rs1)), "r"((uint64_t)(rs2)) : "memory")',
        f'#define {prefix}_FENCE() __asm__ volatile("fence" : : : "memory")',
        "",
        f"#endif /* MERLIN_{prefix}_ISA_H */",
        "",
    ]
    return "\n".join(lines)


def materialize(target: str, name: str, *, facts_path: Path | None = None, contract: Mapping | None = None) -> Path:
    """Write the header for ``target`` (once per content) and return its path."""
    from merlin.common.paths import build_dir
    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl import facts as rtl_facts

    if not name or Path(name).name != name or not name.endswith(".h"):
        raise IsaHeaderError(f"generated header name must be a plain .h file name, got {name!r}")
    path = Path(facts_path) if facts_path is not None else rtl_facts.rtl_facts_path(target)
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise IsaHeaderError(f"{target}: RTL facts {path} cannot be read: {exc}") from exc
    declared = (document.get("facts") or {}).get("target") if isinstance(document, dict) else None
    if declared not in (None, target):
        raise IsaHeaderError(f"{path}: facts describe {declared!r}, not {target!r}")
    selected = dict(contract) if contract is not None else target_registry.resolve(target).load_contract()
    text = render(target, facts=document, contract=selected, facts_sha256=hashlib.sha256(raw).hexdigest())
    digest = hashlib.sha256(text.encode()).hexdigest()
    out = build_dir() / "isa_headers" / target / digest[:16] / name
    if out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == digest:
        return out
    if out.exists() or out.is_symlink():
        raise IsaHeaderError(f"{out} exists with different bytes; refusing to rewrite a content-addressed header")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(f".{digest[:8]}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(out)
    return out
