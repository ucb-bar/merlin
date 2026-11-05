"""Private source/ELF-bound output layout and coherent terminal-dump decoding.

This module does not select a transport, run a simulator, compare a golden, or grade.
The public caller-layout inspector remains answer-free and is reused unchanged.
"""

from __future__ import annotations

import json
import stat
import struct
from itertools import product
from math import prod
from pathlib import Path

from .caller_layout import _canonical, _ordinary_member, _sha256, inspect_caller_layout

_COHERENT_OUTPUT_DTYPES = frozenset({"i8", "i16", "i32", "i64", "f32"})
_MAX_SIGNATURE_REGION_BYTES = 256 * 1024 * 1024


def _elf_output_symbols(
    raw: bytes, expected: dict[str, int], *, aliases: tuple[str, ...] = ()
) -> tuple[dict[str, int], dict[str, int]]:
    """Resolve unique, sized object symbols inside writable ELF64 load segments.

    Read the ELF's own symbol table rather than parsing compiler C or inferring a
    physical word width from the symbol size. A stripped or ambiguous image has
    no output-address proof.
    """
    if len(raw) < 64 or raw[:6] != b"\x7fELF\x02\x01" or raw[6] != 1:
        raise ValueError("coherent output requires an ELF64 little-endian image")
    header = struct.unpack_from("<HHIQQQIHHHHHH", raw, 16)
    if header[0] not in (2, 3):
        raise ValueError("coherent output requires a linked executable or shared ELF image")
    phoff, shoff, phentsize, phnum, shentsize, shnum = (
        header[4],
        header[5],
        header[8],
        header[9],
        header[10],
        header[11],
    )

    def section(offset: int, count: int, width: int, label: str) -> None:
        if not offset or not count or count > (len(raw) - offset) // width or offset > len(raw):
            raise ValueError(f"coherent output ELF has an incomplete {label} table")

    if phentsize != 56 or shentsize != 64:
        raise ValueError("coherent output ELF has unsupported table entry widths")
    section(phoff, phnum, phentsize, "program")
    section(shoff, shnum, shentsize, "section")
    load = []
    for index in range(phnum):
        kind, flags, fileoff, address, _, filesz, memsz, _ = struct.unpack_from("<IIQQQQQQ", raw, phoff + index * 56)
        if kind == 1:
            if filesz > memsz or fileoff > len(raw) or filesz > len(raw) - fileoff or address + memsz > 1 << 64:
                raise ValueError("coherent output ELF has an invalid load segment")
            if flags & 6 == 6 and memsz:
                load.append((address, address + memsz))
    sections = [struct.unpack_from("<IIQQQQIIQQ", raw, shoff + index * 64) for index in range(shnum)]
    symtabs = [row for row in sections if row[1] == 2]
    if len(symtabs) != 1:
        raise ValueError("coherent output ELF needs one unstripped symbol table")
    symtab = symtabs[0]
    symoff, symsize, strindex, syment = symtab[4], symtab[5], symtab[6], symtab[9]
    if syment != 24 or strindex >= shnum or sections[strindex][1] != 3:
        raise ValueError("coherent output ELF has malformed symbol metadata")
    stroff, strsize = sections[strindex][4:6]
    if (
        symoff > len(raw)
        or symsize > len(raw) - symoff
        or symsize % syment
        or stroff > len(raw)
        or strsize > len(raw) - stroff
    ):
        raise ValueError("coherent output ELF has truncated symbol metadata")
    names = {name.encode("ascii"): name for name in (*expected, *aliases)}
    found: dict[str, int] = {}
    found_aliases: dict[str, int] = {}
    for offset in range(symoff, symoff + symsize, syment):
        nameoff, info, _, section_index, address, size = struct.unpack_from("<IBBHQQ", raw, offset)
        if nameoff >= strsize:
            raise ValueError("coherent output ELF has a symbol outside its string table")
        end = raw.find(b"\x00", stroff + nameoff, stroff + strsize)
        if end < 0:
            raise ValueError("coherent output ELF has an unterminated symbol name")
        name = names.get(raw[stroff + nameoff : end])
        if name is None:
            continue
        if name in aliases:
            if name in found_aliases or info >> 4 != 1 or not 0 < section_index < shnum:
                raise ValueError(f"coherent output signature alias {name!r} is duplicate, non-global, or undefined")
            found_aliases[name] = address
            continue
        if name in found or info & 15 != 1 or not 0 < section_index < shnum or size != expected[name]:
            raise ValueError(f"coherent output symbol {name!r} is duplicate, untyped, or wrongly sized")
        owner = sections[section_index]
        if owner[2] & 3 != 3 or not owner[3] <= address < address + size <= owner[3] + owner[5]:
            raise ValueError(f"coherent output symbol {name!r} is not in writable allocated storage")
        if sum(start <= address < address + size <= stop for start, stop in load) != 1:
            raise ValueError(f"coherent output symbol {name!r} is not in one writable load segment")
        found[name] = address
    if set(found) != set(expected):
        raise ValueError("coherent output ELF omits a required sized symbol")
    if set(found_aliases) != set(aliases):
        raise ValueError("coherent output ELF omits a required signature alias")
    intervals = sorted((address, address + expected[name]) for name, address in found.items())
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("coherent output ELF aliases distinct output allocations")
    return found, found_aliases


def admit_coherent_output_layout(
    *,
    submission: Path,
    command_buffer_member: str,
    target: str,
    facts_path: Path,
    elf_path: Path,
    expected_elf_sha256: str,
) -> dict:
    """Bind selected caller output geometry to exact writable ELF allocations.

    This is an answer-free source/ELF admission primitive, not a transport
    selection, simulator execution, numerical grade, or certificate.
    """
    from merlin.llvmlower.c_runtime import DT_BYTES
    from merlin.perf.structural_transitions import StaticStridedLayout
    from merlin.runtime.fp8_formats import float_format_of

    receipt = inspect_caller_layout(
        submission=submission, command_buffer_member=command_buffer_member, target=target, facts_path=facts_path
    )
    cb_raw = _ordinary_member(Path(submission), command_buffer_member).read_bytes()
    if _sha256(cb_raw) != receipt["command_buffer_sha256"]:
        raise ValueError("caller command buffer changed after selected layout inspection")
    cb = json.loads(cb_raw)
    abi = cb["kernel_abi"]
    outputs = abi.get("outputs")
    accesses = {arg["tensor"]: arg["access"] for arg in abi["args"]}
    if (
        not isinstance(outputs, list)
        or not outputs
        or any(type(name) is not str or not name.isascii() or not name.isidentifier() for name in outputs)
        or len(set(outputs)) != len(outputs)
        or set(outputs) != {name for name, spec in cb["tensors"].items() if spec.get("role") == "output"}
        or any(accesses.get(name) not in ("write", "readwrite") for name in outputs)
    ):
        raise ValueError("coherent output requires a closed writable pointer roster")
    rows = {row["tensor"]: row for row in receipt["tensors"]}
    checked = []
    for name in outputs:
        row = rows[name]
        dtype, shape = row["dtype"], row["logical_shape"]
        if shape != cb["tensors"][name].get("shape") or dtype not in _COHERENT_OUTPUT_DTYPES:
            raise ValueError("coherent output logical dtype or shape differs from its declaration")
        width = DT_BYTES[dtype]
        floating = float_format_of(dtype) is not None
        if (dtype == "f32") != floating:
            raise ValueError("coherent output has no registered float-code decoder")
        if type(width) is not int or width not in (1, 2, 4, 8):
            raise ValueError("coherent output has no supported physical storage word")
        size = row["storage_elements"] * width
        StaticStridedLayout(
            tuple(shape), tuple(row["logical_strides_elements"]), size, row["offset_elements"]
        ).validate(dtype)
        if receipt["policy"]["mode"] == "legacy_aligned_row_major_v1":
            alignment = receipt["policy"]["row_alignment_elements"]
            logical_rows, logical_cols = prod(shape[:-1]), shape[-1]
            physical_rows = ((logical_rows + alignment - 1) // alignment) * alignment
            physical_cols = ((logical_cols + alignment - 1) // alignment) * alignment
            strides = [prod(shape[axis + 1 : -1]) * physical_cols for axis in range(len(shape) - 1)] + [1]
            if (
                row["physical_extents"] != [physical_rows, physical_cols]
                or row["logical_strides_elements"] != strides
                or row["storage_elements"] != physical_rows * physical_cols
                or row["offset_elements"] != 0
            ):
                raise ValueError("coherent output legacy address map differs from selected alignment")
        checked.append(
            {
                "tensor": name,
                "dtype": dtype,
                "logical_shape": list(shape),
                "physical_extents": list(row["physical_extents"]),
                "logical_strides_elements": list(row["logical_strides_elements"]),
                "offset_elements": row["offset_elements"],
                "storage_elements": row["storage_elements"],
                "physical_word_bytes": width,
                "physical_word_kind": "float_code" if floating else "signed_integer",
                "symbol": (row["symbol"] if receipt["schema"] == "public_caller_layout_receipt_v2" else f"T_{name}"),
                "bytes": size,
            }
        )
    elf_path = Path(elf_path)
    if elf_path.is_symlink() or not elf_path.is_file() or not stat.S_ISREG(elf_path.stat().st_mode):
        raise ValueError("coherent output requires an ordinary linked ELF")
    elf_raw = elf_path.read_bytes()
    if (
        type(expected_elf_sha256) is not str
        or len(expected_elf_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_elf_sha256)
        or _sha256(elf_raw) != expected_elf_sha256
    ):
        raise ValueError("coherent output ELF differs from selected linked bytes")
    symbols, _ = _elf_output_symbols(elf_raw, {row["symbol"]: row["bytes"] for row in checked})
    for row in checked:
        row["address"] = symbols[row["symbol"]]
        if row["address"] % row["physical_word_bytes"]:
            raise ValueError("coherent output ELF symbol is misaligned for its declared physical word")
    if _sha256(elf_path.read_bytes()) != expected_elf_sha256:
        raise ValueError("coherent output ELF changed during symbol inspection")
    after = inspect_caller_layout(
        submission=submission, command_buffer_member=command_buffer_member, target=target, facts_path=facts_path
    )
    if _canonical(after) != _canonical(receipt):
        raise ValueError("selected caller layout changed during ELF inspection")
    return {
        "schema": ("coherent_output_layout_admission_v2"
                   if receipt["schema"] == "public_caller_layout_receipt_v2"
                   else "coherent_output_layout_admission_v1"),
        "status": "layout_only",
        "scope": "selected layout and linked writable storage only; no execution or numerical grade",
        "command_buffer_sha256": receipt["command_buffer_sha256"],
        "rtl_facts_sha256": receipt["rtl_facts_sha256"],
        "provider_sha256": receipt["provider_sha256"],
        "core_layout_sha256": receipt["core_layout_sha256"],
        "policy_sha256": receipt["policy_sha256"],
        "caller_layout_sha256": _sha256(_canonical(receipt)),
        "layout_policy": receipt["policy"],
        "elf_sha256": expected_elf_sha256,
        "byte_order": "little",
        "outputs": checked,
    }


def _verified_layout(
    admission: dict,
    *,
    submission: Path,
    command_buffer_member: str,
    target: str,
    facts_path: Path,
    elf_path: Path,
) -> dict:
    if not isinstance(admission, dict) or admission.get("schema") not in {
        "coherent_output_layout_admission_v1", "coherent_output_layout_admission_v2",
    }:
        raise ValueError("output has no complete selected layout admission")
    current = admit_coherent_output_layout(
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
        expected_elf_sha256=admission.get("elf_sha256"),
    )
    try:
        unchanged = _canonical(current) == _canonical(admission)
    except (TypeError, ValueError):
        unchanged = False
    if not unchanged:
        raise ValueError("selected output layout admission changed")
    return current


def admit_fixed_address_output_layout(
    *,
    admission: dict,
    submission: Path,
    command_buffer_member: str,
    target: str,
    facts_path: Path,
    elf_path: Path,
) -> dict:
    """Preflight fixed-address ELF output storage before any native execution.

    Layout-only inspection accepts ET_DYN for diagnostic inspection. An actual
    memory readback cannot use those symbol addresses without a separately
    verified load bias, which this contract deliberately does not support.
    """
    current = _verified_layout(
        admission,
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
    )
    elf_path = Path(elf_path)
    if elf_path.is_symlink() or not elf_path.is_file():
        raise ValueError("fixed-address readback requires an ordinary ELF")
    raw = elf_path.read_bytes()
    if len(raw) < 18 or raw[16:18] != b"\x02\x00" or _sha256(raw) != current["elf_sha256"]:
        raise ValueError("fixed-address readback requires the exact linked ET_EXEC ELF")
    if _sha256(elf_path.read_bytes()) != current["elf_sha256"]:
        raise ValueError("fixed-address readback ELF changed during preflight")
    return {
        "schema": "fixed_address_output_layout_v1",
        "status": "prelaunch_only",
        "layout_admission_sha256": _sha256(_canonical(current)),
        "elf_sha256": current["elf_sha256"],
        "outputs": [
            {"symbol": row["symbol"], "address": row["address"], "bytes": row["bytes"]} for row in current["outputs"]
        ],
    }


def decode_coherent_output_dump(
    *,
    admission: dict,
    fixed_address_admission: dict,
    submission: Path,
    command_buffer_member: str,
    target: str,
    facts_path: Path,
    elf_path: Path,
    dump_path: Path,
) -> dict:
    """Read every logical output value from a complete, coherent terminal dump.

    Padding is present in the physical region but is not a logical output.
    The caller retains ownership of execution identity and golden comparison.
    """
    from merlin.common.digest import sha256_file
    from merlin.perf.whole_model_gsim import read_dump

    current = _verified_layout(
        admission,
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
    )
    fixed = admit_fixed_address_output_layout(
        admission=admission,
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
    )
    try:
        unchanged = _canonical(fixed) == _canonical(fixed_address_admission)
    except (TypeError, ValueError):
        unchanged = False
    if not unchanged:
        raise ValueError("coherent output fixed-address preflight changed before dump decoding")
    outputs = admission["outputs"]
    expected = sorted((row["address"], row["bytes"]) for row in outputs)
    dump_path = Path(dump_path)
    if dump_path.is_symlink() or not dump_path.is_file() or not stat.S_ISREG(dump_path.stat().st_mode):
        raise ValueError("coherent output dump is not an ordinary complete file")
    if dump_path.stat().st_size != 48 + 16 * len(expected) + sum(size for _, size in expected):
        raise ValueError("coherent output dump size differs from its exact output roster")
    dump_sha256 = sha256_file(dump_path)
    parsed = read_dump(dump_path, expected)
    if parsed["source"] != "coherent":
        raise ValueError("coherent output dump was not read through the coherent port")
    if (
        dump_path.stat().st_size != 48 + 16 * len(expected) + sum(size for _, size in expected)
        or sha256_file(dump_path) != dump_sha256
    ):
        raise ValueError("coherent output dump changed during decoding")
    return {
        "schema": "coherent_output_values_v1",
        "source": "coherent",
        "layout_admission_sha256": _sha256(_canonical(current)),
        "elf_sha256": current["elf_sha256"],
        "dump_sha256": dump_sha256,
        "outputs": _decode_physical_regions(outputs, parsed["regions"]),
    }


def _decode_physical_regions(outputs: list[dict], regions: dict[tuple[int, int], bytes]) -> dict[str, list]:
    """Decode every logical coordinate from exact, independently admitted physical bytes."""
    from merlin.runtime.backends.base import decode_float_readback

    result: dict[str, list[list]] = {}
    dtypes: dict[str, str] = {}
    for row in outputs:
        shape, strides = row["logical_shape"], row["logical_strides_elements"]
        width, dtype = row["physical_word_bytes"], row["dtype"]
        blob = regions[(row["address"], row["bytes"])]
        cols = shape[-1]
        flat = []
        for index in product(*(range(dim) for dim in shape)):
            offset = row["offset_elements"] + sum(i * s for i, s in zip(index, strides, strict=True))
            start = offset * width
            value = int.from_bytes(
                blob[start : start + width], "little", signed=row["physical_word_kind"] == "signed_integer"
            )
            flat.append(value)
        result[row["tensor"]] = [flat[index : index + cols] for index in range(0, len(flat), cols)]
        dtypes[row["tensor"]] = dtype
    return decode_float_readback(result, dtypes)


def admit_htif_signature_bounds(
    *,
    admission: dict,
    submission: Path,
    command_buffer_member: str,
    target: str,
    facts_path: Path,
    elf_path: Path,
) -> dict:
    """Preflight an exact, contiguous output-only ELF signature window.

    Stock HTIF can allocate memory from the begin/end alias range. This check must
    precede launch; decoding rechecks the same exact source and ELF bytes later.
    """
    current = _verified_layout(
        admission,
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
    )
    fixed = admit_fixed_address_output_layout(
        admission=admission,
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
    )
    outputs = current["outputs"]
    ordered = sorted(outputs, key=lambda row: row["address"])
    if any(left["address"] + left["bytes"] != right["address"] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("HTIF signature output symbols do not exactly tile one window")
    begin = ordered[0]["address"]
    end = ordered[-1]["address"] + ordered[-1]["bytes"]
    size = end - begin
    if size <= 0 or size > _MAX_SIGNATURE_REGION_BYTES:
        raise ValueError("HTIF signature output window exceeds its bounded physical extent")
    elf_path = Path(elf_path)
    elf_raw = elf_path.read_bytes()
    if _sha256(elf_raw) != current["elf_sha256"]:
        raise ValueError("HTIF signature ELF changed before alias inspection")
    _, aliases = _elf_output_symbols(
        elf_raw, {row["symbol"]: row["bytes"] for row in ordered}, aliases=("begin_signature", "end_signature")
    )
    if aliases != {"begin_signature": begin, "end_signature": end}:
        raise ValueError("HTIF signature aliases do not exactly bound the selected outputs")
    if _sha256(elf_path.read_bytes()) != current["elf_sha256"]:
        raise ValueError("HTIF signature ELF changed during alias inspection")
    bounds = {
        "schema": "htif_signature_bounds_v1" if len(outputs) == 1 else "htif_signature_bounds_v2",
        "status": "prelaunch_only",
        "fixed_address_admission_sha256": _sha256(_canonical(fixed)),
        "layout_admission_sha256": _sha256(_canonical(current)),
        "elf_sha256": current["elf_sha256"],
        "begin": begin,
        "end": end,
        "bytes": size,
        "signature_granularity_bytes": 1,
        "signature_file_bytes": 3 * size,
    }
    if len(outputs) == 1:
        bounds["symbol"] = ordered[0]["symbol"]
    else:
        bounds["outputs"] = [
            {"symbol": row["symbol"], "address": row["address"], "bytes": row["bytes"]} for row in ordered
        ]
    return bounds


def decode_htif_signature(
    *,
    admission: dict,
    bounds_admission: dict,
    submission: Path,
    command_buffer_member: str,
    target: str,
    facts_path: Path,
    elf_path: Path,
    signature_path: Path,
) -> dict:
    """Decode a complete output-only Spike signature, without judging exit or DONE.

    The caller must execute this exact preflight before launch, select the same
    ELF, and separately prove normal exit and a completed harness protocol.
    """
    from merlin.common.digest import sha256_file

    bounds = admit_htif_signature_bounds(
        admission=admission,
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
    )
    try:
        unchanged = _canonical(bounds) == _canonical(bounds_admission)
    except (TypeError, ValueError):
        unchanged = False
    if not unchanged:
        raise ValueError("HTIF signature prelaunch bounds changed before decoding")
    current = _verified_layout(
        admission,
        submission=submission,
        command_buffer_member=command_buffer_member,
        target=target,
        facts_path=facts_path,
        elf_path=elf_path,
    )
    signature_path = Path(signature_path)
    if (
        signature_path.is_symlink()
        or not signature_path.is_file()
        or not stat.S_ISREG(signature_path.stat().st_mode)
        or signature_path.stat().st_size != bounds["signature_file_bytes"]
    ):
        raise ValueError("HTIF signature is absent, indirect, or the wrong exact length")
    signature_sha256 = sha256_file(signature_path)
    raw = signature_path.read_bytes()
    if len(raw) != bounds["signature_file_bytes"]:
        raise ValueError("HTIF signature changed during bounded read")
    digits = b"0123456789abcdef"
    physical = bytearray(bounds["bytes"])
    for index in range(bounds["bytes"]):
        first, second, newline = raw[3 * index : 3 * index + 3]
        if first not in digits or second not in digits or newline != 10:
            raise ValueError("HTIF signature is not canonical lowercase byte-per-line hex")
        physical[index] = (digits.index(first) << 4) | digits.index(second)
    if (
        signature_path.stat().st_size != bounds["signature_file_bytes"]
        or sha256_file(signature_path) != signature_sha256
    ):
        raise ValueError("HTIF signature changed during decoding")
    regions = {
        (row["address"], row["bytes"]): bytes(
            physical[row["address"] - bounds["begin"] : row["address"] - bounds["begin"] + row["bytes"]]
        )
        for row in current["outputs"]
    }
    value_schema = (
        "htif_signature_output_values_v1" if len(current["outputs"]) == 1 else "htif_signature_output_values_v2"
    )
    return {
        "schema": value_schema,
        "source": "htif_signature",
        "bounds_admission_sha256": _sha256(_canonical(bounds)),
        "layout_admission_sha256": _sha256(_canonical(current)),
        "elf_sha256": current["elf_sha256"],
        "signature_sha256": signature_sha256,
        "outputs": _decode_physical_regions(current["outputs"], regions),
    }
