"""The LLVM data layout of a cross target, read from the compiler that compiles for it.

MLIR translates a module to LLVM IR under the module's ``llvm.data_layout``, or under LLVM's default
layout when it carries none -- and the default aligns 64-bit integers and doubles to 4 bytes. Every
load and store the translation emits then states ``align 4``, which the cross compiler keeps even after
it substitutes its own layout; on a vector host an under-aligned 64-bit access has no legal vector form,
so a loop over int64 or f64 data is not vectorized at all. The layout is a property of the target
(triple and ISA flags), so it is ASKED of the compiler with the build's own cross flags -- never written
here. Nothing here names a target.
"""

from __future__ import annotations

import hashlib
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_file
from merlin.common.text_tokens import source_tokens

_PREFIX = 'target datalayout = "'
_KNOWN: dict[tuple[str, ...], str] = {}
_LOCK = threading.Lock()


def parse(llvm_ir: str) -> str | None:
    """The layout string of a ``target datalayout = "..."`` line in ``llvm_ir``, or ``None``."""
    for line in llvm_ir.splitlines():
        stripped = line.strip()
        if stripped.startswith(_PREFIX) and stripped.endswith('"'):
            return stripped[len(_PREFIX) : -1]
    return None


def default_index_bits(layout: str) -> int:
    """The default pointer's index width when it agrees with its representation width.

    LLVM spells a default pointer as ``p:size:abi[:preferred[:index]]`` or
    ``p0:...``. Omitted index width equals size. MLIR index lowering also
    determines memref descriptor field widths, so a differing GEP index width
    cannot alone authorize this producer's C-facing ABI. An absent default
    pointer declaration has no checked width, even if LLVM has a default.
    """
    selected = []
    for field in layout.split("-"):
        parts = field.split(":")
        if parts[0] not in {"p", "p0"}:
            continue
        if len(parts) not in {3, 4, 5} or any(not part.isdecimal() for part in parts[1:]):
            raise ValueError("compiler data layout has malformed default pointer fields")
        size, abi = int(parts[1]), int(parts[2])
        preferred = int(parts[3]) if len(parts) >= 4 else abi
        index = int(parts[4]) if len(parts) == 5 else size
        if min(size, abi, preferred, index) <= 0 or index > size:
            raise ValueError("compiler data layout has invalid default pointer/index widths")
        if index != size:
            raise ValueError("compiler pointer and GEP index widths differ; memref ABI is unverified")
        selected.append(index)
    if len(selected) != 1:
        raise ValueError("compiler data layout has no unique declared default pointer index width")
    return selected[0]


def selected_index_bits(observed: object) -> int | None:
    """Check a versioned compiler observation; consumers still bind its provenance.

    V1 requires an explicit default pointer field. V2 additionally retains actual
    compiler/LLVM probe output for an implicit default; neither supplies source,
    numerical or linked-build permission.
    """
    fields = {
        "schema",
        "compiler_requested",
        "compiler_resolved",
        "compiler_sha256",
        "cross_flags",
        "data_layout",
        "index_bits",
        "scope",
    }
    if not isinstance(observed, Mapping):
        return None
    version = observed.get("schema")
    if version == "merlin.selected-index-lowering.v2":
        fields.add("pointer_observation")
    elif version != "merlin.selected-index-lowering.v1":
        return None
    if set(observed) != fields or (
        type(observed.get("index_bits")) is not int
        or not all(
            type(observed.get(key)) is str and bool(observed[key])
            for key in ("compiler_requested", "compiler_resolved", "data_layout", "scope")
        )
        or not is_sha256(observed.get("compiler_sha256"))
        or type(observed.get("cross_flags")) is not list
        or not observed["cross_flags"]
        or any(type(flag) is not str or not flag for flag in observed["cross_flags"])
    ):
        return None
    try:
        if version == "merlin.selected-index-lowering.v1":
            bits = default_index_bits(observed["data_layout"])
        else:
            if _has_default_pointer(observed["data_layout"]):
                return None
            bits = _implicit_bits(
                observed["data_layout"],
                observed["pointer_observation"],
                observed["compiler_resolved"],
                observed["cross_flags"],
            )
        return bits if bits == observed["index_bits"] else None
    except (ValueError, TypeError):
        return None


_POINTER_SOURCE = "const unsigned merlin_pointer_representation_bits = sizeof(void *) * __CHAR_BIT__;\n"


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _has_default_pointer(layout):
    return any(field.split(":")[0] in {"p", "p0"} for field in layout.split("-"))


def _llvm_rows(ir):
    """Tokenize bounded LLVM records, respecting quoted strings and ';' comments."""
    rows = []
    for line in ir.splitlines():
        quoted, position = False, 0
        while position < len(line):
            char = line[position]
            if char == "\\" and quoted:
                position += 2
                continue
            if char == '"':
                quoted = not quoted
            elif char == ";" and not quoted:
                line = line[:position]
                break
            position += 1
        tokens = [token.text for token in source_tokens(line)]
        if tokens:
            rows.append(tokens)
    return rows


def _representation_bits(ir, layout):
    rows = [row for row in _llvm_rows(ir) if row[:2] == ["@", "merlin_pointer_representation_bits"]]
    if parse(ir) != layout or len(rows) != 1 or rows[0][:3] != ["@", "merlin_pointer_representation_bits", "="]:
        raise ValueError("compiler did not report one pointer representation under its selected layout")
    row = rows[0][3:]
    prefixes = {
        "dso_local",
        "dso_preemptable",
        "local_unnamed_addr",
        "unnamed_addr",
        "hidden",
        "protected",
        "internal",
        "private",
        "weak",
        "weak_odr",
        "linkonce",
        "linkonce_odr",
    }
    while row and row[0] in prefixes:
        row = row[1:]
    if (
        len(row) < 3
        or row[0] != "constant"
        or not row[1].startswith("i")
        or not row[1][1:].isdecimal()
        or not row[2].isdecimal()
    ):
        raise ValueError("compiler pointer representation is not one constant integer definition")
    bits, suffix = int(row[2]), row[3:]
    while suffix:
        if suffix[0] != ",":
            raise ValueError("compiler pointer representation has unsupported declaration tokens")
        following = suffix[1:]
        end = following.index(",") if "," in following else len(following)
        group, suffix = following[:end], following[end:]
        alignment = len(group) == 2 and group[0] == "align" and group[1].isdecimal() and int(group[1]) > 0
        metadata = len(group) == 4 and group[0] == group[2] == "!" and group[3].isdecimal()
        if not (alignment or metadata):
            raise ValueError("compiler pointer representation has unsupported declaration suffix")
    if not 0 < bits <= 128:
        raise ValueError("compiler pointer representation exceeds the bounded observer")
    return bits


def _gep_source(layout, bits):
    # No inbounds/nuw/nsw promise: the offset is allowed to wrap. LLVM truncates
    # it to the GEP index width. The high representation bit survives exactly
    # when that width equals the pointer representation (LangRef Data Layout).
    return (
        f'target datalayout = "{layout}"\n'
        f"define i{bits + 1} @merlin_pointer_index() {{\n"
        f"  %p = getelementptr i8, ptr null, i{bits + 1} {1 << (bits - 1)}\n"
        f"  %n = ptrtoint ptr %p to i{bits + 1}\n"
        f"  ret i{bits + 1} %n\n}}\n"
    )


def _check_gep(ir, layout, bits):
    if parse(ir) != layout:
        raise ValueError("implicit pointer LLVM probe changed its selected layout")
    rows = _llvm_rows(ir)
    starts = [index for index, row in enumerate(rows) if row[0] == "define" and "merlin_pointer_index" in row]
    if len(starts) != 1:
        raise ValueError("implicit pointer LLVM probe has no unique function definition")
    start = starts[0]
    signature = rows[start][1:]
    if signature and signature[0] == "noundef":
        signature = signature[1:]
    if signature[:5] != [f"i{bits + 1}", "@", "merlin_pointer_index", "(", ")"]:
        raise ValueError("implicit pointer LLVM probe changed its complete function signature")
    suffix = signature[5:]
    while suffix and suffix[0] in {"local_unnamed_addr", "unnamed_addr"}:
        suffix = suffix[1:]
    if len(suffix) >= 2 and suffix[0] == "#" and suffix[1].isdecimal():
        suffix = suffix[2:]
    if suffix != ["{"]:
        raise ValueError("implicit pointer LLVM probe has unsupported function qualifiers")
    body, position = [], start + 1
    while position < len(rows) and rows[position] != ["}"]:
        body.append(rows[position])
        position += 1
    if position == len(rows):
        raise ValueError("implicit pointer LLVM probe has an unterminated body")
    if body and len(body[0]) == 2 and body[0][-1] == ":":
        body = body[1:]
    if body != [["ret", f"i{bits + 1}", str(1 << (bits - 1))]]:
        raise ValueError(
            "compiler pointer and LLVM GEP index widths differ or remain unobserved; memref ABI is unverified"
        )
    return bits


def _probe_argv(clang, flags, language):
    return [str(clang), *flags, "-O1", "-S", "-emit-llvm", "-x", language, "-", "-o", "-"]


def _implicit_bits(layout, evidence, compiler, flags):
    fields = {
        "schema",
        "representation_source",
        "representation_ir",
        "representation_ir_sha256",
        "gep_source",
        "gep_source_sha256",
        "gep_ir",
        "gep_ir_sha256",
        "representation_argv",
        "gep_argv",
        "observer_source",
        "observer_sha256",
    }
    strings = fields - {"representation_argv", "gep_argv"}
    if (
        type(evidence) is not dict
        or set(evidence) != fields
        or evidence.get("schema") != "merlin.implicit-pointer-observation.v1"
        or any(type(evidence[key]) is not str for key in strings)
        or evidence["representation_source"] != _POINTER_SOURCE
        or not evidence["observer_source"]
        or not is_sha256(evidence["observer_sha256"])
        or evidence["representation_argv"] != _probe_argv(compiler, flags, "c")
        or evidence["gep_argv"] != _probe_argv(compiler, flags, "ir")
    ):
        raise ValueError("implicit pointer width requires complete original compiler/LLVM probe records")
    for name in ("representation_ir", "gep_source", "gep_ir"):
        if not is_sha256(evidence[name + "_sha256"]) or _hash(evidence[name]) != evidence[name + "_sha256"]:
            raise ValueError("implicit pointer probe text differs from its recorded digest")
    bits = _representation_bits(evidence["representation_ir"], layout)
    if evidence["gep_source"] != _gep_source(layout, bits):
        raise ValueError("implicit pointer LLVM probe differs from the selected source/layout")
    return _check_gep(evidence["gep_ir"], layout, bits)


def _probe(clang, flags, source, language):
    done = subprocess.run(
        _probe_argv(clang, flags, language), input=source, capture_output=True, text=True, timeout=120
    )
    if done.returncode:
        raise RuntimeError(f"{clang} failed selected {language} pointer observation: {done.stderr[-500:]}")
    return done.stdout


def _observe_implicit(clang, flags, layout, observer_source, observer_sha256):
    representation = _probe(clang, flags, _POINTER_SOURCE, "c")
    bits = _representation_bits(representation, layout)
    source = _gep_source(layout, bits)
    result = _probe(clang, flags, source, "ir")
    if sha256_file(observer_source) != observer_sha256:
        raise ValueError("selected implicit-pointer observer source changed during observation")
    evidence = {
        "schema": "merlin.implicit-pointer-observation.v1",
        "representation_source": _POINTER_SOURCE,
        "representation_ir": representation,
        "representation_ir_sha256": _hash(representation),
        "gep_source": source,
        "gep_source_sha256": _hash(source),
        "gep_ir": result,
        "gep_ir_sha256": _hash(result),
        "representation_argv": _probe_argv(clang, flags, "c"),
        "gep_argv": _probe_argv(clang, flags, "ir"),
        "observer_source": str(observer_source),
        "observer_sha256": observer_sha256,
    }
    return _implicit_bits(layout, evidence, str(clang), flags), evidence


def _query(clang: str | Path, cross_flags: Sequence[str]) -> str:
    flags = [str(flag) for flag in cross_flags if str(flag) != "-c"]
    done = subprocess.run(
        [str(clang), *flags, "-S", "-emit-llvm", "-x", "c", "-", "-o", "-"],
        input="",
        capture_output=True,
        text=True,
        timeout=120,
    )
    layout = parse(done.stdout) if done.returncode == 0 else None
    if not layout:
        raise RuntimeError(f"{clang} states no data layout for {flags}: {done.stderr[-500:]}")
    return layout


def observe_index_width(clang: str | Path, cross_flags: Sequence[str]) -> dict:
    """Fresh compiler-owned lowering selection for exact cross flags.

    The compiler executable's bytes are pinned before and after observation.
    This does not authenticate its dynamic libraries or the later MLIR runtime.
    """
    import shutil

    requested = str(clang)
    executable = Path(requested)
    if not executable.is_file() and executable.name == requested:
        found = shutil.which(requested)
        if found:
            executable = Path(found)
    resolved = executable.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("selected cross compiler is not a regular file")
    before = sha256_file(resolved)
    flags = [str(flag) for flag in cross_flags]
    if "-c" in flags:
        raise ValueError("selected cross flags cannot suppress LLVM data-layout observation")
    observer = Path(__file__).resolve()
    observer_sha256 = sha256_file(observer)
    layout = _query(resolved, flags)
    evidence = None
    if _has_default_pointer(layout):
        bits = default_index_bits(layout)
    else:
        bits, evidence = _observe_implicit(resolved, flags, layout, observer, observer_sha256)
    if sha256_file(resolved) != before:
        raise ValueError("selected cross compiler changed during layout observation")
    return {
        "schema": "merlin.selected-index-lowering.v1" if evidence is None else "merlin.selected-index-lowering.v2",
        "compiler_requested": requested,
        "compiler_resolved": str(resolved),
        "compiler_sha256": before,
        "cross_flags": flags,
        "data_layout": layout,
        "index_bits": bits,
        "scope": "compiler binary and exact cross flags only; no transitive toolchain closure",
        **({"pointer_observation": evidence} if evidence is not None else {}),
    }


def of(clang: str | Path, cross_flags: Sequence[str]) -> str:
    """The data layout ``clang`` uses for ``cross_flags``; raises when it states none."""
    key = (str(clang), *map(str, cross_flags))
    with _LOCK:
        if key in _KNOWN:
            return _KNOWN[key]
    layout = _query(clang, cross_flags)
    with _LOCK:
        _KNOWN[key] = layout
    return layout
