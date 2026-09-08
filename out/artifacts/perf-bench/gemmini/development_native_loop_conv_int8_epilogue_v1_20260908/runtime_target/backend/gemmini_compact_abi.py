"""Compile-only facts for the existing target caller's compact pointer ABI.

This does not render a caller, execute an object, authorize prepacking, or prove
that arbitrary submitted LLVM obeys the original caller's alignment contract.
Only address-space-zero C data pointers are supported.  No facts come from the
Python host ABI.  The compiler and selected headers are trusted host inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

from merlin.runtime.compact_binding import CompactTargetABI


def _document_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _layout(text):
    """Read compiler-generated module properties, refusing implicit pointer layouts."""
    layouts = [line.split('"')[1] for line in text.splitlines()
               if line.startswith('target datalayout = "')]
    triples = [line.split('"')[1] for line in text.splitlines()
               if line.startswith('target triple = "')]
    _require(len(layouts) == len(triples) == 1, "missing or ambiguous compiler data layout/triple")
    parts = layouts[0].split("-")
    pointers = [part.split(":")[1:] for part in parts if part.startswith(("p:", "p0:"))]
    _require(len(pointers) == 1, "explicit unique address-space-zero pointer layout required")
    fields = pointers[0]
    _require(2 <= len(fields) <= 4 and all(x.isdecimal() and int(x) > 0 for x in fields),
             "malformed pointer layout")
    _require(parts[0] in ("e", "E"), "missing compiler byte order")
    _require(not any(part.startswith("ni:") and "0" in part.split(":")[1:] for part in parts),
             "non-integral address-space-zero pointers unsupported")
    _require(not any(part[:1] in ("P", "G", "A") and part[1:] != "0" for part in parts),
             "nonzero default pointer address spaces unsupported")
    size, alignment = map(int, fields[:2])
    index = int(fields[3]) if len(fields) == 4 else size  # LLVM DataLayout's explicit p-field rule.
    _require(index <= size, "pointer index wider than pointer representation")
    return {"data_layout": layouts[0], "triple": triples[0], "pointer_bits": size,
            "pointer_alignment_bits": alignment, "pointer_index_bits": index,
            "byte_order": "little" if parts[0] == "e" else "big", "address_space": 0}


def _dependencies(text, cwd):
    target, marker, rest = text.replace("\\\n", " ").partition(":")
    _require(marker and target.strip() == "merlin_abi_dependencies", "malformed compiler dependencies")
    names = shlex.split(rest, posix=True)
    _require(names and len(names) <= 256, "missing or excessive compiler dependencies")
    paths = {str((Path(cwd) / name).resolve()) for name in names}
    _require(all(Path(path).is_file() for path in paths), "compiler dependency is not a regular file")
    return {path: _file_sha(path) for path in sorted(paths)}


def _symbol_sizes(text):
    facts = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[-1].startswith("merlin_abi_"):
            continue
        _require(len(fields) == 4 and fields[1].isdecimal() and fields[2] in ("B", "D", "R"),
                 "malformed target fact symbol")
        name = fields[-1].removeprefix("merlin_abi_")
        _require(name not in facts, "duplicate target fact symbol")
        facts[name] = int(fields[1])
    return facts


def _agree(layout, facts, dtype_widths):
    expected = {"char_bits", "pointer_bits", "pointer_alignment", "byte_order"}
    expected |= {f"{kind}_{i}" for i in range(len(dtype_widths)) for kind in ("element_bits", "alignment")}
    _require(set(facts) == expected and all(type(x) is int and x > 0 for x in facts.values()),
             "missing, extra, or invalid compiled target facts")
    _require(facts["char_bits"] == 8, "compact storage uses octets; target CHAR_BIT differs")
    _require(facts["pointer_bits"] == layout["pointer_bits"], "kernel/harness pointer width mismatch")
    _require(facts["pointer_alignment"] * facts["char_bits"] == layout["pointer_alignment_bits"],
             "kernel/harness pointer alignment mismatch")
    _require(facts["byte_order"] == (1 if layout["byte_order"] == "little" else 2),
             "kernel/harness byte order mismatch")
    for i, width in enumerate(dtype_widths):
        _require(facts[f"element_bits_{i}"] == width, "harness container/declared dtype width mismatch")
        align = facts[f"alignment_{i}"]
        _require(align & (align - 1) == 0, "non-power-of-two container alignment unsupported")


@dataclass(frozen=True)
class DerivedCompactABI:
    target_abi: CompactTargetABI
    evidence_json: str

    def to_evidence(self):
        return json.loads(self.evidence_json)


def derive_compact_target_abi(command_buffer, compact_contract, *, workdir, timeout_s=30):
    """Derive the current caller's ABI guarantees, never a candidate's claimed facts.

    The returned per-base widths refer to C data pointers in address space zero.
    The caller must independently establish that the original and compact emitted
    functions use that address space, bind the transformation receipt to their
    exact bytes, and check the eventual linker allocations.  No cache is used.
    """
    from . import gemmini, gemmini_codegen_mlir
    from merlin.llvmlower import codegen, toolchain
    from merlin.targetgen.capsule_dram import dtype_bits

    harness_build_recipe = gemmini.harness_build_recipe
    container_for = gemmini_codegen_mlir.container_for

    _require(type(timeout_s) in (int, float) and 0 < timeout_s <= 60, "ABI fact deadline must be <=60 seconds")
    started = time.monotonic()
    deadline = started + timeout_s
    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="compact_abi_", dir=root))
    args = command_buffer.get("kernel_abi", {}).get("args", [])
    _require(command_buffer.get("kernel_abi", {}).get("kind") == "whole_program"
             and 0 < len(args) <= 4096, "bounded whole-program pointer ABI required")
    _require(compact_contract.get("schema") == "compact_pointer_entry_v1", "unsupported compact contract")
    dtypes = [command_buffer["tensors"][arg["tensor"]]["dtype"] for arg in args]
    unique = sorted(set(dtypes))
    _require(len(unique) <= 16, "too many container types for bounded ABI probe")
    containers = [container_for(dtype) for dtype in unique]
    widths = [dtype_bits(dtype) for dtype in unique]
    recipe = harness_build_recipe()
    kernel_compiler = Path(toolchain.clang()).resolve()
    gcc = Path(recipe.compiler).resolve()
    kernel_flags = [*codegen.RISCV_FLAGS, recipe.march()]
    includes = [value for path in recipe.include_roots for value in ("-I", str(path))]
    source = work / "facts.c"
    # Match the actual renderer's top-level include. A direct params include can
    # resolve differently from this header's nested quoted include under shadows.
    declarations = ["#include <limits.h>", '#include "include/gemmini_testutils.h"',
        "char merlin_abi_char_bits[CHAR_BIT];",
        "char merlin_abi_pointer_bits[sizeof(void*) * CHAR_BIT];",
        "char merlin_abi_pointer_alignment[__alignof__(void*)];",
        "#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__",
        "char merlin_abi_byte_order[1];",
        "#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__",
        "char merlin_abi_byte_order[2];", "#else", '#error "unknown byte order"', "#endif"]
    for i, container in enumerate(containers):
        declarations += [container.decl(f"probe_{i}", 1),
            f"char merlin_abi_element_bits_{i}[sizeof(probe_{i}[0]) * CHAR_BIT];",
            f"char merlin_abi_alignment_{i}[__alignof__(probe_{i})];"]
    source.write_text("\n".join(declarations) + "\n")
    commands = []

    def run(argv):
        _require(time.monotonic() < deadline, "ABI fact deadline exhausted")
        index = len(commands)
        stdout, stderr = work / f"command_{index}.stdout", work / f"command_{index}.stderr"
        record = {"argv": [str(x) for x in argv], "stdout": str(stdout), "stderr": str(stderr)}
        commands.append(record)
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(record["argv"], cwd=work, stdout=out, stderr=err, start_new_session=True)
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline or max(stdout.stat().st_size, stderr.stat().st_size) > 4 * 1024 * 1024:
                        raise ValueError("ABI fact command exceeded deadline/output bound")
                    time.sleep(0.01)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record["returncode"] = process.returncode
                (work / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        _require(stdout.stat().st_size <= 4 * 1024 * 1024 and stderr.stat().st_size <= 4 * 1024 * 1024,
                 "ABI fact command output bound exceeded")
        _require(process.returncode == 0, f"ABI fact command failed; see {stderr}")
        return stdout.read_text()

    pins = {str(path): _file_sha(path) for path in (kernel_compiler, gcc, source, Path(__file__).resolve(),
        Path(gemmini.__file__).resolve(), Path(gemmini_codegen_mlir.__file__).resolve(),
        Path(codegen.__file__).resolve(), Path(toolchain.__file__).resolve())}
    helpers = {}
    for name in ("cc1", "as", "nm"):
        selected = run([gcc, *recipe.cflags, f"-print-prog-name={name}"]).strip()
        resolved = Path(selected) if Path(selected).is_absolute() else Path(shutil.which(selected) or "")
        _require(resolved.is_file(), f"unresolved harness compiler helper {name}")
        helpers[name] = resolved.resolve()
        pins[str(helpers[name])] = _file_sha(helpers[name])
    dependency_cmd = [gcc, *recipe.cflags, *includes, "-M", "-MT", "merlin_abi_dependencies", str(source)]
    dependencies = _dependencies(run(dependency_cmd), work)
    selected_headers = [path for path in dependencies if Path(path).name == "gemmini_params.h"]
    _require(len(selected_headers) == 1, "missing or ambiguous selected target container header")
    pins.update(dependencies)
    layout_text = run([kernel_compiler, *kernel_flags, "-S", "-emit-llvm", "-x", "c", "/dev/null", "-o", "-"])
    (work / "target_data_layout.ll").write_text(layout_text)
    layout = _layout(layout_text)
    obj, depfile = work / "facts.o", work / "facts.d"
    compile_command = recipe.compile_command(source=source, output=obj)
    compile_command[0] = str(gcc)  # Execute the resolved/pinned binary, not a mutable symlink.
    run([*compile_command, "-MD", "-MF", depfile,
         "-MT", "merlin_abi_dependencies"])
    _require(_dependencies(depfile.read_text(), work) == dependencies, "selected headers changed during ABI compilation")
    facts = _symbol_sizes(run([helpers["nm"], "-S", "--radix=d", "--defined-only", obj]))
    _agree(layout, facts, widths)
    _require(all(_file_sha(path) == digest for path, digest in pins.items()), "ABI tool/source pins changed during derivation")

    bases = compact_contract.get("bases", [])
    bindings = compact_contract.get("bindings", [])
    _require(0 < len(bases) <= len(args) and len(bindings) == len(args), "compact base/binding coverage mismatch")
    _require(all(type(binding.get("argument_index")) is int and type(binding.get("base_index")) is int
                 for binding in bindings), "non-integer compact binding indices")
    _require(sorted(binding["argument_index"] for binding in bindings) == list(range(len(args))),
             "compact original argument coverage is not exact")
    _require(all(0 <= binding["base_index"] < len(bases) for binding in bindings), "compact base index outside bounds")
    alignments = tuple(facts[f"alignment_{unique.index(dtype)}"] for dtype in dtypes)
    base_alignments, base_access = [], []
    for i in range(len(bases)):
        indices = [binding["argument_index"] for binding in bindings if binding["base_index"] == i]
        accesses = {args[j]["access"] for j in indices}
        _require(indices and len(accesses) == 1 and accesses <= {"read", "write", "readwrite"},
                 "compact bases must retain homogeneous original access")
        base_alignments.append(max(alignments[j] for j in indices))
        base_access.append(next(iter(accesses)))
    evidence = {"schema": "compiled_compact_target_abi_v1", "scope": "original_caller_container_guarantees",
        "command_buffer_sha256": _document_sha(command_buffer), "compact_contract_sha256": _document_sha(compact_contract),
        "kernel_layout": layout, "harness_facts": facts, "dtype_order": unique,
        "argument_dtype_order": dtypes, "argument_access_order": [arg["access"] for arg in args],
        "source_pins": pins, "kernel_flags": kernel_flags, "harness_flags": list(recipe.cflags),
        "harness_include_roots": [str(x) for x in recipe.include_roots],
        "selected_container_header": selected_headers[0], "commands": commands,
        "object_sha256": _file_sha(obj), "layout_artifact_sha256": _file_sha(work / "target_data_layout.ll"),
        "pointer_address_space_required": 0, "emitted_address_space_verified": False,
        "stronger_emitted_access_alignment_verified": False, "linker_allocations_verified": False,
        "prepack_authorized": False, "target_executed": False, "runtime_addresses_verified": False,
        "elapsed_seconds": time.monotonic() - started}
    authority_sha = _document_sha(evidence)
    _require(time.monotonic() < deadline, "ABI fact deadline exhausted")
    target_abi = CompactTargetABI(tuple(layout["pointer_index_bits"] for _ in bases),
        tuple(base_alignments), alignments, tuple(base_access), "compiled-target-abi-sha256:" + authority_sha)
    result = DerivedCompactABI(target_abi, json.dumps(evidence, sort_keys=True))
    (work / "receipt.json").write_text(json.dumps({**evidence, "authority_sha256": authority_sha,
        "target_abi": {field: getattr(target_abi, field) for field in target_abi.__dataclass_fields__}}, indent=2) + "\n")
    return result
