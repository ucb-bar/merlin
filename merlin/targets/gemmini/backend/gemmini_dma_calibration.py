"""Target-owned, derivation-backed Gemmini DMA calibration kernels.

This module deliberately does not copy Gemmini instruction codes or packed register layouts.  It asks
the configured target header to expand ``gemmini_config_{ld,st}`` and
``gemmini_extended_{mvin,mvout}``, compiles those expansions to LLVM IR, and imports only the resulting
inline-assembly recipes into the same LLVM-dialect MLIR lowering used by the production backend.

The requested byte count is a *payload extent*.  It is never reported as physical traffic: byte-named
RTL counters remain raw readings until their direction and unit are independently validated.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Iterator, Mapping

from merlin.targetgen.capability_discovery import parse_c_header

from .gemmini_codegen import CodegenError


SCHEMA = "gemmini_dma_calibration_emitter_v1"
DIRECTIONS = ("read", "write", "copy")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _temporary_root() -> Path:
    from merlin.common.paths import artifacts_dir

    root = artifacts_dir() / "perf-bench" / "gemmini" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CodegenError(f"{name} must be a positive integer")
    return value


def _input_element_bytes(rtl_facts: Mapping[str, Any]) -> int:
    body = rtl_facts.get("facts")
    datapaths = body.get("datapaths") if isinstance(body, Mapping) else None
    if not isinstance(datapaths, list):
        raise CodegenError("RTL facts do not establish an input datapath")
    matches = [row for row in datapaths
               if isinstance(row, Mapping) and row.get("name") == "input"]
    if len(matches) != 1:
        raise CodegenError("RTL facts do not establish exactly one input datapath")
    dtype = matches[0].get("dtype")
    if not isinstance(dtype, str) or not dtype.startswith("i") or not dtype[1:].isdigit():
        raise CodegenError("RTL facts do not establish an integer input element width")
    width = int(dtype[1:])
    if width <= 0 or width % 8:
        raise CodegenError("RTL-derived input width is not byte-addressable")
    return width // 8


def _target_headers() -> tuple[Path, Path]:
    from . import gemmini

    include = gemmini.rocc_tests_dir() / "include"
    main = include / "gemmini.h"
    params = include / "gemmini_params.h"
    if not main.is_file() or not params.is_file():
        raise CodegenError("configured target's generated Gemmini headers are unavailable")
    return main, params


def max_command_payload_bytes(rtl_facts: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Return the target header's own per-command byte bound, cross-checked with RTL dtype."""
    element_bytes = _input_element_bytes(rtl_facts)
    _main, params = _target_headers()
    model = parse_c_header(params)
    macro = model.macro("MAX_BYTES")
    value = macro.int_value if macro is not None else None
    if value is None or value <= 0:
        raise CodegenError("generated target header has no literal positive MAX_BYTES capability")
    if value % element_bytes:
        raise CodegenError(
            "generated target MAX_BYTES is not divisible by the RTL-derived input element width")
    raw = params.read_bytes()
    return value, {
        "path": str(params), "sha256": _sha256(raw), "macro": macro.name,
        "line": macro.line, "literal": macro.body, "element_bytes": element_bytes,
    }


def derived_transfer_ladder(rtl_facts: Mapping[str, Any], *, points: int) -> tuple[int, ...]:
    """Create probe coordinates only from an emitted-header capability, then probe every coordinate."""
    points = _positive_int(points, name="points")
    maximum, _receipt = max_command_payload_bytes(rtl_facts)
    return tuple(maximum * multiple for multiple in range(1, points + 1))


def _clang_context() -> tuple[list[str], dict[str, Any]]:
    """Build clang flags from the configured target build recipe and GCC installation."""
    from merlin.llvmlower.toolchain import clang
    from . import gemmini

    recipe = gemmini.harness_build_recipe()
    gcc = Path(recipe.compiler)
    clang_path = Path(clang())
    if not gcc.is_file() or not clang_path.is_file():
        raise CodegenError("configured target GCC or repository clang is unavailable")

    def query(argument: str) -> str:
        proc = subprocess.run([str(gcc), argument], capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise CodegenError(f"could not query target GCC with {argument}")
        return proc.stdout.strip()

    triple = query("-dumpmachine")
    gcc_include = Path(query("-print-file-name=include"))
    gcc_fixed = Path(query("-print-file-name=include-fixed"))
    libc = Path(query("-print-file-name=libc.a"))
    target_include = libc.resolve().parents[1] / "include"
    include_roots = [Path(path) for path in recipe.include_roots]
    system_roots = [path for path in (target_include, gcc_include, gcc_fixed) if path.is_dir()]
    march = next((flag for flag in recipe.cflags if flag.startswith("-march=")), None)
    if march is None:
        raise CodegenError("target build recipe does not declare its ISA march")
    definitions = [flag for flag in recipe.cflags if flag.startswith("-D")]
    args = [str(clang_path), f"--target={triple}", march, "-std=gnu99", "-O2",
            *definitions]
    for root in include_roots:
        args.extend(("-I", str(root)))
    for root in system_roots:
        args.extend(("-isystem", str(root)))
    return args, {
        "clang": str(clang_path), "clang_sha256": _sha256(clang_path.read_bytes()),
        "gcc": str(gcc), "gcc_sha256": _sha256(gcc.read_bytes()), "target_triple": triple,
        "march": march, "include_roots": [str(path) for path in include_roots],
    }


def _compile_source(source: str) -> tuple[str, dict[str, Any]]:
    base, toolchain = _clang_context()
    pre = subprocess.run([*base, "-E", "-P", "-x", "c", "-"], input=source,
                         capture_output=True, text=True)
    if pre.returncode != 0:
        raise CodegenError(f"target-header preprocessing failed: {pre.stderr[-1200:]}")
    compiled = subprocess.run([*base, "-S", "-emit-llvm", "-x", "c", "-", "-o", "-"],
                              input=source, capture_output=True, text=True)
    if compiled.returncode != 0:
        raise CodegenError(f"target-header LLVM extraction failed: {compiled.stderr[-1200:]}")
    main, params = _target_headers()
    receipt = {
        "source_sha256": _sha256(source.encode("utf-8")),
        "preprocessed_source_sha256": _sha256(pre.stdout.encode("utf-8")),
        "llvm_ir_sha256": _sha256(compiled.stdout.encode("utf-8")),
        "headers": [
            {"path": str(path), "sha256": _sha256(path.read_bytes())}
            for path in (main, params)
        ],
        "toolchain": toolchain,
    }
    return compiled.stdout, receipt


@dataclass(frozen=True)
class _Asm:
    template: str
    constraints: str
    operands: tuple[str | int, ...]


def _parse_asm_call(line: str) -> _Asm:
    marker = 'asm sideeffect "'
    start = line.find(marker)
    if start < 0:
        raise CodegenError("LLVM probe line is not an inline-assembly call")
    rest = line[start + len(marker):]
    template, separator, rest = rest.partition('", "')
    if not separator:
        raise CodegenError("could not delimit target-header assembly template")
    constraints, separator, arguments = rest.partition('"(')
    if not separator or ")" not in arguments:
        raise CodegenError("could not delimit target-header assembly operands")
    arguments = arguments.rsplit(")", 1)[0]
    operands: list[str | int] = []
    if arguments.strip():
        for argument in arguments.split(", "):
            _typ, separator, value = argument.strip().partition(" ")
            if not separator:
                raise CodegenError(f"malformed target-header assembly operand {argument!r}")
            if value in {"%src", "%dst"}:
                operands.append(value[1:])
            else:
                try:
                    operands.append(int(value, 0))
                except ValueError as exc:
                    raise CodegenError(
                        f"target-header assembly operand is neither a probe pointer nor a literal: "
                        f"{value!r}") from exc
    return _Asm(template, constraints, tuple(operands))


def _probe_asm(llvm_ir: str) -> tuple[_Asm, ...]:
    inside = False
    calls: list[_Asm] = []
    for line in llvm_ir.splitlines():
        stripped = line.strip()
        if stripped.startswith("define ") and "@probe(" in stripped:
            inside = True
            continue
        if not inside:
            continue
        if stripped == "}":
            break
        if " call " in f" {stripped} " and "asm sideeffect" not in stripped:
            raise CodegenError("target header did not constant-fold to a self-contained asm recipe")
        if "asm sideeffect" in stripped:
            calls.append(_parse_asm_call(stripped))
    if not inside or not calls:
        raise CodegenError("target compiler emitted no probe inline-assembly recipe")
    return tuple(calls)


def _source(direction: str, *, elements: int, element_bytes: int) -> str:
    stride = elements * element_bytes
    operations: list[str] = []
    if direction in {"read", "copy"}:
        operations.extend((
            f"  gemmini_config_ld({stride});",
            f"  gemmini_extended_mvin(src, 0, {elements}, 1);",
        ))
    if direction in {"write", "copy"}:
        operations.extend((
            f"  gemmini_config_st({stride});",
            f"  gemmini_extended_mvout(dst, 0, {elements}, 1);",
        ))
    return ("#include \"include/gemmini.h\"\n"
            "void probe(void *src, void *dst) {\n" + "\n".join(operations) + "\n}\n")


def _witness_source(kind: str) -> str:
    operation = ("  gemmini_extended_mvin(src, 0, 1, 1);" if kind == "read" else
                 "  gemmini_extended_mvout(dst, 0, 1, 1);")
    return ("#include \"include/gemmini.h\"\n"
            "void probe(void *src, void *dst) {\n" + operation + "\n}\n")


@cache
def _direction_templates() -> tuple[str, str, dict[str, Any]]:
    read_ir, read_receipt = _compile_source(_witness_source("read"))
    write_ir, write_receipt = _compile_source(_witness_source("write"))
    read = _probe_asm(read_ir)
    write = _probe_asm(write_ir)
    if len(read) != 1 or len(write) != 1 or read[0].template == write[0].template:
        raise CodegenError("target header does not yield distinguishable single mvin/mvout recipes")
    return read[0].template, write[0].template, {
        "read_witness": read_receipt, "write_witness": write_receipt,
    }


@cache
def _chunk_recipe(direction: str, *, elements: int,
                  element_bytes: int) -> tuple[tuple[_Asm, ...], dict[str, Any]]:
    source = _source(direction, elements=elements, element_bytes=element_bytes)
    llvm_ir, receipt = _compile_source(source)
    calls = _probe_asm(llvm_ir)
    read_template, write_template, witnesses = _direction_templates()
    observed_read = sum(call.template == read_template for call in calls)
    observed_write = sum(call.template == write_template for call in calls)
    expected = {
        "read": (1, 0, 2),
        "write": (0, 1, 2),
        "copy": (1, 1, 4),
    }[direction]
    if (observed_read, observed_write, len(calls)) != expected:
        raise CodegenError(
            f"target-header {direction} recipe is not direction-pure: got "
            f"mvin={observed_read}, mvout={observed_write}, asm={len(calls)}")
    receipt.update({
        "payload_bytes": elements * element_bytes,
        "elements": elements,
        "direction_purity": {
            "mvin_calls": observed_read, "mvout_calls": observed_write,
            "proof": "target-compiler expansion compared with isolated header-macro witnesses",
            "witnesses_sha256": _canonical_sha256(witnesses),
        },
    })
    return calls, receipt


def _escape_mlir(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def emit_dma_kernel_mlir(direction: str, payload_bytes: int,
                         rtl_facts: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    """Emit a direction-pure kernel and its derivation receipt.

    ``payload_bytes`` is the requested byte extent across all DMA commands.  The receipt keeps
    physical traffic explicitly unmeasured.
    """
    if direction not in DIRECTIONS:
        raise CodegenError(f"direction must be one of {DIRECTIONS}")
    payload_bytes = _positive_int(payload_bytes, name="payload_bytes")
    element_bytes = _input_element_bytes(rtl_facts)
    if payload_bytes % element_bytes:
        raise CodegenError(
            "payload byte count is not an exact multiple of the RTL-derived input element width")
    command_bytes, max_receipt = max_command_payload_bytes(rtl_facts)
    chunks: list[int] = []
    remaining = payload_bytes
    while remaining:
        chunk = min(remaining, command_bytes)
        if chunk % element_bytes:
            raise CodegenError("derived DMA chunk is not a whole RTL input element")
        chunks.append(chunk)
        remaining -= chunk

    args = (["src"] if direction == "read" else ["dst"] if direction == "write"
            else ["src", "dst"])
    declarations = ", ".join(f"%{name}: !llvm.ptr" for name in args)
    body: list[str] = []
    values = 0

    def fresh() -> str:
        nonlocal values
        values += 1
        return f"%v{values}"

    pointers: dict[str, str] = {}
    for name in args:
        value = fresh()
        body.append(f"    {value} = llvm.ptrtoint %{name} : !llvm.ptr to i64")
        pointers[name] = value

    receipts: list[dict[str, Any]] = []
    header_custom_instruction_count = 0
    offset = 0
    for chunk in chunks:
        calls, receipt = _chunk_recipe(
            direction, elements=chunk // element_bytes, element_bytes=element_bytes)
        receipts.append(receipt)
        header_custom_instruction_count += len(calls)
        for call in calls:
            operands: list[str] = []
            for operand in call.operands:
                if isinstance(operand, str):
                    if operand not in pointers:
                        raise CodegenError(f"header recipe unexpectedly uses %{operand}")
                    value = pointers[operand]
                    if offset:
                        constant = fresh()
                        body.append(f"    {constant} = llvm.mlir.constant({offset} : i64) : i64")
                        adjusted = fresh()
                        body.append(f"    {adjusted} = llvm.add {value}, {constant} : i64")
                        value = adjusted
                    operands.append(value)
                else:
                    value = fresh()
                    body.append(f"    {value} = llvm.mlir.constant({operand} : i64) : i64")
                    operands.append(value)
            suffix = ((" " + ", ".join(operands) + " : (" + ", ".join("i64" for _ in operands)
                       + ") -> ()") if operands else " : () -> ()")
            body.append(
                f'    llvm.inline_asm has_side_effects "{_escape_mlir(call.template)}", '
                f'"{_escape_mlir(call.constraints)}"{suffix}')
        offset += chunk
    body.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
    body.append("    llvm.return")
    text = (f"module {{\n  llvm.func @gemmini_dma_kernel({declarations}) {{\n"
            + "\n".join(body) + "\n  }\n}\n")
    receipt = {
        "schema": SCHEMA, "status": "accepted", "direction": direction,
        "requested_payload_bytes": payload_bytes,
        "physical_traffic_bytes": {
            "status": "unmeasured", "value": None,
            "why": "only a validated RTL counter binding can establish physical traffic",
        },
        "rtl_input_element_bytes": element_bytes,
        "command_payload_limit": max_receipt,
        "chunks_payload_bytes": chunks,
        "header_expansion_receipts_sha256": _canonical_sha256(receipts),
        "header_custom_instruction_count": header_custom_instruction_count,
        "emitted_mlir_sha256": _sha256(text.encode("utf-8")),
        "arguments": args,
    }
    return text, args, receipt


def build_dma_object(direction: str, payload_bytes: int, rtl_facts: Mapping[str, Any],
                     workdir: str | Path) -> tuple[Path, dict[str, Any]]:
    """Lower through the production MLIR→LLVM pipeline and compile a RISC-V object."""
    from merlin.llvmlower import codegen
    from merlin.llvmlower.custom_isa import disassemble
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    mlir, _arguments, receipt = emit_dma_kernel_mlir(direction, payload_bytes, rtl_facts)
    llvm_ir = lower_to_llvm_ir(mlir, workdir=work)
    llvm_path = work / "gemmini_dma_kernel.ll"
    llvm_path.write_text(llvm_ir, encoding="utf-8")
    obj = Path(codegen.compile_ll(llvm_path, work / "gemmini_dma_kernel.o", "riscv"))
    raw_disassembly = disassemble(obj)
    disassembly_lines = raw_disassembly.splitlines()
    symbol = next((index for index, line in enumerate(disassembly_lines)
                   if "<gemmini_dma_kernel>:" in line), None)
    if symbol is None:
        raise CodegenError("compiled DMA object has no gemmini_dma_kernel symbol in disassembly")
    kernel_disassembly = "\n".join(disassembly_lines[symbol:]) + "\n"
    custom_lines = [line.strip() for line in disassembly_lines[symbol:] if ".insn" in line]
    if len(custom_lines) != receipt["header_custom_instruction_count"]:
        raise CodegenError(
            "compiled DMA object custom-instruction count disagrees with target-header recipe: "
            f"object={len(custom_lines)}, header={receipt['header_custom_instruction_count']}")
    receipt = dict(receipt, llvm_ir_sha256=_sha256(llvm_ir.encode("utf-8")),
                   object_sha256=_sha256(obj.read_bytes()), stage="compile",
                   object_kernel_disassembly=kernel_disassembly,
                   object_kernel_disassembly_sha256=_sha256(kernel_disassembly.encode("utf-8")),
                   object_custom_instruction_count=len(custom_lines),
                   object_custom_instruction_lines=custom_lines)
    return obj, receipt


def probe_dma_capability(direction: str, payload_bytes: int, rtl_facts: Mapping[str, Any],
                         *, stage: str) -> dict[str, Any]:
    """Return UNKNOWN on any missing derivation/tool, never a guessed fallback."""
    try:
        if stage == "emission":
            _text, _args, receipt = emit_dma_kernel_mlir(direction, payload_bytes, rtl_facts)
            return dict(receipt, stage=stage)
        if stage != "compile":
            raise ValueError("stage must be 'emission' or 'compile'")
        with tempfile.TemporaryDirectory(
                prefix="dma-compile-probe-", dir=_temporary_root()) as directory:
            _object, receipt = build_dma_object(direction, payload_bytes, rtl_facts, directory)
            return receipt
    except Exception as exc:  # unavailable derivation is evidence for UNKNOWN, not a fallback ABI
        return {
            "schema": SCHEMA, "status": "unknown", "direction": direction,
            "requested_payload_bytes": payload_bytes, "stage": stage,
            "physical_traffic_bytes": {"status": "unmeasured", "value": None},
            "why": f"{type(exc).__name__}: {exc}",
        }


@contextmanager
def _measurement_environment(protocol: str, *, counter_unit: str | None = "BYTES") \
        -> Iterator[dict[str, Any]]:
    from .gemmini_codegen_mlir import _measurement_c_fragments

    previous = {name: os.environ.get(name) for name in
                ("MERLIN_HW_COUNTERS", "MERLIN_HW_COUNTER_UNIT", "MERLIN_CACHE_STATE")}
    try:
        os.environ["MERLIN_HW_COUNTERS"] = "1"
        if counter_unit is None:
            os.environ.pop("MERLIN_HW_COUNTER_UNIT", None)
        else:
            if not isinstance(counter_unit, str) or not counter_unit.strip():
                raise CodegenError("counter_unit must be a nonempty header unit or None")
            os.environ["MERLIN_HW_COUNTER_UNIT"] = counter_unit.strip()
        protocol_to_request: dict[str, str] = {}
        for request in ("cold", "warm"):
            os.environ["MERLIN_CACHE_STATE"] = request
            row = _measurement_c_fragments("")
            protocol_to_request[row["cache_protocol"]] = request
        if protocol not in protocol_to_request:
            raise CodegenError(f"measurement protocol {protocol!r} is not emitted by this harness")
        os.environ["MERLIN_CACHE_STATE"] = protocol_to_request[protocol]
        yield _measurement_c_fragments("  gemmini_dma_kernel(DMA_ARGS);")
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _harness(direction: str, payload_bytes: int, protocol: str, *,
             counter_unit: str | None = "BYTES") -> tuple[str, dict[str, Any]]:
    args = "(void*)dma_src" if direction == "read" else "(void*)dma_dst"
    if direction == "copy":
        args = "(void*)dma_src, (void*)dma_dst"
    declaration = "void *" if direction != "copy" else "void *, void *"
    call = f"  gemmini_dma_kernel({args});"
    with _measurement_environment(protocol, counter_unit=counter_unit) as fragments:
        warmup = fragments["warmup"].replace("DMA_ARGS", args)
        source = (
            "#include <stdint.h>\n#include <stdio.h>\n#include \"include/gemmini_testutils.h\"\n"
            + fragments["include"]
            + f"extern void gemmini_dma_kernel({declaration});\n"
            + f"static uint8_t dma_src[{payload_bytes} ? {payload_bytes} : 1] row_align(1);\n"
            + f"static uint8_t dma_dst[{payload_bytes} ? {payload_bytes} : 1] row_align(1);\n"
            + "int main() {\n"
            + f"  for (uint64_t i = 0; i < {payload_bytes}; ++i) {{ dma_src[i] = (uint8_t)(i + 1); dma_dst[i] = 0; }}\n"
            + warmup + fragments["prologue"]
            + "  uint64_t c0 = read_cycles();\n" + call + "\n  uint64_t c1 = read_cycles();\n"
            + fragments["epilogue"]
            + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
            + '  printf("METRIC cycle_window_gemmini_region 1\\n");\n'
            + (f"  uint64_t mismatches = 0; for (uint64_t i = 0; i < {payload_bytes}; ++i) "
               "mismatches += dma_src[i] != dma_dst[i];\n"
               '  printf("METRIC copy_mismatches %lu\\n", (unsigned long)mismatches);\n'
               if direction == "copy" else "")
            + '  printf("DONE\\n");\n  return 0;\n}\n')
        conditions = {key: fragments[key] for key in
                      ("cache_state", "cache_state_observed", "cache_protocol",
                       "requested_cache_condition")}
    return source, conditions


def run_dma_calibration(direction: str, payload_bytes: int, rtl_facts: Mapping[str, Any], *,
                        protocol: str, simulator: str = "verilator", timeout: int = 600,
                        workdir: str | Path | None = None,
                        counter_unit: str | None = "BYTES") -> dict[str, Any]:
    """Compile, link, and execute one RTL calibration point.

    ``counter_unit=None`` selects the target header's complete joint-occupancy family.  A nonempty
    unit selects that header unit family.  The default preserves the byte-family API used by the DMA
    rate calibration; neither selection assigns semantic roles to the raw counter names.
    """
    from merlin.perf import hw_counters
    from . import gemmini

    oracle = gemmini.ORACLE.get(simulator)
    if not isinstance(oracle, Mapping) or oracle.get("derived_from_rtl") is not True:
        raise CodegenError("DMA performance calibration requires an RTL-derived simulator")
    work = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(
        prefix="dma-run-", dir=_temporary_root()))
    work.mkdir(parents=True, exist_ok=True)
    obj, emitter = build_dma_object(direction, payload_bytes, rtl_facts, work)
    harness, conditions = _harness(
        direction, payload_bytes, protocol, counter_unit=counter_unit)
    harness_path = work / "gemmini_dma_harness.c"
    harness_path.write_text(harness, encoding="utf-8")
    recipe = gemmini.harness_build_recipe()
    elf = work / "gemmini_dma_calibration.elf"
    command = [str(recipe.compiler), *recipe.cflags]
    for root in recipe.include_roots:
        command.extend(("-I", str(root)))
    command.extend(("-T", str(recipe.link_script), str(harness_path), str(obj), "-o", str(elf)))
    command.extend(str(path) for path in recipe.support_sources)
    linked = subprocess.run(command, capture_output=True, text=True)
    if linked.returncode != 0:
        raise CodegenError(f"DMA calibration link failed: {linked.stderr[-1600:]}")
    console = gemmini.run_elf(elf, simulator=simulator, timeout=timeout)
    _outputs, raw = gemmini.parse_output(console)
    discovery = hw_counters.counters_for_target("gemmini")
    measured_schema = hw_counters.parse_counter_schema(console)
    if discovery.get("status") != "derived" or measured_schema != discovery.get("header_sha256"):
        raise CodegenError("measured DMA byte-counter schema does not match target discovery")
    readings = hw_counters.parse_counter_output(console)
    if not readings:
        raise CodegenError("RTL DMA calibration returned no byte-counter readings")
    counter_report: dict[str, Any] = {
        "discovery": discovery,
        "selection": {"kind": "joint_occupancy" if counter_unit is None else "unit",
                      "unit": counter_unit},
        "readings": readings,
        "measured_header_sha256": measured_schema,
    }
    if counter_unit is None:
        occupancy = hw_counters.derive_occupancy_counters(
            Path(discovery["header"]).read_text(encoding="utf-8", errors="replace"))
        selected = set(occupancy.by_combination.values())
        if not occupancy.complete() or set(readings) != selected:
            raise CodegenError(
                "RTL DMA occupancy readings do not exactly cover the derived joint partition")
        partition = gemmini.counter_partition_inputs()
        if partition.get("status") != "available":
            raise CodegenError(partition.get(
                "why", "CIRCT occupancy partition evidence is unavailable"))
        codes = hw_counters.event_codes(
            Path(discovery["header"]).read_text(encoding="utf-8", errors="replace"))
        overlap = hw_counters.eta_from_counters(
            readings, occupancy, hw_text=partition["hw_text"], codes=codes,
            module=partition["module"], counter_module=partition["counter_module"],
            measurement_cycles=raw.get("cycles"), source=partition["source"])
        proof = overlap.get("partition_proof") if isinstance(overlap, Mapping) else None
        cycles = raw.get("cycles")
        if (not isinstance(proof, Mapping) or proof.get("status") != "proved"
                or isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in readings.values())
                or sum(readings.values()) > cycles):
            raise CodegenError(str(overlap.get(
                "why", "joint occupancy is not a proved partition inside its cycle window")))
        counter_report.update({"occupancy": occupancy.to_dict(), "overlap": overlap,
                               "event_codes": {
                                   name: codes[name]
                                   for name in occupancy.by_combination.values()},
                               "partition": {key: partition[key] for key in
                                             ("module", "counter_module", "source")}})
    else:
        selected = hw_counters.counters_with_unit(
            Path(discovery["header"]).read_text(encoding="utf-8", errors="replace"),
            counter_unit)
        if set(readings) != set(selected):
            raise CodegenError(
                "RTL DMA unit readings do not exactly cover the selected header family")
        counter_report["selected_counters"] = selected

    result = {
        "status": "measured", "direction": direction,
        "requested_payload_bytes": payload_bytes,
        # Names in a software header are not unit proofs.  In fact the first differential smoke of
        # this path falsified byte identity: one 64-byte read command produced RDMA_BYTES_REC=256,
        # while one 64-byte write produced WDMA_BYTES_SENT=80.  Preserve readings as raw evidence and
        # leave physical traffic UNKNOWN until the CIRCT binding proves the accumulator semantics.
        "raw_counter_readings": {"readings": readings,
                                 "counter_header_sha256": measured_schema},
        "physical_traffic_bytes": {
            "status": "unknown", "value": None,
            "why": "byte-named RTL counter direction/unit has not passed semantic validation",
        },
        "cycles": raw.get("cycles"), "measurement_conditions": conditions,
        "oracle": dict(oracle), "emitter": emitter,
        "elf": str(elf), "elf_sha256": _sha256(elf.read_bytes()), "console": console,
        "correct": (raw.get("copy_mismatches") == 0 if direction == "copy" else True),
        "counters": counter_report,
    }
    if counter_unit is None:
        result["physical_traffic_bytes"] = {
            "status": "not_measured", "value": None,
            "why": "this pass selected joint occupancy, not a physical-byte family",
        }
    return result
