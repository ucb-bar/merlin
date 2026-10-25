"""Exact-ELF ISA and executed source-bound composition audits.

Retained instruments use the original functional run. Numerical and timing
verdicts remain independent obligations.
"""

import bisect
import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from merlin.compile.model_execution_inputs import file_sha256
from merlin.llvmlower.source_compute_groups import form_exact_groups


def _function_ranges(elf, symbols):
    output = subprocess.check_output(["readelf", "--symbols", "--wide", str(elf)], text=True)
    functions = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[3] == "FUNC" and fields[7] in symbols and fields[6] != "UND":
            start, size = int(fields[1], 16), int(fields[2])
            if size <= 0:
                continue
            span = (start, start + size)
            if fields[7] in functions and functions[fields[7]] != span:
                raise ValueError("UNKNOWN: routed symbol has ambiguous ELF function ranges")
            functions[fields[7]] = span
    ranges = sorted((start, end, symbol) for symbol, (start, end) in functions.items())
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("UNKNOWN: routed ELF function ranges overlap")
    return ranges


def _trace_word(line):
    fields = line.split()
    if not fields or fields[0] != b"core":
        return None
    # Spike -l disassembly precedes the --log-commits retirement record.
    if len(fields) >= 3 and fields[2].startswith(b"0x"):
        return None
    if (
        len(fields) < 5
        or not fields[1].endswith(b":")
        or not fields[1][:-1].isdigit()
        or not fields[2].isdigit()
        or not fields[3].startswith(b"0x")
        or not fields[4].startswith(b"(0x")
        or not fields[4].endswith(b")")
    ):
        raise ValueError("UNKNOWN: malformed retired instruction record")
    try:
        return int(fields[3], 16), int(fields[4][1:-1], 16)
    except ValueError as error:
        raise ValueError("UNKNOWN: invalid PC/instruction word in execution trace") from error


def _update_instruction_identity(digest, pc, word, kernel):
    if type(pc) is not int or type(word) is not int or (kernel is not None and type(kernel) is not str):
        raise ValueError("UNKNOWN: decoded instruction lacks its executed PC/word/kernel identity")
    digest.update(json.dumps([pc, word, kernel], separators=(",", ":")).encode() + b"\n")


def _integer_writes(tokens, registers):
    """Replay Spike's xN/value and padded x N/value integer write spellings."""
    writes = iter(tokens)
    for name in writes:
        if not name.startswith(b"x"):
            continue
        number = next(writes, b"") if name == b"x" else name[1:]
        value = next(writes, b"")
        if not number.isdigit() or not 0 <= int(number) < len(registers) or not value.startswith(b"0x"):
            raise ValueError("UNKNOWN: malformed committed integer register write")
        try:
            raw = int(value, 16)
        except ValueError as error:
            raise ValueError("UNKNOWN: invalid committed integer register value") from error
        if int(number):
            registers[int(number)] = raw
        elif raw != 0:
            raise ValueError("UNKNOWN: committed zero register is nonzero")


def _requires_runtime_bindings(value):
    if isinstance(value, dict):
        return value.get("kind") == "argbase" or any(_requires_runtime_bindings(v) for v in value.values())
    return isinstance(value, list) and any(_requires_runtime_bindings(v) for v in value)


def _composition_runtime_abi(native, declaration, routed):
    """Reopen this build's logical harness interface and compiled pointer ABI.

    RISC-V psABI a0..a7 are host calling-convention registers, independent of
    accelerator encodings. Stack-passed pointers and other host ABIs refuse.
    Logical roles join the original source boundary to the actual kernel order.
    """
    from merlin.common import mlir_query as mq
    from merlin.common.quant_formats import storage_bits
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.runtime.host_provider import read_host_function_abi

    from .frozen_placement import _program
    from .native_model_execution import _pinned_native_file

    try:
        kernels = native["build"]["output"]["device_artifacts"]["kernels"]
        header = subprocess.check_output(
            ["readelf", "--file-header", str(_pinned_native_file(native, "elf"))], text=True
        )
        host = {
            line.partition(":")[0].strip(): line.partition(":")[2].strip()
            for line in header.splitlines()
            if ":" in line
        }
        if host.get("Class") != "ELF64" or host.get("Machine") != "RISC-V":
            raise ValueError("linked ELF has no supported host pointer ABI")
        source = _program(declaration, Path(native["capsule_directory"]))
        module = mq.parse(source.read_text())
        function = next(op for op in module.body.block.ops if mq.op_name(op) == "func.func")
        returns = list(function.body.block.ops)[-1].operands
        roster, counters = {}, Counter()
        for index, row in enumerate(declaration["inputs"]):
            role = row["role"]
            roster[f"{role}_{counters[role]}"] = (index, row["shape"], row["dtype"])
            counters[role] += 1
        for index, value in enumerate(returns):
            shape, dtype = mq.type_shape_dtype(value.type)
            roster[f"out_{index}"] = (len(declaration["inputs"]) + index, shape, dtype)
        result = {}
        for route in routed:
            found = [k for k in kernels.values() if k["kernel_symbol"] == route["symbol"]]
            if len(found) != 1:
                raise ValueError("kernel ABI is ambiguous")
            kernel = found[0]
            interface = _pinned_native_file(kernel["artifacts"], "logical_interface")
            buffers, refusal = logical_buffers(interface.read_text())
            if not buffers or refusal or len(buffers) > 8 or {b.role for b in buffers} != set(roster):
                raise ValueError("logical ABI does not cover the original source boundary")
            llvm = _pinned_native_file(kernel["artifacts"], "llvm_ir")
            compilation = json.loads(_pinned_native_file(kernel["custody"], "compilation").read_text())
            rename = json.loads(_pinned_native_file(kernel["custody"], "rename").read_text())
            if any(
                record.get("status") != "completed" or record.get("returncode") != 0 for record in (compilation, rename)
            ):
                raise ValueError("kernel ABI has no successful compilation/symbol binding")
            triples = [arg.partition("=")[2] for arg in compilation["argv"] if arg.startswith("--target=")]
            if len(triples) != 1 or not triples[0].startswith("riscv64-"):
                raise ValueError("compiled kernel has no supported host pointer ABI")
            renames = [arg.partition("=")[2] for arg in rename["argv"] if arg.startswith("--redefine-sym=")]
            original = [value.partition("=")[0] for value in renames if value.partition("=")[2] == route["symbol"]]
            if len(original) != 1:
                raise ValueError("compiled entry ABI is not bound to its linked symbol")
            abi = read_host_function_abi(llvm, original[0])
            if abi.result != "void" or abi.variadic or abi.arguments != ("ptr",) * len(buffers):
                raise ValueError("compiled entry does not have the logical pointer ABI")
            arguments = []
            for register, buffer in zip(range(10, 18), buffers):
                index, shape, dtype = roster[buffer.role]
                if list(buffer.shape) != shape or buffer.dtype != dtype:
                    raise ValueError("kernel logical operand differs from its source role")
                bits = storage_bits(dtype)
                if bits % 8 or any(type(size) is not int or size <= 0 for size in shape):
                    raise ValueError("logical pointer storage is unresolved")
                arguments.append(dict(arg_index=index, register=register, byte_extent=math.prod(shape) * (bits // 8)))
            result[route["symbol"]] = arguments
        return result
    except (KeyError, TypeError, ValueError, StopIteration, OSError) as error:
        raise ValueError(f"UNKNOWN: runtime buffer ABI is unavailable: {error}") from error


def _entry_buffers(arguments, registers):
    bindings = []
    for argument in arguments:
        register, extent, index = (argument[key] for key in ("register", "byte_extent", "arg_index"))
        if (
            type(register) is not int
            or not 0 < register < len(registers)
            or type(extent) is not int
            or extent <= 0
            or type(index) is not int
            or index < 0
        ):
            raise ValueError("UNKNOWN: runtime buffer ABI argument is unresolved")
        base = registers[register]
        if type(base) is not int or base <= 0:
            raise ValueError("UNKNOWN: runtime argument base has no committed register observation")
        bindings.append(dict(arg_index=index, base=base, byte_extent=extent))
    ordered = sorted(bindings, key=lambda row: row["base"])
    if len({row["arg_index"] for row in bindings}) != len(bindings) or any(
        a["base"] + a["byte_extent"] > b["base"] for a, b in zip(ordered, ordered[1:])
    ):
        raise ValueError("UNKNOWN: runtime logical buffers overlap or repeat")
    return bindings


def decode_retirement_trace(*, elf, trace, routed, target, runtime_abi=None):
    """Decode actual operands from per-hart committed writes, never guessed zeros."""
    from merlin.kernels.decode.rocc import fields_of
    from merlin.targetgen.rocc.decode import _semantics

    semantics = _semantics(target)
    isa = semantics.isa_constants(target)
    opcode = isa.get("CUSTOM_OPCODE")
    if type(opcode) is not int:
        raise ValueError("UNKNOWN: executed composition has no selected custom opcode")
    bindings = {row["symbol"]: row for row in routed}
    if len(bindings) != len(routed):
        raise ValueError("UNKNOWN: executed composition has ambiguous kernel bindings")
    ranges = _function_ranges(elf, set(bindings))
    starts = [row[0] for row in ranges]
    registers, instructions, command_harts = {}, [], set()
    digest = hashlib.sha256()
    active_buffers = {}
    with Path(trace).open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = _trace_word(line)
            if record is None:
                continue
            parts = line.split()
            hart = int(parts[1][:-1])
            regs = registers.setdefault(hart, [0, *([None] * 31)])
            pc, word = record
            index = bisect.bisect_right(starts, pc) - 1
            owner = ranges[index][2] if index >= 0 and pc < ranges[index][1] else None
            if runtime_abi and owner in runtime_abi and pc == ranges[index][0]:
                active_buffers[(hart, owner)] = _entry_buffers(runtime_abi[owner], regs)
            identity = {
                "pc": pc,
                "word": word,
                "kernel": owner,
                "region_id": bindings[owner]["region_id"] if owner else None,
            }
            if runtime_abi and owner in runtime_abi:
                if (hart, owner) not in active_buffers:
                    raise ValueError("UNKNOWN: runtime buffer binding has no observed kernel entry")
                identity["runtime_buffer_bindings"] = active_buffers[(hart, owner)]
            fields = fields_of(word)
            if fields["opcode"] == opcode:
                command_harts.add(hart)
                if len(command_harts) != 1:
                    raise ValueError("UNKNOWN: composition commands interleave multiple harts")
                first, second = regs[fields["rs1"]], regs[fields["rs2"]]
                if first is None or second is None:
                    raise ValueError("UNKNOWN: executed command reads an unobserved integer register")
                rs1, rs2 = {"kind": "const", "raw": first}, {"kind": "const", "raw": second}
                kind, decoded = semantics.decode_instruction(fields["funct"], rs1, rs2, isa)
                instructions.append(
                    {**identity, "class": kind, "funct": fields["funct"], "rs1": rs1, "rs2": rs2, "decoded": decoded}
                )
            # Standard RISC-V FENCE transport, independent of the selected device ISA.
            elif owner is not None and fields["opcode"] == 15 and word >> 12 & 7 == 0:
                instructions.append({**identity, "class": "FENCE"})
            _integer_writes(parts[5:], regs)
    result = {"instructions": instructions, "retirement_trace_sha256": digest.hexdigest()}
    if type(isa.get("FUNCT3")) is int:
        result["abi"] = {"custom_opcode": hex(opcode), "funct3": hex(isa["FUNCT3"])}
    return result


def _require_executed_sequence(decoded, observed):
    digest = hashlib.sha256()
    for row in decoded["instructions"]:
        if row.get("class") == "FENCE" and row.get("funct") is None:
            continue
        _update_instruction_identity(digest, row.get("pc"), row.get("word"), row.get("kernel"))
    if digest.hexdigest() != observed.get("instruction_sequence_sha256"):
        raise ValueError("UNKNOWN: decoded composition is not the executed PC/word/kernel sequence")


def _retirement_counts(trace, ranges, opcode, names):
    from merlin.kernels.decode.rocc import fields_of

    starts = [row[0] for row in ranges]
    counts, entries, outside = Counter(), Counter(), Counter()
    selectors = {row[2]: Counter() for row in ranges}
    digest = hashlib.sha256()
    instruction_digest = hashlib.sha256()
    unknown = Counter()
    with Path(trace).open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = _trace_word(line)
            if record is None:
                continue
            pc, word = record
            index = bisect.bisect_right(starts, pc) - 1
            owner = ranges[index] if index >= 0 and pc < ranges[index][1] else None
            if owner and pc == owner[0]:
                entries[owner[2]] += 1
            fields = fields_of(word)
            if fields["opcode"] != opcode:
                continue
            _update_instruction_identity(instruction_digest, pc, word, owner[2] if owner else None)
            if fields["funct"] not in names:
                unknown[fields["funct"]] += 1
            if owner:
                counts[owner[2]] += 1
                selectors[owner[2]][str(fields["funct"])] += 1
            else:
                outside[hex(pc)] += 1
    return dict(
        counts=counts,
        entries=entries,
        outside=outside,
        selectors=selectors,
        unknown=unknown,
        trace_sha256=digest.hexdigest(),
        instruction_sequence_sha256=instruction_digest.hexdigest(),
    )


def _instruction_report(expected, observed, elf, opcode):
    counts, entries = observed["counts"], observed["entries"]
    outside, selectors, unknown = observed["outside"], observed["selectors"], observed["unknown"]
    kernels = [
        {
            "symbol": symbol,
            "bound_roots": expected[symbol],
            "observed_entries": entries[symbol],
            "native_instructions": counts[symbol],
            "functs": dict(selectors.get(symbol, {})),
        }
        for symbol in sorted(expected)
    ]
    issues = []
    if unknown:
        issues.append("UNKNOWN: observed custom command selector absent from selected RTL table")
    if outside:
        issues.append("native instruction outside source-bound kernel functions")
    if any(row["observed_entries"] < row["bound_roots"] or row["native_instructions"] <= 0 for row in kernels):
        issues.append("routed bindings lack observed kernel entries or native instructions")
    return {
        "schema": "run14.observed_native_family_work.v1",
        "status": "fail" if issues else "pass",
        "pass_": not issues,
        "issues": issues,
        "kernels": kernels,
        "native_instructions": sum(counts.values()),
        "outside_kernel_instructions": sum(outside.values()),
        "outside_kernel_pcs": dict(outside),
        "unknown_selectors": dict(unknown),
        "elf_sha256": hashlib.sha256(Path(elf).read_bytes()).hexdigest(),
        "trace_sha256": observed["trace_sha256"],
        "instruction_sequence_sha256": observed["instruction_sequence_sha256"],
        "custom_opcode": opcode,
    }


def audit_case_instructions(*, elf, trace, routed, facts):
    """Count executed custom commands inside and outside source-bound ELF kernels.

    Port of the functional harness's run14 counter. Inputs must be the original
    grader's selected ELF, live retirement trace and region_offload.routed data.
    A static ELF scan is deliberately insufficient to credit an observed entry.
    """
    from merlin.common.facts_view import interface

    try:
        table = interface(facts.get("facts", facts), "funct_decode_table") or {}
        opcode = table.get("custom_opcode")
        names = {int(code) for code in table.get("names", {})}
        if type(opcode) is not int or not names:
            raise ValueError("UNKNOWN: selected RTL custom instruction table is absent")
        if (
            not isinstance(routed, list)
            or not routed
            or any(
                not isinstance(row, Mapping) or not isinstance(row.get("symbol"), str) or not row["symbol"]
                for row in routed
            )
        ):
            raise ValueError("UNKNOWN: no source-bound routed kernels")
        expected = Counter(row["symbol"] for row in routed)
        ranges = _function_ranges(elf, set(expected))
        observed = _retirement_counts(trace, ranges, opcode, names)
        return _instruction_report(expected, observed, elf, opcode)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        return {
            "schema": "run14.observed_native_family_work.v1",
            "status": "UNKNOWN",
            "pass_": False,
            "issues": [str(error)],
            "outside_kernel_instructions": None,
        }


def require_observed_kernels(audit, symbols):
    """A missing counter or unobserved routed kernel cannot certify a composition."""
    if (
        not isinstance(audit, Mapping)
        or audit.get("status") != "pass"
        or audit.get("pass_") is not True
        or type(audit.get("outside_kernel_instructions")) is not int
        or audit["outside_kernel_instructions"] != 0
    ):
        raise ValueError("UNKNOWN: composition needs a clean observed outside-kernel instruction audit")
    rows = audit.get("kernels")
    if not isinstance(rows, list):
        raise ValueError("UNKNOWN: observed audit lacks kernel identities")
    for symbol in symbols:
        found = [row for row in rows if row.get("symbol") == symbol]
        if len(found) != 1:
            raise ValueError("UNKNOWN: composition kernel is not uniquely observed")
        row = found[0]
        if (
            type(row.get("native_instructions")) is not int
            or row["native_instructions"] <= 0
            or type(row.get("bound_roots")) is not int
            or row["bound_roots"] <= 0
            or type(row.get("observed_entries")) is not int
            or row["observed_entries"] < row["bound_roots"]
        ):
            raise ValueError("UNKNOWN: routed composition kernel did not execute")


def audit_native_composition(native, *, policy, facts, target):
    """Reopen original pinned composition instruments; never accept a cached pass.

    The execution owner supplies these byte pins. Without an executed decoded
    trace, routed source identities and a host-intermediate def-use audit the
    composition remains unverified. Ordinary capsules need none of these files.
    """
    import yaml

    from .frozen_placement import _program, host_chain_consumers, read
    from .native_model_execution import _pinned_native_file
    from .trace_check import check

    declaration_path = _pinned_native_file(policy, "capsule_declaration")
    declaration = yaml.safe_load(declaration_path.read_text())
    expected = declaration["expected"]
    if expected != policy["composition_expected"]:
        raise ValueError("frozen composition requirements changed")
    sidecar = json.loads(_pinned_native_file(native, "region_offload").read_text())
    host = json.loads(_pinned_native_file(native, "host_chain_audit").read_text())
    if host.get("status") != "verified":
        raise ValueError("UNKNOWN: host chain def-use audit is unverified")
    decoded = json.loads(_pinned_native_file(native, "composition_trace").read_text())
    observed = audit_case_instructions(
        elf=_pinned_native_file(native, "elf"),
        trace=_pinned_native_file(native, "retirement_trace"),
        routed=sidecar["routed"],
        facts=facts,
    )
    if decoded.get("retirement_trace_sha256") != observed.get("trace_sha256"):
        raise ValueError("UNKNOWN: decoded composition stream is not bound to the executed trace")
    _require_executed_sequence(decoded, observed)
    runtime_abi = None
    if _requires_runtime_bindings(expected.get("mode_parameters", {})):
        runtime_abi = _composition_runtime_abi(
            {**native, "capsule_directory": str(declaration_path.parent)}, declaration, sidecar["routed"]
        )
    replayed = decode_retirement_trace(
        elf=_pinned_native_file(native, "elf"),
        trace=_pinned_native_file(native, "retirement_trace"),
        routed=sidecar["routed"],
        target=target,
        runtime_abi=runtime_abi,
    )
    fields = ("pc", "word", "kernel", "region_id", "rs1", "rs2")

    def operands(stream):
        return [tuple(row.get(key) for key in fields) for row in stream["instructions"] if row.get("funct") is not None]

    if operands(decoded) != operands(replayed):
        raise ValueError("UNKNOWN: decoded composition operands differ from committed register replay")
    decoded = replayed
    frozen = read(declaration, declaration_path.parent)
    consumers = host_chain_consumers(
        _program(declaration, declaration_path.parent).read_text(), frozen, expected["composition"]
    )
    if host.get("intermediate_consumers") != consumers:
        raise ValueError("UNKNOWN: host intermediate audit differs from the original source def-use graph")
    manifest_path = Path(sidecar["manifest"])
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != sidecar["artifact_sha256"]["manifest"]
    ):
        raise ValueError("UNKNOWN: routed manifest differs from its sidecar byte pin")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bindings") != sidecar["routed"]:
        raise ValueError("UNKNOWN: routed bindings differ from the pinned source manifest")
    if host.get("source_sha256") != frozen["program_sha256"]:
        raise ValueError("UNKNOWN: host chain audit identifies different source bytes")
    decoded["composition_evidence"] = {
        "frozen_placement": frozen,
        "source_sha256": host["source_sha256"],
        "routed_source_sha256": manifest.get("source_sha256"),
        "routed": sidecar["routed"],
        "host_intermediate_consumers": host.get("intermediate_consumers"),
        "native_instruction_audit": observed,
    }
    result = check(decoded, expected, target=target)
    if result["status"] != "pass":
        raise ValueError(f"composition instruction/placement audit failed: {result['violations']}")
    return {"trace_check": result, "native_instruction_audit": observed}


def audit_native_instruction_tier(native, *, target, module, policy, allow_empty_instructions=False):
    from .native_model_execution import NativeModelExecutionError, _digest, _pinned_native_file

    try:
        elf = _pinned_native_file(native, "elf")
        from merlin.common.facts_view import interface as facts_interface
        from merlin.compile.model_execution_inputs import selected_firrtl
        from merlin.perf.isa_prohibition import scan_elf
        from merlin.targetgen import elf_lanes
        from merlin.targetgen.rocc.decode import decode_module
        from merlin.targetgen.rtl.facts import observed_facts

        selected = native.get("rtl_facts") or {}
        if (
            not isinstance(selected, Mapping)
            or not isinstance(selected.get("path"), str)
            or not isinstance(selected.get("config"), str)
            or selected_firrtl(selected["path"], target=target, config=selected["config"]) != selected
        ):
            raise NativeModelExecutionError("L1 lacks selected byte-bound RTL ISA facts")
        facts_path = Path(selected["path"])
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        table = facts_interface(facts.get("facts"), "funct_decode_table") or {}
        opcode = table.get("custom_opcode")
        names = {int(key) for key in (table.get("names") or {})}
        if type(opcode) is not int or not names:
            raise NativeModelExecutionError("selected ISA has no complete custom command table")
        blob = elf.read_bytes()
        sections = elf_lanes.executable_sections(blob)
        if not sections:
            raise NativeModelExecutionError("linked candidate ELF has no executable sections")
        custom_count = 0
        for _name, offset, size, address in sections:
            for _at, word in elf_lanes.instruction_words(blob[offset : offset + size], address):
                if word & 0x7F == opcode:
                    custom_count += 1
                    if word >> 25 not in names:
                        raise NativeModelExecutionError("linked candidate ELF has an unknown custom selector")
        with observed_facts(target, facts, path=facts_path):
            decoded = decode_module(module, target=target)
            instructions = decoded.get("instructions")
            if (
                not isinstance(instructions, list)
                or (not instructions and allow_empty_instructions is not True)
                or any(row.get("class") == "UNKNOWN" for row in instructions)
            ):
                raise NativeModelExecutionError("candidate LLVM has missing/unknown selected ISA decode")
            classes = {str(row.get("class")) for row in instructions}
            required = set(policy["expected_instruction_classes"])
            if not required.issubset(classes):
                raise NativeModelExecutionError(
                    f"candidate LLVM omits frozen required instruction classes {sorted(required - classes)}"
                )
            if policy["prohibited_instruction_roles"]:
                prohibition = scan_elf(elf, target=target, roles=policy["prohibited_instruction_roles"])
                if prohibition.get("status") != "measured" or prohibition.get("clean") is not True:
                    raise NativeModelExecutionError("linked candidate ELF has prohibited or unmeasured ISA roles")
            else:
                prohibition = {
                    "status": "not_applicable",
                    "clean": True,
                    "reason": "frozen reviewed policy explicitly prohibits no roles",
                }
            composition = (
                audit_native_composition(native, policy=policy, facts=facts, target=target)
                if "composition_expected" in policy
                else None
            )
        return {
            "status": "pass",
            "scope": "selected ISA decode, frozen class coverage and whole-ELF prohibited roles",
            "observed_classes": sorted(classes),
            "required_classes": sorted(required),
            "prohibition": prohibition,
            "whole_elf_known_custom_commands": custom_count,
            "rtl_facts": selected,
            "elf": _digest(elf),
            **({"composition_audit": composition} if composition is not None else {}),
        }
    except Exception as exc:  # noqa: BLE001 -- unknown ISA legality is not a pass
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


def run_selected_functional(backend, elf, *, timeout, retirement_trace=None):
    """Apply the selected process ceiling to the owned functional transport."""
    from merlin.common.execution_deadline import ExecutionDeadline
    from merlin.targetgen.contract.compile import _selected_run_timeout

    bounded = _selected_run_timeout(backend, "spike", timeout, ExecutionDeadline.start(timeout))
    if retirement_trace is not None:
        invocation = getattr(backend, "functional_trace_invocation", None)
        if not callable(invocation):
            raise ValueError("UNKNOWN: selected functional engine supplies no committed trace invocation")
        argv, environment = invocation(elf)
        if argv.count(str(elf)) != 1 or argv.count("-l") != 1 or argv.count("--log-commits") != 1:
            raise ValueError("UNKNOWN: functional trace invocation lacks exact ELF/commit-log selection")
        with Path(retirement_trace).open("xb") as trace:
            process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=trace, env=environment, timeout=bounded)
        if process.returncode != 0:
            raise ValueError("UNKNOWN: composition functional trace execution did not complete")
        return process.stdout
    declared_runner = getattr(backend, "run_declared_model_elf", None)
    if callable(declared_runner):
        return declared_runner(elf, timeout=bounded, capture_bytes=True)
    return backend.run_elf(elf, simulator="spike", timeout=bounded, capture_bytes=True)


def _composition_routes(record, *, policy, target):
    """Join actual linked function names to members of the original source groups."""
    import yaml

    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as groups_api

    from .frozen_placement import _program
    from .native_model_execution import _pinned_native_file

    declaration_path = _pinned_native_file(policy, "capsule_declaration")
    declaration = yaml.safe_load(declaration_path.read_text())
    source = _program(declaration, declaration_path.parent)
    bundle = Path(record["runtime_inputs"])
    if file_sha256(source) != file_sha256(bundle / "model.mlir"):
        raise ValueError("UNKNOWN: composition model differs from its frozen source")
    module = mq.parse(source.read_text())
    functions = [op for op in module.body.block.ops if mq.op_name(op) == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("UNKNOWN: composition needs one original source block")
    operations = [op for op in functions[0].body.block.ops if mq.op_name(op) != "func.return"]
    indices = {op: index for index, op in enumerate(operations)}
    groups, _ = form_exact_groups(module, target, bundle, groups_api.form_groups)
    members = {group.index: sorted(indices[op] for op in group.members) for group in groups}
    joined = record["source_device_join"]
    kernels = record["build"]["output"]["device_artifacts"]["kernels"]
    bindings = []
    for assigned in joined["assigned"]:
        source_group = joined["joined_groups"].get(
            assigned["group"], joined["joined_groups"].get(str(assigned["group"]))
        )
        bindings.append(
            {
                "symbol": kernels[assigned["symbol"]]["kernel_symbol"],
                "region_id": assigned["source_region_id"],
                "source_op_indices": members[source_group],
            }
        )
    return {"source_sha256": file_sha256(source), "bindings": bindings}


def retain_owned_composition(record, *, policy, target, retirement_trace, directory):
    import yaml

    from .frozen_placement import _program, host_chain_consumers, read
    from .native_model_execution import _digest, _pinned_native_file

    manifest = _composition_routes(record, policy=policy, target=target)
    declaration_path = _pinned_native_file(policy, "capsule_declaration")
    declaration = yaml.safe_load(declaration_path.read_text())
    frozen = read(declaration, declaration_path.parent)
    host = {
        "status": "verified",
        "source_sha256": frozen["program_sha256"],
        "intermediate_consumers": host_chain_consumers(
            _program(declaration, declaration_path.parent).read_text(),
            frozen,
            policy["composition_expected"]["composition"],
        ),
    }
    elf = Path(record["build"]["output"]["elf"])
    decoded = decode_retirement_trace(elf=elf, trace=retirement_trace, routed=manifest["bindings"], target=target)
    pins = {"elf": _digest(elf), "retirement_trace": _digest(retirement_trace)}
    documents = {"composition_manifest": manifest, "host_chain_audit": host, "composition_trace": decoded}
    for role, document in documents.items():
        path = directory / f"{role}.json"
        path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
        pins[role] = _digest(path)
    sidecar = directory / "region_offload.json"
    sidecar.write_text(
        json.dumps(
            {
                "routed": manifest["bindings"],
                "manifest": pins["composition_manifest"]["path"],
                "artifact_sha256": {"manifest": pins["composition_manifest"]["sha256"]},
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    pins["region_offload"] = _digest(sidecar)
    return pins


def audit_owned_composition(record, *, policy, facts, target):
    from merlin.targetgen.rtl.facts import observed_facts

    from .native_model_execution import _digest, _pinned_native_file

    expected = _composition_routes(record, policy=policy, target=target)
    sidecar = json.loads(_pinned_native_file(record, "region_offload").read_text())
    manifest = json.loads(_pinned_native_file(record, "composition_manifest").read_text())
    if manifest != expected or sidecar.get("routed") != expected["bindings"]:
        raise ValueError("UNKNOWN: retained composition routing differs from original source groups")
    if record.get("elf") != _digest(Path(record["build"]["output"]["elf"])):
        raise ValueError("UNKNOWN: composition trace identifies a different linked ELF")
    with observed_facts(target, facts):
        return audit_native_composition(record, policy=policy, facts=facts, target=target)


def complete_owned_composition(record, *, policy, facts, target, retirement_trace, directory):
    """Retain and audit the completed run under the same selected RTL facts."""
    from merlin.targetgen.rtl.facts import observed_facts

    path = Path(facts["path"])
    document = json.loads(path.read_text())
    with observed_facts(target, document, path=path):
        pins = retain_owned_composition(
            record, policy=policy, target=target, retirement_trace=retirement_trace, directory=directory
        )
        audit = audit_owned_composition({**record, **pins}, policy=policy, facts=document, target=target)
    return pins, audit


def selected_execution_facts(path, *, target):
    from merlin.compile.model_execution_inputs import selected_execution_firrtl

    return selected_execution_firrtl(path, target=target)
