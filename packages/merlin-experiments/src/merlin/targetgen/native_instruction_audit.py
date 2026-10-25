"""Exact-ELF ISA and executed source-bound composition audits.

Numerical and timing verdicts remain independent obligations.
"""

import bisect
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path


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


def _require_executed_sequence(decoded, observed):
    digest = hashlib.sha256()
    for row in decoded["instructions"]:
        if row.get("class") == "FENCE" and row.get("funct") is None:
            continue
        _update_instruction_identity(digest, row.get("pc"), row.get("word"), row.get("kernel"))
    if digest.hexdigest() != observed.get("instruction_sequence_sha256"):
        raise ValueError("UNKNOWN: decoded composition is not the executed PC/word/kernel sequence")


def _decode_composition_operands(decoded, target):
    from merlin.kernels.decode.rocc import fields_of
    from merlin.targetgen.rocc.decode import _semantics

    semantics = _semantics(target)
    isa = semantics.isa_constants(target)
    for row in decoded["instructions"]:
        if row.get("class") == "FENCE" and row.get("funct") is None:
            continue
        fields = fields_of(row["word"])
        if fields["opcode"] != isa["CUSTOM_OPCODE"] or row.get("funct") != fields["funct"]:
            raise ValueError("UNKNOWN: decoded composition instruction differs from its executed word")
        row["class"], row["decoded"] = semantics.decode_instruction(fields["funct"], row["rs1"], row["rs2"], isa)


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
    _decode_composition_operands(decoded, target)
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
