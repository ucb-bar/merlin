"""Online descriptive Spike commit aggregation over one verified ELF entry.

Only lexical entry PCs are attributed. Outside-entry activity is retained as
unattributed; callee work, numerical behavior and all physical costs are UNKNOWN.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.analysis.optimization.candidates import _census_process as process
from merlin.analysis.optimization.candidates.compiled_features import linked_entry_extent
from merlin.analysis.perf.measurement import execution_traces as ET
from merlin.common.process import invocation_records as I
from merlin.common.process import ownership as owned_children
from merlin.development.targetgen.grading.elf_lanes import executable_sections
from merlin.runtime.backends.registry import get_backend
from merlin.runtime.backends.rocc import chipyard_backend as chipyard_rocc
from merlin.runtime.views import native_simulator as F
from merlin.target.isa.rocc import rtl_checks_generic as G
from merlin.target.isa.rocc import semantics
from merlin.target.registry import experiment_descriptor as target_experiment
from merlin.target.rtl import facts as RF
from merlin.target.simulators import spike_runtime as spike_extension

# ---- _dynamic_stream ---------------------------------------------------------------------------


class CensusUnavailable(ValueError):
    """An incomplete or unsupported census must never expose partial counts."""


class CensusCleanupUnknown(BaseException):
    """Abort the enclosing job while its real descendant owner remains live."""


def _hex(value):
    token = value.removeprefix("(").removesuffix(")")
    if not token.startswith("0x"):
        raise CensusUnavailable("unsupported committed hexadecimal value")
    try:
        number = int(token, 16)
    except ValueError as error:
        raise CensusUnavailable("malformed committed hexadecimal value") from error
    if not 0 <= number < 1 << 64:
        raise CensusUnavailable("committed value exceeds supported register width")
    return number


class OnlineCensus:
    def __init__(self, *, elf_bytes, entry_symbol, isa, local_fields, limits):
        limits.verify()
        if len(elf_bytes) > limits.max_elf_bytes:
            raise CensusUnavailable("linked ELF exceeds census byte budget")
        self.entry = linked_entry_extent(elf_bytes, entry_symbol)
        address, size = self.entry["address"], self.entry["size_bytes"]
        if type(size) is not int or size <= 0:
            raise CensusUnavailable("linked entry has no finite retained extent")
        sections = [
            (offset, start)
            for _, offset, length, start in executable_sections(elf_bytes)
            if start <= address and address + size <= start + length
        ]
        if len(sections) != 1:
            raise CensusUnavailable("linked entry is outside its unique executable section")
        offset, start = sections[0]
        self.code = elf_bytes[offset + address - start : offset + address - start + size]
        self.opcode = isa.get("CUSTOM_OPCODE")
        if type(self.opcode) is not int or not 0 <= self.opcode < 128:
            raise CensusUnavailable("selected RTL facts supply no custom transport opcode")
        self.isa, self.local_fields, self.limits = isa, tuple(local_fields), limits
        if len(self.local_fields) > limits.max_bins:
            raise CensusUnavailable("local-address declarations exceed census state budget")
        self.regs, self.retired, self.inside = {}, 0, 0
        self.host, self.commands, self.requests, self.memory, self.local = (Counter() for _ in range(5))
        self.accelerator_memory = Counter()
        self.bins, self.state_bytes = 0, 0

    def _add(self, counter, key):
        if key not in counter:
            self.bins += 1
            # Bound each stored key and the complete serialized census state.
            # Process RSS is separately bounded by the selected process owner.
            self.state_bytes += len(str(key).encode()) + 512
            if self.bins > self.limits.max_bins or self.state_bytes > self.limits.max_state_bytes:
                raise CensusUnavailable("online census exceeds its state budget")
        counter[key] += 1

    def line(self, raw):
        if len(raw.encode("utf-8")) > self.limits.max_line_bytes:
            raise CensusUnavailable("commit line exceeds census line budget")
        parts = raw.split()
        if not parts or parts[0] != "core":
            return
        if len(parts) < 3:
            raise CensusUnavailable("malformed core trace record")
        if parts[2].startswith("0x"):
            return  # The paired disassembly line is not another retirement.
        if len(parts) < 5 or not parts[2].isascii() or not parts[2].isdigit():
            raise CensusUnavailable("unsupported committed core record")
        if not parts[1].endswith(":") or not parts[1][:-1].isascii() or not parts[1][:-1].isdigit():
            raise CensusUnavailable("malformed commit hart identity")
        hart = int(parts[1][:-1])
        if hart not in self.regs:
            if len(self.regs) >= self.limits.max_harts:
                raise CensusUnavailable("trace exceeds selected hart state budget")
            self.regs[hart] = [0, *([None] * 31)]
        registers = self.regs[hart]
        pc, instruction = _hex(parts[3]), _hex(parts[4])
        self.retired += 1
        if self.retired > self.limits.max_retired:
            raise CensusUnavailable("trace exceeds retired-instruction budget")
        at = pc - self.entry["address"]
        inside = 0 <= at < self.entry["size_bytes"]
        custom = False
        if inside:
            # These are the standard RISC-V variable-width transport bits,
            # shared with executed_trace; device opcode values come from ISA.
            width = 4 if instruction & 3 == 3 else 2
            if instruction & 31 == 31 or at % 2 or at + width > len(self.code):
                raise CensusUnavailable("unsupported or out-of-range committed entry instruction")
            if instruction != int.from_bytes(self.code[at : at + width], "little"):
                raise CensusUnavailable("committed entry bytes disagree with selected linked ELF")
            self.inside += 1
            if width == 4 and instruction & 127 == self.opcode:
                custom = True
                first, second = registers[(instruction >> 15) & 31], registers[(instruction >> 20) & 31]
                if first is None or second is None:
                    raise CensusUnavailable("executed command has an unobserved source register")
                funct = instruction >> 25
                kind, decoded = semantics.decode_instruction(
                    funct, {"raw": first, "kind": "const"}, {"raw": second, "kind": "const"}, self.isa
                )
                if type(kind) is not str or not kind or kind == "UNKNOWN" or len(kind.encode()) > 128:
                    raise CensusUnavailable("executed command has unsupported selected ISA semantics")
                self._add(self.commands, kind)
                self._add(self.requests, (kind, funct, first, second))
                for classes, field, rows_field in self.local_fields:
                    if kind not in classes:
                        continue
                    value, rows = decoded.get(field), decoded.get(rows_field) if rows_field else None
                    if type(value) is not int or not 0 <= value < 1 << 64:
                        raise CensusUnavailable("local-address operand could not be decoded")
                    if rows_field and (type(rows) is not int or not 0 <= rows < 1 << 64):
                        raise CensusUnavailable("local-address extent could not be decoded")
                    self._add(self.local, (kind, field, value, rows_field, rows))
            else:
                self._add(self.host, pc)
        writes = iter(parts[5:])
        for name in writes:
            if name == "mem":
                address = _hex(next(writes, ""))
                if inside:
                    self._add(self.accelerator_memory if custom else self.memory, address)
            elif name.startswith("x"):
                number = next(writes, "") if name == "x" else name[1:]
                if not number.isascii() or not number.isdigit() or not 0 <= int(number) < 32:
                    raise CensusUnavailable("unsupported integer-register write")
                value = _hex(next(writes, ""))
                if int(number):
                    registers[int(number)] = value
                elif value:
                    raise CensusUnavailable("committed zero-register write is inconsistent")

    def finish(self):
        if not self.retired or not self.inside:
            raise CensusUnavailable("complete trace did not retire the verified linked entry")
        return {
            "entry": {**self.entry, "instruction_bytes_verified": True},
            "retired_instructions": {
                "whole_trace": self.retired,
                "entry": self.inside,
                "outside_entry": self.retired - self.inside,
            },
            "host": {
                "retired_in_entry": sum(self.host.values()),
                "by_pc": [{"pc": pc, "count": count} for pc, count in sorted(self.host.items())],
                "scope": "verified lexical entry PCs only; harness/setup/readback excluded",
                "outside_entry_callee_work": "UNKNOWN",
            },
            "accelerator": {
                "commands_in_entry": sum(self.commands.values()),
                "by_class": dict(sorted(self.commands.items())),
                "operand_requests": [
                    {"class": kind, "funct": funct, "rs1": first, "rs2": second, "count": count}
                    for (kind, funct, first, second), count in sorted(self.requests.items())
                ],
            },
            "host_memory_requests": {
                "requests_in_entry": sum(self.memory.values()),
                "by_address": [{"address": address, "count": count} for address, count in sorted(self.memory.items())],
                "physical_bytes": None,
            },
            "accelerator_memory_requests": {
                "requests_in_entry": sum(self.accelerator_memory.values()),
                "by_address": [
                    {"address": address, "count": count} for address, count in sorted(self.accelerator_memory.items())
                ],
                "scope": "only requests present in commit records; total and physical traffic UNKNOWN",
            },
            "local_address_activity": {
                "status": "observed" if self.local_fields else "UNKNOWN",
                "requests": [
                    {
                        "class": kind,
                        "address_field": field,
                        "raw_address": value,
                        "rows_field": rows_field,
                        "raw_rows": rows,
                        "count": count,
                    }
                    for (kind, field, value, rows_field, rows), count in sorted(self.local.items())
                ],
                "scope": "decoded requested addresses/extents; allocation and physical traffic UNKNOWN",
            },
        }


# ---- dynamic_census ----------------------------------------------------------------------------
# Bounded descriptive census of the original selected Spike trace route.
#
# The host's original namespace owner issues a live launch for one ELF. Saved
# commands or receipts cannot select execution. These data grant no numerical,
# timing, traffic, utilization, emitter-integrity or replay authority.

SCHEMA = "merlin.dynamic_census.v1"
WALL_600S_PROTOCOL = "merlin.census.resources.wall600s.v1"
WALL_900S_PROTOCOL = "merlin.census.resources.wall900s.v1"
FEATURES = (
    "/dynamic/entry/host_retired",
    "/dynamic/entry/accelerator_commands",
    "/dynamic/entry/host_memory_requests",
    "/dynamic/entry/operand_request_distinct",
    "/dynamic/entry/local_address_request_distinct",
    "/dynamic/outside_entry/retired",
)
_ISSUED = {}


@dataclass(frozen=True)
class CensusLimits:
    wall_s: float = 30.0
    max_retired: int = 2_000_000
    max_ram_bytes: int = 1 << 30
    max_output_bytes: int = 32 << 20
    max_output_files: int = 1024
    max_line_bytes: int = 8192
    max_bins: int = 4096
    max_state_bytes: int = 4 << 20
    max_harts: int = 8
    max_elf_bytes: int = 64 << 20
    max_tool_bytes: int = 128 << 20
    wall_resource_protocol: str | None = None

    def verify(self):
        if self.wall_resource_protocol is not None and (
            type(self.wall_resource_protocol) is not str
            or self.wall_resource_protocol not in (WALL_600S_PROTOCOL, WALL_900S_PROTOCOL)
        ):
            raise CensusUnavailable("census needs its exact explicit wall resource protocol")
        maximum = (
            900
            if self.wall_resource_protocol == WALL_900S_PROTOCOL
            else 600
            if self.wall_resource_protocol == WALL_600S_PROTOCOL
            else 120
        )
        if type(self.wall_s) not in (int, float) or not math.isfinite(self.wall_s) or not 0 < self.wall_s <= maximum:
            raise CensusUnavailable("census needs an explicit bounded wall deadline")
        for name, value in asdict(self).items():
            if name not in ("wall_s", "wall_resource_protocol") and (type(value) is not int or value <= 0):
                raise CensusUnavailable("census needs explicit positive integer resource limits")
        if self.max_line_bytes > self.max_output_bytes or self.max_bins * 512 > self.max_state_bytes:
            raise CensusUnavailable("census line or state bounds exceed selected storage budgets")

    @classmethod
    def for_timeout(cls, timeout_s, *, wall_resource_protocol=None):
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise CensusUnavailable("census has no remaining parent deadline")
        maximum = (
            900.0
            if wall_resource_protocol == WALL_900S_PROTOCOL
            else 600.0
            if wall_resource_protocol == WALL_600S_PROTOCOL
            else 30.0
        )
        result = cls(wall_s=min(maximum, timeout_s), wall_resource_protocol=wall_resource_protocol)
        result.verify()
        return result


def _limits_record(limits):
    """Keep default limit records unchanged; bind the opt-in resource class."""
    result = asdict(limits)
    if result["wall_resource_protocol"] is None:
        del result["wall_resource_protocol"]
    return result


def _hash(value, *, max_bytes=8 << 20):
    data = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > max_bytes:
        raise CensusUnavailable("selected census metadata exceeds its byte budget")
    return hashlib.sha256(data).hexdigest()


def _selection(elf, target, limits):
    backend = get_backend(target)
    if Path(backend.__file__).resolve() != Path(chipyard_rocc.__file__).resolve():
        raise CensusUnavailable("selected functional trace backend is outside the original neutral route")
    selected_semantics = backend.rocc_semantics
    if Path(selected_semantics.__file__).resolve() != Path(semantics.__file__).resolve():
        raise CensusUnavailable("functional census has no independently selected generic ISA decoder")
    invocation = ET.trace_invocation(target, elf)
    if invocation is None:
        raise CensusUnavailable("original backend has no functional trace invocation")
    argv, environment = invocation
    # The commit log alone (no "-l" execution/disassembly lines): every retirement is one commit line.
    if argv.count(str(elf)) != 1 or "-l" in argv or argv.count("--log-commits") != 1:
        raise CensusUnavailable("original functional invocation has unsupported trace or ELF operands")
    extensions = [Path(token.partition("=")[2]).resolve(strict=True) for token in argv if token.startswith("--extlib=")]
    declared = spike_extension.declared_extension(target)
    library_key = extensions[0].name if len(extensions) == 1 else None
    if not extensions:
        flags, library_dir = backend.spike_extension()
        names = [flag.partition("=")[2] for flag in flags if flag.startswith("--extension=")]
        if len(names) != 1 or list(flags) != argv[1 : 1 + len(flags)]:
            raise CensusUnavailable("original default functional model has an unsupported library selection")
        extensions = [(Path(library_dir) / ("lib" + names[0] + ".so")).resolve(strict=True)]
        library_key = names[0]
    if len(extensions) != 1:
        raise CensusUnavailable("original Spike extension lacks its unchanged selected library")
    extension = F.pin(extensions[0], max_bytes=limits.max_tool_bytes)
    if declared is not None and extension["sha256"] != declared.get("sha256"):
        raise CensusUnavailable("selected functional extension differs from the original contract bytes")
    tool = F.pin(Path(argv[0]).resolve(strict=True), max_bytes=limits.max_tool_bytes)
    original_model = backend.simulator_identity("spike")
    if (
        not isinstance(original_model, dict)
        or not original_model.get("digest")
        or original_model.get("binary") != tool["sha256"]
        or original_model.get("extensions", {}).get(library_key) != extension["sha256"]
    ):
        raise CensusUnavailable("functional model differs from its original selected identity")
    entry = backend.harness_build_recipe().require_kernel_stack_frame().entry_symbol
    facts_path = RF.find_facts(target)
    if facts_path is None:
        raise CensusUnavailable("original selected RTL facts are unavailable; census cannot extract them")
    facts_pin = F.pin(facts_path.resolve(), max_bytes=8 << 20)
    facts = RF.load_facts(target, regenerate=False)
    with RF.observed_facts(target, facts, facts_path):
        isa = semantics.isa_constants(target)
    contract = target_experiment.load_capability_manifest(target).contract
    protocol = G.protocol_for(target)
    local = tuple(
        (
            tuple(sorted(protocol.classes(str(row["role"])))),
            str(row["address"]),
            str(row["rows"]) if row.get("rows") else "",
        )
        for row in protocol.local_addresses
    )
    if any(len(field.encode()) > 128 or len(rows.encode()) > 128 for _, field, rows in local):
        raise CensusUnavailable("local-address field names exceed census metadata budget")
    sources = tuple(
        F.pin(Path(module.__file__).resolve(), max_bytes=limits.max_tool_bytes)
        for module in (I, F, owned_children, process, ET, semantics, G, RF, spike_extension, target_experiment, backend)
    ) + (
        F.pin(Path(__file__).resolve(), max_bytes=limits.max_tool_bytes),
        F.pin(Path(inspect.getsourcefile(OnlineCensus)).resolve(), max_bytes=limits.max_tool_bytes),
    )
    identity = {
        "elf": F.pin(elf, max_bytes=limits.max_elf_bytes),
        "tool": tool,
        "extension": extension,
        "original_model_sha256": original_model["digest"],
        "sources": list(sources),
        "argv_sha256": _hash(argv),
        "original_environment": I.environment_identity(environment),
        "entry_symbol": entry,
        "isa_sha256": _hash(isa),
        "contract_sha256": _hash(contract),
        "facts_sha256": _hash(facts),
        "facts_artifact": facts_pin,
        "local_fields_sha256": _hash(local),
    }
    return identity, argv, environment, isa, local


class CensusLaunch:
    """Exact live one-call selection, issued only by the fixed host wrapper."""

    def __init__(
        self,
        *,
        original_owner,
        elf,
        executed_elf,
        target,
        directory,
        prefix,
        limits,
        writable_roots,
        owner_sources,
        issuer,
    ):
        self.original_owner, self.elf, self.executed_elf = original_owner, elf, executed_elf
        self.target, self.directory, self.prefix, self.limits = target, directory, prefix, limits
        self._issuer, self.closed, self.started = issuer, True, False
        self.writable_roots = tuple(F.canonical(path) for path in writable_roots)
        self._owner_sources = tuple(F.pin(path, max_bytes=limits.max_tool_bytes) for path in owner_sources)
        self.identity, selected, environment, self.isa, self.local = _selection(elf, target, limits)
        self._owner_argv = tuple(original_owner.argv)
        self._owner_workspace = Path(original_owner.workspace)
        self._owner_source = F.pin(
            Path(inspect.getsourcefile(type(original_owner))).resolve(), max_bytes=limits.max_tool_bytes
        )
        if F.pin(executed_elf, max_bytes=limits.max_elf_bytes)["sha256"] != self.identity["elf"]["sha256"]:
            raise CensusUnavailable("boxed census ELF differs from the actual original linked product")
        self.environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "LANG": "C",
            "LD_LIBRARY_PATH": environment.get("LD_LIBRARY_PATH", ""),
        }
        payload = [str(executed_elf) if token == str(elf) else token for token in selected]
        self.argv = [
            *prefix,
            "--ro-bind",
            str(executed_elf),
            str(executed_elf),
            "--chdir",
            str(executed_elf.parent),
            "--",
            "/usr/bin/env",
            "-i",
            *(key + "=" + value for key, value in sorted(self.environment.items())),
            *payload,
        ]
        self.namespace_source = F.pin(
            Path(shutil.which(prefix[0]) or prefix[0]).resolve(strict=True), max_bytes=limits.max_tool_bytes
        )
        self.argv[0] = self.namespace_source["path"]
        self._command, self._environment = tuple(self.argv), dict(self.environment)
        self._identity, self._limits = _hash(self.identity), _limits_record(limits)

    def verify(self):
        if _ISSUED.get(self._issuer) is not self:
            raise CensusUnavailable("census needs its exact live host-issued namespace owner")
        if (
            tuple(self.original_owner.argv) != self._owner_argv
            or Path(self.original_owner.workspace) != self._owner_workspace
            or F.pin(Path(self._owner_source["path"]), max_bytes=self.limits.max_tool_bytes) != self._owner_source
            or F.pin(Path(self.namespace_source["path"]), max_bytes=self.limits.max_tool_bytes) != self.namespace_source
            or F.pin(self.executed_elf, max_bytes=self.limits.max_elf_bytes)["sha256"] != self.identity["elf"]["sha256"]
            or _selection(self.elf, self.target, self.limits)[0] != self.identity
            or _hash(self.identity) != self._identity
            or _hash(self.isa) != self.identity["isa_sha256"]
            or _hash(self.local) != self.identity["local_fields_sha256"]
            or tuple(self.argv) != self._command
            or self.environment != self._environment
            or _limits_record(self.limits) != self._limits
            or any(F.pin(Path(pin["path"]), max_bytes=self.limits.max_tool_bytes) != pin for pin in self._owner_sources)
        ):
            raise CensusUnavailable("original census source/tool/ELF/namespace selection changed")
        return self.identity


def _issue_launch(
    *, original_owner, elf, executed_elf, target, directory, prefix, limits, writable_roots, owner_sources
):
    """Private seam for the fixed original-policy wrapper; never a broker action."""
    if type(limits) is not CensusLimits or type(prefix) is not tuple or not prefix or "--" in prefix:
        raise CensusUnavailable("census requires a closed original namespace prefix and limits")
    limits.verify()
    issuer = object()
    launch = CensusLaunch(
        original_owner=original_owner,
        elf=F.canonical(elf),
        executed_elf=F.canonical(executed_elf),
        target=target,
        directory=F.canonical(directory),
        prefix=prefix,
        limits=limits,
        issuer=issuer,
        writable_roots=writable_roots,
        owner_sources=owner_sources,
    )
    _ISSUED[issuer] = launch
    launch.verify()
    return launch


def unknown(reason, *, limits=None):
    return {
        "schema": SCHEMA,
        "status": "UNKNOWN",
        "reason": reason,
        "limits": _limits_record(limits) if type(limits) is CensusLimits else None,
        "candidate_features": {"status": "UNKNOWN", "features": dict.fromkeys(FEATURES)},
        "observations": None,
        "cycles": None,
        "physical_traffic": None,
        "utilization": None,
        "authority": "none",
    }


def profile_elf(elf, *, target, directory, limits, launch=None):
    """Execute only the original host-issued one-ELF boxed trace selection."""
    try:
        if type(launch) is not CensusLaunch or type(limits) is not CensusLimits:
            raise CensusUnavailable("original one-ELF boxed census launch is unavailable")
        if (F.canonical(elf), target, F.canonical(directory), limits) != (
            launch.elf,
            launch.target,
            launch.directory,
            launch.limits,
        ):
            raise CensusUnavailable("census launch belongs to another exact request")
        identity = launch.verify()
        census = OnlineCensus(
            elf_bytes=launch.executed_elf.read_bytes(),
            entry_symbol=identity["entry_symbol"],
            isa=launch.isa,
            local_fields=launch.local,
            limits=limits,
        )
        dependencies = (
            tuple(Path(pin["path"]) for pin in identity["sources"])
            + (
                Path(identity["tool"]["path"]),
                Path(identity["extension"]["path"]),
                launch.elf,
                Path(launch._owner_source["path"]),
                Path(identity["facts_artifact"]["path"]),
            )
            + tuple(Path(pin["path"]) for pin in launch._owner_sources)
        )
        with I.observe(
            launch.directory,
            stage="bounded_functional_census",
            argv=launch.argv,
            cwd=launch.directory,
            env={},
            inputs=(launch.executed_elf,),
            dependencies=dependencies,
        ) as recorded:
            try:
                launch.closed = False
                returncode = process.capture(
                    launch.argv,
                    environment={},
                    cwd=launch.directory,
                    limits=limits,
                    record=recorded,
                    census=census,
                    writable_roots=launch.writable_roots,
                    readonly_files=(launch.executed_elf,),
                )
            finally:
                launch.started = getattr(recorded, "census_started", False)
                launch.closed = not getattr(recorded, "census_started", False) or getattr(
                    recorded, "census_cleanup_complete", False
                )
            launch.verify()
            observations = census.finish()
            recorded.complete_streams(returncode)
            actual = I.verify(recorded.path)
        features = (
            observations["host"]["retired_in_entry"],
            observations["accelerator"]["commands_in_entry"],
            observations["host_memory_requests"]["requests_in_entry"],
            len(observations["accelerator"]["operand_requests"]),
            len(observations["local_address_activity"]["requests"])
            if observations["local_address_activity"]["status"] == "observed"
            else None,
            observations["retired_instructions"]["outside_entry"],
        )
        return {
            "schema": SCHEMA,
            "status": "observed",
            "limits": _limits_record(limits),
            "identity": {
                "selection_sha256": _hash(identity),
                "elf_sha256": identity["elf"]["sha256"],
                "invocation_sha256": F.digest(recorded.path),
                "tool_sha256": identity["tool"]["sha256"],
                "extension_sha256": identity["extension"]["sha256"],
                "stdout_sha256": actual["stdout"]["sha256"],
                "stderr_sha256": actual["stderr"]["sha256"],
                "executed_environment": I.environment_identity(launch.environment),
            },
            "candidate_features": {"status": "descriptive", "features": dict(zip(FEATURES, features, strict=True))},
            "observations": observations,
            "completeness": {
                "stream_eof": True,
                "truncated": False,
                "descendants_reaped": True,
                "emitter_integrity": "UNKNOWN",
                "complete_call_attribution": "UNKNOWN",
            },
            "resources": {
                "ram": "per-process RLIMIT_AS; aggregate bound belongs to original external owner",
                "disk": "bounded raw capture and online state; enclosing scratch quota is caller-owned",
            },
            "cycles": None,
            "physical_traffic": None,
            "utilization": None,
            "numerical_verdict": "none; original numerical gates remain required",
            "authority": "none",
        }
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError, TypeError) as error:
        # Private details are retained only in the original host output owner.
        if type(launch) is CensusLaunch:
            (launch.directory / "census_failure_private.json").write_text(
                json.dumps({"category": type(error).__name__, "detail": str(error)[:500]})
            )
        return unknown("bounded original census unavailable: " + type(error).__name__, limits=limits)


__all__ = [
    "CensusCleanupUnknown",
    "CensusLaunch",
    "CensusLimits",
    "WALL_600S_PROTOCOL",
    "WALL_900S_PROTOCOL",
    "profile_elf",
    "unknown",
]
