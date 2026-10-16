"""Bounded descriptive census of the original selected Spike trace route.

The host's original namespace owner issues a live launch for one ELF. Saved
commands or receipts cannot select execution. These data grant no numerical,
timing, traffic, utilization, emitter-integrity or replay authority.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common import owned_children
from merlin.runtime import _native_files as F
from merlin.runtime.backends import chipyard_rocc
from merlin.runtime.backends.base import get_backend
from merlin.targetgen import rtl_checks_generic as G
from merlin.targetgen import spike_extension, target_experiment
from merlin.targetgen.rocc import semantics
from merlin.targetgen.rtl import facts as RF

from . import _census_process as process
from . import executed_trace as ET
from ._dynamic_stream import CensusCleanupUnknown, CensusUnavailable, OnlineCensus

SCHEMA = "merlin.dynamic_census.v1"
__all__ = ["CensusCleanupUnknown", "CensusLaunch", "CensusLimits", "profile_elf", "unknown"]
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

    def verify(self):
        if type(self.wall_s) not in (int, float) or not math.isfinite(self.wall_s) or not 0 < self.wall_s <= 120:
            raise CensusUnavailable("census needs an explicit bounded wall deadline")
        for name, value in asdict(self).items():
            if name != "wall_s" and (type(value) is not int or value <= 0):
                raise CensusUnavailable("census needs explicit positive integer resource limits")
        if self.max_line_bytes > self.max_output_bytes or self.max_bins * 512 > self.max_state_bytes:
            raise CensusUnavailable("census line or state bounds exceed selected storage budgets")

    @classmethod
    def for_timeout(cls, timeout_s):
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise CensusUnavailable("census has no remaining parent deadline")
        result = cls(wall_s=min(30.0, timeout_s))
        result.verify()
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
    if argv.count(str(elf)) != 1 or argv.count("-l") != 1 or argv.count("--log-commits") != 1:
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
        self._identity, self._limits = _hash(self.identity), asdict(limits)

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
            or asdict(self.limits) != self._limits
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
        "limits": asdict(limits) if type(limits) is CensusLimits else None,
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
            "limits": asdict(limits),
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
