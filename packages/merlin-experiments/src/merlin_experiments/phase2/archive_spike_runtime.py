"""Explicit operator-selected count runtime; no event prices or compiler grants."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.runtime import _native_files as F
from merlin_experiments.phase1 import annotated_spike_evidence as E

from .contracts import StageGateError, canonical_json, document_sha256

MAX_TOOL_BYTES = 256 << 20
MAX_METADATA_BYTES = 64 << 20
COUNT_OBSERVATION_120S_PROTOCOL = "merlin.archive_count_observation.resources.120s.v1"
COUNT_OBSERVATION_600S_PROTOCOL = "merlin.archive_count_observation.resources.600s.v1"
COUNT_OBSERVATION_900S_PROTOCOL = "merlin.archive_count_observation.resources.900s.v1"


def metadata(path, *, maximum=MAX_METADATA_BYTES):
    pin = F.pin(Path(path), max_bytes=maximum)
    value = json.loads(Path(path).read_bytes())
    if F.pin(Path(path), max_bytes=maximum) != pin:
        raise StageGateError("archive screening metadata changed during read")
    return value, pin


@dataclass(frozen=True, eq=False)
class ArchiveSpikeRuntime:
    """Frozen exact tools and original command, selected by the host before search.

    ``functional_argv`` is the original functional runtime command WITHOUT an
    ELF. No author document can select this owner. Readonly namespace admission
    and fresh complete original numerics remain separate requirements.
    """

    review_id: str
    functional_argv: tuple[str, ...]
    observer: Path
    disassembler: Path
    model_path: Path
    source_manifest: Path
    libraries: tuple[Path, ...]
    dependencies: tuple[Path, ...]
    entry_symbol: str
    completion_symbol: str
    timing_json: bytes
    files: tuple[tuple[str, str], ...]
    model_json: bytes
    observation_resource_protocol: str | None = None

    def record(self):
        return {
            "schema": ("merlin.phase2.archive_spike_runtime.v4"
                       if self.observation_resource_protocol == COUNT_OBSERVATION_900S_PROTOCOL
                       else "merlin.phase2.archive_spike_runtime.v3"
                       if self.observation_resource_protocol == COUNT_OBSERVATION_600S_PROTOCOL
                       else "merlin.phase2.archive_spike_runtime.v2" if self.observation_resource_protocol is not None
                       else "merlin.phase2.archive_spike_runtime.v1"),
            "review_id": self.review_id,
            "functional_argv": list(self.functional_argv),
            "observer": str(self.observer),
            "disassembler": str(self.disassembler),
            "model": str(self.model_path),
            "source_manifest": str(self.source_manifest),
            "libraries": [str(p) for p in self.libraries],
            "dependencies": [str(p) for p in self.dependencies],
            "entry_symbol": self.entry_symbol,
            "completion_symbol": self.completion_symbol,
            "timing_selection": json.loads(self.timing_json),
            "files": [{"path": p, "sha256": d} for p, d in self.files],
            "scope": "count-only public kernel-call observation; native event timing unqualified",
            **({"observation_resources": {
                "protocol": self.observation_resource_protocol,
                "per_command_wall_seconds": (
                    900 if self.observation_resource_protocol == COUNT_OBSERVATION_900S_PROTOCOL else
                    600 if self.observation_resource_protocol == COUNT_OBSERVATION_600S_PROTOCOL else 120
                ),
                "parent_deadline_preserved": True,
            }} if self.observation_resource_protocol is not None else {}),
        }

    @property
    def sha256(self):
        return document_sha256(self.record())

    def verify(self):
        if type(self) is not ArchiveSpikeRuntime:
            raise StageGateError("archive screening requires the exact reviewed count runtime")
        if self.observation_resource_protocol is not None and (
            type(self.observation_resource_protocol) is not str
            or self.observation_resource_protocol not in (
                COUNT_OBSERVATION_120S_PROTOCOL, COUNT_OBSERVATION_600S_PROTOCOL, COUNT_OBSERVATION_900S_PROTOCOL
            )
        ):
            raise StageGateError("archive count runtime requires its explicit bounded observation resource protocol")
        for path, digest in self.files:
            if F.pin(Path(path), max_bytes=MAX_TOOL_BYTES)["sha256"] != digest:
                raise StageGateError("archive screening runtime source/tool/model changed")
        if canonical_json(json.loads(self.model_path.read_bytes())) != self.model_json:
            raise StageGateError("archive screening model changed")
        return self.record()

    def observation_limits(self, remaining_s):
        """Resource-only opt-in; the same cap brackets both count commands."""
        from merlin.perf.dynamic_census import WALL_600S_PROTOCOL, WALL_900S_PROTOCOL, CensusLimits

        self.verify()
        protocol = {COUNT_OBSERVATION_600S_PROTOCOL: WALL_600S_PROTOCOL,
                    COUNT_OBSERVATION_900S_PROTOCOL: WALL_900S_PROTOCOL}.get(self.observation_resource_protocol)
        limits = CensusLimits.for_timeout(remaining_s, wall_resource_protocol=protocol)
        if self.observation_resource_protocol == COUNT_OBSERVATION_120S_PROTOCOL:
            limits = replace(limits, wall_s=min(120.0, remaining_s))
            limits.verify()
        return limits

    @property
    def execution_files(self):
        """Only the frozen runtime leaves needed inside the child namespace."""
        return tuple(
            dict.fromkeys(
                (
                    Path(self.functional_argv[0]),
                    self.observer,
                    self.disassembler,
                    self.model_path,
                    *self.libraries,
                    *self.dependencies,
                )
            )
        )

    def commands(self, elf, report, scope):
        """Preserve the original ISA/memory/extension command, adding observation only."""
        self.verify()
        argv = list(self.functional_argv)
        original = [*argv, str(elf)]
        argv[0] = str(self.observer)
        last_library = max(i for i, token in enumerate(argv) if token.startswith("--extlib="))
        argv.insert(last_library + 1, "--extlib=" + str(self.libraries[-1]))
        argv.extend(
            [
                "--timing-model=" + str(self.model_path),
                "--timing-report=" + str(report),
                "--timing-scope-binding=" + str(scope),
                "--timing-input-sha256=" + json.loads(scope.read_bytes())["input_sha256"],
                str(elf),
            ]
        )
        return original, argv


def freeze_runtime(
    *,
    review_id,
    functional_argv,
    observer,
    disassembler,
    model_path,
    source_manifest,
    libraries,
    dependencies,
    entry_symbol,
    completion_symbol,
    timing_selection,
    observation_resource_protocol=None,
):
    """Review explicit runtime bytes; this supplies no hardware correspondence certificate."""
    if not isinstance(review_id, str) or not review_id.strip():
        raise StageGateError("archive count runtime needs explicit operator review attribution")
    if (
        type(functional_argv) is not tuple
        or not functional_argv
        or any(type(v) is not str or not v or "\0" in v for v in functional_argv)
        or any(v in {"-l", "--log-commits"} or v.startswith("--timing-") for v in functional_argv)
    ):
        raise StageGateError("archive count runtime needs the original nontracing functional command without an ELF")
    if any(not isinstance(p, Path) for p in (observer, disassembler, model_path, source_manifest)):
        raise StageGateError("archive runtime tool selections require explicit paths")
    if (
        type(libraries) is not tuple
        or len(libraries) < 2
        or type(dependencies) is not tuple
        or any(not isinstance(p, Path) for p in (*libraries, *dependencies))
    ):
        raise StageGateError("archive count runtime needs ordered functional libraries then one timing provider")
    model, model_pin = metadata(model_path)
    manifest, _ = metadata(source_manifest)
    expected_scope = E.kernel_call_scope(timing_selection)
    if "owned_model_scope" in timing_selection:
        from .owned_model_observation import scope_digest
        expected_scope = scope_digest(timing_selection)
        if entry_symbol != "merlin_run_multi" or completion_symbol != "return_after_synchronous_dispatch":
            raise StageGateError("owned runtime must select its real call and synchronous return contract")
    if (
        model.get("schema") != E.MODEL_SCHEMA
        or model.get("profile") != "phase1-public"
        or model.get("scope") != {"kind": "counter_pc_pair"}
        or model.get("domain", {}).get("feature_engine_sha256") != manifest.get("sha256")
        or model.get("domain", {}).get("scope_sha256") != expected_scope
    ):
        raise StageGateError("archive count runtime differs from reviewed public source/scope model")
    original_libraries = [v.partition("=")[2] for v in functional_argv if v.startswith("--extlib=")]
    if original_libraries != [str(p) for p in libraries[:-1]]:
        raise StageGateError("archive observer must preserve original ordered functional libraries")
    if [F.pin(p, max_bytes=MAX_TOOL_BYTES)["sha256"] for p in libraries] != model.get("libraries"):
        raise StageGateError("archive count runtime libraries differ from the selected model")
    if [v.partition("=")[2] for v in functional_argv if v.startswith("--isa=")] != [model["cpu"]["isa"]]:
        raise StageGateError("archive count runtime ISA differs from the selected model")
    for symbol in (entry_symbol, completion_symbol):
        if type(symbol) is not str or not symbol.isidentifier():
            raise StageGateError("archive count runtime requires reviewed plain entry/completion symbols")
    timing_pins = timing_selection.get("counter_semantics", {}).get("source_pins", [])
    if not timing_pins:
        raise StageGateError("archive counter selection lacks original source pins")
    for pin in timing_pins:
        if F.pin(Path(pin["path"]), max_bytes=MAX_TOOL_BYTES) != pin:
            raise StageGateError("archive counter source changed")
    paths = tuple(
        dict.fromkeys(
            (
                Path(functional_argv[0]),
                observer,
                disassembler,
                model_path,
                source_manifest,
                *libraries,
                *dependencies,
                *(Path(p["path"]) for p in timing_pins),
            )
        )
    )
    files = tuple((str(path), F.pin(path, max_bytes=MAX_TOOL_BYTES)["sha256"]) for path in paths)
    if dict(files)[str(model_path)] != model_pin["sha256"]:
        raise StageGateError("archive model changed during runtime selection")
    result = ArchiveSpikeRuntime(
        review_id,
        functional_argv,
        observer,
        disassembler,
        model_path,
        source_manifest,
        libraries,
        dependencies,
        entry_symbol,
        completion_symbol,
        canonical_json(timing_selection),
        files,
        canonical_json(model),
        observation_resource_protocol,
    )
    result.verify()
    return result
