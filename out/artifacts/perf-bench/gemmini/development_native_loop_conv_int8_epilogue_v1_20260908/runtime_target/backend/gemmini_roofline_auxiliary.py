"""Target-owned empty and joint-occupancy calibration paths for Gemmini.

No counter spelling or engine role is declared here.  The counter layout and its exhaustive boolean
partition come from the shipped header and elaborated CIRCT.  Resource roles are then identified by
actual differential RTL runs: direction-pure read/write/copy kernels identify the movement tokens,
and the remaining token exercised by a native compute probe identifies compute.  Any ambiguity is an
UNKNOWN result, never a guessed label.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from merlin.perf import hw_counters
from merlin.targetgen.rtl import mlc_bridge

from .gemmini_codegen import CodegenError


SCHEMA = "gemmini_roofline_auxiliary_v1"
_TARGET = "gemmini"
_COUNTER_ENV = ("MERLIN_HW_COUNTERS", "MERLIN_HW_COUNTER_UNIT", "MERLIN_CACHE_STATE")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _temporary_root() -> Path:
    from merlin.common.paths import artifacts_dir

    root = artifacts_dir() / "perf-bench" / _TARGET / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _exact_inputs(rtl_facts: Mapping[str, Any], rtl_facts_sha256: str) -> tuple[str, Path, str]:
    if not _is_sha256(rtl_facts_sha256):
        raise CodegenError("exact RTL-facts file SHA-256 is required")
    inputs = rtl_facts.get("inputs")
    recorded = inputs.get("core_hw_sha256") if isinstance(inputs, Mapping) else None
    circt_path = mlc_bridge.core_hw_mlir(_TARGET)
    if not _is_sha256(recorded) or circt_path is None or not Path(circt_path).is_file():
        raise CodegenError("RTL facts do not identify an available elaborated CIRCT artifact")
    resolved = Path(circt_path)
    actual = _sha256(resolved.read_bytes())
    if actual != recorded:
        raise CodegenError("RTL facts do not match the active elaborated CIRCT bytes")
    return rtl_facts_sha256, resolved, actual


def empty_command_buffer() -> dict[str, Any]:
    """Return the structurally empty compiler input used for the shared-cost baseline."""
    return {"tensors": {}, "commands": []}


@contextmanager
def _measurement_environment(protocol: str, *, counters: bool,
                             counter_unit: str | None = None) -> Iterator[dict[str, Any]]:
    from .gemmini_codegen_mlir import _measurement_c_fragments

    previous = {name: os.environ.get(name) for name in _COUNTER_ENV}
    try:
        if counters:
            os.environ["MERLIN_HW_COUNTERS"] = "1"
            if counter_unit is None:
                os.environ.pop("MERLIN_HW_COUNTER_UNIT", None)
            else:
                os.environ["MERLIN_HW_COUNTER_UNIT"] = counter_unit
        else:
            os.environ.pop("MERLIN_HW_COUNTERS", None)
            os.environ.pop("MERLIN_HW_COUNTER_UNIT", None)
        protocol_to_request: dict[str, str] = {}
        for requested in ("cold", "warm"):
            os.environ["MERLIN_CACHE_STATE"] = requested
            fragment = _measurement_c_fragments("")
            observed = fragment.get("cache_protocol")
            if isinstance(observed, str) and observed:
                protocol_to_request[observed] = requested
        if protocol not in protocol_to_request:
            raise CodegenError(f"measurement protocol {protocol!r} is not emitted by this harness")
        os.environ["MERLIN_CACHE_STATE"] = protocol_to_request[protocol]
        yield _measurement_c_fragments("")
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def probe_empty_workload(rtl_facts: Mapping[str, Any], *, rtl_facts_sha256: str,
                         stage: str) -> dict[str, Any]:
    """Probe the actual native compiler on a zero-command program."""
    from . import gemmini_codegen_mlir as codegen

    base: dict[str, Any] = {"schema": SCHEMA, "kind": "empty_workload", "stage": stage}
    if stage not in {"emission", "compile"}:
        return {**base, "status": "unknown", "why": "stage must be emission or compile"}
    try:
        facts_sha256, circt_path, circt_sha256 = _exact_inputs(
            rtl_facts, rtl_facts_sha256)
        command = empty_command_buffer()
        emitted, arguments = codegen.emit_kernel_mlir(command)
        if arguments:
            raise CodegenError("structurally empty compiler program unexpectedly has arguments")
        record = {
            **base, "status": "accepted", "rtl_facts_sha256": facts_sha256,
            "circt_core_hw": {"path": str(circt_path), "sha256": circt_sha256},
            "command_buffer": command, "command_buffer_sha256": _canonical_sha256(command),
            "emitted_mlir_sha256": _sha256(emitted.encode("utf-8")),
        }
        if stage == "compile":
            with tempfile.TemporaryDirectory(prefix="empty-probe-", dir=_temporary_root()) as root:
                obj = codegen.build_object(command, root)
                record["object_sha256"] = _sha256(Path(obj).read_bytes())
        record["artifact_sha256"] = _canonical_sha256(record)
        return record
    except Exception as exc:  # unavailable compiler evidence remains UNKNOWN
        return {**base, "status": "unknown", "why": f"{type(exc).__name__}: {exc}"}


def probe_joint_occupancy(rtl_facts: Mapping[str, Any], *,
                          rtl_facts_sha256: str,
                          measurement_protocol: str) -> dict[str, Any]:
    """Prove that the native harness can measure one complete CIRCT-backed partition."""
    from . import gemmini
    from .gemmini_codegen_mlir import _measurement_c_fragments

    base: dict[str, Any] = {"schema": SCHEMA, "kind": "joint_occupancy"}
    try:
        facts_sha256, circt_path, circt_sha256 = _exact_inputs(
            rtl_facts, rtl_facts_sha256)
        with _measurement_environment(measurement_protocol, counters=True):
            _measurement_c_fragments("")
        discovery = hw_counters.counters_for_target(_TARGET)
        if discovery.get("status") != "derived":
            raise CodegenError(str(discovery.get("why", "counter discovery is unavailable")))
        header_path = Path(discovery["header"])
        header_bytes = header_path.read_bytes()
        if _sha256(header_bytes) != discovery.get("header_sha256"):
            raise CodegenError("counter header changed after discovery")
        header = header_bytes.decode("utf-8")
        occupancy = hw_counters.derive_occupancy_counters(header)
        if not occupancy.complete():
            raise CodegenError("target header does not expose a complete occupancy partition")
        partition = gemmini.counter_partition_inputs()
        if partition.get("status") != "available":
            raise CodegenError(str(partition.get("why", "CIRCT partition input is unavailable")))
        codes = hw_counters.event_codes(header)
        proof = hw_counters.prove_occupancy_partition_from_circt(
            partition["hw_text"], occupancy, codes, module=partition["module"],
            counter_module=partition["counter_module"], source=partition["source"])
        if proof.get("status") != "proved" or proof.get("sha256") != circt_sha256:
            raise CodegenError(str(proof.get("why", "occupancy partition was not proved")))
        record = {
            **base, "status": "accepted", "rtl_facts_sha256": facts_sha256,
            "circt_core_hw": {"path": str(circt_path), "sha256": circt_sha256},
            "counter_header": {"path": str(header_path),
                               "sha256": discovery["header_sha256"]},
            "counter_layout": occupancy.to_dict(),
            "codes": {name: codes[name] for name in occupancy.by_combination.values()},
            "partition_proof": proof,
        }
        record["artifact_sha256"] = _canonical_sha256(record)
        return record
    except Exception as exc:  # no partial layout is promoted into a capability
        return {**base, "status": "unknown", "why": f"{type(exc).__name__}: {exc}"}


def _compute_command_buffer(rtl_facts: Mapping[str, Any], multiple: int) -> dict[str, Any]:
    from .calibration_capabilities import _compute_probe_buffer, _mesh_shape

    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple <= 0:
        raise CodegenError("compute tile multiple must be a positive integer")
    shape = _mesh_shape(rtl_facts)
    if shape is None:
        raise CodegenError("RTL facts do not identify one positive primary compute array")
    return _compute_probe_buffer(*shape, multiple)


def run_empty_workload(protocol: str, *, simulator: str = "verilator", timeout: int = 600,
                       workdir: str | Path | None = None) -> dict[str, Any]:
    """Execute the structurally empty program through the production compiler and RTL harness."""
    from . import gemmini_codegen_mlir as codegen

    command = empty_command_buffer()
    with _measurement_environment(protocol, counters=False):
        result = codegen.run_on_spike(command, workdir=workdir, simulator=simulator, timeout=timeout)
    cycles = result.get("metrics", {}).get("cycles")
    if (result.get("correct") is not True or isinstance(cycles, bool)
            or not isinstance(cycles, int) or cycles <= 0):
        raise CodegenError("empty compiler path did not return a correct positive RTL cycle window")
    elf = Path(result.get("elf", ""))
    if not elf.is_file():
        raise CodegenError("empty compiler path did not retain its exact ELF")
    return {**result, "cycles": cycles, "command_buffer": command,
            "command_buffer_sha256": _canonical_sha256(command),
            "elf_sha256": _sha256(elf.read_bytes())}


def run_compute_probe(rtl_facts: Mapping[str, Any], multiple: int, protocol: str, *,
                      simulator: str = "verilator", timeout: int = 600,
                      workdir: str | Path | None = None,
                      counter_unit: str | None = None) -> dict[str, Any]:
    """Execute one RTL-derived native compute probe, optionally with joint occupancy."""
    from . import gemmini_codegen_mlir as codegen

    command = _compute_command_buffer(rtl_facts, multiple)
    with _measurement_environment(protocol, counters=True, counter_unit=counter_unit):
        result = codegen.run_on_spike(command, workdir=workdir, simulator=simulator, timeout=timeout)
    cycles = result.get("metrics", {}).get("cycles")
    if (result.get("correct") is not True or isinstance(cycles, bool)
            or not isinstance(cycles, int) or cycles <= 0):
        raise CodegenError("compute probe did not return a correct positive RTL cycle window")
    if counter_unit is None:
        report = result.get("counters")
        occupancy = report.get("occupancy") if isinstance(report, Mapping) else None
        discovery = report.get("discovery") if isinstance(report, Mapping) else None
        overlap = report.get("overlap") if isinstance(report, Mapping) else None
        proof = overlap.get("partition_proof") if isinstance(overlap, Mapping) else None
        combinations = occupancy.get("by_combination") if isinstance(occupancy, Mapping) else None
        codes = discovery.get("event_codes") if isinstance(discovery, Mapping) else None
        selected_names = (set(combinations.values()) if isinstance(combinations, Mapping) else set())
        if (not selected_names or not isinstance(codes, Mapping)
                or not selected_names <= set(codes) or not isinstance(proof, Mapping)
                or proof.get("status") != "proved"
                or not all(isinstance(proof.get(key), str) and proof.get(key)
                           for key in ("module", "counter_module", "source"))):
            raise CodegenError(
                "compute occupancy report lacks a complete CIRCT-backed partition identity")
        normalized = dict(report)
        normalized["event_codes"] = {name: codes[name] for name in selected_names}
        normalized["partition"] = {
            key: proof[key] for key in ("module", "counter_module", "source")}
        result = dict(result, counters=normalized)
    elf = Path(result.get("elf", ""))
    if not elf.is_file():
        raise CodegenError("compute probe did not retain its exact ELF")
    return {**result, "cycles": cycles, "command_buffer": command,
            "command_buffer_sha256": _canonical_sha256(command),
            "elf_sha256": _sha256(elf.read_bytes())}


def _active_engines(result: Mapping[str, Any], expected_layout: Mapping[str, Any]) -> set[str]:
    report = result.get("counters")
    layout = report.get("occupancy") if isinstance(report, Mapping) else None
    readings = report.get("readings") if isinstance(report, Mapping) else None
    if layout != expected_layout or not isinstance(readings, Mapping):
        raise CodegenError("differential probe did not retain the exact common occupancy layout")
    combinations = layout.get("by_combination")
    engines = layout.get("engines")
    if not isinstance(combinations, Mapping) or not isinstance(engines, list):
        raise CodegenError("differential occupancy layout is incomplete")
    expected_names = set(combinations.values())
    cycles = result.get("cycles")
    if (set(readings) != expected_names or isinstance(cycles, bool) or not isinstance(cycles, int)
            or cycles <= 0 or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                                 for value in readings.values())
            or sum(readings.values()) > cycles):
        raise CodegenError("differential readings do not fit their own positive cycle window")
    active: set[str] = set()
    for combination, counter_name in combinations.items():
        if readings[counter_name] <= 0:
            continue
        tokens = combination.split("+") if isinstance(combination, str) else []
        if not tokens or any(token not in engines for token in tokens):
            raise CodegenError("occupancy combination is not over the declared engine set")
        active.update(tokens)
    return active


def derive_resource_kinds(probes: Mapping[str, Mapping[str, Any]], *,
                          rtl_facts_sha256: str, circt_hw_sha256: str) -> dict[str, Any]:
    """Derive ResourceKind strings from differential activity, never counter spellings."""
    base: dict[str, Any] = {"schema": SCHEMA, "kind": "differential_resource_roles",
                            "rtl_facts_sha256": rtl_facts_sha256,
                            "circt_hw_sha256": circt_hw_sha256}
    required = ("read", "write", "copy", "compute")
    if set(probes) != set(required) or not _is_sha256(rtl_facts_sha256) \
            or not _is_sha256(circt_hw_sha256):
        return {**base, "status": "unknown",
                "why": "requires all four probes and exact RTL/CIRCT identities"}
    try:
        first_report = probes[required[0]].get("counters")
        layout = first_report.get("occupancy") if isinstance(first_report, Mapping) else None
        if not isinstance(layout, Mapping):
            raise CodegenError("read probe has no derived occupancy layout")
        support = {name: _active_engines(probes[name], layout) for name in required}
        read, write, copy, compute = (support[name] for name in required)
        engines = set(layout.get("engines", []))
        movement = read | write
        remaining = engines - movement
        if (len(read) != 1 or len(write) != 1 or read == write or copy != movement
                or len(remaining) != 1 or compute != engines):
            raise CodegenError(
                "pure DMA/compute activity does not uniquely identify movement and compute tokens")
        kinds = {engine: ("movement" if engine in movement else "compute")
                 for engine in sorted(engines)}
        evidence = {name: {
            "cycles": probes[name].get("cycles"),
            "command_buffer_sha256": probes[name].get("command_buffer_sha256"),
            "elf_sha256": probes[name].get("elf_sha256"),
            "emitter_sha256": _canonical_sha256(probes[name].get("emitter")),
            "counter_readings_sha256": _canonical_sha256(
                probes[name].get("counters", {}).get("readings")),
            "active_engines": sorted(support[name]),
        } for name in required}
        record = {**base, "status": "proved", "counter_layout": dict(layout),
                  "kinds": kinds, "probe_evidence": evidence,
                  "method": "direction_pure_dma_plus_native_compute_differential_v1"}
        record["artifact_sha256"] = _canonical_sha256(record)
        return record
    except Exception as exc:
        return {**base, "status": "unknown", "why": f"{type(exc).__name__}: {exc}"}


def run_joint_occupancy_probe(rtl_facts: Mapping[str, Any], *, protocol: str,
                              payload_bytes: int, compute_multiple: int,
                              rtl_facts_sha256: str,
                              simulator: str = "verilator", timeout: int = 600,
                              workdir: str | Path | None = None) -> dict[str, Any]:
    """Run the four differential probes and return a content-linked composition candidate."""
    from . import gemmini_dma_calibration as dma

    facts_sha256, circt_path, circt_sha256 = _exact_inputs(
        rtl_facts, rtl_facts_sha256)
    root = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(
        prefix="joint-occupancy-", dir=_temporary_root()))
    root.mkdir(parents=True, exist_ok=True)
    probes = {
        direction: dma.run_dma_calibration(
            direction, payload_bytes, rtl_facts, protocol=protocol, simulator=simulator,
            timeout=timeout, workdir=root / direction, counter_unit=None)
        for direction in ("read", "write", "copy")
    }
    probes["compute"] = run_compute_probe(
        rtl_facts, compute_multiple, protocol, simulator=simulator, timeout=timeout,
        workdir=root / "compute", counter_unit=None)
    roles = derive_resource_kinds(
        probes, rtl_facts_sha256=facts_sha256, circt_hw_sha256=circt_sha256)
    return {
        "status": "measured" if roles.get("status") == "proved" else "unknown",
        "rtl_facts_sha256": facts_sha256,
        "circt_core_hw": {"path": str(circt_path), "sha256": circt_sha256},
        "measurement_protocol": protocol,
        "probes": probes,
        "resource_role_binding": roles,
        "composition_measurement": probes["compute"],
        **({} if roles.get("status") == "proved" else {"why": roles.get("why")}),
    }
