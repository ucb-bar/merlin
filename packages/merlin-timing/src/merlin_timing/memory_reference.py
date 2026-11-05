"""Pinned memory-model selection and cycle-by-cycle adapter comparison.

This module runs no tools and supplies no timing approximation. Matching source
settings is preparation; matching traces establishes only those exact traces,
never full simulator equivalence or a calibrated whole-system model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from . import contracts as C

SELECTION = "merlin.memory_reference_selection.v1"
PREPARED = "merlin.memory_reference_preparation.v1"
TRACE = "merlin.memory_boundary_trace.v1"
AXI_FIELDS = {"data_bits", "address_bits", "id_bits", "max_read_bytes", "max_write_bytes",
              "max_flight", "id_reuse", "address_base", "address_size"}
REQUIRED_SOURCES = {"configuration", "timing_model", "dram_model", "llc_model", "runtime_settings", "build_receipt"}


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise C.ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_settings(text: str) -> dict[str, int]:
    """Read explicit FASED +mm_ settings; never infer absent hardware defaults."""
    result = {}
    for line in text.splitlines():
        for token in line.partition("#")[0].split():
            if not token.startswith("+mm_"):
                raise C.ContractError("memory runtime settings must contain only explicit +mm_ assignments")
            key, equal, value = token.partition("=")
            if not equal or not value.isascii() or not value.isdecimal() or key in result:
                raise C.ContractError("memory runtime settings require unique nonnegative integer assignments")
            if not key[4:] or not all(c.isascii() and (c.isalnum() or c == "_") for c in key[4:]):
                raise C.ContractError("invalid memory runtime setting name")
            result[key] = int(value)
    if not result:
        raise C.ContractError("explicit memory runtime settings are missing")
    return result


def prepare(selection: dict, *, source_root: str | Path) -> dict:
    """Verify selected files/settings without asserting deployment equivalence."""
    C.closed(selection, {"schema", "configuration", "axi", "sources"}, "memory reference selection")
    if selection["schema"] != SELECTION:
        raise C.ContractError("unsupported memory reference selection")
    C.text(selection["configuration"], "memory model configuration")
    axi = selection["axi"]
    C.closed(axi, AXI_FIELDS, "memory AXI contract")
    for key, value in axi.items():
        if key in {"max_flight", "id_reuse"} and value is None:
            continue
        _integer(value, key, minimum=0 if key == "address_base" else 1)
    if axi["data_bits"] % 8 or axi["address_base"] + axi["address_size"] > 1 << axi["address_bits"]:
        raise C.ContractError("memory AXI width or address range is inconsistent")
    sources = selection["sources"]
    if not isinstance(sources, dict) or not REQUIRED_SOURCES <= sources.keys():
        raise C.ContractError("memory reference omits configuration, model, runtime or original build sources")
    root, pins = Path(source_root).resolve(strict=True), {}
    for role, pin in sources.items():
        C.text(role, "source role")
        C.closed(pin, {"path", "sha256"}, "memory source pin")
        path = root / C.text(pin["path"], "source path")
        expected = C.sha(pin["sha256"], "memory source digest")
        actual = _file(path)
        if expected != actual:
            raise C.ContractError(f"memory source bytes changed: {role}")
        pins[role] = {"path": str(path.resolve()), "sha256": actual}
    settings = runtime_settings(Path(pins["runtime_settings"]["path"]).read_text())
    identity = {
        "configuration": selection["configuration"], "axi": axi,
        "settings": settings, "source_sha256s": {k: v["sha256"] for k, v in sorted(pins.items())},
    }
    return {
        "schema": PREPARED, "memory_reference_sha256": C.digest(identity),
        **identity, "sources": pins, "selection_sha256": C.digest(selection),
        "source_settings_verified": True, "native_adapter_qualified": False,
        "deployment_equivalence": "NOT_ESTABLISHED",
        "required_checks": [
            "Bind this source/configuration and all runtime settings to the deployed build.",
            "Run identical target-cycle stimuli through the reference and adapter.",
            "Compare AXI admission, response timing, ordering, payloads and backpressure.",
            "Qualify complete numerically checked calls separately from boundary traces.",
        ],
    }


def _ports(value):
    C.closed(value, {"inputs", "outputs"}, "memory trace ports")
    for direction, ports in value.items():
        if not isinstance(ports, dict) or not ports:
            raise C.ContractError("memory trace needs explicit input and output ports")
        for name, specification in ports.items():
            C.text(name, "port name")
            C.closed(specification, {"width", "valid_when"}, "port specification")
            _integer(specification["width"], "port width", minimum=1)
            if specification["width"] > 4096:
                raise C.ContractError("memory trace port width exceeds 4096 bits")
            qualifier = specification["valid_when"]
            if qualifier is not None and (
                direction != "outputs" or qualifier not in ports
                or ports[qualifier] != {"width": 1, "valid_when": None}
            ):
                raise C.ContractError("payload qualifier must name an unconditional one-bit output")


def _header(stream):
    line = stream.readline(1024 * 1024)
    if not line.endswith("\n"):
        raise C.ContractError("missing or oversized memory trace header")
    header = json.loads(line)
    C.closed(header, {"schema", "memory_reference_sha256", "initial_state_sha256", "scope_sha256",
                      "stimulus_sha256", "engine_sha256", "ports"}, "memory trace header")
    if header["schema"] != TRACE:
        raise C.ContractError("unsupported memory boundary trace")
    for key in header.keys() - {"schema", "ports"}:
        C.sha(header[key], key)
    _ports(header["ports"])
    return header


def _frames(stream, ports):
    index = 0
    while line := stream.readline(1024 * 1024):
        if not line.endswith("\n"):
            raise C.ContractError("oversized or incomplete memory trace frame")
        frame = json.loads(line)
        C.closed(frame, {"cycle", "inputs", "outputs"}, "memory trace frame")
        if type(frame["cycle"]) is not int or frame["cycle"] != index:
            raise C.ContractError("memory trace must record every target cycle starting at zero")
        for direction in ("inputs", "outputs"):
            C.closed(frame[direction], set(ports[direction]), "memory trace signal values")
            for name, value in frame[direction].items():
                _integer(value, name)
                if value >= 1 << ports[direction][name]["width"]:
                    raise C.ContractError("memory trace signal exceeds declared width")
        yield frame
        index += 1


def compare(reference_path: str | Path, candidate_path: str | Path, *, maximum_cycles=1_000_000) -> dict:
    """Compare target cycles, including backpressure; cap diagnostics, not errors."""
    _integer(maximum_cycles, "maximum cycles", minimum=1)
    before = [_file(reference_path), _file(candidate_path)]
    mismatches, samples, cycles = 0, [], 0
    with Path(reference_path).open() as left, Path(candidate_path).open() as right:
        a, b = _header(left), _header(right)
        if any(a[key] != b[key] for key in a if key != "engine_sha256"):
            raise C.ContractError("memory traces differ in model, initial state, scope, stimulus or ports")
        for x, y in itertools.zip_longest(_frames(left, a["ports"]), _frames(right, b["ports"])):
            if x is None or y is None:
                raise C.ContractError("memory traces contain different target-cycle counts")
            cycles += 1
            if cycles > maximum_cycles:
                raise C.ContractError("memory trace exceeds declared cycle budget")
            if x["inputs"] != y["inputs"]:
                raise C.ContractError("memory traces were driven by different inputs")
            for name, specification in a["ports"]["outputs"].items():
                qualifier = specification["valid_when"]
                meaningful = qualifier is None or x["outputs"][qualifier] or y["outputs"][qualifier]
                if meaningful and x["outputs"][name] != y["outputs"][name]:
                    mismatches += 1
                    if len(samples) < 20:
                        samples.append({"cycle": x["cycle"], "signal": name,
                                        "reference": x["outputs"][name], "candidate": y["outputs"][name]})
    if not cycles:
        raise C.ContractError("empty traces cannot establish equivalence")
    if before != [_file(reference_path), _file(candidate_path)]:
        raise C.ContractError("memory traces changed while being compared")
    return {
        "schema": "merlin.memory_boundary_comparison.v1",
        "memory_reference_sha256": a["memory_reference_sha256"],
        "initial_state_sha256": a["initial_state_sha256"], "scope_sha256": a["scope_sha256"],
        "stimulus_sha256": a["stimulus_sha256"],
        "reference_engine_sha256": a["engine_sha256"], "candidate_engine_sha256": b["engine_sha256"],
        "reference_trace_sha256": before[0], "candidate_trace_sha256": before[1],
        "target_cycles": cycles, "signal_mismatches": mismatches, "mismatch_samples": samples,
        "equivalent_for_trace": mismatches == 0, "whole_system_qualified": False,
    }
