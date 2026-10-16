"""Pin an explicit deployment declaration for the shared RoCC execution tools.

Only hashes may be omitted from the ordinary contract shape. No tools are run,
no hardware facts or software semantics are invented, and no review or native
qualification is granted. The prepared document remains operator-owned data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.paths import artifacts_dir

from . import chipyard_rocc_bound as backend


def _pin(row):
    backend._fields(row, {"path"}, {"sha256"})
    path = backend._path(row["path"])
    digest = backend._digest(path)
    if "sha256" in row:
        backend._sha(row["sha256"])
        backend._require(row["sha256"] == digest, "declared execution member changed")
    row["sha256"] = digest


def _resources(rows, roster):
    backend._require(type(rows) is list and rows, "deployment requires explicit core runtime IDs")
    for row in rows:
        backend._fields(row, {"id"}, {"sha256"})
        backend._require(
            type(row["id"]) is str and row["id"] in roster, "deployment includes an unsupported core runtime ID"
        )
        pin = {"path": str(backend._resource(row["id"], roster))}
        if "sha256" in row:
            pin["sha256"] = row["sha256"]
        _pin(pin)
        row["sha256"] = pin["sha256"]


def select_native_timeout(contract, timeout_s):
    """Derive only an explicit host process-resource variant of a contract."""
    backend._require(type(contract) is dict, "resource selection requires an original contract")
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    backend._require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "native resource selection requires a declared data-bound execution family",
    )
    block = runner.get("chipyard_rocc")
    backend._require(
        type(block) is dict and type(block.get("engines")) is dict,
        "native resource selection requires declared execution engines",
    )
    resources = dict(block.get("execution_resources", {}))
    resources["max_native_timeout_s"] = timeout_s
    backend._execution_timeout_ceiling({"execution_resources": resources})
    engines = {name: engine for name, engine in block["engines"].items() if name in {"gsim", "verilator"}}
    backend._require(bool(engines), "native resource selection requires a declared native engine")
    for engine in engines.values():
        backend._require(
            type(engine) is dict and "max_timeout_s" in engine,
            "native resource selection requires an existing engine deadline",
        )
        engine["max_timeout_s"] = timeout_s
    block["execution_resources"] = resources
    return selected


def select_model_spike_timeout(contract, timeout_s):
    """Select a bounded whole-model Spike process budget without changing instructions or numerics."""
    backend._require(type(contract) is dict, "model Spike resource selection requires an original contract")
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    backend._require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "model Spike resource selection requires a declared data-bound execution family",
    )
    block = runner.get("chipyard_rocc")
    backend._require(
        type(block) is dict and type(block.get("engines")) is dict
        and type(block["engines"].get("spike")) is dict,
        "model Spike resource selection requires a declared Spike engine",
    )
    resources = dict(block.get("execution_resources", {}))
    resources.setdefault("max_native_timeout_s", 600)
    resources["max_model_spike_timeout_s"] = timeout_s
    backend._model_spike_timeout_ceiling({"execution_resources": resources})
    block["engines"]["spike"]["max_timeout_s"] = timeout_s
    block["execution_resources"] = resources
    return selected


def select_gsim_readback(contract, transport):
    """Derive only the declared full-value gSIM transport, preserving semantics."""
    backend._require(type(contract) is dict, "readback selection requires an original contract")
    from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, COHERENT_PACKET_V1, COHERENT_PACKET_V2, FULL_VALUES_B64

    backend._require(
        type(transport) is str and transport in (COHERENT_DUMP_V1, COHERENT_PACKET_V1, COHERENT_PACKET_V2),
        "readback selection requires complete coherent values",
    )
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    backend._require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "readback selection requires a declared data-bound execution family",
    )
    block = runner.get("chipyard_rocc")
    backend._require(
        type(block) is dict and type(block.get("engines")) is dict,
        "readback selection requires declared execution engines",
    )
    engine = block["engines"].get("gsim")
    abi = selected.get("harness_abi")
    backend._require(
        type(engine) is dict and type(abi) is dict and abi.get("readback_transport") == FULL_VALUES_B64,
        "readback selection requires an original B64 logical ABI and declared gSIM engine",
    )
    backend._require(
        engine.get("readback_transport", transport) == transport,
        "readback selection cannot replace another engine transport",
    )
    if transport in (COHERENT_PACKET_V1, COHERENT_PACKET_V2):
        toolchain = block.get("toolchain")
        backend._require(type(toolchain) is dict, "packet selection requires the original declared toolchain")
        headers = toolchain.get("headers")
        backend._units(headers, backend._HEADERS)
        backend._require(
            {"htif", "out_b64"} <= {row["id"] for row in headers},
            "packet selection requires the original B64 console header roster",
        )
        for identifier in ("out_bin", "out_bin_memory", *(("out_packet_v2",) if transport == COHERENT_PACKET_V2 else ())):
            matches = [row for row in headers if row["id"] == identifier]
            if matches:
                _resources(matches, backend._HEADERS)
            else:
                row = {"id": identifier}
                _resources([row], backend._HEADERS)
                headers.append(row)
    engine["readback_transport"] = transport
    return selected


def select_operand_protocol(contract, selection):
    """Select a pinned detached observer without changing original semantic data."""
    backend._require(type(contract) is dict, "operand selection requires an original contract")
    backend._require(
        "rocc_operand_selection" not in contract and "rocc_operand_protocol" not in contract,
        "operand selection cannot replace an existing selected observer",
    )
    from merlin.targetgen.rocc.selected_operand_protocol import _json, _read_pin, bind_selected_observer
    from merlin.targetgen.rocc.semantics import BoundSemantics

    selected = copy.deepcopy(contract)
    if type(selection) is not dict or set(selection) != {"original_facts", "sidecar"}:
        raise ValueError("operand selection requires only original facts and a pinned sidecar")
    facts = _json(_read_pin(selection["original_facts"], 16 * 1024 * 1024), 16 * 1024 * 1024)
    original = BoundSemantics(target=contract.get("name"), contract=contract, facts=facts)
    bind_selected_observer(original, selection)
    selected["rocc_operand_selection"] = copy.deepcopy(selection)
    return selected


def prepare_contract(*, target, contract, facts):
    """Fill missing hashes and validate all selected members without execution.

    The caller must separately review the declarations and establish verified
    source/capture closure and native launch evidence. Existing pins are checked,
    never replaced, and the caller's input objects remain unchanged.
    """
    backend._require(type(contract) is dict, "deployment contract must be a mapping")
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    backend._require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "deployment must explicitly select the shared execution family",
    )
    block = runner.get("chipyard_rocc")
    backend._require(type(block) is dict, "deployment requires an explicit execution block")
    toolchain = block.get("toolchain")
    backend._require(type(toolchain) is dict, "deployment requires an explicit toolchain")
    for name in ("compiler", "link_script"):
        _pin(toolchain.get(name))
    _resources(toolchain.get("runtime_units"), backend._RUNTIME)
    _resources(toolchain.get("headers"), backend._HEADERS)
    engines = block.get("engines")
    backend._require(
        type(engines) is dict and engines and set(engines) <= {"spike", "gsim", "verilator"},
        "deployment requires a supported explicit engine roster",
    )
    for engine, declaration in engines.items():
        backend._require(type(declaration) is dict, "deployment engine must be a mapping")
        _pin(declaration.get("binary"))
        if engine in {"gsim", "verilator"}:
            _pin(declaration.get("receipt"))
    if "spike" in engines:
        extension = runner.get("spike_extension")
        backend._fields(extension, {"extension_name", "extlib"}, {"sha256"})
        pin = {"path": extension["extlib"]}
        if "sha256" in extension:
            pin["sha256"] = extension["sha256"]
        _pin(pin)
        extension["sha256"] = pin["sha256"]
    bound = backend.bind(target=target, contract=selected, facts=facts)
    bound.verify_execution_inputs()
    return selected


def _read_document(path):
    path = backend._path(str(path))
    backend._regular(path)
    backend._require(path.stat().st_size <= 8 * 1024 * 1024, "deployment input exceeds the bounded document limit")
    with path.open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    backend._require(len(raw) <= 8 * 1024 * 1024, "deployment input exceeds the bounded document limit")
    document = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
    backend._require(type(document) is dict, "deployment input must be a data mapping")
    return document, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = backend._path(str(args.output))
        backend._require(
            output.is_relative_to(artifacts_dir().resolve()), "prepared contract must be beneath the artifacts root"
        )
        backend._require(not output.exists(), "prepared contract destination already exists")
        contract, contract_digest = _read_document(args.contract)
        facts, facts_digest = _read_document(args.facts)
        prepared = prepare_contract(target=args.target, contract=contract, facts=facts)
        for path, expected in ((args.contract, contract_digest), (args.facts, facts_digest)):
            backend._require(backend._digest(path) == expected, "deployment input changed during preparation")
        raw = (json.dumps(prepared, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        backend._require(
            output.parent.resolve() == output.parent
            and not any(p.is_symlink() for p in (output.parent, *output.parent.parents)),
            "prepared contract destination is not canonical",
        )
        with output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"target": args.target, "sha256": hashlib.sha256(raw).hexdigest(), "native_executed": False}))
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
