"""Contract-selected execution of prebuilt RoCC simulators and logical LLVM kernels.

This module contains no target compiler or compute implementation. Runtime units
are a closed set of core console/startup primitives; simulator and software ABI
choices come from the selected contract. File/transcript checks establish local
identity, not emitter correspondence, hardware timing or runtime qualification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import artifacts_dir, checkout_root, python_source_dir
from merlin.target.contract.harness.readback_policy import (
    COHERENT_DUMP_V1,
    COHERENT_PACKET_V1,
    COHERENT_PACKET_V2,
    FULL_VALUES_B64,
)

# ---- chipyard_rocc_bound -----------------------------------------------------------------------

_RUNTIME = {
    "startup": "spike/crt.S",
    "htif_console": "spike/htif.c",
    "printf": "spike/printf_min.c",
    "libc": "spike/libc_min.c",
}
_HEADERS = {
    "htif": "spike/htif.h",
    "out_b64": "out_b64.h",
    "out_bin": "out_bin.h",
    "out_bin_memory": "out_bin_memory.h",
    "out_packet_v2": "out_packet_v2.h",
}
_CFLAGS = frozenset(
    {
        "-O0",
        "-O1",
        "-O2",
        "-O3",
        "-Os",
        "-Og",
        "-g",
        "-ffreestanding",
        "-nostdlib",
        "-nostartfiles",
        "-fno-builtin",
        "-fno-tree-vectorize",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-tree-loop-distribute-patterns",
        "-ffunction-sections",
        "-fdata-sections",
        "-fno-pie",
        "-fno-pic",
        "-fno-stack-protector",
        "-mno-relax",
        "-std=c11",
        "-std=gnu11",
    }
)
_LDFLAGS = frozenset({"-Wl,--gc-sections", "-Wl,--no-relax", "-Wl,--build-id=none", "-no-pie"})
_READOUT = {"scalar_abi", "epilogue_capability", "operand_sum", "stage_routes"}
_SCOPE = "selected process and complete console only; hardware, timing and runtime qualification unestablished"


class RoCCExecutionError(ValueError):
    """A selected execution input is missing, unsupported or changed."""


def _require(condition, message):
    if not condition:
        raise RoCCExecutionError(message)


def _fields(value, required, optional=()):
    _require(
        type(value) is dict and set(required) <= set(value) <= set(required) | set(optional),
        "execution selection has unsupported or missing fields",
    )


def _positive(value):
    _require(type(value) is int and 0 < value < 2**63, "execution budget requires a positive bounded integer")


def _path(value):
    _require(
        type(value) is str and "\0" not in value and Path(value).is_absolute() and ".." not in Path(value).parts,
        "execution selection requires an explicit absolute path",
    )
    return Path(value)


def _sha(value):
    _require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "execution selection requires an exact SHA256 digest",
    )


def _pin_shape(value):
    _fields(value, {"path", "sha256"})
    _path(value["path"])
    _sha(value["sha256"])


def _regular(path):
    try:
        okay = (
            not any(p.is_symlink() for p in (path, *path.parents))
            and path.resolve() == path
            and stat.S_ISREG(path.stat().st_mode)
        )
    except OSError:
        okay = False
    _require(okay, "selected execution member is not a canonical regular file")


def _digest(path):
    _regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_pin(pin, *, executable=False):
    path = _path(pin["path"])
    _require(_digest(path) == pin["sha256"], "selected execution member changed")
    _require(not executable or os.access(path, os.X_OK), "selected execution tool is not executable")
    return path


def _resource(identifier, roster):
    # Do not honor runtime/repo overrides that could turn an ID into vendor code.
    owner = checkout_root()
    root = owner / "src/merlin/runtime/native" if owner is not None else python_source_dir() / "merlin/_data/runtime"
    return root / "baremetal" / roster[identifier]


def _units(rows, roster):
    _require(type(rows) is list and rows, "execution recipe requires explicit core runtime selections")
    names = []
    for row in rows:
        _fields(row, {"id", "sha256"})
        _require(
            type(row["id"]) is str and row["id"] in roster, "execution recipe includes an unsupported runtime unit"
        )
        _sha(row["sha256"])
        names.append(row["id"])
    _require(len(set(names)) == len(names), "execution recipe repeats a runtime unit")


def _strings(value):
    _require(
        type(value) is list and all(type(x) is str and x and "\0" not in x for x in value),
        "execution selection requires exact string arguments",
    )


def _validate_toolchain(block, abi):
    _fields(
        block,
        {
            "compiler",
            "runtime_units",
            "headers",
            "link_script",
            "load_address",
            "cflags",
            "ldflags",
            "kernel_stack_frame",
        },
    )
    _pin_shape(block["compiler"])
    _pin_shape(block["link_script"])
    _positive(block["load_address"])
    _units(block["runtime_units"], _RUNTIME)
    _units(block["headers"], _HEADERS)
    _require(
        {"startup", "htif_console"} <= {r["id"] for r in block["runtime_units"]}
        and "htif" in {r["id"] for r in block["headers"]},
        "execution recipe omits the selected startup or console ABI",
    )
    required_headers = (
        {"htif", "out_b64"}
        if abi.readback_policy.transport == FULL_VALUES_B64
        else {"htif", "out_bin", "out_bin_memory"}
    )
    _require(
        required_headers <= {row["id"] for row in block["headers"]},
        "execution recipe omits its complete selected readback header roster",
    )
    _strings(block["cflags"])
    _strings(block["ldflags"])
    flags = block["cflags"]
    for prefix in ("-march=", "-mabi="):
        _require(
            sum(flag.startswith(prefix) and len(flag) > len(prefix) for flag in flags) == 1,
            "execution recipe requires one explicit ISA and ABI selection",
        )
    for prefix in ("--target=", "-fuse-ld="):
        _require(
            sum(flag.startswith(prefix) for flag in flags) <= 1,
            "execution recipe contains ambiguous compiler selectors",
        )
    for flag in flags:
        _require(
            flag in _CFLAGS
            or any(
                flag.startswith(prefix)
                and flag[len(prefix) :].isascii()
                and all(c.isalnum() or c in "_-." for c in flag[len(prefix) :])
                and flag[len(prefix) :]
                for prefix in ("-march=", "-mabi=", "-mcmodel=", "--target=", "-fuse-ld=")
            ),
            "execution recipe contains an unsupported compiler option",
        )
    _require(
        all(flag in _LDFLAGS for flag in block["ldflags"]), "execution recipe contains an unsupported linker option"
    )
    stack = block["kernel_stack_frame"]
    _fields(stack, {"entry_symbol", "max_static_bytes"})
    _require(stack["entry_symbol"] == abi.pointer_abi.entry_symbol, "execution recipe and harness entry differ")
    _positive(stack["max_static_bytes"])


def _execution_timeout_ceiling(block):
    """Only reviewed host selection can extend the legacy process ceiling."""
    if "execution_resources" not in block:
        return 600
    resources = block["execution_resources"]
    _fields(resources, {"max_native_timeout_s"}, {"max_model_spike_timeout_s"})
    ceiling = resources["max_native_timeout_s"]
    _require(
        type(ceiling) is int and 0 < ceiling <= 86400,
        "selected native process ceiling must be an integer within 1..86400 seconds",
    )
    return ceiling


def _model_spike_timeout_ceiling(block):
    """A separately selected process bound for whole-model functional Spike runs."""
    _execution_timeout_ceiling(block)
    resources = block.get("execution_resources", {})
    ceiling = resources.get("max_model_spike_timeout_s", 600)
    _require(
        type(ceiling) is int and 0 < ceiling <= 172800,
        "selected model Spike process ceiling must be an integer within 1..172800 seconds",
    )
    return ceiling


def _validate_engine(engine, block, *, timeout_ceiling=600, spike_timeout_ceiling=600):
    required = {"binary", "argv", "environment", "cwd", "max_timeout_s", "max_console_bytes", "failure_markers"}
    _fields(block, required | ({"receipt"} if engine in {"gsim", "verilator"} else set()), {"readback_transport"})
    if "readback_transport" in block:
        _require(
            engine == "gsim"
            and block["readback_transport"] in {COHERENT_DUMP_V1, COHERENT_PACKET_V1, COHERENT_PACKET_V2},
            "selected readback override requires the coherent GSim full-value transport",
        )
    _pin_shape(block["binary"])
    if engine in {"gsim", "verilator"}:
        _pin_shape(block["receipt"])
    _path(block["cwd"])
    _strings(block["argv"])
    _require(
        block["argv"].count("{elf}") == 1
        and all(token == "{elf}" or "{" not in token and "}" not in token for token in block["argv"]),
        "execution command requires exactly one ELF operand",
    )
    _require(
        type(block["environment"]) is dict
        and all(
            type(k) is str and k and "=" not in k and "\0" not in k and type(v) is str and "\0" not in v
            for k, v in block["environment"].items()
        ),
        "execution requires an explicit process environment",
    )
    _positive(block["max_timeout_s"])
    _require(
        block["max_timeout_s"] <= (spike_timeout_ceiling if engine == "spike" else timeout_ceiling),
        "execution deadline exceeds the bounded process limit",
    )
    _positive(block["max_console_bytes"])
    _strings(block["failure_markers"])
    if engine == "spike":
        _require(
            any(token.startswith("--isa=") and len(token) > 6 for token in block["argv"])
            and not any(token.startswith(("--extension", "--extlib")) for token in block["argv"]),
            "functional execution requires an explicit ISA and a single selected extension",
        )


def _validate_readout(block):
    _fields(block, (), _READOUT)
    for key in ("scalar_abi", "operand_sum"):
        if key in block:
            _require(block[key] is None or type(block[key]) is dict, "readout observation is not a data mapping")
    for key in ("epilogue_capability", "stage_routes"):
        if key in block:
            _require(
                block[key] is None or type(block[key]) is list and all(type(r) is dict for r in block[key]),
                "readout observation is not a complete data roster",
            )


#: Simulated cycles an operator-budgeted gSIM run may reach in this process tree. A measurement whose
#: wall budget outlasts the selected command's ``+max-cycles`` hang bound sets it (derived from that
#: budget and the engine's observed rate), so the cap does not end the run before the budget does. It
#: only raises the selected cap; the selected wall deadline still bounds the process.
GSIM_BUDGET_MAX_CYCLES_ENV = "MERLIN_GSIM_BUDGET_MAX_CYCLES"


def _gsim_budget_max_cycles():
    raw = os.environ.get(GSIM_BUDGET_MAX_CYCLES_ENV, "").strip()
    if not raw:
        return None
    _require(raw.isdigit() and 0 < int(raw) <= (1 << 63) - 1, "budgeted gSIM cycle bound must be a positive integer")
    return int(raw)


def _gsim_startup_transport():
    selected = os.environ.get("MERLIN_GSIM_STARTUP_PRELOAD")
    _require(selected in {None, "", "0", "1"}, "selected gSIM startup preload must be 0 or 1")
    return "original_dram_elf_preload_v1" if selected == "1" else None


def _bounded_process(argv, *, cwd, env, timeout, max_bytes, observed):
    """Keep complete stdout frames separate from bounded process diagnostics."""
    process = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    )
    data = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as poll:
            poll.register(process.stdout, selectors.EVENT_READ, "stdout")
            poll.register(process.stderr, selectors.EVENT_READ, "stderr")
            while poll.get_map():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise subprocess.TimeoutExpired(
                        argv, timeout, output=bytes(data["stdout"]), stderr=bytes(data["stderr"])
                    )
                for key, _ in poll.select(min(left, 0.1)):
                    total = len(data["stdout"]) + len(data["stderr"])
                    chunk = os.read(key.fileobj.fileno(), min(65536, max_bytes + 1 - total))
                    if not chunk:
                        poll.unregister(key.fileobj)
                    else:
                        data[key.data].extend(chunk)
                        _require(
                            len(data["stdout"]) + len(data["stderr"]) <= max_bytes,
                            "execution console exceeds its selected byte budget",
                        )
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        observed.complete(
            subprocess.CompletedProcess(
                argv, process.returncode, stdout=bytes(data["stdout"]), stderr=bytes(data["stderr"])
            )
        )
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    result = subprocess.CompletedProcess(
        argv, process.returncode, stdout=bytes(data["stdout"]), stderr=bytes(data["stderr"])
    )
    observed.complete(result)
    return result


def _execution_binding(facts, config):
    from merlin.target.rtl.extraction.unit_correspondence import ExecutionBindingError, execution_binding

    try:
        return execution_binding(facts, config=config)
    except ExecutionBindingError as exc:
        raise RoCCExecutionError(
            "selected contract or original facts identify another target or configuration"
        ) from exc


def bind(*, target, contract, facts):
    """Bind already selected data once; no plugin, discovery, binary check or build."""
    from merlin.runtime.backends.rocc.bound_backend import BoundRoCCBackend

    _require(
        type(target) is str and target and type(contract) is dict and type(facts) is dict and facts,
        "RoCC execution requires an explicit contract and original facts",
    )
    try:
        snapshot = json.dumps(
            {"contract": contract, "facts": facts}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise RoCCExecutionError("execution selection is not finite serializable data") from exc
    selected = json.loads(snapshot)
    contract, facts = selected["contract"], selected["facts"]
    runner = contract.get("runner")
    _require(type(runner) is dict and runner.get("backend") == "chipyard_rocc", "RoCC execution family is not selected")
    _require(not contract.get("plugin"), "data-bound runtime selection cannot grant executable provider hooks")
    block = runner.get("chipyard_rocc")
    _fields(block, {"version", "config", "toolchain", "engines"}, {"readout", "execution_resources"})
    _require(
        type(block["version"]) is int and block["version"] == 1 and type(block["config"]) is str and block["config"],
        "RoCC execution requires a supported version and explicit configuration",
    )
    runtime = contract.get("runtime")
    if runtime is not None:
        _require(type(runtime) is dict, "selected runtime declarations must be a mapping")
        if "rtl_sim_config" in runtime:
            _require(
                runtime["rtl_sim_config"] == block["config"], "selected runtime and execution configurations differ"
            )
    inputs = facts.get("inputs")
    body = facts.get("facts")
    consistency = facts.get("source_consistency")
    _require(
        type(inputs) is dict and type(body) is dict and type(body.get("source")) is dict and type(consistency) is dict,
        "selected original facts omit their source configuration binding",
    )
    if "source" in facts:
        _require(type(facts["source"]) is dict, "selected original facts have an ambiguous source configuration")
    _require(
        contract.get("name") == target and inputs.get("target") == target,
        "selected contract or original facts identify another target or configuration",
    )
    # Facts of the runner's own configuration bind directly. Facts of another configuration bind
    # only through their verified simulation correspondence to this runner's configuration; the
    # engine receipt must then carry that simulation's FIRRTL (checked with the engine identity).
    binding = _execution_binding(facts, block["config"])
    _sha(inputs.get("fir_sha256"))
    _sha(binding["firrtl"]["sha256"])
    from merlin.runtime.harness.pointer_storage import validate_contract
    from merlin.target.isa.rocc.semantics import bind as bind_semantics

    abi = validate_contract(contract)
    _require(abi.readback_policy.transport == FULL_VALUES_B64, "selected execution has no coherent readback transport")
    _validate_toolchain(block["toolchain"], abi)
    _require(
        type(block["engines"]) is dict and block["engines"] and set(block["engines"]) <= {"spike", "gsim", "verilator"},
        "RoCC execution requires explicitly supported engines",
    )
    timeout_ceiling = _execution_timeout_ceiling(block)
    spike_timeout_ceiling = _model_spike_timeout_ceiling(block)
    for engine, engine_block in block["engines"].items():
        _validate_engine(
            engine, engine_block, timeout_ceiling=timeout_ceiling, spike_timeout_ceiling=spike_timeout_ceiling
        )
    if "spike" in block["engines"]:
        extension = runner.get("spike_extension")
        _fields(extension, {"extension_name", "extlib", "sha256"})
        _require(
            type(extension["extension_name"]) is str
            and extension["extension_name"]
            and all(c.isalnum() or c == "_" for c in extension["extension_name"]),
            "functional execution requires a plain selected extension name",
        )
        _path(extension["extlib"])
        _sha(extension["sha256"])
    if "readout" in block:
        _validate_readout(block["readout"])
    semantics = bind_semantics(target=target, contract=contract, facts=facts)
    return BoundRoCCBackend(target, snapshot, semantics)


# ---- rocc_selection ----------------------------------------------------------------------------
# Pin an explicit deployment declaration for the shared RoCC execution tools.
#
# Only hashes may be omitted from the ordinary contract shape. No tools are run,
# no hardware facts or software semantics are invented, and no review or native
# qualification is granted. The prepared document remains operator-owned data.

"""Pin an explicit deployment declaration for the shared RoCC execution tools.

Only hashes may be omitted from the ordinary contract shape. No tools are run,
no hardware facts or software semantics are invented, and no review or native
qualification is granted. The prepared document remains operator-owned data.
"""


def _pin(row):
    _fields(row, {"path"}, {"sha256"})
    path = _path(row["path"])
    digest = _digest(path)
    if "sha256" in row:
        _sha(row["sha256"])
        _require(row["sha256"] == digest, "declared execution member changed")
    row["sha256"] = digest


def _resources(rows, roster):
    _require(type(rows) is list and rows, "deployment requires explicit core runtime IDs")
    for row in rows:
        _fields(row, {"id"}, {"sha256"})
        _require(type(row["id"]) is str and row["id"] in roster, "deployment includes an unsupported core runtime ID")
        pin = {"path": str(_resource(row["id"], roster))}
        if "sha256" in row:
            pin["sha256"] = row["sha256"]
        _pin(pin)
        row["sha256"] = pin["sha256"]


def select_native_timeout(contract, timeout_s):
    """Derive only an explicit host process-resource variant of a contract."""
    _require(type(contract) is dict, "resource selection requires an original contract")
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    _require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "native resource selection requires a declared data-bound execution family",
    )
    block = runner.get("chipyard_rocc")
    _require(
        type(block) is dict and type(block.get("engines")) is dict,
        "native resource selection requires declared execution engines",
    )
    resources = dict(block.get("execution_resources", {}))
    resources["max_native_timeout_s"] = timeout_s
    _execution_timeout_ceiling({"execution_resources": resources})
    engines = {name: engine for name, engine in block["engines"].items() if name in {"gsim", "verilator"}}
    _require(bool(engines), "native resource selection requires a declared native engine")
    for engine in engines.values():
        _require(
            type(engine) is dict and "max_timeout_s" in engine,
            "native resource selection requires an existing engine deadline",
        )
        engine["max_timeout_s"] = timeout_s
    block["execution_resources"] = resources
    return selected


def select_model_spike_timeout(contract, timeout_s):
    """Select a bounded whole-model Spike process budget without changing instructions or numerics."""
    _require(type(contract) is dict, "model Spike resource selection requires an original contract")
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    _require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "model Spike resource selection requires a declared data-bound execution family",
    )
    block = runner.get("chipyard_rocc")
    _require(
        type(block) is dict and type(block.get("engines")) is dict and type(block["engines"].get("spike")) is dict,
        "model Spike resource selection requires a declared Spike engine",
    )
    resources = dict(block.get("execution_resources", {}))
    resources.setdefault("max_native_timeout_s", 600)
    resources["max_model_spike_timeout_s"] = timeout_s
    _model_spike_timeout_ceiling({"execution_resources": resources})
    block["engines"]["spike"]["max_timeout_s"] = timeout_s
    block["execution_resources"] = resources
    return selected


def select_gsim_readback(contract, transport):
    """Derive only the declared full-value gSIM transport, preserving semantics."""
    _require(type(contract) is dict, "readback selection requires an original contract")
    from merlin.target.contract.harness.readback_policy import (
        COHERENT_DUMP_V1,
        COHERENT_PACKET_V1,
        COHERENT_PACKET_V2,
        FULL_VALUES_B64,
    )

    _require(
        type(transport) is str and transport in (COHERENT_DUMP_V1, COHERENT_PACKET_V1, COHERENT_PACKET_V2),
        "readback selection requires complete coherent values",
    )
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    _require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "readback selection requires a declared data-bound execution family",
    )
    block = runner.get("chipyard_rocc")
    _require(
        type(block) is dict and type(block.get("engines")) is dict,
        "readback selection requires declared execution engines",
    )
    engine = block["engines"].get("gsim")
    abi = selected.get("harness_abi")
    _require(
        type(engine) is dict and type(abi) is dict and abi.get("readback_transport") == FULL_VALUES_B64,
        "readback selection requires an original B64 logical ABI and declared gSIM engine",
    )
    _require(
        engine.get("readback_transport", transport) == transport,
        "readback selection cannot replace another engine transport",
    )
    if transport in (COHERENT_PACKET_V1, COHERENT_PACKET_V2):
        toolchain = block.get("toolchain")
        _require(type(toolchain) is dict, "packet selection requires the original declared toolchain")
        headers = toolchain.get("headers")
        _units(headers, _HEADERS)
        _require(
            {"htif", "out_b64"} <= {row["id"] for row in headers},
            "packet selection requires the original B64 console header roster",
        )
        identifiers = ("out_bin", "out_bin_memory", *(("out_packet_v2",) if transport == COHERENT_PACKET_V2 else ()))
        for identifier in identifiers:
            matches = [row for row in headers if row["id"] == identifier]
            if matches:
                _resources(matches, _HEADERS)
            else:
                row = {"id": identifier}
                _resources([row], _HEADERS)
                headers.append(row)
    engine["readback_transport"] = transport
    return selected


def select_operand_protocol(contract, selection):
    """Select a pinned detached observer without changing original semantic data."""
    _require(type(contract) is dict, "operand selection requires an original contract")
    _require(
        "rocc_operand_selection" not in contract and "rocc_operand_protocol" not in contract,
        "operand selection cannot replace an existing selected observer",
    )
    from merlin.target.isa.rocc.operand_protocol import _json, _read_pin, bind_selected_observer
    from merlin.target.isa.rocc.semantics import BoundSemantics

    selected = copy.deepcopy(contract)
    if type(selection) is not dict or set(selection) != {"original_facts", "sidecar"}:
        raise ValueError("operand selection requires only original facts and a pinned sidecar")
    facts = _json(_read_pin(selection["original_facts"], 16 * 1024 * 1024), 16 * 1024 * 1024)
    original = BoundSemantics(target=contract.get("name"), contract=contract, facts=facts)
    bind_selected_observer(original, selection)
    selected["rocc_operand_selection"] = copy.deepcopy(selection)
    return selected


def relocate_operand_protocol(contract, selection):
    """Relocate an existing observer only to byte-identical facts and sidecar files."""
    from merlin.target.isa.rocc.operand_protocol import _json, _read_pin, bind_selected_observer
    from merlin.target.isa.rocc.semantics import BoundSemantics

    if type(contract) is not dict or "rocc_operand_protocol" in contract:
        raise ValueError("observer relocation requires an unchanged selected contract")
    previous = contract.get("rocc_operand_selection")
    if any(type(row) is not dict or set(row) != {"original_facts", "sidecar"} for row in (previous, selection)):
        raise ValueError("observer relocation requires the complete original and relocated selections")
    originals = {
        key: _read_pin(previous[key], limit)
        for key, limit in (("original_facts", 16 * 1024 * 1024), ("sidecar", 2 * 1024 * 1024))
    }
    for key, raw in originals.items():
        if _read_pin(selection[key], 16 * 1024 * 1024 if key == "original_facts" else 2 * 1024 * 1024) != raw:
            raise ValueError("observer relocation cannot change facts or sidecar bytes")
    original = BoundSemantics(
        target=contract.get("name"), contract=contract, facts=_json(originals["original_facts"], 16 * 1024 * 1024)
    )
    bind_selected_observer(original, selection)
    for key, raw in originals.items():
        if _read_pin(previous[key], 16 * 1024 * 1024 if key == "original_facts" else 2 * 1024 * 1024) != raw:
            raise ValueError("original operand selection changed during relocation")
    selected = copy.deepcopy(contract)
    selected["rocc_operand_selection"] = copy.deepcopy(selection)
    return selected


def prepare_contract(*, target, contract, facts):
    """Fill missing hashes and validate all selected members without execution.

    The caller must separately review the declarations and establish verified
    source/capture closure and native launch evidence. Existing pins are checked,
    never replaced, and the caller's input objects remain unchanged.
    """
    _require(type(contract) is dict, "deployment contract must be a mapping")
    selected = copy.deepcopy(contract)
    runner = selected.get("runner")
    _require(
        type(runner) is dict and runner.get("backend") == "chipyard_rocc",
        "deployment must explicitly select the shared execution family",
    )
    block = runner.get("chipyard_rocc")
    _require(type(block) is dict, "deployment requires an explicit execution block")
    toolchain = block.get("toolchain")
    _require(type(toolchain) is dict, "deployment requires an explicit toolchain")
    for name in ("compiler", "link_script"):
        _pin(toolchain.get(name))
    _resources(toolchain.get("runtime_units"), _RUNTIME)
    _resources(toolchain.get("headers"), _HEADERS)
    engines = block.get("engines")
    _require(
        type(engines) is dict and engines and set(engines) <= {"spike", "gsim", "verilator"},
        "deployment requires a supported explicit engine roster",
    )
    for engine, declaration in engines.items():
        _require(type(declaration) is dict, "deployment engine must be a mapping")
        _pin(declaration.get("binary"))
        if engine in {"gsim", "verilator"}:
            _pin(declaration.get("receipt"))
    if "spike" in engines:
        extension = runner.get("spike_extension")
        _fields(extension, {"extension_name", "extlib"}, {"sha256"})
        pin = {"path": extension["extlib"]}
        if "sha256" in extension:
            pin["sha256"] = extension["sha256"]
        _pin(pin)
        extension["sha256"] = pin["sha256"]
    bound = bind(target=target, contract=selected, facts=facts)
    bound.verify_execution_inputs()
    return selected


def _read_document(path):
    path = _path(str(path))
    _regular(path)
    _require(path.stat().st_size <= 8 * 1024 * 1024, "deployment input exceeds the bounded document limit")
    with path.open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    _require(len(raw) <= 8 * 1024 * 1024, "deployment input exceeds the bounded document limit")
    document = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
    _require(type(document) is dict, "deployment input must be a data mapping")
    return document, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = _path(str(args.output))
        _require(
            output.is_relative_to(artifacts_dir().resolve()), "prepared contract must be beneath the artifacts root"
        )
        _require(not output.exists(), "prepared contract destination already exists")
        contract, contract_digest = _read_document(args.contract)
        facts, facts_digest = _read_document(args.facts)
        prepared = prepare_contract(target=args.target, contract=contract, facts=facts)
        for path, expected in ((args.contract, contract_digest), (args.facts, facts_digest)):
            _require(_digest(path) == expected, "deployment input changed during preparation")
        raw = (json.dumps(prepared, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        _require(
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


# ---- engine_identity_memo ----------------------------------------------------------------------
# Reuse a validated engine identity while every file it pins is unchanged.
#
# Validating a prebuilt simulator's identity hashes its binary, its build receipt and every file the
# receipt pins. A grade asks for that identity once per completed test case (to route its certificate
# job), so the same unchanged files were hashed hundreds of times per grade: measured, about 60% of a
# grade's completion processing, delaying its first verdict by about 20 minutes. A value is reused only
# while the stat identity of every pinned file and directory tree is unchanged (any content change moves
# the file's ctime, which an unprivileged process cannot set), never for a file changed within
# :data:`SETTLED_NS`, and never for longer than :data:`MAX_AGE_S`, after which it is validated in full.

MAX_AGE_S = 300.0
SETTLED_NS = 2_000_000_000
_CACHE: dict[tuple, tuple[tuple, float, Any]] = {}
_LOCK = threading.Lock()


def _stat_row(path: str) -> tuple:
    st = os.lstat(path)
    return (path, st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def signature(paths: Iterable[str]) -> tuple | None:
    """Stat identity of each path, and of every entry below a directory; ``None`` if any is unreadable."""
    rows = []
    try:
        for raw in sorted(set(paths)):
            rows.append(_stat_row(raw))
            if os.path.isdir(raw) and not os.path.islink(raw):
                for directory, dirnames, filenames in os.walk(raw):
                    dirnames.sort()
                    for name in sorted(dirnames) + sorted(filenames):
                        rows.append(_stat_row(os.path.join(directory, name)))
    except OSError:
        return None
    return tuple(rows)


def pinned_paths(document: Any) -> list[str]:
    """Every absolute ``path`` of a ``{path, sha256}`` pin anywhere in a JSON document."""
    found: list[str] = []
    stack = [document]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            path = item.get("path")
            if isinstance(path, str) and isinstance(item.get("sha256"), str) and os.path.isabs(path):
                found.append(path)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def receipt_paths(receipt: str | Path) -> list[str]:
    """The receipt itself and every file it pins; an unreadable receipt pins nothing extra."""
    try:
        document = json.loads(Path(receipt).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        document = None
    return [str(receipt), *pinned_paths(document)]


def cached(key: tuple, paths: Iterable[str], compute: Callable[[], Any]) -> Any:
    """``compute()``, reused while ``paths`` keep the signature they had when it was computed."""
    paths = list(paths)
    before = signature(paths)
    now = time.monotonic()
    with _LOCK:
        entry = _CACHE.get(key)
    if before is not None and entry is not None and entry[0] == before and now - entry[1] < MAX_AGE_S:
        return copy.deepcopy(entry[2])
    value = compute()
    settled = before is not None and all(
        row[5] < time.time_ns() - SETTLED_NS and row[6] < time.time_ns() - SETTLED_NS for row in before
    )
    if settled and signature(paths) == before:
        with _LOCK:
            _CACHE[key] = (before, now, copy.deepcopy(value))
    return value


def clear() -> None:
    with _LOCK:
        _CACHE.clear()
