"""Original native public-program receipts projected into normal paired rows.

This does not execute a second evaluator or issue an engine certificate. The
original model grade must already pass. Actual native source/ELF/caller/console
and engine receipt bytes are reopened before the ordinary certificate gate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from merlin.targetgen.public_program_timing import SCHEMA, is_public_program, verify_selection


def selection(capsule, rtl_identity, *, simulator="gsim"):
    if not is_public_program(capsule):
        return None
    pin = dict(rtl_identity.get("rtl_facts") or {})
    path = Path(str(pin.get("path") or ""))
    if not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError("public program requires the selected ordinary RTL facts")
    document = json.loads(path.read_bytes())
    body = document.get("facts") or {}
    config = (body.get("source") or {}).get("config")
    if "simulation_correspondence" in body:
        from merlin.targetgen.rtl.unit_correspondence import execution_binding

        simulation = (body["simulation_correspondence"].get("configs") or {}).get("simulation") or {}
        config = execution_binding(document, config=simulation.get("config"))["config"]
    value = {
        "schema": SCHEMA,
        "simulator": simulator,
        "rtl_facts": pin,
        "board_config": config,
        "complete_call_timing": simulator == "gsim",
    }
    verify_selection(capsule, value)
    return value


def _reopen(pin):
    if not isinstance(pin, Mapping) or not {"path", "sha256"} <= set(pin):
        raise ValueError("actual public native product has no original byte pin")
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("actual public native product is unavailable or indirect")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != pin["sha256"] or ("size_bytes" in pin and path.stat().st_size != pin["size_bytes"]):
        raise ValueError("actual public native product changed")
    return path


def project(capsule, grade, *, target, readback_policy=None):
    """Return actual timing evidence only after the original complete grade."""
    if not is_public_program(capsule) or grade.get("status") != "pass":
        raise ValueError("public native timing requires the original complete passing public grade")
    verify_selection(capsule, grade.get("public_program_execution"))
    if (grade.get("model_execution_check") or {}).get("kind") != "candidate_whole_program_execution":
        raise ValueError("public timing cannot borrow a runner-owned host-graph verdict")
    native = grade.get("candidate_native_execution") or {}
    timing = native.get("complete_call_timing") or {}
    from merlin.runtime.backends import base as backends
    from merlin.runtime.complete_call_timing import SCHEMA as TIMER_SCHEMA

    backend = backends.get_backend(target)
    resolver = getattr(backend, "complete_call_timing", None)
    plan = resolver() if callable(resolver) else None
    if plan is None or timing.get("schema") != TIMER_SCHEMA:
        raise ValueError("public timing has no selected original complete-call counter/clock")
    expected = plan.verify()
    if {key: timing.get(key) for key in expected} != expected:
        raise ValueError("public timing selection changed after the original run")
    if timing.get("elf") != native.get("elf"):
        raise ValueError("complete-call timer does not bind the actual candidate ELF")
    products = native.get("build_artifacts")
    required_products = {
        "kernel.llvm.mlir",
        "kernel.ll",
        "kernel.o",
        "kernel.stack_frame.json",
        "kernel.abi.json",
        "harness.c",
    }
    if type(products) is not dict or not required_products <= set(products) or len(products) > 4096:
        raise ValueError("public timing lacks its bounded actual object/caller/build product roster")
    for pin in [
        *products.values(),
        *native.get("source", {}).values(),
        *native.get("host_build_sources", []),
        native.get("elf"),
        native.get("console"),
        timing.get("caller"),
        *grade.get("candidate_emission", {}).values(),
    ]:
        _reopen(pin)
    tier = native.get("tiers", {}).get("L3") or {}
    if tier.get("status") != "pass" or tier.get("engine") != "gsim":
        raise ValueError("public timing did not execute the original selected RTL tier")
    citation = tier.get("engine_citation") or {}
    from merlin.targetgen import gsim_emulator as GE

    current = GE.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    if current != citation or current.get("receipt_status") != "bound":
        raise ValueError("public timing engine bytes or original receipt changed")
    receipt = citation.get("receipt") or {}
    if receipt.get("schema_version") != GE.STRICT_RECEIPT_SCHEMA:
        raise ValueError("public timing has no original bound engine receipt")
    console_path = _reopen(native["console"])
    cb_path = _reopen(grade["candidate_emission"]["command_buffer"])
    cb = json.loads(cb_path.read_bytes())
    if console_path.stat().st_size > 64 << 20:
        raise ValueError("complete public output console exceeds its normal bounded projection")
    from merlin.targetgen.contract.readback_policy import read_console, selected

    readback_policy = selected(readback_policy)
    console = read_console(console_path, policy=readback_policy)
    # Coherent memory readout has separately selected complete output objects.
    # Until they have a fixed normal replay producer, this projection refuses it.
    if native.get("readback_memory_l3") is not None:
        raise ValueError("public complete timing projection lacks a coherent-output replay producer")
    observed, metrics = backend.parse_output(console)
    from merlin.runtime.commandbuffer import declared_output_dtypes

    from .gsim_workload import OUTPUT_ENCODING, encode_declared_outputs

    observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
    from merlin.targetgen.capsule_golden import compare
    from merlin.targetgen.golden_store import load_golden

    declaration_path = _reopen(native["source"]["capsule_declaration"])
    import yaml

    declaration = yaml.safe_load(declaration_path.read_text())
    golden = load_golden(declaration_path.parent)
    if set(observed) != set(golden["outputs"]) or set(observed) != set(cb["kernel_abi"]["outputs"]):
        raise ValueError("public native output names differ from the original complete result roster")
    comparison = compare(
        golden["outputs"],
        observed,
        declaration["numeric_policy"],
        golden_source=str(golden.get("golden_source") or "independent_capsule"),
    )
    if comparison.get("status") != "pass":
        raise ValueError("actual complete public outputs no longer pass the original numerical contract")
    output_sha, output_tensors = encode_declared_outputs(observed, cb)
    if native.get("output_observation") != {
        "sha256": output_sha,
        "encoding": OUTPUT_ENCODING,
        "tensors": output_tensors,
    }:
        raise ValueError("public native complete output bytes or roster changed")
    cycles = metrics.get("cycles")
    if (
        type(cycles) is not int
        or not 0 < cycles <= plan.max_delta_cycles
        or cycles != tier.get("metrics", {}).get("cycles")
    ):
        raise ValueError("public native console lacks its actual complete-call cycle delta")
    return {
        "engine": "gsim",
        "status": "pass",
        "elf": str(native["elf"]["path"]),
        "elf_sha256": native["elf"]["sha256"],
        "cycles": cycles,
        "output_sha256": output_sha,
        "output_encoding": OUTPUT_ENCODING,
        "output_tensors": output_tensors,
        "binary_sha256": citation["binary_sha256"],
        "firrtl_sha256": receipt["firrtl_sha256"],
        "model_sha256": receipt["model_manifest_sha256"],
        "observed_engine_binaries": {"status": "observed", "digests": [citation["binary_sha256"]]},
        "derived_from_rtl": True,
        "cycle_accurate": True,
        "reused_measurement": False,
        "oracle": {"kind": "candidate_whole_program_same_elf", "engine_citation": citation},
        "complete_call_timing": timing,
        "actual_console": native["console"],
        "execution_invocations": native.get("execution_invocations"),
        "actual_source": native["source"],
        "actual_candidate": grade["candidate_emission"],
    }
