"""Pinned actual Kernel configuration hoisting and its complete restricted projection.

This is trusted Python correspondence, not a Lean proof of parsing or an effect
admission. Supported here: one top-level nonempty flat loop, one constant pure
setup prefix, arbitrary retained sequential calls before/inside/after that loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lean_graph_state_capture import REVISION, external_runtime, file_ref


def capture(reference, source, target, contract_path, destination):
    reference, destination = Path(reference).resolve(), Path(destination).resolve()
    revision = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError("configuration producer revision differs from selected source")
    subprocess.run(
        ["git", "-C", str(reference), "diff", "--exit-code", REVISION, "--", "src/merlin"],
        check=True,
        capture_output=True,
    )
    from merlin.compile.scheduling.configuration import ConfigurationAdapter, ConfigurationEffects
    from merlin.compile.scheduling.configuration_loops import (
        hoist_loop_configuration,
        verify_loop_configuration_rewrite,
    )
    from merlin.sched.ir import Call, Const, Expr, Loop, check_structure, parse_kernel

    record = json.loads(Path(contract_path).read_text())
    rules = {row["instruction"]: row for row in record["rules"]}
    if len(rules) != len(record["rules"]) or not rules:
        raise ValueError("unique nonempty original configuration rules required")
    for row in rules.values():
        if not all(
            row[key] is True
            for key in ("deterministic_total", "complete_state_effects", "sequential_atomic_complete", "ordered")
        ):
            raise ValueError("total complete ordered sequential effect admission required")
        if len(row["semantic_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["semantic_sha256"]):
            raise ValueError("explicit operation semantic identity required")
        for key in ("state_reads", "state_writes", "arguments"):
            if any(type(item) is not str or not item for item in row[key]) or len(set(row[key])) != len(row[key]):
                raise ValueError("unique complete state cells and argument slots required")

    def decode(call):
        row = rules.get(call.instr)
        if row is None or [name for name, _ in call.args] != row["arguments"]:
            raise ValueError("unknown instruction or changed actual argument slots")
        if any((call.unit, call.produces, call.awaits, call.stages, call.sets, call.assumes)):
            raise ValueError("this configuration projection requires undecorated sequential calls")
        if any(not isinstance(value, Expr) for _, value in call.args):
            raise ValueError("this configuration projection supports integer expressions only")
        writes = []
        for field, argument in row["configuration_writes"]:
            value = call.arg(argument)
            if type(value) is not Const or type(value.value) is not int:
                raise ValueError("configuration values must decode from actual integer constants")
            writes.append((field, value.value))
        reads = tuple(tuple(pair) for pair in row["configuration_reads"])
        effect = ConfigurationEffects(reads, tuple(writes), tuple(row["clobbers"]), row["ordered"], row["pure_write"])
        if any("configuration:" + key not in row["state_reads"] for key, _ in reads) or any(
            "configuration:" + key not in row["state_writes"] for key in [*(key for key, _ in writes), *row["clobbers"]]
        ):
            raise ValueError("whole-state footprint omits decoded persistent configuration effects")
        if effect.pure_write and (
            row["state_reads"]
            or sorted(row["state_writes"]) != sorted("configuration:" + key for key, _ in writes)
            or sorted(row["arguments"]) != sorted(argument for _, argument in row["configuration_writes"])
        ):
            raise ValueError("pure setup must mean exactly the decoded field overwrite")
        return effect

    adapter = ConfigurationAdapter(decode, record["configuration_identity"], record["qualification_identity"])
    original = parse_kernel(Path(source).read_text())
    if check_structure(original):
        raise ValueError("source Kernel structure rejected")
    loops = [(index, item) for index, item in enumerate(original.body) if type(item) is Loop]
    if len(loops) != 1:
        raise ValueError("supported configuration boundary needs exactly one top-level loop")
    loop_position, loop = loops[0]
    if not loop.body or any(type(item) is not Call for item in loop.body):
        raise ValueError("supported configuration boundary needs a flat call body")
    static = (*original.body[:loop_position], *loop.body, *original.body[loop_position + 1 :])
    if len(static) > 4096 or any(type(call) is not Call for call in static):
        raise ValueError("complete bounded flat static call roster required")
    effects = [decode(call) for call in static]
    setup = len(original.body[:loop_position])
    if not effects[setup].pure_write or (len(loop.body) > 1 and effects[setup + 1].pure_write):
        raise ValueError("supported configuration prefix is exactly one pure setup call")
    if target is None:
        actual_result = hoist_loop_configuration(original, adapter)
        actual, producer = (
            actual_result.kernel,
            "merlin.compile.scheduling.configuration_loops.hoist_loop_configuration",
        )
    else:
        actual, producer = parse_kernel(Path(target).read_text()), "supplied-actual-configuration-target"
    verification = verify_loop_configuration_rewrite(original, actual, adapter)
    if verification["hoisted_static_calls"] != 1:
        raise ValueError("supported checked boundary requires an actual single changed prefix")
    cells = sorted(
        {key for call in static for key in (*rules[call.instr]["state_reads"], *rules[call.instr]["state_writes"])}
    )

    def footprint(call):
        return {
            "reads": [cells.index(key) for key in rules[call.instr]["state_reads"]],
            "writes": [cells.index(key) for key in rules[call.instr]["state_writes"]],
        }

    # Match every actual call to its original static occurrence in order. Only
    # the verified setup crosses the single matching loop boundary.
    expected_body = (
        *original.body[:loop_position],
        loop.body[0],
        type(loop)(loop.var, loop.extent, loop.body[1:]),
        *original.body[loop_position + 1 :],
    )
    from dataclasses import replace

    if replace(original, body=expected_body) != actual:
        raise ValueError("actual complete Kernel differs from the supported exact hoist shape")
    body = list(range(setup + 1, setup + len(loop.body)))
    earlier, later = list(range(setup)), list(range(setup + len(loop.body), len(static)))
    source_code = (
        [{"call": index} for index in earlier]
        + [{"loop": loop.extent, "calls": [setup, *body]}]
        + [{"call": index} for index in later]
    )
    target_code = (
        [{"call": index} for index in earlier]
        + [{"call": setup}, {"loop": loop.extent, "calls": body}]
        + [{"call": index} for index in later]
    )
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "before.mk").write_text(original.text())
    (destination / "after.mk").write_text(actual.text())
    sources = []
    for name, module in tuple(sys.modules.items()):
        if (name == "merlin" or name.startswith("merlin.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(reference / "src"):
                raise ValueError("configuration producer import escaped pinned source")
            sources.append(file_ref(path))
    report = {
        "schema": "merlin.graph-state.configuration-projection.v1",
        "producer": producer,
        "producer_revision": revision,
        "projection_checked": True,
        "public_acceptance_issued": False,
        "source_contract_approved": False,
        "native_fidelity_proved": False,
        "original_source": file_ref(source),
        "original_contract": file_ref(contract_path),
        "source_digest": original.digest(),
        "target_digest": actual.digest(),
        "producer_sources": sorted(sources, key=lambda row: row["path"]),
        "external_runtime": external_runtime(),
        "producer_verification": verification,
        "cells": cells,
        "footprints": [footprint(call) for call in static],
        "before": source_code,
        "after": target_code,
        "move": {
            "earlier": earlier,
            "setup": setup,
            "body": body,
            "extent": loop.extent,
            "later": later,
            "fields": [cells.index("configuration:" + key) for key, _ in effects[setup].writes],
        },
        "field_values": [[cells.index("configuration:" + key), value] for key, value in effects[setup].writes],
        "projection_trust": "pinned production parser and full rewrite verifier; "
        "primary-owned restricted complete Kernel projection",
        "assumptions": [
            "actual instructions implement the original complete total sequential Action models",
            "configuration field identities are exact nonaliasing state cells",
            "pure setup implements exactly the decoded constant field overwrite with no observer or other effects",
            "all remaining calls preserve setup fields, including hidden aliases and observers",
            "integer constant interpretation and IR-to-state correspondence are admitted independently",
        ],
    }
    (destination / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("reference", "source", "contract", "destination"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    try:
        result = capture(args.reference, args.source, args.target, args.contract, args.destination)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        args.destination.mkdir(parents=True, exist_ok=True)
        result = {"projection_checked": False, "reason": {"type": type(error).__name__, "detail": str(error)}}
        (args.destination / "capture.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"projection_checked": result["projection_checked"]}))


if __name__ == "__main__":
    main()
