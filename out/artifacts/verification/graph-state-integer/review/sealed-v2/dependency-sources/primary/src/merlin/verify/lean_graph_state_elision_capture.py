"""Pinned actual configuration elimination with complete bit-exact Kernel intake.

This standalone subprocess imports only the separately pinned production tree.
Call/operand decoding and physical effect admission remain trusted host context;
the primary checker rechecks full source/result omission in the Lean model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


class CorrespondenceRefusal(ValueError):
    def __init__(self, reason, production):
        super().__init__(reason)
        self.production = production


def file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def exact(value, depth=0):
    if depth > 64:
        raise ValueError("complete Kernel operand depth exceeds the explicit intake budget")
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is str:
        return ["string", value]
    if type(value) is float:
        return ["float64-bits", struct.pack(">d", value).hex()]
    if type(value) is tuple:
        return ["tuple", [exact(item, depth + 1) for item in value]]
    if is_dataclass(value) and type(value).__module__.startswith("merlin.sched.ir"):
        return [
            type(value).__module__ + "." + type(value).__name__,
            [[field.name, exact(getattr(value, field.name), depth + 1)] for field in fields(value)],
        ]
    raise ValueError("unsupported complete Kernel operand representation")


def capture(reference, source, target, contract_path, destination, *, enabled=True):
    reference, destination = Path(reference).resolve(), Path(destination).resolve()
    revision = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError("actual producer source revision differs")
    subprocess.run(
        ["git", "-C", str(reference), "diff", "--exit-code", REVISION, "--", "src/merlin"],
        check=True,
        capture_output=True,
    )
    sys.dont_write_bytecode = True

    from merlin.compile.scheduling.configuration import (
        ConfigurationAdapter,
        ConfigurationEffects,
        eliminate_redundant_configuration,
        verify_configuration_rewrite,
        verify_configuration_state,
    )
    from merlin.sched.ir import Call, Const, Kernel, check_structure, parse_kernel

    record = json.loads(Path(contract_path).read_text())
    if (
        type(record) is not dict
        or set(record) != {"schema", "configuration_identity", "qualification_identity", "rules"}
        or record["schema"] != "merlin.graph-state-elision.decoder.v1"
    ):
        raise ValueError("versioned independently selected complete decoder contract required")
    rules = {row["instruction"]: row for row in record["rules"]}
    if len(rules) != len(record["rules"]):
        raise ValueError("duplicate independently selected instruction decoder")
    for row in rules.values():
        if set(row) != {
            "instruction",
            "arguments",
            "reads",
            "writes",
            "clobbers",
            "ordered",
            "pure_write",
            "no_other_effects",
            "non_failing",
            "semantic_sha256",
        }:
            raise ValueError("closed original instruction interpretation required")
        if len(set(row["arguments"])) != len(row["arguments"]) or any(
            type(name) is not str for name in row["arguments"]
        ):
            raise ValueError("complete unique ordered call argument names required")
        for flag in ("ordered", "pure_write", "no_other_effects", "non_failing"):
            if type(row[flag]) is not bool:
                raise ValueError("effect/admission flags require exact booleans")
        identity = row["semantic_sha256"]
        if type(identity) is not str or len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
            raise ValueError("explicit original operation interpretation identity required")

    def integer(call, argument):
        value = call.arg(argument)
        if type(value) is not Const or type(value.value) is not int:
            raise ValueError("configuration decoder requires actual integer constant operands")
        return value.value

    def field_name(call, selected):
        if type(selected) is str and selected:
            return selected
        if type(selected) is dict and set(selected) == {"prefix", "argument"} and type(selected["prefix"]) is str:
            return selected["prefix"] + str(integer(call, selected["argument"]))
        raise ValueError("configuration field needs an exact selected decoder")

    def value(call, selected):
        if type(selected) is dict and set(selected) == {"argument"}:
            return integer(call, selected["argument"])
        if type(selected) is dict and set(selected) == {"constant"} and type(selected["constant"]) is int:
            return selected["constant"]
        raise ValueError("configuration value needs an actual operand or selected literal requirement")

    def decode(call):
        row = rules.get(call.instr)
        if row is None:
            return None
        if [name for name, _ in call.args] != row["arguments"]:
            raise ValueError("actual call argument order/membership differs from selected decoder")
        # This decoder has no admission for execution/token/staging/state
        # annotations. Preserve them as opaque barriers instead of treating a
        # configuration-only rule as a complete meaning of those extra effects.
        if any((call.unit, call.produces, call.awaits, call.stages, call.sets, call.assumes)):
            return None
        reads = tuple((field_name(call, key), value(call, selected)) for key, selected in row["reads"])
        writes = tuple((field_name(call, key), value(call, selected)) for key, selected in row["writes"])
        return ConfigurationEffects(
            reads, writes, tuple(field_name(call, key) for key in row["clobbers"]), row["ordered"], row["pure_write"]
        )

    adapter = ConfigurationAdapter(decode, record["configuration_identity"], record["qualification_identity"])
    original = parse_kernel(Path(source).read_text())
    if (
        type(original) is not Kernel
        or check_structure(original)
        or any(type(call) is not Call for call in original.body)
    ):
        raise ValueError("complete straight-line structurally valid Kernel required")
    if len(original.body) > 4096:
        raise ValueError("complete actual call count exceeds explicit budget")
    if target is None:
        optimized = eliminate_redundant_configuration(original, adapter, enabled=enabled)
        actual, producer = optimized.kernel, "merlin.compile.scheduling.configuration.eliminate_redundant_configuration"
    else:
        actual, producer = parse_kernel(Path(target).read_text()), "supplied-actual-result"
    if check_structure(actual) or any(type(call) is not Call for call in actual.body):
        raise ValueError("actual output must be a complete straight-line valid Kernel")
    production = verify_configuration_rewrite(original, actual, adapter)
    if canonical(exact((original.name, original.args, original.attrs))) != canonical(
        exact((actual.name, actual.args, actual.attrs))
    ):
        raise CorrespondenceRefusal("complete Kernel invocation interface/metadata changed", production)
    before_effects, after_effects = [decode(call) for call in original.body], [decode(call) for call in actual.body]
    fields_used = sorted(
        {
            name
            for effect in [*before_effects, *after_effects]
            if effect is not None
            for name in [*(key for key, _ in (*effect.reads, *effect.writes)), *effect.clobbers]
        }
    )
    if len(fields_used) > 256:
        raise ValueError("actual complete configuration field domain exceeds explicit budget")

    def atom(index, call, effect):
        row = rules.get(call.instr)
        empty_dependencies = not any((call.unit, call.produces, call.awaits, call.stages, call.sets, call.assumes))
        eligible = row is not None and row["no_other_effects"] and row["non_failing"] and empty_dependencies
        mapped = (
            None
            if effect is None
            else {
                "reads": [[fields_used.index(key), value] for key, value in effect.reads],
                "writes": [[fields_used.index(key), value] for key, value in effect.writes],
                "clobbers": [fields_used.index(key) for key in effect.clobbers],
                "ordered": effect.ordered,
                "pureWrite": effect.pure_write,
            }
        )
        return {"id": index, "token": canonical(exact(call)), "effect": mapped, "eraseAllowed": eligible}

    before = [
        atom(index, call, effect)
        for index, (call, effect) in enumerate(zip(original.body, before_effects, strict=True))
    ]
    after = []
    cursor = 0
    for index, call in enumerate(actual.body):
        candidate = atom(0, call, after_effects[index])
        while cursor < len(before) and before[cursor]["token"] != candidate["token"]:
            cursor += 1
        if cursor == len(before):
            raise CorrespondenceRefusal(
                "actual output changed/reordered a complete bit-exact retained call or observer", production
            )
        candidate["id"] = cursor
        if candidate != before[cursor]:
            raise ValueError("actual retained call decoder/admission changed")
        after.append(candidate)
        cursor += 1
    try:
        established = {"status": "checked", "report": verify_configuration_state(original, adapter)}
    except ValueError as error:
        established = {"status": "not_established", "reason": str(error)}
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "before.mk").write_bytes(Path(source).read_bytes())
    (destination / "after.mk").write_text(actual.text() if target is None else Path(target).read_text())
    modules = []
    for name, module in tuple(sys.modules.items()):
        if (name == "merlin" or name.startswith("merlin.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(reference / "src"):
                raise ValueError("producer/decoder import escaped pinned source root")
            modules.append(file_ref(path))
    result = {
        "schema": "merlin.graph-state-elision.actual-projection.v1",
        "projection_checked": True,
        "producer": producer,
        "producer_revision": revision,
        "selected_contract": file_ref(contract_path),
        "decoder_contract": record,
        "admission_approved": False,
        "owned_projection": file_ref(__file__),
        "before": before,
        "after": after,
        "configuration_fields": fields_used,
        "source_signature": canonical(exact((original.name, original.args, original.attrs))),
        "actual_signature": canonical(exact((actual.name, actual.args, actual.attrs))),
        "source_digest": original.digest(),
        "actual_digest": actual.digest(),
        "production_conformance": production,
        "read_establishment": established,
        "genuine_changed_boundary": before != after,
        "removed_calls": len(before) - len(after),
        "selected_enabled": enabled if target is None else None,
        "producer_sources": sorted(modules, key=lambda row: row["path"]),
        "producer_external_runtime": {
            "python": file_ref(sys.executable),
            "python_version": sys.version,
            "loaded_sources": sorted(
                {
                    str(Path(module.__file__).resolve()): file_ref(module.__file__)
                    for module in tuple(sys.modules.values())
                    if getattr(module, "__file__", None)
                    and Path(module.__file__).is_file()
                    and not Path(module.__file__).resolve().is_relative_to(reference / "src")
                    and Path(module.__file__).resolve() != Path(__file__).resolve()
                }.values(),
                key=lambda row: row["path"],
            ),
        },
        "inputs": [file_ref(destination / "before.mk"), file_ref(destination / "after.mk")],
        "trust": [
            "actual pinned Kernel parser and primary bit-exact correspondence/operand decoder are trusted",
            "qualified removable setter implements exact non-failing overwrite and NO other observation",
            "ordered retained effects completely describe configuration writes/clobbers/frame on successful return",
            "retained common calls can fail; their actual error and terminal state interpretation must match",
            "read establishment is separate from source execution/typing/poison/native qualification",
        ],
        "public_acceptance_issued": False,
        "native_fidelity_proved": False,
        "source_success_proved": False,
    }
    (destination / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("reference", "source", "contract", "destination"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--disabled", action="store_true")
    args = parser.parse_args()
    try:
        result = capture(
            args.reference, args.source, args.target, args.contract, args.destination, enabled=not args.disabled
        )
    except (ValueError, TypeError, KeyError, IndexError) as error:
        args.destination.mkdir(parents=True, exist_ok=True)
        result = {
            "projection_checked": False,
            "reason": {"type": type(error).__name__, "detail": str(error)},
            "public_acceptance_issued": False,
        }
        if isinstance(error, CorrespondenceRefusal):
            result["production_conformance_before_independent_refusal"] = error.production
        for role, path in (("before", args.source), ("after", args.target)):
            if path is not None:
                (args.destination / (role + ".mk")).write_bytes(path.read_bytes())
        (args.destination / "projection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"projection_checked": result["projection_checked"]}))


if __name__ == "__main__":
    main()
