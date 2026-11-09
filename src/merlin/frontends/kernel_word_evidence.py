"""Source-word stage byte custody, independent of conversion/numeric proof."""

import hashlib

_POINTER_FLAGS = ["--triton-to-unstructured=offset-bit-width=64"]
_COMPUTE_FLAGS = ["--triton-arith-to-linalg=pids-to-func-args=false tt-to-func-func=false",
                  "--remove-dead-values", "--canonicalize"]
_TRANSPORT = {"--mlir-disable-threading", "--verify-each", "--mlir-print-op-generic",
              "--mlir-print-debuginfo", "--mlir-print-local-scope"}


def _record(raw):
    raw = raw.encode() if isinstance(raw, str) else raw
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _native(stage, name, flags, source, output):
    if not isinstance(stage, dict):
        raise ValueError("source-word native stage must be an object")
    command = stage.get("command", [])
    pipeline = [flag for flag in (command if isinstance(command, list) else [])
                if isinstance(flag, str) and flag.startswith("--") and flag not in _TRANSPORT]
    if (stage.get("schema") != "t2m.triton-shared-stage.v1" or stage.get("stage") != name
            or stage.get("stage_completed") is not True
            or not isinstance(command, list) or any(type(flag) is not str for flag in command) or pipeline != flags
            or stage.get("artifacts", {}).get("source.ttir") != _record(source)
            or stage.get("artifacts", {}).get("lowered.mlir") != _record(output)):
        raise ValueError("source-word native stage does not bind its declared pipeline/input/output")


def checked_word_stages(stage, source, output, memory, inventory):
    """Check declared stage chains; producer owns typed event reconstruction."""
    if not isinstance(stage, dict):
        raise ValueError("source-word evidence must be an object")
    if stage.get("schema") == "t2m.triton-shared-stage.v1":
        if memory.get("word_conversion") is not None:
            raise ValueError("source-word evidence version differs from its representation declaration")
        _native(stage, "source-word-tensor", _POINTER_FLAGS + _COMPUTE_FLAGS, source, output)
        return
    if (stage.get("schema") != "t2m.source-word-stages.v2" or stage.get("stage_completed") is not True
            or stage.get("input") != _record(source) or stage.get("output") != _record(output)
            or stage.get("reconstruction") != "same-region-same-event-order"
            or memory.get("word_conversion") != "native-isolated-events-v2"
            or not isinstance(stage.get("events"), dict) or not isinstance(stage.get("compute"), dict)):
        raise ValueError("source-word event conversion lacks its versioned input/output chain")
    events, compute = stage["events"], stage["compute"]
    if (any(type(events.get(key)) is not str for key in ("input", "output"))
            or type(compute.get("input")) is not str or not isinstance(events.get("bindings"), list)
            or not events["bindings"]):
        raise ValueError("source-word event conversion lacks its actual intermediate bytes")
    _native(events.get("native_stage"), "source-word-events", _POINTER_FLAGS, events["input"], events["output"])
    _native(compute.get("native_stage"), "source-word-compute", _COMPUTE_FLAGS, compute["input"], output)
    event_tool = events["native_stage"].get("optimizer", {}).get("sha256")
    if not isinstance(event_tool, str) or len(event_tool) != 64 or event_tool != compute["native_stage"].get("optimizer", {}).get("sha256"):
        raise ValueError("source-word event and computation stages used different optimizers")
    symbols = set()
    for binding in events["bindings"]:
        if not isinstance(binding, dict):
            raise ValueError("source-word event binding must be an object")
        name = binding.get("symbol")
        atomic = binding.get("opcode") in ("tt.atomic_rmw", "tt.atomic_cas")
        if (not isinstance(name, str) or not name or name in symbols
                or binding.get("opcode") not in ("tt.load", "tt.store", "tt.atomic_rmw", "tt.atomic_cas")
                or not isinstance(binding.get("source_ids"), list) or not binding["source_ids"]
                or any(not isinstance(s, str) or s not in inventory for s in binding["source_ids"])
                or any(not isinstance(binding.get(key), list) or any(type(t) is not str for t in binding[key])
                       for key in ("input_types", "result_types"))
                or len(binding["input_types"]) < 2
                or len(binding["result_types"]) != (0 if binding["opcode"] == "tt.store" else 1)
                or atomic and (memory.get("effect_profile") not in ("private-invocation-integer-atomics-v1", "private-invocation-typed-atomics-v2")
                               or len(binding["input_types"]) not in (3, 4) or binding.get("fill") != "unchanged")
                or binding.get("opcode") == "tt.atomic_cas" and len(binding["input_types"]) != 4
                or binding.get("fill") not in ("unchanged", "source-mask-select-after-zero-filled-load")):
            raise ValueError("source-word event binding differs from the declared source/type inventory")
        symbols.add(name)
