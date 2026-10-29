"""Observe plural pointer-call profiling ABIs in actual compiled bridges.

These are counter boundaries, not borrowed effects or ownership permissions.
Every mapping reopens the completed model's linked shim and its observed C
compilation, then matches the entire original generated forwarding function.
"""

import json
import re
from pathlib import Path

from merlin.common import invocation_record
from merlin.common.digest import sha256_file
from merlin.runtime.whole_model_readback import ReadbackError

from .sibling_source_route import borrowed_bridge_source


def _reopen(row, pins):
    path = Path(row["path"])
    size = row.get("bytes", row.get("size_bytes"))
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
            or not path.is_file() or sha256_file(path) != row["sha256"]
            or size is not None and path.stat().st_size != size):
        raise ReadbackError("profile compiled sibling bridge changed")
    pin = dict(path=str(path), sha256=row["sha256"], bytes=path.stat().st_size)
    if str(path) in pins and pins[str(path)] != pin:
        raise ReadbackError("profile sibling bridge has conflicting observations")
    pins[str(path)] = pin
    return path


def _bridge(program, record, pins):
    source = _reopen(record["source"], pins)
    obj = _reopen(record["object"], pins)
    if not any(row["path"] == str(obj) and row["sha256"] == record["object"]["sha256"]
               for row in program["objects"]):
        raise ReadbackError("profile sibling shim is absent from the completed final link")
    observed = invocation_record.verify(_reopen(record["compilation"], pins))
    if (observed.get("stage") != "device_shim_compilation"
            or observed["inputs"] != [dict(path=str(source), sha256=record["source"]["sha256"])]
            or observed["outputs"] != [dict(path=str(obj), sha256=record["object"]["sha256"])]):
        raise ReadbackError("profile sibling bridge lacks its actual C compilation")
    for row in (*observed["inputs"], *observed["outputs"], *observed["dependencies"],
                observed["executable"], observed["stdout"], observed["stderr"]):
        _reopen(row, pins)
    return source.read_text()


def completed_sibling_bindings(program, routing, undefined):
    """Derive existing ``callees`` bindings without recompiling model code.

    The input program is the original completed-program reader's result. Source
    route names are joined through the complete emitted C bridge, never by a
    suffix convention. The actual model object must reference that void entry.
    Returned pins retain observations only; no serialized lifetime is adopted.
    """
    pins, cache, bindings = {}, {}, {}
    sidecar = program["receipt"]["output"]["device_sidecar"]
    if json.loads(_reopen(sidecar, pins).read_text()) != routing:
        raise ReadbackError("profile sibling routing differs from the completed build")
    kernels = (program["receipt"]["output"].get("device_artifacts") or {}).get("kernels", {})
    for symbol in sorted({row["symbol"] for row in routing.get("routed", ())}):
        custody = kernels.get(symbol, {}).get("custody", {}).get("sibling_source")
        if custody is None:
            continue
        record = custody["descriptor_bridge"]
        key = json.dumps(record, sort_keys=True)
        if key not in cache:
            cache[key] = _bridge(program, record, pins)
        source = cache[key]
        call = routing.get("call_buffers", {}).get(symbol)
        if (not isinstance(call, list) or not 0 < len(call) <= 64
                or any(not isinstance(row, dict) or not isinstance(row.get("role"), str)
                       or not isinstance(row.get("shape"), list) or not row["shape"]
                       or any(type(d) is not int or d <= 0 for d in row["shape"]) for row in call)):
            raise ReadbackError("profile sibling route lost its complete positional buffers")
        results = [i for i, row in enumerate(call) if row["role"].startswith("out_")]
        if len(results) < 2:
            raise ReadbackError("profile sibling bridge requires its complete plural results")
        candidates = set(re.findall(r"^void _mlir_ciface_([A-Za-z_][A-Za-z0-9_]*)\(", source, re.M))
        matches = [name for name in candidates
                   if source.count(borrowed_bridge_source(symbol, name, call, results)) == 1]
        if len(matches) != 1:
            raise ReadbackError("profile sibling has no unique complete compiled forwarding bridge")
        borrowed = matches[0]
        if f"_mlir_ciface_{borrowed}" not in undefined:
            raise ReadbackError("profile sibling bridge is not referenced by the actual model object")
        if any(row["borrowed_symbol"] == borrowed for row in bindings.values()):
            raise ReadbackError("profile sibling entry has ambiguous source routes")
        bindings[symbol] = dict(borrowed_symbol=borrowed, buffers=len(call),
                                argument_ranks=[len(row["shape"]) for row in call])
    for row in tuple(pins.values()):
        _reopen(row, pins)
    return bindings, list(pins.values())
