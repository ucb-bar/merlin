"""Observe the exact prepared export's buffer roles, effects and storage aliases.

This leaf runs in the selected capture interpreter. It imports no framework at
module initialization and changes no program or producer installation. Its
observations need separate original-caller consent before buffers are frozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import operator
import struct
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "merlin.capture.export-state.v1"


class ExportStateUnavailable(ValueError):
    """No complete, read-only, unaliased prepared state observation is available."""


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ExportStateUnavailable(f"state observation needs its regular conversion artifact: {name}")
    return path.read_bytes()


def _tensor(tensor, torch):
    if (type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided
            or tensor.device.type != "cpu" or tensor.is_conj() or tensor.is_neg()):
        raise ExportStateUnavailable("buffer observation requires plain CPU strided tensor storage")
    storage = tensor.untyped_storage()
    version = tensor._version
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if tensor._version != version:
        raise ExportStateUnavailable("state tensor changed while observing its payload")
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
        "strides": list(tensor.stride()), "storage_offset": tensor.storage_offset(),
        "storage_bytes": storage.nbytes(), "payload_bytes": len(raw), "payload_sha256": _sha(raw),
        "version": version,
    }, storage._cdata


def _outputs(exported):
    rows = []
    for ordinal, spec in enumerate(exported.graph_signature.output_specs):
        kind = getattr(spec.kind, "name", None)
        if kind != "USER_OUTPUT":
            raise ExportStateUnavailable(f"prepared export has non-user output effect: {kind}")
        rows.append({"ordinal": ordinal, "kind": kind, "argument": getattr(spec.arg, "name", None),
                     "target": spec.target})
    if not rows:
        raise ExportStateUnavailable("prepared export has no complete output roster")
    return rows


def _effects(exported):
    rows = []
    if any(name for name, _ in exported.graph_module.named_modules()):
        raise ExportStateUnavailable("nested export modules require a separately supported effect observation")
    for ordinal, node in enumerate(exported.graph.nodes):
        if node.op in {"placeholder", "output"}:
            continue
        if node.op != "call_function":
            raise ExportStateUnavailable(f"prepared graph has an unknown effect kind: {node.op}")
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            if node.target is operator.getitem and len(node.args) == 2 and type(node.args[1]) is int:
                rows.append({"ordinal": ordinal, "target": str(node.target), "schema": None, "writes": []})
                continue
            raise ExportStateUnavailable("prepared graph has an unknown operator schema")
        writes = [arg.name for arg in schema.arguments if arg.alias_info is not None and arg.alias_info.is_write]
        if writes:
            raise ExportStateUnavailable(f"prepared operator writes its arguments: {node.target}")
        rows.append({"ordinal": ordinal, "target": str(node.target), "schema": str(schema), "writes": writes})
    return rows


def observe_export_state(conversion, inputs, *, directory):
    """Observe the returned prepared EP immediately, before further execution.

    The returned EP retains its source storage identities: the pinned producer's
    externalizer clones local file payloads and does not replace EP.state_dict.
    A graph-only snapshot copies no original tensor storage. No raw-output-only
    inference, re-export, mutation removal or buffer-name convention is used.
    """
    import torch
    from m2m.capture.trace import snapshot_exported_program
    from torch.export.graph_signature import InputKind

    exported = getattr(conversion, "exported_program", None)
    trace = getattr(conversion, "capture_trace", None)
    if not isinstance(exported, torch.export.ExportedProgram) or not isinstance(trace, dict):
        raise ExportStateUnavailable("conversion has no actual prepared export and complete trace")
    if (getattr(conversion, "ok", None) is not True or trace.get("status") != "complete"
            or trace.get("blockers") or exported.range_constraints
            or getattr(conversion, "path_taken", None) != "fx_importer"):
        raise ExportStateUnavailable("prepared state requires a complete static FX-import conversion")
    original_ir = conversion.mlir_text.encode("utf-8")
    if trace.get("mlir", {}).get("sha256") != _sha(original_ir):
        raise ExportStateUnavailable("conversion trace does not bind its actual emitted IR")
    prepared = (trace.get("graphs") or {}).get("prepared")
    if not isinstance(prepared, dict) or not isinstance(prepared.get("sha256"), str):
        raise ExportStateUnavailable("conversion trace lacks its exact prepared graph")
    outputs, effects = _outputs(exported), _effects(exported)
    root = Path(directory).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ExportStateUnavailable("conversion artifact directory must be real")
    manifest_bytes = _file(root, "weights.safetensors.manifest.json")
    manifest = json.loads(manifest_bytes)
    specs = tuple(exported.graph_signature.input_specs)
    if not isinstance(manifest, dict) or set(manifest) != {str(i) for i in range(len(specs))}:
        raise ExportStateUnavailable("prepared signature and externalized argument roster differ")
    if type(inputs) is not tuple or any(not isinstance(value, torch.Tensor) for value in inputs):
        raise ExportStateUnavailable("state observation requires the actual flat runtime tensor inputs")
    state = dict(exported.state_dict)
    constants = dict(exported.constants)
    tensors, rows, groups, runtime = [], [], {}, iter(inputs)
    for ordinal, spec in enumerate(specs):
        kind, target, argument = spec.kind, spec.target, getattr(spec.arg, "name", None)
        item = manifest[str(ordinal)]
        if not isinstance(item, dict) or not isinstance(argument, str):
            raise ExportStateUnavailable("prepared input has no exact tensor argument/manifest mapping")
        if kind == InputKind.USER_INPUT:
            tensor = next(runtime, None)
            if item != {"kind": "input", "name": argument}:
                raise ExportStateUnavailable("runtime argument differs from its actual exported placeholder")
        elif kind in (InputKind.PARAMETER, InputKind.BUFFER):
            tensor = state.get(target)
            role = "buffer" if kind == InputKind.BUFFER else "param"
            if (item.get("kind") != role or item.get("stub") or item.get("error")
                    or (item.get("weight") != target and item != {"kind": role, "name": argument})):
                raise ExportStateUnavailable("prepared stored argument differs from its original role/target")
        elif kind == InputKind.CONSTANT_TENSOR:
            tensor = constants.get(target)
            if item.get("kind") not in {"constant", "input"} or item.get("name") != argument:
                raise ExportStateUnavailable("prepared constant differs from its exported placeholder")
        else:
            raise ExportStateUnavailable("prepared signature contains an unsupported argument kind")
        if tensor is None:
            raise ExportStateUnavailable("prepared tensor has no original state/input owner")
        tensor_record, owner = _tensor(tensor, torch)
        tensors.append(tensor)
        group = groups.setdefault(owner, len(groups))
        rows.append({"ordinal": ordinal, "kind": kind.name, "target": target, "argument": argument,
                     "storage_key": item.get("weight", argument), "storage_group": group, **tensor_record})
    if next(runtime, None) is not None:
        raise ExportStateUnavailable("runtime tensors exceed the exact prepared signature")
    for row in rows:
        if row["kind"] == "BUFFER" and sum(other["storage_group"] == row["storage_group"] for other in rows) != 1:
            raise ExportStateUnavailable("selected buffer aliases another source state/runtime tensor")
    # Snapshot only graph/metadata; preserve all actual tensor owners in the copy
    # memo so observing a large model does not duplicate its parameter storage.
    # The importer may uniquify a reserved FX node.name after its prepared
    # snapshot (for example input -> input_1). Targets and signature arguments
    # must still correspond exactly. Restore names only on this observation
    # copy, retaining the actual live spelling as separately recorded evidence.
    placeholders = [node for node in exported.graph.nodes if node.op == "placeholder"]
    originals = [node for node in prepared.get("nodes", ()) if node.get("op") == "placeholder"]
    if (len(placeholders) != len(rows) or len(originals) != len(rows)
            or len({node.name for node in placeholders}) != len(rows)
            or len({row["argument"] for row in rows}) != len(rows)):
        raise ExportStateUnavailable("prepared graph and signature have different placeholder rosters")
    placeholder_names = []
    for node, row, original in zip(placeholders, rows, originals, strict=True):
        if (str(node.target) != row["argument"] or original.get("target") != row["argument"]
                or original.get("input_kind") != row["kind"] or original.get("input_target") != row["target"]):
            raise ExportStateUnavailable("prepared placeholder target differs from the signature argument")
        placeholder_names.append({"ordinal": row["ordinal"], "name": node.name, "target": str(node.target),
                                  "argument": row["argument"], "prepared_name": original["target"]})
    graph = copy.deepcopy(exported.graph_module, {id(tensor): tensor for tensor in tensors})
    for node, row in zip((node for node in graph.graph.nodes if node.op == "placeholder"), rows, strict=True):
        node.name = row["argument"]
    view = SimpleNamespace(graph_module=graph, graph_signature=exported.graph_signature,
                           range_constraints=exported.range_constraints,
                           _m2m_trace_eliminations=getattr(exported, "_m2m_trace_eliminations", ()))
    if snapshot_exported_program(view, stage="prepared")["sha256"] != prepared["sha256"]:
        raise ExportStateUnavailable("returned prepared export differs from the imported prepared graph")
    # Independently corroborate externally stored buffer bytes. Extra-archive
    # buffers retain their actual payload hash for the later typed materializer;
    # no evaluator archive needs to be opened by this observer.
    weights = root / "weights.safetensors"
    if weights.is_symlink() or not weights.is_file():
        raise ExportStateUnavailable("externalized state archive must be a regular selected file")
    with weights.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ExportStateUnavailable("externalized state has no safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if size > 64 * 1024 * 1024:
            raise ExportStateUnavailable("externalized state header exceeds the observation bound")
        header = json.loads(stream.read(size))
        for row in rows:
            item = manifest[str(row["ordinal"])]
            if row["kind"] != "BUFFER" or "weight" not in item:
                continue
            entry = header.get(item["weight"])
            if (not isinstance(entry, dict) or entry.get("shape") != row["shape"]
                    or item.get("shape") != row["shape"] or item.get("dtype") != row["dtype"]):
                raise ExportStateUnavailable("stored buffer shape/dtype differs from the actual export")
            offsets = entry.get("data_offsets")
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(i) is not int for i in offsets) or not 0 <= offsets[0] <= offsets[1]
                    or offsets[1] - offsets[0] != row["payload_bytes"]):
                raise ExportStateUnavailable("stored buffer has no exact observed payload extent")
            stream.seek(8 + size + offsets[0])
            if _sha(stream.read(row["payload_bytes"])) != row["payload_sha256"]:
                raise ExportStateUnavailable("stored buffer differs from the prepared export's original bytes")
    for tensor, row in zip(tensors, rows, strict=True):
        identity_fields = {"ordinal", "kind", "target", "argument", "storage_key", "storage_group"}
        if _tensor(tensor, torch)[0] != {key: value for key, value in row.items() if key not in identity_fields}:
            raise ExportStateUnavailable("prepared source state changed during observation")
    if _file(root, "weights.safetensors.manifest.json") != manifest_bytes:
        raise ExportStateUnavailable("externalized argument manifest changed during observation")
    return {"schema": SCHEMA, "status": "observed", "source_sha256": _sha(original_ir),
            "prepared_graph_sha256": prepared["sha256"], "manifest_sha256": _sha(manifest_bytes),
            "inputs": rows, "outputs": outputs, "operator_effects": effects,
            "placeholder_names": placeholder_names,
            "observer_source_sha256": _sha(Path(__file__).read_bytes()),
            "snapshot_source_sha256": _sha(Path(snapshot_exported_program.__code__.co_filename).read_bytes()),
            "scope": "prepared export/state observation only; original-caller freeze consent and independent "
                     "compiler/reference admission are required"}
