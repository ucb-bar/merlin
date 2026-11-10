"""Route authenticated application output words into declared invocation inputs.

Edges require identical shape and dtype.
This only loads caller-owned buffers; it grants no comparison, device residency,
in-place update, or hardware qualification authority.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np


_STORAGE = {
    "bf16": "uint16", "fp8_e4m3": "uint8", "f16": "float16",
    "f32": "float32", "f64": "float64", "i1": "uint8",
    "i8": "uint8", "i16": "int16", "i32": "int32", "i64": "int64",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate application record field: {key}")
        result[key] = value
    return result


def _ports(record, role):
    interface = record.get("interface")
    if not isinstance(interface, dict):
        raise ValueError("application record needs an explicit interface")
    names, specs = interface.get(role), interface.get(role[:-1] + "_types")
    if (not isinstance(names, list) or not isinstance(specs, list) or len(names) != len(specs)
            or any(not isinstance(name, str) or not name for name in names)
            or (role == "inputs" and len(set(names)) != len(names))):
        raise ValueError(f"application {role} need declared names and explicit tensor types")
    for spec in specs:
        if (not isinstance(spec, dict) or not isinstance(spec.get("dtype"), str)
                or spec["dtype"] not in _STORAGE
                or not isinstance(spec.get("shape"), list)
                or any(type(n) is not int or n < 0 for n in spec["shape"])):
            raise ValueError(f"application {role} have an unsupported tensor signature")
    return specs


def _authenticated_outputs(producer_root: Path, execution_root: Path):
    """Read and authenticate every published output, including unselected ones."""
    compiled_bytes = (Path(producer_root) / "compiled.json").read_bytes()
    receipt_bytes = (Path(execution_root) / "execution.json").read_bytes()
    archive_bytes = (Path(execution_root) / "actual.npz").read_bytes()
    producer = json.loads(compiled_bytes, object_pairs_hook=_unique_object)
    receipt = json.loads(receipt_bytes, object_pairs_hook=_unique_object)
    if not isinstance(producer, dict) or not isinstance(receipt, dict):
        raise ValueError("application bindings require object records")
    if producer.get("status") != "COMPILED" or receipt.get("status") != "EXECUTED_UNCOMPARED":
        raise ValueError("producer requires a compiled application and completed execution receipt")
    if receipt.get("compiled_sha256") != sha256(compiled_bytes).hexdigest():
        raise ValueError("producer execution compiled publication differs")
    for field in ("source_request_digest", "source_model_sha256"):
        if not isinstance(producer.get(field), str) or not producer[field] or receipt.get(field) != producer[field]:
            raise ValueError(f"producer execution identity differs: {field}")
    outputs = _ports(producer, "outputs")
    metadata = receipt.get("outputs")
    if not isinstance(metadata, list) or len(metadata) != len(outputs):
        raise ValueError("producer execution output roster differs")
    actual = []
    with np.load(BytesIO(archive_bytes), allow_pickle=False) as archive:
        expected = [f"output_{i}" for i in range(len(outputs))]
        if len(archive.files) != len(expected) or set(archive.files) != set(expected):
            raise ValueError("producer actual archive output roster differs")
        for i, spec in enumerate(outputs):
            value, row = archive[f"output_{i}"], metadata[i]
            if (not isinstance(row, dict) or value.dtype != np.dtype(_STORAGE[spec["dtype"]])
                    or list(value.shape) != spec["shape"]
                    or row.get("dtype") != value.dtype.str or row.get("shape") != list(value.shape)
                    or row.get("sha256") != sha256(value.tobytes()).hexdigest()):
                raise ValueError(f"producer output_{i} bytes or tensor signature differ")
            actual.append(value.copy(order="C"))
    identities = {
        "producer_compiled_sha256": sha256(compiled_bytes).hexdigest(),
        "producer_execution_sha256": sha256(receipt_bytes).hexdigest(),
        "producer_actual_sha256": sha256(archive_bytes).hexdigest(),
        "source_request_digest": producer["source_request_digest"],
        "source_model_sha256": producer["source_model_sha256"],
    }
    return outputs, actual, metadata, identities


def greedy_token_from_bf16(logits: np.ndarray) -> np.ndarray:
    """Select the last sequence row's lowest-index maximum from bf16[1,S,V].

    BF16 storage words are widened exactly to float32 before selection. NaNs in
    the logits publication are refused; equal maxima, including signed zeros
    and infinities, select the lowest vocabulary index.
    """
    if (not isinstance(logits, np.ndarray) or logits.dtype != np.dtype("uint16")
            or logits.ndim != 3 or logits.shape[0] != 1
            or logits.shape[1] < 1 or logits.shape[2] < 1):
        raise ValueError("greedy logits require nonempty bf16[1, sequence, vocabulary]")
    values = (logits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    if np.isnan(values).any():
        raise ValueError("greedy logits contain NaN")
    return np.array([[int(np.argmax(values[0, -1]))]], dtype=np.int64)


def load_output_bindings(
    producer_root: Path, execution_root: Path, consumer_record: dict, bindings: dict[int, int],
    *, greedy: dict | None = None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Authenticate all outputs, then route exact edges and optional greedy input.

    ``bindings`` maps consumer input indices to producer output indices. Optional
    ``greedy`` contains ``input_index`` and ``output_index`` for an example-owned
    bf16[1,S,V] last-row argmax into i64[1,1]. Every producer output is checked
    before either operation; returned arrays own contiguous storage.
    """
    if (not isinstance(bindings, dict) or (not bindings and greedy is None)
            or any(type(i) is not int or type(j) is not int or i < 0 or j < 0
                   for i, j in bindings.items())):
        raise ValueError("output bindings need nonempty nonnegative integer index pairs")
    if not isinstance(consumer_record, dict):
        raise ValueError("application bindings require object records")
    inputs = _ports(consumer_record, "inputs")
    if greedy is not None:
        if (not isinstance(greedy, dict) or set(greedy) != {"input_index", "output_index"}
                or any(type(greedy[key]) is not int or greedy[key] < 0 for key in greedy)
                or greedy["input_index"] in bindings):
            raise ValueError("greedy binding needs distinct nonnegative integer input and output indices")
    outputs, actual, metadata, identities = _authenticated_outputs(producer_root, execution_root)
    edges = list(bindings.items()) + ([] if greedy is None else [(greedy["input_index"], greedy["output_index"])])
    if any(i >= len(inputs) or j >= len(outputs) for i, j in edges):
        raise ValueError("output binding index exceeds the declared application ABI")
    selected, provenance = {}, []
    for target, source in sorted(bindings.items()):
        incoming, outgoing = inputs[target], outputs[source]
        if incoming["shape"] != outgoing["shape"] or incoming["dtype"] != outgoing["dtype"]:
            raise ValueError("output binding shape or dtype differs across application ABIs")
        key = f"input_{target}"
        selected[key] = actual[source].copy(order="C")
        provenance.append({**identities, "input_index": target, "output_index": source,
                           "dtype": outgoing["dtype"], "shape": outgoing["shape"],
                           "sha256": metadata[source]["sha256"]})
    if greedy is not None:
        target, source = greedy["input_index"], greedy["output_index"]
        incoming, outgoing = inputs[target], outputs[source]
        if incoming["dtype"] != "i64" or incoming["shape"] != [1, 1]:
            raise ValueError("greedy token input requires i64[1,1]")
        if outgoing["dtype"] != "bf16":
            raise ValueError("greedy logits require bf16")
        token = greedy_token_from_bf16(actual[source])
        selected[f"input_{target}"] = token
        provenance.append({**identities, "operation": "greedy_last_token",
                           "input_index": target, "output_index": source,
                           "dtype": "i64", "shape": [1, 1],
                           "sha256": sha256(token.tobytes()).hexdigest(),
                           "producer_output_sha256": metadata[source]["sha256"],
                           "source_dtype": "bf16", "source_shape": outgoing["shape"],
                           "sequence_index": actual[source].shape[1] - 1,
                           "token_id": int(token[0, 0]), "tie_break": "lowest_index"})
    return selected, provenance
