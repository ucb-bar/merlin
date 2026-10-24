"""Grade observable compiled semantics while declaring fields absent from its ABI."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from merlin.targetgen.core_aten_batch_grade import _DTYPES, _EXPECTED_DTYPES, _decode_storage


def tensor_paths(value, path):
    if isinstance(value, Mapping) and value.get("kind") == "tensor":
        return [(path, value)]
    if isinstance(value, Mapping) and value.get("kind") == "tuple":
        return [leaf for i, item in enumerate(value["items"]) for leaf in tensor_paths(item, f"{path}[{i}]")]
    if isinstance(value, Mapping):
        return [leaf for key, item in value.items() for leaf in tensor_paths(item, f"{path}.{key}" if path else key)]
    if isinstance(value, list):
        return [leaf for i, item in enumerate(value) for leaf in tensor_paths(item, f"{path}[{i}]")]
    return []


def full_observation(record, raw, readback):
    def _encode(value):
        if isinstance(value, list):
            return [_encode(v) for v in value]
        if isinstance(value, complex):
            return dict(kind="complex", real=_encode(value.real), imag=_encode(value.imag))
        if isinstance(value, float) and (not math.isfinite(value) or (value == 0 and math.copysign(1, value) < 0)):
            return dict(kind="float", value=repr(value))
        return value

    boundary = record["semantic_boundary"]
    metadata = readback["metadata"]
    aliases = readback["aliases"]
    before = readback["pre_bytes"]

    def document(index, abi):
        layout = metadata[index]
        shape = list(layout["shape"])
        stride = list(layout["stride"])
        offset = layout["storage_offset"]
        dtype = abi["dtype"]
        if dtype.startswith("complex<"):
            ndtype = np.complex64 if dtype == "complex<f32>" else np.complex128
            logical = np.dtype(ndtype).name
            # Complex tensor lowering exposes a trailing real/imag component axis.
            # Interpret this representation only after checking the actual memref.
            if not shape or shape[-1] != 2 or stride[-1] != 1 or offset % 2 or any(s % 2 for s in stride[:-1]):
                raise ValueError("compiled complex result has incompatible component layout")
            shape, stride, offset = shape[:-1], [s // 2 for s in stride[:-1]], offset // 2
        else:
            ndtype, logical = _DTYPES[dtype], _EXPECTED_DTYPES[dtype]
        values = (
            np.frombuffer(raw[index], ndtype) if dtype.startswith("complex<") else _decode_storage(raw[index], dtype)
        ).reshape(shape)
        return dict(
            kind="tensor",
            dtype=logical,
            shape=shape,
            stride=stride,
            storage_offset=offset,
            requires_grad=layout["requires_grad"],
            values=_encode(values.tolist()),
        )

    # Only the argument/output TREE is reused. Tensor metadata and values come from readback.
    def replace(value, documents):
        if isinstance(value, dict) and value.get("kind") == "tensor":
            return next(documents)
        if isinstance(value, dict):
            return {key: replace(item, documents) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item, documents) for item in value]
        return value

    post_docs = [
        document(index, abi) for index, abi in zip(boundary["post_indices"], boundary["post_abi"], strict=True)
    ]

    def consume_tree(builder, tree, docs):
        documents = iter(docs)
        result = builder(tree, documents)
        sentinel = object()
        if next(documents, sentinel) is not sentinel:
            raise ValueError("boundary documents remain after reconstructing tree")
        return result

    post = consume_tree(replace, record["case"]["arguments"], post_docs)
    output_docs = [
        document(index, abi) for index, abi in zip(record["output_indices"], record["output_abi"], strict=True)
    ]

    # Scalar return wrappers have one rank-zero tensor ABI. Preserve the source pytree kind.
    def output_tree(value, documents):
        if isinstance(value, dict) and value.get("kind") == "tensor":
            return next(documents)
        if isinstance(value, dict) and value.get("kind") == "tuple":
            return {**value, "items": [output_tree(item, documents) for item in value["items"]]}
        if isinstance(value, list):
            return [output_tree(item, documents) for item in value]
        tensor = next(documents)
        number = tensor["values"]
        if isinstance(value, bool):
            return bool(number)
        if isinstance(value, int):
            return int(number)
        return number

    output = consume_tree(output_tree, record["case"]["expected"], output_docs)
    paths = tensor_paths(record["case"]["arguments"], "")
    mutated = []
    for (path, initial), observed, source_index, result_index in zip(
        paths, post_docs, record["input_indices"], boundary["post_indices"], strict=True
    ):
        # The corpus defines mutation by byte equality and dtype/shape/stride (not offset).
        if bytes.fromhex(before[str(source_index)]) != raw[result_index] or any(
            initial.get(key) != observed.get(key) for key in ("dtype", "shape", "stride")
        ):
            mutated.append(path)
    output_paths = tensor_paths(output, "result")
    relationships = []
    tensor_documents = {id(doc) for _, doc in output_paths}
    tensor_indices = [
        index for index, doc in zip(record["output_indices"], output_docs, strict=True) if id(doc) in tensor_documents
    ]
    for (path, _), index in zip(output_paths, tensor_indices, strict=True):
        for (input_path, _), post_index in zip(paths, boundary["post_indices"], strict=True):
            if post_index in aliases[index]:
                relationships.append(f"{path}->{input_path}")
    return dict(
        status="executed",
        execution_kind="simulator",
        output=output,
        post_arguments=post,
        mutated_arguments=sorted(mutated),
        output_input_aliases=sorted(relationships),
        evidence={"substrate": "spike_functional", "boundary": "compiled_result_memrefs"},
    )


def validate_full(record, raw, readback):
    from merlin.targetgen.core_aten_batch_grade import _grade_output, _leaves
    from merlin.targetgen.core_aten_eval import _without_values

    if not record.get("semantic_boundary"):
        return None
    if readback is None:
        return dict(status="ungradable", semantic_scope="unknown", reason="exported boundary lacks semantic readback")
    scope = "full"
    try:
        indices = [*record["output_indices"], *record["semantic_boundary"]["post_indices"]]
        imposed = sorted({field for i in indices for field in readback["metadata"][i].get("imposed_by_harness", [])})
        # No other field has an admitted unobserved representation.
        if any(field != "requires_grad" for field in imposed):
            raise ValueError("unsupported imposed boundary field")
        for i in indices:
            layout = readback["metadata"][i]
            if (layout["requires_grad"] is None) != ("requires_grad" in layout.get("imposed_by_harness", [])):
                raise ValueError("autograd observation disagrees with scope marker")
        if imposed:
            scope = "values+mutation+observed-metadata"
        observation = full_observation(record, raw, readback)
        case = record["case"]
        if case["comparison"] not in ("metadata", "torch_close"):
            raise ValueError("unknown comparison policy")

        def metadata_matches(actual, expected):
            if isinstance(actual, dict) and isinstance(expected, dict):
                excluded = (
                    {"requires_grad"}
                    if actual.get("kind") == "tensor" and actual.get("requires_grad") is None
                    else set()
                )
                keys = actual.keys() - excluded
                return keys == expected.keys() - excluded and all(
                    metadata_matches(actual[key], expected[key]) for key in keys
                )
            if isinstance(actual, list) and isinstance(expected, list):
                return len(actual) == len(expected) and all(
                    metadata_matches(a, e) for a, e in zip(actual, expected, strict=True)
                )
            return actual == expected

        metadata_errors = []
        for key, expected in [("output", case["expected"]), ("post_arguments", case["post_arguments"])]:
            if not metadata_matches(_without_values(observation[key]), _without_values(expected)):
                metadata_errors.append(f"{key} metadata mismatch")
        if sorted(observation["output_input_aliases"]) != sorted(case["output_input_aliases"]):
            metadata_errors.append(
                f"output_input_aliases mismatch: expected {case['output_input_aliases']}, "
                f"got {observation['output_input_aliases']}"
            )

        value_errors = []
        if case["comparison"] == "torch_close":
            for boundary_index, (indices, abis, expected) in enumerate(
                [
                    (record["output_indices"], record["output_abi"], _leaves(case["expected"])),
                    (
                        record["semantic_boundary"]["post_indices"],
                        record["semantic_boundary"]["post_abi"],
                        [doc for _, doc in tensor_paths(case["post_arguments"], "")],
                    ),
                ]
            ):
                if boundary_index == 0 and (record.get("native_floor_for_values") or {}).get("status") == "pass":
                    continue  # The independent exact byte/error gate owns native device values.
                for index, abi, doc in zip(indices, abis, expected, strict=True):
                    shape = readback["metadata"][index]["shape"]
                    if abi["dtype"].startswith("complex<"):
                        shape = shape[:-1]
                    verdict = _grade_output(raw[index], {**abi, "shape": shape}, doc, case)
                    if verdict["status"] != "pass":
                        value_errors.append(f"boundary value mismatch: {verdict}")
        if sorted(observation["mutated_arguments"]) != sorted(case["mutated_arguments"]):
            value_errors.append(
                f"mutated_arguments mismatch: expected {case['mutated_arguments']}, "
                f"got {observation['mutated_arguments']}"
            )
        errors = metadata_errors + value_errors
        return {
            "status": "mismatch" if errors else "pass",
            "semantic_scope": scope,
            "imposed_by_harness": imposed,
            "values_mutation_status": "mismatch" if value_errors else "pass",
            "values_mutation_reason": "; ".join(value_errors),
            "observed_metadata_status": "mismatch" if metadata_errors else "pass",
            "observation": observation,
            **({"reason": "; ".join(errors)} if errors else {}),
        }
    except (KeyError, IndexError, ValueError, TypeError, StopIteration) as exc:
        return {"status": "ungradable", "semantic_scope": scope, "reason": f"incomplete boundary readback: {exc}"}
