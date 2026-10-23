"""Deterministic backing storage for native CPU reads of portable inputs."""

from math import prod

from merlin.targetgen.capture_refusals import InputStorageUnavailable


def _geometry(tensor):
    if not isinstance(tensor, dict) or tensor.get("kind") != "tensor":
        raise ValueError("raw input storage requires a canonical tensor document")
    shape, stride, offset = tensor["shape"], tensor["stride"], tensor["storage_offset"]
    if (
        len(shape) != len(stride)
        or any(type(v) is not int or v < 0 for v in [*shape, *stride, offset])
        or "values" not in tensor
    ):
        raise ValueError("raw input storage requires recorded static nonnegative tensor geometry")
    return shape, stride, offset


def _raw_coverage(tensor, rows):
    shape, stride, offset = _geometry(tensor)
    span = offset + (1 + sum((n - 1) * s for n, s in zip(shape, stride)) if all(shape) else 0)
    # Nonnegative strides cannot fill a gap below the next sorted stride. Until
    # that gap, shifted copies of the covered interval overlap or touch exactly.
    prefix = 1 if all(shape) else 0
    dimensions = sorted(zip(stride, shape)) if prefix else ()
    for step, extent in dimensions:
        if prefix >= rows or (step > prefix and extent > 1):
            break
        prefix += (extent - 1) * step
    if prefix >= rows:
        return None
    missing = offset + prefix
    return {
        "shape": shape,
        "stride": stride,
        "storage_offset": offset,
        "required_raw_range": [offset, offset + rows],
        "first_unrecorded_address": missing,
        "recorded_storage_span": span,
        "cause": "raw_read_out_of_bounds" if missing >= span else "unrecorded_storage_hole",
    }


def native_input_storage_plan(overload, arguments):
    """Describe zero-initialized backing spans required by native CPU reads.

    The portable host ABI defines every unrecorded backing byte as zero. Raw
    pointer reads extend the allocation, never the logical shape or strides.
    CPU layer-norm backward reads one statistic per row from data_ptr[row],
    independently of the logical statistics strides.
    """
    if overload != "aten.native_layer_norm_backward.default":
        return None

    def argument(index, name):
        return positional[index] if index < len(positional) else arguments["kwargs"][name]

    try:
        positional = arguments["args"]["items"]
        source = argument(1, "input")
        shape, _, _ = _geometry(source)
        normalized = argument(2, "normalized_shape")
        if isinstance(normalized, dict) and normalized.get("kind") == "tuple":
            normalized = normalized["items"]
        if (
            not normalized
            or any(type(v) is not int or v <= 0 for v in normalized)
            or len(normalized) > len(shape)
            or shape[-len(normalized) :] != normalized
        ):
            raise ValueError("raw statistic row count requires a matching positive normalized shape")
        rows = prod(shape) // prod(normalized)
        tensors = []

        def collect(value):
            if isinstance(value, dict):
                if value.get("kind") == "tensor":
                    tensors.append(value)
                else:
                    for child in value.values():
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(arguments)
        reads = []
        for index, name in ((3, "mean"), (4, "rstd")):
            document = argument(index, name)
            _, _, offset = _geometry(document)
            reads.append({
                "argument": name,
                "argument_index": index,
                "input_index": next(i for i, tensor in enumerate(tensors) if tensor is document),
                "access": "contiguous_elements_from_data_pointer",
                "element_count": rows,
                "minimum_storage_elements": offset + rows,
                "previously_unrecorded": _raw_coverage(document, rows),
            })
    except (KeyError, TypeError, ValueError, StopIteration) as exc:
        raise InputStorageUnavailable(
            str(exc), details={"overload": overload, "cause": "unrepresentable_input_storage"}
        ) from exc
    return {
        "schema": "merlin.host_input_storage.v1",
        "unrecorded_bytes": "zero",
        "allocation": "union_of_logical_and_native_read_spans",
        "authority": "documented_native_cpu_access_pattern",
        "source_adaptation": "logical_gather_with_zero_for_unrecorded_addresses",
        "raw_reads": reads,
    }


def apply_native_input_storage(source, case):
    """Carry the native backing span to the compiled caller input allocator."""
    plan = native_input_storage_plan(case["overload"], case["arguments"])
    if plan is None:
        return source
    import json

    from xdsl.dialects.builtin import StringAttr

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.result_layout import ATTRIBUTE

    module = parse_mlir_text(source)
    if ATTRIBUTE not in module.attributes:
        raise InputStorageUnavailable("native raw reads require a captured input storage layout")
    layout = json.loads(module.attributes[ATTRIBUTE].data)
    for read in plan["raw_reads"]:
        record = layout["inputs"][read["input_index"]]
        record["minimum_storage_elements"] = max(
            record.get("minimum_storage_elements", 0), read["minimum_storage_elements"]
        )
    module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout, sort_keys=True))
    return str(module) + "\n"


def native_argument_projection(plan, tensors):
    """Render the logical gather equivalent of the documented raw pointer reads.

    This avoids asking a frontend FakeTensor to retain spare backing capacity.
    Original caller tensors stay inputs and post-call observations; only the
    native kernel's read-only statistic arguments use the zero-completed view.
    """
    import itertools

    if plan is None:
        return ""
    statements = []
    for read in plan["raw_reads"]:
        i = read["input_index"]
        document = tensors[i]
        shape, strides, offset = _geometry(document)
        addresses = {}
        for coordinate in itertools.product(*(range(n) for n in shape)):
            address = offset + sum(c * s for c, s in zip(coordinate, strides))
            addresses.setdefault(address, coordinate)
        if prod(shape) < read["element_count"]:
            raise InputStorageUnavailable("native statistic shape has fewer elements than its row reads")
        if prod(shape) == 0:
            continue
        elements = []
        for index in range(prod(shape)):
            coordinate = addresses.get(offset + index)
            elements.append(
                f"input_{i}.reshape(-1)[{sum(c * prod(shape[j + 1:]) for j, c in enumerate(coordinate))}:"
                f"{1 + sum(c * prod(shape[j + 1:]) for j, c in enumerate(coordinate))}]" if coordinate is not None
                else f"torch.zeros((1,), dtype=input_{i}.dtype)"
            )
        expression = f"torch.cat([{', '.join(elements)}]).reshape({tuple(shape)!r})"
        # The static loader decoder constructs tuple positional arguments.
        slot, name = read["argument_index"], read["argument"]
        statements.append(
            f"arguments['args'] = tuple({expression} if j == {slot} else value "
            f"for j, value in enumerate(arguments['args'])) if len(arguments['args']) > {slot} "
            f"else arguments['args']"
        )
        statements.append(f"if len(arguments['args']) <= {slot}: arguments['kwargs'][{name!r}] = {expression}")
    return "\n        ".join(statements)
