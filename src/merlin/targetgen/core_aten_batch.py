"""Build one runnable MLIR bundle from independently captured Core ATen cases.

The corpus supplies the denominator and expected documents. A case is included whenever
its capture has a clean Linalg module, regardless of capture provenance labels. Cases
without such a module remain explicit unavailable records in the result map.
"""

from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_DTYPES = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    # NPZ carries widened values; c_runtime packs them into the declared bf16 storage.
    "bf16": np.float32,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.bool_,
}
_COMPLEX_DTYPES = {"complex<f32>": np.complex64, "complex<f64>": np.complex128}


def _json_bytes(document: Any) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=True) + "\n").encode("utf-8")


def _tensor_type(abi: dict[str, Any]) -> str:
    shape = [*abi["shape"]]
    dtype = abi["dtype"]
    if dtype in _COMPLEX_DTYPES:
        shape.append(2)
        dtype = dtype[len("complex<") : -1]
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported batch ABI dtype {dtype!r}")
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError(f"unsupported batch ABI shape {shape!r}")
    return "tensor<" + "".join(f"{dim}x" for dim in shape) + dtype + ">"


def _tensor_documents(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and value.get("kind") == "tensor":
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _tensor_documents(child)]
    if isinstance(value, list):
        return [item for child in value for item in _tensor_documents(child)]
    return []


def _complex_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("kind") == "float":
        return float(value["value"])
    if isinstance(value, dict) and value.get("kind") == "complex":
        return complex(_complex_value(value["real"]), _complex_value(value["imag"]))
    if isinstance(value, list):
        return [_complex_value(item) for item in value]
    return value


def _physical_abi(abi: dict[str, Any]) -> dict[str, Any]:
    if abi["dtype"] in _COMPLEX_DTYPES:
        return {"dtype": abi["dtype"][len("complex<") : -1], "shape": [*abi["shape"], 2]}
    return {"dtype": abi["dtype"], "shape": [*abi["shape"]]}


def _matching_brace(source: str, start: int, opener: str = "{", closer: str = "}") -> int:
    depth = 0
    quoted = False
    escaped = False
    for pos in range(start, len(source)):
        char = source[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError("unbalanced function braces")


def _split_top(text: str) -> list[str]:
    """Split on commas outside (), <>, [] and {} nesting and outside quoted strings."""
    items, depth, quoted, escaped, start = [], 0, False, False, 0
    for pos, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "(<[{":
            depth += 1
        elif char == ">" and pos > 0 and text[pos - 1] == "-":
            pass  # A function-type arrow does not close an angle-delimited type.
        elif char in ")>]}":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:pos].strip())
            start = pos + 1
    tail = text[start:].strip()
    return [*items, tail] if tail or items else []


def _generic_function_parts(source: str) -> tuple[list[str], str, list[str], list[str], str]:
    """The generic-form spelling of :func:`_function_parts` (``"func.func"() <{...}> ({...})``).

    Every op keeps its exact bytes; only the entry block label and terminal return are read.
    """
    if not source.lstrip().startswith('"builtin.module"()'):
        raise ValueError("generic capture must be one builtin.module")
    module_open = source.index("({", source.index('"builtin.module"()')) + 1
    module_close = _matching_brace(source, module_open)
    forward = []
    position = module_open
    while True:
        at = source.find('"func.func"() <{', position, module_close)
        if at < 0:
            break
        props_open = at + len('"func.func"() <')
        props_close = _matching_brace(source, props_open)
        if 'sym_name = "forward"' in _split_top(source[props_open + 1 : props_close]):
            forward.append((at, props_close))
        position = props_close
    if len(forward) != 1:
        raise ValueError("capture must contain exactly one @forward function")
    at, props_close = forward[0]
    if not source.startswith("> ({", props_close + 1):
        raise ValueError("missing @forward body")
    body_start = props_close + len("> ({")
    body_end = _matching_brace(source, body_start)
    op_end = source.find("\n", body_end)
    op_end = module_close if op_end < 0 or op_end > module_close else op_end
    body = source[body_start + 1 : body_end]
    names = []
    first = body.lstrip("\n")
    if first.lstrip().startswith("^"):
        label, sep, rest = first.partition("\n")
        label = label.strip()
        if not label.endswith(":") or "(" not in label:
            raise ValueError("unsupported @forward entry block")
        for arg in _split_top(label[label.index("(") + 1 : label.rindex(")")]):
            name, sep, _ = arg.partition(":")
            if not sep or not name.strip().startswith("%"):
                raise ValueError(f"unsupported @forward argument {arg!r}")
            names.append(name.strip())
        body = rest
    marker = '"func.return"('
    if body.count(marker) != 1:
        raise ValueError("capture must have one terminal func.return")
    return_at = body.rfind(marker)
    terminal = body[return_at:].strip()
    close = _matching_brace(terminal, len(marker) - 1, "(", ")")
    values = _split_top(terminal[len(marker) : close])
    signature = terminal[close + 1 :].strip()
    if signature.startswith("{"):
        signature = signature[_matching_brace(signature, 0) + 1 :].strip()
    if "\n" in terminal or not signature.startswith(": (") or not signature.endswith("-> ()"):
        raise ValueError("unsupported terminal func.return")
    types = _split_top(signature[len(": (") : -len("-> ()")].strip()[:-1])
    if not values or any(not value.startswith("%") for value in values) or len(types) != len(values):
        raise ValueError("unsupported returned SSA values")
    line_start = source.rfind("\n", module_open, at) + 1
    declarations = source[module_open + 1 : line_start] + source[op_end + 1 : module_close]
    return names, body[:return_at], values, types, declarations


def _function_parts(source: str) -> tuple[list[str], str, list[str], list[str], str]:
    """Extract the one forward body and terminal return from model2MLIR's module.

    Accepts the custom form and the generic form model2MLIR prints for traced captures.
    """
    mark = "func.func @forward("
    if mark not in source and '"func.func"()' in source:
        return _generic_function_parts(source)
    if source.count(mark) != 1:
        raise ValueError("capture must contain exactly one @forward function")
    start = source.index(mark) + len(mark)
    close = source.find(")", start)
    if close < 0:
        raise ValueError("missing @forward argument list")
    arguments = source[start:close]
    names = []
    if arguments.strip():
        for arg in arguments.split(","):
            name, sep, _ = arg.strip().partition(":")
            if not sep or not name.startswith("%"):
                raise ValueError(f"unsupported @forward argument {arg!r}")
            names.append(name)
    body_start = source.find("{", close)
    if body_start < 0:
        raise ValueError("missing @forward body")
    body_end = _matching_brace(source, body_start)
    body = source[body_start + 1 : body_end]
    marker = "func.return "
    if body.count(marker) != 1:
        raise ValueError("capture must have one terminal func.return")
    return_at = body.rfind(marker)
    terminal = body[return_at:].strip()
    if "\n" in terminal or ":" not in terminal:
        raise ValueError("unsupported terminal func.return")
    values = [item.strip() for item in terminal[len(marker) :].split(":", 1)[0].split(",")]
    if not values or any(not value.startswith("%") for value in values):
        raise ValueError("unsupported returned SSA values")
    types = [item.strip() for item in terminal.split(":", 1)[1].split(",")]
    module_start = source.index("{")
    if "attributes" in source[:module_start]:
        module_start = source.index("{", _matching_brace(source, module_start) + 1)
    declarations = source[module_start + 1 : source.index(mark)] + source[body_end + 1 : source.rfind("}")]
    return names, body[:return_at], values, types, declarations


def caller_input_storage(source: str, records: list[dict]) -> str:
    """Bind legacy normalized calls to original caller input storage at execution.

    Input shapes, strides and offsets come only from original argument documents.
    Directly forwarded input observations retain that input's storage layout.
    Computed results remain natural compiler observations; no expected post-call
    metadata or answers supply this contract. Existing capture contracts win.
    """
    from xdsl.dialects.builtin import StringAttr

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.result_layout import ATTRIBUTE

    module = parse_mlir_text(source)
    if ATTRIBUTE in module.attributes:
        return source
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    arguments = list(function.body.block.args)
    returned = next(op for op in function.body.block.ops if op.name == "func.return")
    inputs = [None] * len(arguments)
    results = [None] * len(returned.operands)
    for record in records:
        documents = _tensor_documents(record["case"]["arguments"])
        if len(documents) != len(record["input_indices"]):
            raise ValueError("caller tensor count differs from compiled input ABI")
        for document, index, abi in zip(documents, record["input_indices"], record["input_abi"], strict=True):
            shape = list(arguments[index].type.get_shape())
            physical = _physical_abi(abi)
            if (
                abi["shape"] != document["shape"]
                or shape != physical["shape"]
                or str(arguments[index].type.element_type) != physical["dtype"]
            ):
                raise ValueError("legacy caller storage requires an unchanged tensor input ABI")
            inputs[index] = {key: copy.deepcopy(document[key]) for key in ("shape", "stride", "storage_offset")}
            if abi["dtype"] in _COMPLEX_DTYPES:
                inputs[index] = dict(
                    shape=physical["shape"],
                    stride=[2 * step for step in document["stride"]] + [1],
                    storage_offset=2 * document["storage_offset"],
                )
        posts = record["semantic_boundary"]["post_indices"]
        for index, post in zip(record["input_indices"], posts, strict=True):
            if returned.operands[post] is arguments[index]:
                # This is an observation buffer, not an inferred original-result
                # alias or a caller-storage write-back grant. Actual result aliases
                # still have to agree with independent compiled descriptor readback.
                results[post] = copy.deepcopy(inputs[index])
    module.attributes[ATTRIBUTE] = StringAttr(json.dumps(dict(inputs=inputs, results=results)))
    module.verify()
    return str(module) + "\n"


def caller_boundary_layouts(source: str, semantic_io: dict, records: list[dict]) -> dict:
    """Restore caller layouts for post results that directly forward input SSA values.

    Exported placeholder metadata can describe normalized compiler tensors. A direct
    input return preserves the caller's logical layout, independently of that packed
    tensor ABI. Mutation results retain their same-conversion exported metadata.
    Only source argument metadata is used here, never expected post-call observations.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        raise ValueError("capture must contain exactly one @forward function")
    block = functions[0].body.block
    returns = [op for op in block.ops if op.name == "func.return"]
    if len(returns) != 1 or returns[0] is not block.last_op:
        raise ValueError("capture must have one terminal func.return")
    names, returned = block.args, returns[0].operands
    boundary = copy.deepcopy(semantic_io)
    for record in records:
        arguments = _tensor_documents(record["case"]["arguments"])
        inputs = record["input_indices"]
        posts = record["semantic_boundary"]["post_indices"]
        if len(arguments) != len(inputs) or len(inputs) != len(posts):
            raise ValueError("caller tensor count differs from semantic boundary")
        for document, input_index, post_index in zip(arguments, inputs, posts):
            entry = boundary["outputs"][post_index]
            if entry.get("role") == "user_input_mutation":
                continue
            if returned[post_index] is not names[input_index]:
                continue
            for key in ("dtype", "shape"):
                if entry[key] != document[key]:
                    raise ValueError("pass-through caller type differs from exported boundary")
            for key in ("stride", "storage_offset", "requires_grad"):
                entry[key] = copy.deepcopy(document[key])
    return boundary


def _rename_ssa(source: str, prefix: str, argument_names: dict[str, str]) -> str:
    """Rename SSA and block labels outside quoted attributes, preserving MLIR punctuation."""
    out: list[str] = []
    pos = 0
    quoted = False
    escaped = False
    while pos < len(source):
        char = source[pos]
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            pos += 1
            continue
        if char == '"':
            quoted = True
            out.append(char)
            pos += 1
            if "".join(out[-12:-1]).endswith("sym_name = "):
                # A generic-form symbol definition is a string; keep it paired with its @references.
                out.append(prefix)
            continue
        if char not in ("%", "^", "@"):
            out.append(char)
            pos += 1
            continue
        end = pos + 1
        while end < len(source) and (source[end].isalnum() or source[end] in "_.$-"):
            end += 1
        token = source[pos:end]
        if len(token) == 1:
            out.append(token)
        elif char == "%":
            out.append(argument_names.get(token, "%" + prefix + token[1:]))
        else:
            out.append(char + prefix + token[1:])
        pos = end
    return "".join(out)


def _write_npz(path: Path, arrays: list[np.ndarray]) -> None:
    """Use fixed ZIP metadata so identical bundles have identical bytes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, array in enumerate(arrays):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"in{index}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, stream.getvalue())


def build_core_aten_batch(
    corpus: dict[str, Any],
    capture_root: str | Path,
    bundle_root: str | Path,
    *,
    selected_overloads: set[str] | None = None,
) -> dict[str, Any]:
    """Write ``model.mlir`` and input/golden sidecars for every case in the corpus.

    The emitted @forward inlines each captured body in corpus order, giving all operands
    and results one positional ABI. The map identifies each case's contiguous input and
    output interval. An unavailable case has no interval and cannot earn a pass.
    """
    identifiers = [str(case.get("case_id") or case["overload"]) for case in corpus["cases"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate case identifiers in batch corpus")
    capture_root = Path(capture_root)
    bundle_root = Path(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    inputs: list[np.ndarray] = []
    arg_types: list[str] = []
    result_types: list[str] = []
    results: list[str] = []
    body_lines: list[str] = []
    declarations: list[str] = []
    records: list[dict[str, Any]] = []
    manifest: dict[str, dict[str, Any]] = {}
    input_order: dict[str, int] = {}
    semantic_outputs = []
    semantic_inputs = []
    semantic_posts = []
    layout_inputs = []
    layout_results = []

    for case in corpus["cases"]:
        overload = case["overload"]
        from merlin.targetgen.core_aten_capture import case_capture_name

        identifier = str(case.get("case_id") or overload)
        directory = capture_root / case_capture_name(case)
        capture_file = directory / "capture.json"
        mlir_file = directory / "capsule.linalg.mlir"
        record: dict[str, Any] = {
            "overload": overload,
            "case_id": identifier,
            "routing": {"lane": "host", "routed": False, "reason": "scalar mode"},
            "case": case,
            "capture_directory": str(directory),
        }
        if selected_overloads is not None and identifier not in selected_overloads:
            record.update(status="not_in_shard", reason="scheduled in a different batch shard")
            records.append(record)
            continue
        if not capture_file.is_file() or not mlir_file.is_file():
            record.update(status="capture_unavailable", reason="no clean Linalg capture artifact")
            if capture_file.is_file():
                capture = json.loads(capture_file.read_text(encoding="utf-8"))
                record["capture_status"] = capture.get("status")
                record["reason"] = capture.get("reason", record["reason"])
            records.append(record)
            continue
        capture = json.loads(capture_file.read_text(encoding="utf-8"))
        meta = capture["capture_meta"]
        if not meta.get("ok") or meta.get("opaque") != 0:
            record.update(status="capture_unavailable", reason="capture is not clean Linalg")
            records.append(record)
            continue
        try:
            names, body, returned, local_result_types, local_declarations = _function_parts(
                mlir_file.read_text(encoding="utf-8")
            )
            from merlin.llvmlower.result_layout import read_layout

            result_layout = read_layout(mlir_file.read_text(encoding="utf-8"))
            abi_in = meta["input_abi"]
            abi_out = meta["output_abi"]
            contract = meta.get("result_contract")
            if contract is not None:
                role_results = contract["results"]
                if len(role_results) != len(returned):
                    raise ValueError("same-conversion result roles differ from emitted results")
                dtype_map = {str(np.dtype(v)): k for k, v in _DTYPES.items()}
                dtype_map.update(bool="i1", complex64="complex<f32>", complex128="complex<f64>")
                dtype_map.update(float32="f32", bfloat16="bf16")
                abi_out = [{"shape": r["shape"], "dtype": dtype_map[r["dtype"]]} for r in role_results]
                user_ordinals = [i for i, r in enumerate(role_results) if r["role"] == "user_output"]
                post_ordinals = []
                post_abi = []
                post_semantics = []
                for i, (name, input_type, input_meta) in enumerate(zip(names, abi_in, contract["inputs"])):
                    mutation = [
                        j
                        for j, r in enumerate(role_results)
                        if r.get("input_index") == i and r["role"] == "user_input_mutation"
                    ]
                    if len(mutation) > 1:
                        raise ValueError("multiple exported mutation results for one input")
                    if mutation:
                        j = mutation[0]
                        post_ordinals.append(j)
                        post_abi.append(abi_out[j])
                        post_semantics.append(role_results[j])
                    else:
                        post_ordinals.append(len(returned))
                        returned.append(name)
                        local_result_types.append(_tensor_type(input_type))
                        abi_out.append(input_type)
                        post_abi.append(input_type)
                        post_semantics.append(input_meta)
                all_semantics = role_results + [
                    contract["inputs"][i] for i, j in enumerate(post_ordinals) if j >= len(role_results)
                ]
            if (
                contract is None
                and len(abi_out) == 1
                and len(returned) > 1
                and len(set(zip(returned, local_result_types))) == 1
            ):
                # Functional export can expose a mutation and a user result as
                # the exact same SSA value. Every possible user-result mapping
                # observes that value; no output-role or shape guess is needed.
                record["result_projection"] = {"emitted_count": len(returned), "proof": "all_results_identical_ssa"}
                returned = returned[:1]
                local_result_types = local_result_types[:1]
            from merlin.common.mlir_query import forward_signature

            _, declared_results = forward_signature(mlir_file)
            abi_out = (
                [
                    {**item, "declared_shape": [(-1 if d < 0 else d) for d in shape]}
                    for item, (shape, _) in zip(abi_out, declared_results)
                ]
                if len(abi_out) == len(declared_results)
                else abi_out
            )
            captured_inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
            if len(names) != len(abi_in) or len(captured_inputs) != len(abi_in):
                raise ValueError("captured argument ABI or input count mismatch")
            if len(returned) != len(abi_out):
                raise ValueError(
                    f"captured result ABI count mismatch: emitted {len(returned)} results, "
                    f"eager ABI has {len(abi_out)}; explicit exported result mapping is absent"
                )
            local_types = [_tensor_type(item) for item in abi_in]
            local_inputs = []
            for index, (value, item) in enumerate(zip(captured_inputs, abi_in)):
                physical = _physical_abi(item)
                if item["dtype"] in _COMPLEX_DTYPES:
                    logical = np.asarray(_complex_value(value), dtype=_COMPLEX_DTYPES[item["dtype"]])
                    array = np.stack((logical.real, logical.imag), axis=-1).astype(_DTYPES[physical["dtype"]])
                else:
                    array = np.asarray(_complex_value(value), dtype=_DTYPES[item["dtype"]])
                if list(array.shape) != physical["shape"]:
                    raise ValueError(f"input shape {list(array.shape)} differs from ABI {physical['shape']}")
                local_inputs.append(array)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            record.update(status="bundle_unavailable", reason=f"{type(exc).__name__}: {exc}")
            records.append(record)
            continue

        ordinal = len([item for item in records if item["status"] == "bundled"])
        prefix = f"c{ordinal}_"
        first_input = len(inputs)
        first_output = len(results)
        argument_names = {name: f"%arg{first_input + index}" for index, name in enumerate(names)}
        renamed_body = _rename_ssa(body, prefix, argument_names)
        declarations.append(_rename_ssa(local_declarations, prefix, {}))
        renamed_returns = [_rename_ssa(name, prefix, argument_names) for name in returned]
        if result_layout is not None:
            layout_inputs.extend(result_layout["inputs"])
            for item in result_layout["results"]:
                entry = {
                    **item,
                    "alias_inputs": [first_input + j for j in item["alias_inputs"]],
                    "alias_results": [first_output + j for j in item.get("alias_results", [])],
                }
                if "input_index" in entry:
                    entry["input_index"] += first_input
                layout_results.append(entry)
            for name in returned[len(result_layout["results"]) :]:
                j = names.index(name)
                # A direct post-call input observation borrows its storage without
                # copying, including read-only expanded inputs.
                layout_results.append(
                    {**result_layout["inputs"][j], "alias_inputs": [first_input + j], "role": "user_output"}
                )
        else:
            layout_inputs.extend([None] * len(names))
            layout_results.extend([None] * len(returned))
        body_lines.append(f"    // {overload}\n" + renamed_body.rstrip() + "\n")
        inputs.extend(local_inputs)
        arg_types.extend(local_types)
        results.extend(renamed_returns)
        result_types.extend(local_result_types)
        user_indices = list(range(first_output, len(results)))
        user_abi = abi_out
        if contract is not None:
            user_indices = [first_output + i for i in user_ordinals]
            user_abi = [abi_out[i] for i in user_ordinals]
            post_indices = [first_output + i for i in post_ordinals]
            # Mutation outputs establish each post-call input's storage identity as well.
            record["semantic_boundary"] = dict(
                post_indices=post_indices, post_abi=post_abi, authority=contract["authority"]
            )
            for i, semantic in enumerate(all_semantics):
                aliases = list(semantic.get("alias_inputs", []))
                if i in post_ordinals:
                    aliases = []
                semantic_outputs.append({**semantic, "alias_outputs": [post_indices[j] for j in aliases]})
            semantic_inputs.extend(range(first_input, len(inputs)))
            semantic_posts.extend(post_indices)
        else:
            semantic_outputs.extend([None] * len(renamed_returns))
        for index, item in enumerate(abi_in):
            global_index = first_input + index
            physical = _physical_abi(item)
            input_order[f"arg{global_index}"] = global_index
            manifest[str(global_index)] = {
                "kind": "input",
                "name": f"arg{global_index}",
                "dtype": physical["dtype"],
                "shape": physical["shape"],
            }
        record.update(
            status="bundled",
            native_eager_boundary=meta.get("native_eager_boundary"),
            native_eager_indices={"outputs": user_indices, "posts": post_indices if contract is not None else []},
            capture_status=capture.get("status"),
            input_indices=list(range(first_input, len(inputs))),
            output_indices=user_indices,
            input_abi=abi_in,
            output_abi=user_abi,
            capture_golden_observation=json.loads((directory / "golden.json").read_text(encoding="utf-8")),
        )
        records.append(record)

    if not results:
        raise ValueError("no clean captures to bundle")
    arguments = ", ".join(f"%arg{i}: {kind}" for i, kind in enumerate(arg_types))
    outputs = result_types[0] if len(result_types) == 1 else "(" + ", ".join(result_types) + ")"
    returns = ", ".join(results)
    return_types = ", ".join(result_types)
    from merlin.llvmlower.result_layout import module_attribute

    layout_attr = (
        module_attribute(dict(schema_version=1, inputs=layout_inputs, results=layout_results))
        if any(item is not None for item in layout_results)
        else ""
    )
    module = (
        "builtin.module"
        + layout_attr
        + " {\n"
        + "".join(declarations)
        + f"  func.func @forward({arguments}) -> {outputs} {{\n"
        + "".join(body_lines)
        + f"    func.return {returns} : {return_types}\n"
        + "  }\n}\n"
    )
    (bundle_root / "model.mlir").write_text(module, encoding="utf-8")
    if any(item.strip() for item in declarations):
        from merlin.llvmlower.generic_form import to_generic_form

        generic = to_generic_form(bundle_root / "model.mlir")
        (bundle_root / "model.mlir").write_bytes(generic.read_bytes())
    semantic_path = bundle_root / "semantic_io.json"
    if semantic_outputs and all(item is not None for item in semantic_outputs):
        semantic_path.write_bytes(
            _json_bytes(
                caller_boundary_layouts(
                    # The custom-form module: the written file may be re-printed generically.
                    module,
                    dict(
                        schema_version=1,
                        outputs=semantic_outputs,
                        input_indices=semantic_inputs,
                        post_indices=semantic_posts,
                    ),
                    [record for record in records if record["status"] == "bundled"],
                )
            )
        )
    else:
        semantic_path.unlink(missing_ok=True)
    _write_npz(bundle_root / "inputs.npz", inputs)
    (bundle_root / "weights.safetensors.manifest.json").write_bytes(_json_bytes(manifest))
    (bundle_root / "input_order.json").write_bytes(_json_bytes(input_order))
    report = {
        "schema_version": 1,
        "case_count": len(records),
        "bundled_count": sum(item["status"] == "bundled" for item in records),
        "input_count": len(inputs),
        "output_count": len(results),
        "denominator_sha256": corpus.get("denominator_sha256"),
        "cases": records,
    }
    (bundle_root / "core_aten_batch_map.json").write_bytes(_json_bytes(report))
    return report
