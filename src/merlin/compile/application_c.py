"""Bind semantic constants and ordered model outputs to compiled C dispatch."""

from __future__ import annotations

from hashlib import sha256
from math import prod

import numpy as np

from merlin.llvmlower.c_runtime import DT_BYTES, embed_array
from merlin.semantic_compiler.model import KernelRequest

from .c_sources import c_source_pair


def emit_application_c(
    source: KernelRequest, driver: dict, driver_files: dict[str, str], name: str, *, capacity: int, parameters=None,
):
    """Stage all outputs until dispatch succeeds; constants never become runtime inputs.

    The caller owns compiled-call contexts and their private scratch. ``capacity``
    bounds the application tensor arena plus output staging, not device memory.
    This interface transports dense, fixed-width element bytes without tensor math.
    Non-f32 ports use an opaque pointer ABI with explicit dtype and shape metadata.
    """
    if any(
        node.type.dtype not in DT_BYTES or (node.type.dtype == "i64" and node.type.numerical_policy != "signed-i64")
        for node in source.nodes
    ):
        raise ValueError("application C needs declared fixed-width source tensors and signed-i64 index policies")
    sizes = {node.id: prod(node.type.shape) * DT_BYTES[node.type.dtype] for node in source.nodes}
    parameter_keys = set()
    parameter_declarations, parameter_implementation, private_headers = "", "", {}
    if parameters is not None:
        parameters.check(source)
        parameter_keys = set(parameters.node_ids)
        if driver.get("read_only_inputs") != list(parameters.node_ids):
            raise ValueError("parameter inputs need explicit read-only dispatch bindings")
        parameter_declarations, parameter_implementation, private_headers = parameters.emit_binding(name)
    inputs = tuple((key, storage) for key, storage in source.input_storages if key not in parameter_keys)
    if any(size > 2**63 - 1 for size in sizes.values()):
        raise ValueError("application tensor exceeds the supported C size domain")
    mixed = any(
        source.node(key).type.dtype != "f32" for key in [*(key for key, _ in source.input_storages), *source.outputs]
    )
    input_type, output_type = (
        ("const void *const *", "void *const *") if mixed else ("const float *const *", "float *const *")
    )
    if type(capacity) is not int or capacity < 0:
        raise ValueError("application tensor byte budget must be a nonnegative integer")
    if driver["inputs"] != [key for key, _ in source.input_storages] or driver["outputs"] != list(source.outputs):
        raise ValueError("application C differs from the source input/output interface")
    driver_name = driver["entry"].removesuffix("_run")
    if not driver_name or driver["entry"] != driver_name + "_run":
        raise ValueError("application C needs an existing serial dispatch driver")
    arena = driver["memory_plan"]["arena_bytes"]
    alignment = driver["memory_plan"]["stats"]["align"]
    minimum_alignment = max(DT_BYTES[node.type.dtype] for node in source.nodes)
    if type(alignment) is not int or alignment < minimum_alignment or alignment & (alignment - 1):
        raise ValueError("application workspace alignment must support its element types and be a power of two")

    def align(value):
        return (value + alignment - 1) // alignment * alignment

    offset = align(arena)
    pointers, constants, staged = {}, [], []
    for index, (key, _) in enumerate(inputs):
        pointers[key] = f"inputs[{index}]"
    if parameters is not None:
        pointers.update((key, f"parameters->data+{entry.offset}u")
                        for key, entry in zip(parameters.node_ids, parameters.entries, strict=True))
    for index, binding in enumerate(sorted(source.constants, key=lambda item: item.node_id)):
        symbol = f"{name}_constant_{index}"
        if binding.encoding == "f32-le-raw":
            values = np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<f4")
            constants.append(
                f"_Alignas({alignment}) static const float {symbol}[{values.size}]={{{embed_array(values, 'f32')}}};"
            )
        elif binding.encoding == "i64-le":
            values = np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<i8")

            def literal(value):
                integer = int(value)
                if integer == -(1 << 63):
                    return "(-INT64_C(9223372036854775807)-INT64_C(1))"
                return ("-" if integer < 0 else "") + f"INT64_C({abs(integer)})"

            initializer = ",".join(literal(value) for value in values)
            constants.append(f"_Alignas({alignment}) static const int64_t {symbol}[{values.size}]={{{initializer}}};")
        else:
            # Decode the serialized little-endian words at compile time and emit
            # native integer carriers. No C floating conversion can canonicalize
            # BF16/FP8 or binary64 NaN payloads, signs or subnormals.
            width = DT_BYTES[source.node(binding.node_id).type.dtype]
            values = np.frombuffer(bytes.fromhex(binding.data_hex), dtype=f"<u{width}")
            initializer = ",".join(f"UINT{width * 8}_C({int(value)})" for value in values)
            constants.append(
                f"_Alignas({alignment}) static const uint{width * 8}_t {symbol}[{values.size}]={{{initializer}}};"
            )
        pointers[binding.node_id] = symbol
    for key in source.outputs:
        if key not in pointers:
            size = sizes[key]
            pointers[key] = f"workspace+{offset}u"
            staged.append({"id": key, "offset": offset, "bytes": size})
            offset = align(offset + size)
    if offset > capacity:
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        raise NativeCompilationError(
            "resource_limit", "application tensor arena and output staging exceed the byte budget"
        )
    bindings = ",".join(f"{{(void *)({pointers[item['id']]}),{item['bytes']}u}}" for item in driver["buffers"])
    input_checks = [
        f"!{name}_valid(inputs[{i}],{sizes[key]}u) || {name}_overlap(inputs[{i}],{sizes[key]}u,workspace,{offset}u) || "
        f"{name}_overlap(inputs[{i}],{sizes[key]}u,state,sizeof(*state))"
        for i, (key, _) in enumerate(inputs)
    ]
    output_checks = []
    for index, key in enumerate(source.outputs):
        output_checks.extend(
            [
                f"!{name}_valid(outputs[{index}],{sizes[key]}u)",
                f"{name}_overlap(outputs[{index}],{sizes[key]}u,workspace,{offset}u)",
                f"{name}_overlap(outputs[{index}],{sizes[key]}u,state,sizeof(*state))",
                *(
                    f"{name}_overlap(outputs[{index}],{sizes[key]}u,inputs[{j}],{sizes[operand]}u)"
                    for j, (operand, _) in enumerate(inputs)
                ),
                *(
                    f"{name}_overlap(outputs[{index}],{sizes[key]}u,outputs[{j}],{sizes[other]}u)"
                    for j, other in enumerate(source.outputs[:index])
                ),
            ]
        )
    parameter_checks = []
    if parameters is not None:
        parameter_checks = [
            f"{name}_overlap(parameters,sizeof(*parameters),state,sizeof(*state))",
            f"{name}_overlap(parameters,sizeof(*parameters),workspace,{offset}u)",
            f"{name}_overlap(parameters->data,parameters->bytes,state,sizeof(*state))",
            f"{name}_overlap(parameters->data,parameters->bytes,workspace,{offset}u)",
        ]
        for pointer, size in [(f"inputs[{i}]", sizes[key]) for i, (key, _) in enumerate(inputs)] + [
            (f"outputs[{i}]", sizes[key]) for i, key in enumerate(source.outputs)
        ]:
            parameter_checks += [f"{name}_overlap({pointer},{size}u,parameters,sizeof(*parameters))",
                                 f"{name}_overlap({pointer},{size}u,parameters->data,parameters->bytes)"]
    format_declarations = ""
    if mixed:
        descriptions = []
        for role, keys in (("inputs", [key for key, _ in inputs]), ("outputs", source.outputs)):
            for index, key in enumerate(keys):
                type_ = source.node(key).type
                element = {"f32": "float", "f64": "double", "i64": "int64_t"}.get(
                    type_.dtype, f"uint{8 * DT_BYTES[type_.dtype]}_t ({type_.dtype} bits)"
                )
                shape = ",".join(map(str, type_.shape))
                descriptions.append(f" * {role}[{index}]: {element}[{shape}].")
        format_declarations = "/* Dense native buffers in this order:\n" + "\n".join(descriptions) + "\n */\n"
    declarations = f"""
#define {name.upper()}_INPUT_COUNT {len(inputs)}u
#define {name.upper()}_OUTPUT_COUNT {len(source.outputs)}u
#define {name.upper()}_WORKSPACE_BYTES {offset}u
#define {name.upper()}_WORKSPACE_ALIGNMENT {alignment}u
/* Zero-initialize state. Preserve state, contexts and scratch after an error
 * until the platform has drained/reset any outstanding device work. */
{parameter_declarations}
{format_declarations}int {name}_invoke({driver_name}_state *, void *workspace, size_t workspace_bytes,
                 {f"const {name}_parameters *parameters," if parameters is not None else ""}
                 {input_type}inputs, {output_type}outputs, void *contexts);
"""
    copy_outputs = "\n".join(
        f"  __builtin_memcpy(outputs[{i}],{pointers[key]},{sizes[key]}u);" for i, key in enumerate(source.outputs)
    )
    interface = f"""#include <stdint.h>
#include <float.h>
{'#include "parameter_sha256.h"' if parameters is not None else ''}
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128,
               "application source tensors require IEEE binary32");
{chr(10).join(constants)}
static int {name}_valid(const void *p,size_t n) {{ return p && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *p,size_t n,const void *q,size_t m) {{
  return (uintptr_t)p<(uintptr_t)q+m && (uintptr_t)q<(uintptr_t)p+n;
}}
{parameter_implementation}
int {name}_invoke({driver_name}_state *state, void *memory, size_t bytes,
                 {f"const {name}_parameters *parameters," if parameters is not None else ""}
                 {input_type}inputs, {output_type}outputs, void *contexts) {{
  (void)inputs;
  if (!state) return {driver_name.upper()}_INVALID;
  if (state->active) return {driver_name.upper()}_BUSY;
  if ({f"!{name}_parameters_ready(parameters) || " if parameters is not None else ""}
      !outputs || {"!inputs || " if inputs else ""}bytes<{offset}u ||
      !{name}_valid(memory,{offset}u) || (uintptr_t)memory%{alignment}u) return {driver_name.upper()}_INVALID;
  unsigned char *workspace=memory;
  if ({name}_overlap(state,sizeof(*state),workspace,{offset}u)) return {driver_name.upper()}_INVALID;
  if ({" || ".join(input_checks + output_checks + parameter_checks) or "0"}) return {driver_name.upper()}_INVALID;
  {driver_name}_buffer buffers[{len(driver["buffers"])}]={{{bindings}}};
  int status={driver_name}_run(state,workspace,{arena}u,buffers,{len(driver["buffers"])}u,contexts);
  if (status) return status;
{copy_outputs}
  return 0;
}}
"""
    files = c_source_pair(
        name,
        public_headers={driver_name + ".h": driver_files[driver_name + ".h"]},
        private_headers=private_headers,
        sources={"application.c": interface},
        declarations=declarations,
    )
    description = {
        "entry": name + "_invoke",
        "inputs": [key for key, _ in inputs],
        "outputs": list(source.outputs),
        "workspace_bytes": offset,
        "workspace_alignment": alignment,
        "output_staging": staged,
        "constants": [
            {
                "node_id": item.node_id,
                "encoding": item.encoding,
                "bytes": len(item.data_hex) // 2,
                "sha256": sha256(bytes.fromhex(item.data_hex)).hexdigest(),
            }
            for item in sorted(source.constants, key=lambda item: item.node_id)
        ],
        "private_region_scratch_included": False,
    }
    if parameters is not None:
        description["model_parameters"] = parameters.record()
    if mixed:
        description.update(
            pointer_abi="opaque_native_dense_buffers",
            input_types=[source.node(key).type.record() for key, _ in inputs],
            output_types=[source.node(key).type.record() for key in source.outputs],
        )
    return description, files
