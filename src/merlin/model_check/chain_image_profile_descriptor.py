"""Observe completed descriptor bridge ABIs without restoring effect owners.

Only profiling names and pointer arities are returned. Source, compiled shim,
native checkpoint and completed lowering witnesses remain separate retained
inputs; serialized effects never become a live lifetime or numerical grant.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from xdsl.dialects import bufferization, func
from xdsl.dialects.builtin import NoneAttr, TensorType, UnitAttr
from xdsl.ir import Block, Region

from merlin.common import invocation_record
from merlin.common.digest import sha256_bytes
from merlin.frontends import linalg_mlir
from merlin.llvmlower import descriptor_result_identity as identities
from merlin.llvmlower import device_shim
from merlin.model_check import chain_image_objects as objects
from merlin.model_check import chain_image_sections as sections
from merlin.runtime.whole_model_readback import ReadbackError


def _functions(module):
    result = {}
    for op in module.body.block.ops:
        if isinstance(op, func.FuncOp):
            name = op.sym_name.data
            if name in result:
                raise ReadbackError("descriptor profile source has duplicate functions")
            result[name] = op
    return result


def _source_binding(row, original, selected):
    """Check the actual typed wrapper, including its returned destination."""
    names = [row[key] for key in ("symbol", "borrowed_symbol", "wrapper_symbol")]
    if any(not isinstance(n, str) or not n.isascii() or not n.isidentifier() for n in names):
        raise ReadbackError("descriptor profile requires ordinary distinct symbols")
    if len(set(names)) != 3 or any(n in original for n in names[1:]):
        raise ReadbackError("descriptor profile wrapper names are ambiguous")
    symbol, borrowed, wrapper = names
    declaration = original.get(symbol)
    if declaration is None or declaration.body.blocks or declaration.arg_attrs is None:
        raise ReadbackError("descriptor profile lacks its original tensor declaration")
    types = tuple(declaration.function_type.inputs)
    output = row["result_argument"]
    if (type(output) is not int or not 0 <= output < len(types) or len(types) > 64
            or any(not isinstance(t, TensorType) for t in types)
            or [str(t) for t in types] != row["argument_types"]
            or tuple(declaration.function_type.outputs) != (types[output],)):
        raise ReadbackError("descriptor profile tensor signature changed")
    accesses = tuple(a.data["bufferization.access"].data for a in declaration.arg_attrs)
    layouts = tuple(a.data.get("bufferization.buffer_layout", NoneAttr()) for a in declaration.arg_attrs)
    if list(accesses) != row["argument_access"] or list(map(str, layouts)) != row.get(
        "argument_layouts", [str(NoneAttr())] * len(types)
    ):
        raise ReadbackError("descriptor profile argument access or layouts changed")
    # Observation data only: never instantiate or reconstruct a checked effect owner.
    observation = SimpleNamespace(symbol=symbol, borrowed_symbol=borrowed, wrapper_symbol=wrapper,
                                  argument_types=types, argument_access=accesses,
                                  argument_layouts=layouts, result_argument=output)
    buffers = identities.argument_buffer_types(observation)
    block = Block(arg_types=types)
    arguments = []
    for index, (argument, typ) in enumerate(zip(block.args, buffers, strict=True)):
        op = linalg_mlir.layout_to_buffer_type().build(
            operands=[argument], result_types=[typ],
            properties={} if index == output else {"read_only": UnitAttr()},
        )
        block.add_op(op)
        arguments.append(op.results[0])
    block.add_op(func.CallOp(borrowed, arguments, []))
    result = bufferization.ToTensorOp(arguments[output], restrict=True, writable=True)
    block.add_ops([result, func.ReturnOp(result.tensor)])
    expected = func.FuncOp(wrapper, (types, (types[output],)), Region(block), visibility="private")
    expected.properties["arg_attrs"] = declaration.arg_attrs
    external = func.FuncOp(borrowed, (buffers, ()), Region(), visibility="private")
    external.attributes["llvm.emit_c_interface"] = UnitAttr()
    for name, value in ((symbol, declaration), (wrapper, expected), (borrowed, external)):
        if name not in selected or not value.clone().is_structurally_equivalent(selected[name]):
            raise ReadbackError(f"descriptor profile source differs from its exact pointer wrapper: {name}")
    return observation, dict(borrowed_symbol=borrowed, buffers=len(types),
                             argument_ranks=[len(t.get_shape()) for t in types], wrapper_symbol=wrapper)


def completed_descriptor_bindings(program, routing, undefined):
    """Read a verified completed program's actual descriptor pointer-call ABI.

    Return existing profiler bindings plus ordinary immutable artifact pins.
    This grants neither original effects nor numerical or timing qualification.
    """
    output = program["receipt"]["output"]
    if not (output.get("descriptor_results") or {}).get("selected"):
        return {}, []
    root, preparation = Path(program["root"]), program["recipe"]["preparation"]
    pins = {}

    def retain(pin):
        value = objects.payload(pin)
        actual = objects.pin(Path(pin["path"]))
        if actual["path"] in pins and pins[actual["path"]] != actual:
            raise ReadbackError("descriptor profile has conflicting artifact identities")
        pins[actual["path"]] = actual
        return value

    def bound(path):
        matches = [p for p in preparation.values() if Path(p["path"]) == path]
        if not matches or any(p != matches[0] for p in matches):
            raise ReadbackError("descriptor profile artifact is absent from completed compilation")
        return retain(matches[0])

    for key in ("receipt_pin", "recipe_pin"):
        retain(program[key])
    if json.loads(bound(root / "device_signatures.json")) != routing:
        raise ReadbackError("descriptor profile routing differs from completed source")
    lower_dir = root / "lower"
    report_bytes = bound(lower_dir / "descriptor-result-identity.json")
    report = json.loads(report_bytes)
    lower = json.loads(bound(lower_dir / "lowering_recipe.json"))
    if (report.get("schema") != "merlin.descriptor_result_preparation.v1"
            or lower.get("status") != "returned"
            or lower.get("descriptor_result_identity", {}).get("report", {}).get("sha256")
            != sha256_bytes(report_bytes)):
        raise ReadbackError("descriptor profile lacks the consumed lowering join")
    actual_llvm = bound(lower_dir / "model.ll")
    if lower.get("returned_llvm_ir") != {"sha256": sha256_bytes(actual_llvm), "bytes": len(actual_llvm)}:
        raise ReadbackError("descriptor profile returned LLVM differs from completed lowering")
    sources = lower["sources"]
    native = json.loads(retain(sources["borrowed_lifetime_report"]))
    checkpoint = retain(sources["borrowed_lifetime_checkpoint"])
    if (native.get("status") != "checked" or native.get("violations") != []
            or native.get("checkpoint_sha256") != sha256_bytes(checkpoint)):
        raise ReadbackError("descriptor profile native checkpoint is absent or changed")
    original = _functions(linalg_mlir.parse_mlir_text(retain(report["original_source"]).decode()))
    selected = _functions(linalg_mlir.parse_mlir_text(retain(report["selected_source"]).decode()))
    shim_source_path, shim_path = root / "device" / "device_shim.c", root / "device" / "device_shim.o"
    shim_source = bound(shim_source_path).decode()
    shim_pin = program["by_name"].get(shim_path.name)
    if shim_pin is None or Path(shim_pin["path"]) != shim_path:
        raise ReadbackError("descriptor profile shim is absent from actual final link")
    retain(shim_pin)
    # Locate its actually observed C compilation, rather than guessing a prototype from a name.
    observations = []
    for pin in preparation.values():
        if Path(pin["path"]).name != "invocation.json":
            continue
        observed = json.loads(objects.payload(pin))
        if any(Path(p["path"]) == shim_path for p in observed.get("outputs", [])):
            observations.append((pin, observed))
    if len(observations) != 1:
        raise ReadbackError("descriptor profile needs one observed compiled shim")
    invocation_pin, observed = observations[0]
    retain(invocation_pin)
    verified = invocation_record.verify(Path(invocation_pin["path"]))
    if (verified != observed or shim_pin["sha256"] not in {p["sha256"] for p in observed["outputs"]}
            or not any(Path(p["path"]) == shim_source_path and p["sha256"] == pins[str(shim_source_path)]["sha256"]
                       for p in observed["inputs"])):
        raise ReadbackError("descriptor profile source differs from observed compiled shim")
    wanted = {r["symbol"] for r in routing.get("routed", [])}
    calls = {r["symbol"]: r for r in report["calls"] if r.get("calls")}
    result = {}
    for row in report["selection"]:
        symbol = row["symbol"]
        if symbol not in wanted or symbol not in calls:
            continue
        observation, binding = _source_binding(row, original, selected)
        actual = "_mlir_ciface_" + binding["borrowed_symbol"]
        if actual not in undefined:
            continue
        if symbol in undefined or "_mlir_ciface_" + symbol in undefined:
            raise ReadbackError("descriptor profile original and borrowed ABI both remain referenced")
        if not {binding["borrowed_symbol"], binding["wrapper_symbol"]} <= set(native["accepted_symbols"]):
            raise ReadbackError("descriptor profile wrapper lacks its actual native observation")
        variants = [device_shim._logical_borrowed_result_wrapper(observation)]
        if len(observation.argument_types) == 3 and len(set(binding["argument_ranks"])) == 1:
            variants.append(device_shim._borrowed_result_wrapper(symbol, binding["argument_ranks"][0],
                                                                binding["borrowed_symbol"]))
        if not any(shim_source.count(variant) == 1 for variant in variants):
            raise ReadbackError("descriptor profile compiled shim has a different pointer prototype")
        if not sections.symbol_bytes(shim_path, actual) or actual in {
            "_mlir_ciface_" + value["borrowed_symbol"] for value in result.values()
        } or symbol in result:
            raise ReadbackError("descriptor profile compiled bridge is absent or ambiguous")
        result[symbol] = binding
    for module in (objects, sections, identities, device_shim, linalg_mlir):
        retain(objects.pin(Path(module.__file__).resolve()))
    retain(objects.pin(Path(__file__).resolve()))
    for pin in tuple(pins.values()):
        objects.reopen(pin)
    return result, list(pins.values())
