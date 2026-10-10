"""Fixed ordinary registered scalar conversion with actual call observations.

The profile observes the selected callable and its actual importer caller. It
does not replace functions, registry entries, argument conversion or products.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

FUNCTIONS = {"aten.mul.Tensor": "decompose_mul_tensor", "aten.div.Tensor": "decompose_div_tensor"}


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _reader(path):
    module = ModuleType("selected_original_scalar_factory")
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _selected_imports(capture, sources):
    if any(name == "m2m" or name.startswith("m2m.") for name in sys.modules):
        raise ValueError("scalar observation requires a fresh explicitly selected upstream import")

    class SourceLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = Path(self.path).absolute()
            payload = path.read_bytes()
            if sources.get(str(path)) != hashlib.sha256(payload).hexdigest():
                raise ValueError("scalar conversion imported bytes outside its selected public source inventory")
            return compile(payload, str(path), "exec")

    class SourceFinder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname != "m2m" and not fullname.startswith("m2m."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if (
                spec is None
                or spec.origin not in sources
                or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
            ):
                raise ValueError("scalar conversion requires original selected Python source owners")
            spec.loader = SourceLoader(fullname, spec.origin)
            return spec

    sys.path.insert(0, str(capture))
    sys.meta_path.insert(0, SourceFinder())


def _literal(value, *, version):
    if type(value) is float:
        return {"kind": "float", "value_hex": value.hex()}
    if version == 2 and type(value) is int and -(1 << 63) <= value < (1 << 63):
        return {"kind": "int", "value": value}
    raise ValueError("registered scalar call has an unsupported original literal kind/range")


def _tensor_descriptor(value):
    import torch

    if type(value) is not torch.Tensor:
        raise ValueError("original scalar output is not a direct Tensor")
    return {
        "kind": "tensor",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "layout": str(value.layout),
        "device": str(value.device),
    }


def _getter(selected):
    if (
        type(selected) is not dict
        or set(selected) != {"path", "sha256"}
        or _pin(selected["path"]) != selected
        or "native_tensor_argument_getter" in sys.modules
    ):
        raise ValueError("integer scalar observation needs its unmodified freshly selected native getter")
    spec = importlib.util.spec_from_file_location("native_tensor_argument_getter", selected["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _promotion(member, examples, getter, live_boxes):
    import torch

    binding = member["original_tensor_binding"]
    if binding is None:
        return None
    request = binding["request"]
    literal = request["literal"]
    if (
        len(examples) != 1
        or literal.get("type") != "int"
        or type(literal.get("value")) is not str
        or str(int(literal["value"])) != literal["value"]
        or not -(1 << 63) <= int(literal["value"]) < (1 << 63)
    ):
        raise ValueError("integer scalar promotion lost its exact signed64 original request")
    scalar = int(literal["value"])
    namespace, name, overload = member["target"].split(".")
    native = getter.observe(namespace + "::" + name, overload, 1, scalar)
    boxed = native.pop("tensor")
    native.update(
        shape=list(boxed.shape),
        dtype=str(boxed.dtype),
        element_bytes=boxed.element_size(),
        literal={"type": "int", "value": str(boxed.item())},
        disjoint_from_prior_live_boxes=all(not torch._C._is_alias_of(boxed, prior) for prior in live_boxes),
    )
    if json.dumps(native, sort_keys=True) != json.dumps(binding["native"], sort_keys=True):
        raise ValueError("fresh wrapped integer conversion differs from its original native Tensor binding")
    live_boxes.append(boxed)
    operation = getattr(getattr(torch.ops.aten, name), overload)
    direct = operation(examples[0], scalar)
    boxed_output = operation(examples[0], boxed)
    return {
        "original_binding": binding,
        "native_binding": native,
        "common_dtype": str(torch.result_type(examples[0], boxed)),
        "input": _tensor_descriptor(examples[0]),
        "outputs": [_tensor_descriptor(direct)],
        "boxed_outputs": [_tensor_descriptor(boxed_output)],
    }


def _convert(model, examples, target, sources, *, version=1, tensor_argument=None):
    import m2m
    from m2m.ir import decompositions as D
    from m2m.ir.import_fx import FXImporter

    function = getattr(D, FUNCTIONS[target])
    importer = FXImporter.import_graph
    if D.DECOMPOSITION_TABLE.get(target) is not function or sys.getprofile() is not None:
        raise ValueError("scalar conversion requires the unmodified selected registry and an unowned profile scope")
    events, entered = [], {}

    def profile(frame, event, returned):
        if frame.f_code is not function.__code__:
            return
        parent = frame.f_back
        if (
            parent is None
            or parent.f_code is not importer.__code__
            or parent.f_locals.get("decomp_fn") is not function
            or parent.f_locals.get("target_str") != target
            or parent.f_locals["self"].dynamic_decompositions
            or D.DECOMPOSITION_TABLE.get(target) is not function
        ):
            raise ValueError("registered scalar callable was reached through an override or unsupported caller")
        if event == "call":
            if entered or events:
                raise ValueError("scalar conversion invoked its original registered callable more than once")
            meta, operands = frame.f_locals["meta"], frame.f_locals["operands"]
            args = meta.get("_fx_args")
            if (
                type(args) is not tuple
                or len(args) != 2
                or type(args[1]) not in ({float, int} if version == 2 else {float})
                or len(operands) != 1
                or meta.get("_fx_kwargs") != {}
                or meta.get("_aten_target") != target
            ):
                raise ValueError("registered scalar call lost its original one-SSA right FloatLiteral binding")
            entered[id(frame)] = {
                "target": target,
                "literal": _literal(args[1], version=version),
                "operand_types": [str(value.type) for value in operands],
                "source_node_id": meta["_m2m_node_id"],
                "origin_node_ids": list(meta["_m2m_origin_ids"]),
                "dynamic_overrides": [],
            }
        elif event == "return":
            row = entered.pop(id(frame), None)
            if row is None or returned is None or returned.result is None:
                raise ValueError("original registered scalar conversion did not produce its actual typed result")
            row.update(
                result_type=str(returned.result.type),
                emitted_operations=[operation.name for operation in returned.ops],
            )
            events.append(row)

    sys.setprofile(profile)
    try:
        converted = m2m.convert(model, examples, backend="fx_importer", level="linalg-on-tensors", capture_trace=True)
    finally:
        sys.setprofile(None)
    if (
        entered
        or len(events) != 1
        or D.DECOMPOSITION_TABLE.get(target) is not function
        or not converted.ok
        or converted.module is None
        or converted.path_taken != "fx_importer"
    ):
        raise ValueError("ordinary selected scalar conversion failed or lost its actual registered invocation")
    converted.module.verify()
    selected = []
    for callable_ in (function, importer):
        pin = _pin(callable_.__code__.co_filename)
        if sources.get(pin["path"]) != pin["sha256"]:
            raise ValueError("executed scalar registry/importer differs from selected source bytes")
        selected.append({"module": callable_.__module__, "name": callable_.__qualname__, **pin})
    registry = {
        "schema": "merlin.native_scalar_binary_registry.v2"
        if version == 2
        else "merlin.native_scalar_binary_registry.v1",
        "target": target,
        "function": selected[0],
        "importer": selected[1],
        "events": events,
    }
    if version == 2:
        registry["tensor_argument"] = tensor_argument
    return converted, registry


def _preflight_promotion(request):
    """Bound the entire native observation roster before any imports/allocation."""
    keys = ("max_tensor_elements", "max_promotion_tensor_elements", "max_total_promotion_tensor_elements")
    if any(type(request[key]) is not int or request[key] < 1 for key in keys):
        raise ValueError("integer promotion requires complete positive native payload reservations")
    total = 0
    if type(request["members"]) is not list:
        raise ValueError("integer promotion requires its complete bounded member roster")
    for member in request["members"]:
        if type(member) is not dict or set(member) != {
            "index",
            "target",
            "source",
            "source_sha256",
            "original_tensor_binding",
            "input",
            "promotion_tensor_elements",
        }:
            raise ValueError("integer promotion changed its complete bounded member fields")
        descriptor = member["input"]
        if (
            type(descriptor) is not dict
            or set(descriptor) != {"kind", "dtype", "shape", "layout", "device"}
            or descriptor["kind"] != "tensor"
            or descriptor["dtype"] != "torch.float32"
            or descriptor["layout"] != "torch.strided"
            or descriptor["device"] != "cpu"
            or type(descriptor["shape"]) is not list
            or not descriptor["shape"]
            or len(descriptor["shape"]) > request["max_tensor_elements"] // 2
        ):
            raise ValueError("integer promotion changed its independently bounded typed input descriptor")
        count = 1
        for extent in descriptor["shape"]:
            if type(extent) is not int or extent < 1 or count > (request["max_tensor_elements"] // 2) // extent:
                raise ValueError("integer promotion geometry exceeds its budget before allocation")
            count *= extent
        cost = 3 * count + 1 if member["original_tensor_binding"] is not None else 0
        if (
            type(member["promotion_tensor_elements"]) is not int
            or member["promotion_tensor_elements"] != cost
            or cost > request["max_promotion_tensor_elements"]
            or total + cost > request["max_total_promotion_tensor_elements"]
        ):
            raise ValueError("integer promotion complete readouts exceed their aggregate preallocation budget")
        total += cost


def observe(request, *, capture, destination):
    version = (
        2 if type(request) is dict and request.get("schema") == "merlin.original_scalar_conversion_request.v2" else 1
    )
    fields = {"schema", "capture_sources", "members", "max_product_bytes"}
    if version == 2:
        fields.update(
            {
                "tensor_argument_getter",
                "max_tensor_elements",
                "max_promotion_tensor_elements",
                "max_total_promotion_tensor_elements",
            }
        )
    if (
        type(request) is not dict
        or set(request) != fields
        or request["schema"]
        not in {"merlin.original_scalar_conversion_request.v1", "merlin.original_scalar_conversion_request.v2"}
        or type(request["max_product_bytes"]) is not int
        or request["max_product_bytes"] < 1
    ):
        raise ValueError("scalar conversion needs its closed preflighted original request")
    if version == 2:
        _preflight_promotion(request)
    sources = {row["path"]: row["sha256"] for row in request["capture_sources"]}
    _selected_imports(capture, sources)
    rows = []
    getter = _getter(request["tensor_argument_getter"]) if version == 2 else None
    live_boxes = []
    for member in request["members"]:
        member_fields = {"index", "target", "source", "source_sha256"}
        if version == 2:
            member_fields.update({"original_tensor_binding", "input", "promotion_tensor_elements"})
        if set(member) != member_fields or member["target"] not in FUNCTIONS:
            raise ValueError("scalar observation changed its complete original member request")
        owner = destination / str(member["index"])
        owner.mkdir(mode=0o700)
        try:
            if _pin(member["source"])["sha256"] != member["source_sha256"]:
                raise ValueError("scalar observation changed its selected original source factory bytes")
            loader = _reader(member["source"])
            model, examples = loader.get_model_and_inputs()
            if version == 2 and (len(examples) != 1 or _tensor_descriptor(examples[0]) != member["input"]):
                raise ValueError("integer scalar factory changed its bounded native input ABI")
            promotion = _promotion(member, examples, getter, live_boxes) if version == 2 else None
            converted, registry = _convert(
                model, examples, member["target"], sources, version=version, tensor_argument=promotion
            )
            products = {
                "source.mlir": converted.mlir_text,
                "trace.json": json.dumps(converted.capture_trace, sort_keys=True, allow_nan=False),
                "registry.json": json.dumps(registry, sort_keys=True, allow_nan=False),
            }
            if any(len(value.encode()) + 1 > request["max_product_bytes"] for value in products.values()):
                raise ValueError("actual registered scalar products exceed their original byte reservation")
            for name, value in products.items():
                (owner / name).write_text(value)
                (owner / name).chmod(0o600)
            rows.append({"index": member["index"], "status": "converted"})
        except (ValueError, RuntimeError, TypeError, OverflowError) as error:
            rows.append({"index": member["index"], "status": "unavailable", "reason": str(error)})
    dependencies = []
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if source is not None and Path(source).is_file():
            dependencies.append({"module": name, **_pin(source)})
    return {
        "schema": "merlin.native_original_scalar_conversion.v2"
        if version == 2
        else "merlin.native_original_scalar_conversion.v1",
        "rows": rows,
        "dependencies": dependencies,
    }


if __name__ == "__main__":
    request_path, capture, destination = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_bytes())
    result = observe(request, capture=capture, destination=destination)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode()) > request["max_product_bytes"]:
        raise ValueError("scalar native observation exceeds its complete bounded output frame")
    (destination / "observation.json").write_text(payload)
    (destination / "observation.json").chmod(0o600)
