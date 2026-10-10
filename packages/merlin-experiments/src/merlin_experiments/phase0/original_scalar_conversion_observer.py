"""Fixed ordinary registered scalar conversion with actual call observations.

The profile observes the selected callable and its actual importer caller. It
does not replace functions, registry entries, argument conversion or products.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
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


def _convert(model, examples, target, sources):
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
                or type(args[1]) is not float
                or len(operands) != 1
                or meta.get("_fx_kwargs") != {}
                or meta.get("_aten_target") != target
            ):
                raise ValueError("registered scalar call lost its original one-SSA right FloatLiteral binding")
            entered[id(frame)] = {
                "target": target,
                "literal": {"kind": "float", "value_hex": args[1].hex()},
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
        "schema": "merlin.native_scalar_binary_registry.v1",
        "target": target,
        "function": selected[0],
        "importer": selected[1],
        "events": events,
    }
    return converted, registry


def observe(request, *, capture, destination):
    if (
        type(request) is not dict
        or set(request) != {"schema", "capture_sources", "members", "max_product_bytes"}
        or request["schema"] != "merlin.original_scalar_conversion_request.v1"
        or type(request["max_product_bytes"]) is not int
        or request["max_product_bytes"] < 1
    ):
        raise ValueError("scalar conversion needs its closed preflighted original request")
    sources = {row["path"]: row["sha256"] for row in request["capture_sources"]}
    _selected_imports(capture, sources)
    rows = []
    for member in request["members"]:
        if set(member) != {"index", "target", "source", "source_sha256"} or member["target"] not in FUNCTIONS:
            raise ValueError("scalar observation changed its complete original member request")
        owner = destination / str(member["index"])
        owner.mkdir(mode=0o700)
        try:
            if _pin(member["source"])["sha256"] != member["source_sha256"]:
                raise ValueError("scalar observation changed its selected original source factory bytes")
            loader = _reader(member["source"])
            model, examples = loader.get_model_and_inputs()
            converted, registry = _convert(model, examples, member["target"], sources)
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
    return {"schema": "merlin.native_original_scalar_conversion.v1", "rows": rows, "dependencies": dependencies}


if __name__ == "__main__":
    request_path, capture, destination = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_bytes())
    result = observe(request, capture=capture, destination=destination)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode()) > request["max_product_bytes"]:
        raise ValueError("scalar native observation exceeds its complete bounded output frame")
    (destination / "observation.json").write_text(payload)
    (destination / "observation.json").chmod(0o600)
