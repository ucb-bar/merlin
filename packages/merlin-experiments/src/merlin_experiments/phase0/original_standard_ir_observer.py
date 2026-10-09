"""Fixed private upstream conversion of complete bounded original sources.

The caller preflights the whole roster. Outputs are actual frontend products
and complete framework values, never a compiler or semantic verdict.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import sys
from pathlib import Path


def _reader(name, path):
    from types import ModuleType

    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def observe(request, *, capture, reference_observer, destination):
    if set(request) != {"schema", "capture_sources", "members", "budget"} or request["schema"] not in {
        "merlin.original_standard_ir_request.v1",
        "merlin.original_standard_ir_request.v2",
    }:
        raise ValueError("standard IR observation needs the complete fixed original request")
    sources = {row["path"]: row["sha256"] for row in request["capture_sources"]}

    class SourceLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            path = Path(self.path).absolute()
            payload = path.read_bytes()
            if sources.get(str(path)) != hashlib.sha256(payload).hexdigest():
                raise ValueError("upstream conversion imported source outside the selected clean inventory")
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
                raise ValueError("upstream conversion requires exact selected Python source owners")
            spec.loader = SourceLoader(fullname, spec.origin)
            return spec

    sys.path.insert(0, str(capture))
    sys.meta_path.insert(0, SourceFinder())
    import m2m

    native = _reader("fixed_reference_observer", reference_observer)
    rows = []
    for member in request["members"]:
        if set(member) != {"index", "source", "metadata", "inputs"} or type(member["index"]) is not int:
            raise ValueError("standard IR member lost its exact original source/input slots")
        owner = destination / str(member["index"])
        owner.mkdir(mode=0o700)
        metadata = json.loads(Path(member["metadata"]).read_bytes())
        stimulus = json.loads(Path(member["inputs"]).read_bytes())
        try:
            actual = native.observe(Path(member["source"]), metadata, stimulus)
            loader = _reader("selected_factory", member["source"])
            model, examples = loader.get_model_and_inputs()
            converted = m2m.convert(
                model, examples, backend="fx_importer", level="linalg-on-tensors", capture_trace=True
            )
            if not converted.ok or converted.module is None or converted.path_taken != "fx_importer":
                raise ValueError("ordinary upstream conversion failed: " + repr(converted.diagnostics))
            converted.module.verify()
            products = {
                "source.mlir": converted.mlir_text,
                "trace.json": json.dumps(converted.capture_trace, sort_keys=True, allow_nan=False),
                "actual.json": json.dumps(actual, sort_keys=True, allow_nan=False),
            }
            if any(len(value.encode()) + 1 > request["budget"]["max_source_bytes"] for value in products.values()):
                raise ValueError("upstream products exceed the explicit complete source byte reservation")
            for name, value in products.items():
                output = owner / name
                output.write_text(value + ("\n" if name != "source.mlir" else ""))
                output.chmod(0o600)
            rows.append({"index": member["index"], "status": "converted"})
        except (ValueError, RuntimeError, TypeError) as error:
            rows.append({"index": member["index"], "status": "unavailable", "reason": str(error)})
    dependencies = []
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if source is not None and Path(source).is_file():
            path = Path(source).resolve()
            dependencies.append(
                {"module": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
    return {
        "schema": "merlin.native_original_standard_ir.v2"
        if request["schema"] == "merlin.original_standard_ir_request.v2"
        else "merlin.native_original_standard_ir.v1",
        "rows": rows,
        "dependencies": dependencies,
    }


if __name__ == "__main__":
    request, capture, reference_observer, destination = map(Path, sys.argv[1:])
    result = observe(
        json.loads(request.read_bytes()),
        capture=capture,
        reference_observer=reference_observer,
        destination=destination,
    )
    output = destination / "observation.json"
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if len(payload.encode()) > json.loads(request.read_bytes())["budget"]["max_observation_bytes"]:
        raise ValueError("upstream observation frame exceeds its explicit byte reservation")
    output.write_text(payload)
    output.chmod(0o600)
