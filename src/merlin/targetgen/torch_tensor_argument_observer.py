"""Fixed public-framework Tensor argument observation, never an IR lowering.

The native getter uses the original dispatcher's schema and public numeric
conversion predicate. A caller cannot enable that predicate. Wrapped numbers
are retained: replacing them with ordinary zero-dimensional tensors can change
promotion. This observation grants neither numerical support nor device effects.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import shutil
import sys
import sysconfig
from pathlib import Path

CPP_SOURCE = r"""#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
py::dict observe(const std::string& name, const std::string& overload,
                 size_t argument, py::object value) {
  const auto symbol = c10::Symbol::fromQualString(name);
  if (!symbol.is_aten()) throw std::invalid_argument("unsupported public namespace");
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
  const auto& schema = op.schema();
  const bool allows = torch::should_allow_numbers_as_tensors(symbol.toUnqualString());
  torch::jit::ToIValueAllowNumbersAsTensors guard(allows);
  auto ivalue = torch::jit::argumentToIValue(schema, argument, value);
  if (!ivalue.isTensor()) throw std::invalid_argument("original argument is not a direct Tensor");
  auto tensor = ivalue.toTensor();
  if (!tensor.defined()) throw std::invalid_argument("undefined Tensor has no scalar binding");
  py::dict result;
  result["source_allows_number"] = allows;
  result["schema"] = c10::toString(schema);
  result["argument_name"] = schema.arguments().at(argument).name();
  result["wrapped_number"] = tensor.unsafeGetTensorImpl()->is_wrapped_number();
  result["tensor"] = py::reinterpret_steal<py::object>(THPVariable_Wrap(tensor));
  return result;
}
PYBIND11_MODULE(native_tensor_argument_getter, module) { module.def("observe", &observe); }
"""

PUBLIC_PATHS = (
    "torch/csrc/jit/python/pybind_utils.h",
    "torch/csrc/utils/python_arg_parser.h",
    "torch/csrc/autograd/python_variable.h",
    "torch/csrc/jit/python/pybind_utils.cpp",
    "torch/csrc/utils/python_arg_parser.cpp",
    "torch/csrc/jit/python/init.cpp",
    "setup.py",
)
PRIMARY_HEADERS = PUBLIC_PATHS[:3]


def sdk():
    import torch

    package = Path(torch.__file__).absolute().parent
    return {
        "torch_version": torch.__version__,
        "reported_git_version": torch.version.git_version,
        "torch_module": str(Path(torch.__file__).absolute()),
        "native_module": str(Path(torch._C.__file__).absolute()),
        "torch_include": str(package / "include"),
        "torch_library": str(package / "lib"),
        "python_include": sysconfig.get_path("include"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    }


def generate_headers(public, output):
    """Execute the exact selected public packaging method without stripping guards."""
    output.mkdir()
    for name in PRIMARY_HEADERS:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(public / name, destination)
    tree = ast.parse((public / "setup.py").read_bytes())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "build_ext"]
    if len(classes) != 1:
        raise ValueError("public packaging source has no unique selected header generator")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "_wrap_headers_with_macro"
    ]
    if len(methods) != 1:
        raise ValueError("public packaging source has no unique selected header method")
    namespace = {"Path": Path, "report": lambda *args: None}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(public / "setup.py"), "exec"), namespace)
    namespace[methods[0].name](None, output)


def _literal(value):
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": str(value)}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value_hex": value.hex()}
    raise ValueError("unobserved exact finite scalar kind")


def _value(record):
    if set(record) == {"type", "value"} and record["type"] == "bool" and type(record["value"]) is bool:
        return record["value"]
    if set(record) == {"type", "value"} and record["type"] == "int" and isinstance(record["value"], str):
        value = int(record["value"])
    elif set(record) == {"type", "value_hex"} and record["type"] == "float":
        value = float.fromhex(record["value_hex"])
    else:
        raise ValueError("unsupported original scalar literal representation")
    if _literal(value) != record:
        raise ValueError("original scalar literal representation is not canonical")
    return value


def observe(request, getter):
    import torch

    if (
        set(request) != {"schema", "graph_sha256", "rows"}
        or request["schema"] != "merlin.original_tensor_argument_request.v1"
    ):
        raise ValueError("Tensor argument observer needs the closed complete original request")
    spec = importlib.util.spec_from_file_location("native_tensor_argument_getter", getter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, live_boxes = [], []
    for original in request["rows"]:
        if set(original) != {"node", "target", "schema", "argument_index", "argument_path", "literal"}:
            raise ValueError("original Tensor argument request has an unsupported row")
        try:
            parts = original["target"].split(".")
            if len(parts) != 3 or parts[0] != "aten" or any(not part.isidentifier() for part in parts):
                raise ValueError("original operator has no supported exact public namespace")
            native = module.observe(
                parts[0] + "::" + parts[1],
                "" if parts[2] == "default" else parts[2],
                original["argument_index"],
                _value(original["literal"]),
            )
            boxed = native.pop("tensor")
            if boxed.shape != torch.Size([]) or not native["wrapped_number"] or not native["source_allows_number"]:
                raise ValueError("native conversion is not the supported wrapped scalar relation")
            native.update(
                shape=list(boxed.shape),
                dtype=str(boxed.dtype),
                element_bytes=boxed.element_size(),
                literal=_literal(boxed.item()),
                disjoint_from_prior_live_boxes=all(not torch._C._is_alias_of(boxed, prior) for prior in live_boxes),
            )
            live_boxes.append(boxed)  # All compared allocations remain simultaneously live.
            if native["schema"] != original["schema"] or native["literal"] != original["literal"]:
                raise ValueError("native schema or literal differs from the exact original")
            rows.append({"request": original, "status": "observed", "native": native})
        except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
            rows.append({"request": original, "status": "unknown", "reason": str(exc)})
    return {
        "schema": "merlin.native_tensor_argument_observation.v1",
        "graph_sha256": request["graph_sha256"],
        "rows": rows,
        "runtime": sdk(),
        "scope": "actual public native argument conversion only; no numeric, physical or whole-effect grant",
    }


def main():
    if sys.argv[1] == "sdk":
        result = sdk()
    elif sys.argv[1] == "headers":
        generate_headers(Path(sys.argv[2]), Path(sys.argv[3]))
        result = {"status": "generated"}
    elif sys.argv[1] == "observe":
        result = observe(json.loads(Path(sys.argv[2]).read_bytes()), Path(sys.argv[3]))
    else:
        raise ValueError("unknown fixed Tensor argument observation stage")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
