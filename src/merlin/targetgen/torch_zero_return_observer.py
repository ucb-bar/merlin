"""Observe the public schema's zero-return Python bridge without running an op.

An empty dispatcher result stack becoming Python None is a source representation
relation. The observation proves no operation purity, exception behavior,
numerical implementation or device effect.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

CPP_SOURCE = r"""#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
py::dict observe(const std::string& name, const std::string& overload) {
  const auto symbol = c10::Symbol::fromQualString(name);
  if (!symbol.is_aten()) throw std::invalid_argument("unsupported public namespace");
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
  const auto& schema = op.schema();
  if (!schema.returns().empty()) throw std::invalid_argument("schema has results");
  torch::jit::Stack stack;
  py::object result = torch::jit::createPyObjectForStack(std::move(stack));
  py::dict observation;
  observation["schema"] = c10::toString(schema);
  observation["return_count"] = schema.returns().size();
  observation["empty_stack_is_none"] = result.is_none();
  return observation;
}
PYBIND11_MODULE(native_zero_return_getter, module) { module.def("observe", &observe); }
"""


def _metadata_is_none(row):
    results, metadata = row["results"], row["result_metadata"]
    return (
        isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        and set(results[0])
        == {"id", "kind", "shape", "dtype", "storage_dtype", "compute_dtype", "device", "layout", "stride"}
        and results[0].get("kind") == "unknown"
        and all(value is None for key, value in results[0].items() if key not in {"id", "kind"})
        and metadata
        == {
            "schema": "m2m.frontend_result_metadata.v1",
            "status": "observed",
            "container": "single",
            "values": [{"result_id": results[0]["id"], "kind": "none"}],
        }
    )


def observe(request, getter):
    import torch

    if (
        set(request) != {"schema", "graph_sha256", "rows"}
        or request["schema"] != "merlin.original_zero_return_request.v1"
        or not isinstance(request["rows"], list)
    ):
        raise ValueError("zero-return observer needs the complete original source request")
    spec = importlib.util.spec_from_file_location("native_zero_return_getter", getter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for original in request["rows"]:
        if set(original) != {"node", "target", "schema", "results", "result_metadata"}:
            raise ValueError("zero-return request has an unsupported original row")
        try:
            parts = original["target"].split(".")
            if len(parts) != 3 or parts[0] != "aten" or any(not part.isidentifier() for part in parts):
                raise ValueError("original operator has no supported exact public namespace")
            native = dict(module.observe(parts[0] + "::" + parts[1], "" if parts[2] == "default" else parts[2]))
            if native != {"schema": original["schema"], "return_count": 0, "empty_stack_is_none": True}:
                raise ValueError("native empty-result bridge differs from the original schema")
            if not _metadata_is_none(original):
                raise ValueError("original metadata does not independently record the complete observed None slot")
            rows.append({"request": original, "status": "observed", "native": native})
        except (RuntimeError, TypeError, ValueError) as exc:
            rows.append({"request": original, "status": "unknown", "reason": str(exc)})
    return {
        "schema": "merlin.native_zero_return_observation.v1",
        "graph_sha256": request["graph_sha256"],
        "rows": rows,
        "runtime": {"torch_version": torch.__version__, "reported_git_version": torch.version.git_version},
        "scope": "empty schema result stack to observed Python None only; operation effects remain unknown",
    }


if __name__ == "__main__":
    result = observe(json.loads(Path(sys.argv[1]).read_bytes()), Path(sys.argv[2]))
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
