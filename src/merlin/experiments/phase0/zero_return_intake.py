"""Actual public native zero-result preparation, observation and source replay.

Reuse the selected public SDK/header preparation, then compile only the fixed
empty-stack bridge observer. JSON does not issue authority; the live schema
intake owns protected source membership and logical relation admission.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen import torch_zero_return_observer as Z
from merlin.targetgen.frontend_operator_effects import original_zero_return_requests

from . import tensor_argument_intake as T
from .rtl_intake import RtlIntakeRefusal, _outside, _plain

SCHEMA = "merlin.independent_zero_return_getter.v1"


def prepare_getter(*, tensor_getter, forbidden, output):
    """The fixed native bridge uses the same independently observed public SDK."""
    T.verify_getter(tensor_getter)
    output.mkdir(mode=0o700)
    observer = module_source_path("merlin.targetgen.torch_zero_return_observer")
    _outside(observer, forbidden)
    sdk = json.loads(Path(tensor_getter["sdk"]).read_bytes())
    cpp, dependencies = output / "getter.cpp", output / "getter.d"
    cpp.write_text(Z.CPP_SOURCE)
    getter = output / ("native_zero_return_getter" + sdk["extension_suffix"])
    _, invocation = T._run(
        T._compile_command(tensor_getter["compiler"], sdk, cpp, getter, dependencies),
        owner=output,
        stage="native_zero_return_getter_build",
        inputs=(observer, cpp, Path(tensor_getter["sdk"]), *(Path(path) for path in tensor_getter["dependency_paths"])),
        outputs=(getter, dependencies),
        timeout=180,
        preexec_fn=T._compile_limit,
    )
    names = shlex.split(dependencies.read_text().split(":", 1)[1].replace("\\\n", " "))
    paths = tuple(sorted({T._dependency_path(path) for path in names}))
    for path in paths:
        _outside(path, forbidden)
    return {
        "schema": SCHEMA,
        "tensor_getter": tensor_getter,
        "observer": str(observer),
        "cpp": str(cpp),
        "getter": str(getter),
        "dependencies": str(dependencies),
        "dependency_paths": [str(path) for path in paths],
        "invocation": invocation,
    }


def verify_getter(record):
    if (
        set(record)
        != {"schema", "tensor_getter", "observer", "cpp", "getter", "dependencies", "dependency_paths", "invocation"}
        or record["schema"] != SCHEMA
        or record["observer"] != str(module_source_path("merlin.targetgen.torch_zero_return_observer"))
        or Path(record["cpp"]).read_text() != Z.CPP_SOURCE
    ):
        raise RtlIntakeRefusal("zero-return getter lost its fixed public native source")
    selected = T.verify_getter(record["tensor_getter"])
    sdk = json.loads(Path(selected["sdk"]).read_bytes())
    invocation = I.require_environment(Path(record["invocation"]), environment=T.ENVIRONMENT)
    output = Path(record["cpp"]).parent
    if (
        invocation["stage"] != "native_zero_return_getter_build"
        or invocation["argv"]
        != T._compile_command(selected["compiler"], sdk, record["cpp"], record["getter"], record["dependencies"])
        or Path(record["getter"]) != output / ("native_zero_return_getter" + sdk["extension_suffix"])
    ):
        raise RtlIntakeRefusal("zero-return getter differs from its actual native public SDK build")
    names = shlex.split(Path(record["dependencies"]).read_text().split(":", 1)[1].replace("\\\n", " "))
    actual_paths = [str(path) for path in sorted({T._dependency_path(path) for path in names})]
    if record["dependency_paths"] != actual_paths:
        raise RtlIntakeRefusal("zero-return getter lost its complete observed non-system dependency roster")
    return record


def observe_returns(*, trace, schema_observation, getter, output):
    request = original_zero_return_requests(trace, schema_observation)
    request_path, observation_path = output / "zero-return-request.json", output / "zero-return-observation.json"
    request_path.write_text(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False))
    result, invocation = T._run(
        [getter["tensor_getter"]["python"], "-I", getter["observer"], str(request_path), getter["getter"]],
        owner=output,
        stage="native_original_zero_return_observation",
        inputs=(request_path, Path(getter["observer"]), Path(getter["getter"]), Path(getter["tensor_getter"]["sdk"])),
    )
    observation_path.write_bytes(result.stdout)
    return {"request": str(request_path), "observation": str(observation_path), "invocation": invocation}


def verify_returns(*, trace, schema_observation, getter, member):
    if set(member) != {"request", "observation", "invocation"}:
        raise RtlIntakeRefusal("zero-return observations need their complete original request and output")
    request = original_zero_return_requests(trace, schema_observation)
    if json.loads(Path(member["request"]).read_bytes()) != request:
        raise RtlIntakeRefusal("zero-return request changed the original schema or full metadata/result slots")
    actual = I.require_environment(Path(member["invocation"]), environment=T.ENVIRONMENT)
    if (
        actual["argv"]
        != [getter["tensor_getter"]["python"], "-I", getter["observer"], member["request"], getter["getter"]]
        or actual["stage"] != "native_original_zero_return_observation"
        or Path(actual["stdout"]["path"]).read_bytes() != _plain(member["observation"]).read_bytes()
    ):
        raise RtlIntakeRefusal("zero-return observation differs from the actual fixed public native invocation")
    return json.loads(Path(member["observation"]).read_bytes())
