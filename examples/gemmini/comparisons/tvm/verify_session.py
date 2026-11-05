#!/usr/bin/env python3
"""Verify a supplied model2MLIR semantic session through ONNX and Relax LLVM on CPU.

The selected local factory owns model and workload choices. This verifier follows
the existing ExternalRuntimeSession protocol; it never selects a paper checkpoint,
casts weights, quantizes, downloads models, or qualifies device timing/model quality.
"""
import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from unittest import mock

from verify_resnet import file_identity, prepare_graph, relax_inventory, safe_path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_protocol(root):
    """Load the actual pure protocol without the unrelated MLIR capture initializer."""
    path = safe_path(root / "m2m/capture/external_runtime.py")
    name = "m2m.capture.external_runtime"
    if name in sys.modules:
        module = sys.modules[name]
        if safe_path(module.__file__) != path:
            raise ValueError("A different model2MLIR session protocol is already loaded")
        return module
    return load_module(name, path)


def factory_export_options(factory, torch, model):
    """Pass the selected model to hooks that explicitly accept it."""
    hook = getattr(factory, "get_onnx_export_options", None)
    if not callable(hook):
        return None
    return hook(torch, model=model) if "model" in inspect.signature(hook).parameters else hook(torch)


@contextmanager
def offline():
    settings = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    def reject(*args, **kwargs):
        raise RuntimeError("Session verification requires local artifacts; network connections are disabled")
    with mock.patch.dict(os.environ, settings), mock.patch.object(socket.socket, "connect", reject), \
            mock.patch.object(socket.socket, "connect_ex", reject), mock.patch.object(socket, "create_connection", reject):
        yield


def bfloat16_dtype():
    try:
        import ml_dtypes
    except ImportError as error:
        raise ValueError("Original BF16 verification requires the optional ml_dtypes dependency") from error
    return ml_dtypes.bfloat16


def array(value, np, torch, *, require_finite=True):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Session tensors must be dense CPU tensors")
        if value.dtype == torch.bfloat16:
            value = value.detach().contiguous().view(torch.uint16).numpy().view(bfloat16_dtype())
        else:
            try:
                value = value.detach().numpy()
            except TypeError as error:
                raise ValueError("Unsupported tensor dtype; implicit precision conversion is forbidden") from error
    elif hasattr(value, "numpy"):
        dtype = str(value.dtype)
        value = value.numpy()
        if dtype == "bfloat16":
            if value.dtype != np.uint16:
                raise ValueError("Expected raw uint16 storage from TVM's BF16 NumPy interface")
            value = value.view(bfloat16_dtype())
    value = np.asarray(value)
    if value.dtype.name not in ("bfloat16", "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"):
        raise ValueError("Unsupported session dtype: " + str(value.dtype))
    if any(extent <= 0 for extent in value.shape) or require_finite and not np.isfinite(value).all():
        raise ValueError("Session tensors require positive extents and finite values")
    return np.array(value, copy=True, order="C")


def torch_array(value, np, torch):
    if value.dtype.name == "bfloat16":
        return torch.from_numpy(value.copy().view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(value.copy())


def tensor_identity(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": int(value.nbytes),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def onnx_tensors(message, onnx):
    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            for child in value if field.label == field.LABEL_REPEATED else (value,):
                yield from onnx_tensors(child, onnx)


def load_onnx_graph(path, onnx):
    """Validate and bind external tensor files before ONNX opens them."""
    path = safe_path(path)
    identities = [file_identity(path, safe_path)]
    graph = onnx.load(str(path), load_external_data=False)
    external = {}
    tensors = list(onnx_tensors(graph, onnx))
    for tensor in tensors:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {entry.key: entry.value for entry in tensor.external_data}
        if len(fields) != len(tensor.external_data) or set(fields) - {"location", "offset", "length", "checksum"}:
            raise ValueError("Unsupported or duplicate ONNX external-data metadata")
        location = Path(fields.get("location", ""))
        if not fields.get("location") or location.is_absolute() or ".." in location.parts:
            raise ValueError("ONNX external weights must stay within their stage directory")
        artifact = safe_path(path.parent / location)
        if not artifact.is_relative_to(path.parent) or not artifact.is_file():
            raise ValueError("ONNX external weights must be local regular files")
        if artifact not in external:
            external[artifact] = file_identity(artifact, safe_path)
        offset, length = fields.get("offset", "0"), fields.get("length")
        if not offset.isdecimal() or length is not None and not length.isdecimal():
            raise ValueError("ONNX external-data ranges must be nonnegative integers")
        size = artifact.stat().st_size
        if int(offset) > size or length is not None and (int(length) < 1 or int(offset) + int(length) > size):
            raise ValueError("ONNX external-data range exceeds its weight file")
    # Path checking supports >2-GiB external-weight models without serializing them.
    onnx.checker.check_model(str(path), full_check=True)
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            onnx.external_data_helper.load_external_data_for_tensor(tensor, str(path.parent))
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
    identities.extend(external.values())
    for identity in identities:
        if file_identity(identity["path"], safe_path) != identity:
            raise AssertionError("ONNX model or external weights changed while loading")
    return graph, identities


def normalize(value, np, torch):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return tensor_identity(array(value, np, torch))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict) or hasattr(value, "items"):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Session metadata keys must be strings")
        return {key: normalize(item, np, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item, np, torch) for item in value]
    raise ValueError("Unsupported session metadata type: " + type(value).__name__)


def parameter_digest(module, torch):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Stage parameters must be dense CPU tensors")
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def endpoint(value):
    return value.program, value.input_index


def outputs(value, np, torch, *, require_finite=True):
    if isinstance(value, torch.Tensor) or hasattr(value, "numpy"):
        return [array(value, np, torch, require_finite=require_finite)]
    if isinstance(value, (tuple, list)) or hasattr(value, "__getitem__"):
        result = list(value)
        if result and all(isinstance(item, torch.Tensor) or hasattr(item, "numpy") for item in result):
            return [array(item, np, torch, require_finite=require_finite) for item in result]
    raise ValueError("Stages must return a tensor or flat nonempty tensor tuple")


class SessionComparisonError(AssertionError):
    def __init__(self, message, diagnostics, tensors=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.tensors = tensors


FAILED_FIXTURE_LIMIT_BYTES = 64 << 20


def write_failure_fixture(error, output_dir):
    """Retain validated failure snapshots without changing the comparison gate."""
    if error.tensors is None:
        return
    import numpy as np
    target, temporary = safe_path(output_dir / "failed_comparison.npz"), safe_path(output_dir / "failed_comparison.npz.tmp")
    created, renamed = False, False
    try:
        if target.exists():
            raise ValueError("Failure fixture already exists")
        arrays, encodings = {}, {}
        for name, value in zip(("actual", "reference"), error.tensors):
            if tensor_identity(value) != error.diagnostics[name]["identity"]:
                raise ValueError("Failure snapshot changed before retention")
            arrays[name] = value.view(np.uint16) if value.dtype.name == "bfloat16" else value
            encodings[name] = "raw_uint16_bfloat16" if value.dtype.name == "bfloat16" else "native_numpy"
        with temporary.open("xb") as stream:
            created = True
            np.savez(stream, **arrays)
        temporary.replace(target)
        renamed = True
        error.diagnostics["fixture_retention"] = {"status": "retained", "artifact": file_identity(target, safe_path), "encodings": encodings,
                                                  "bytes": sum(value.nbytes for value in error.tensors), "limit_bytes": FAILED_FIXTURE_LIMIT_BYTES}
    except Exception as failure:
        if created:
            temporary.unlink(missing_ok=True)
        if renamed:
            target.unlink(missing_ok=True)
        error.diagnostics["fixture_retention"] = {"status": "failed", "error": str(failure), "limit_bytes": FAILED_FIXTURE_LIMIT_BYTES}


def compare(actual, expected, np, rtol, atol, *, context=None):
    def summary(value):
        finite = bool(np.isfinite(value).all())
        return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite,
                "identity": tensor_identity(value) if finite else None}

    diagnostics = {**(context or {}), "actual": summary(actual), "reference": summary(expected),
                   "rtol": rtol if np.isfinite(rtol) else str(rtol), "atol": atol if np.isfinite(atol) else str(atol),
                   "max_absolute_error": None, "mismatch_count": None}
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not diagnostics["actual"]["finite"] or not diagnostics["reference"]["finite"]:
        diagnostics["reason"] = "shape/dtype/finite-value mismatch"
        raise SessionComparisonError("Session tensor shape/dtype/finite-value mismatch", diagnostics)
    floating = np.issubdtype(actual.dtype, np.floating) or actual.dtype.name == "bfloat16"
    if floating:
        with np.errstate(over="ignore", invalid="ignore"):
            left, right = actual.astype(np.float64), expected.astype(np.float64)
            matched = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
            error = float(np.max(np.abs(left - right))) if actual.size else 0.0
        diagnostics["max_absolute_error"] = error if np.isfinite(error) else None
        if not np.isfinite(error):
            diagnostics["metric_unavailable"] = "absolute difference exceeds float64 range"
    else:
        matched = actual == expected
        # Python integers preserve differences beyond float64's exact integer range.
        diagnostics["max_absolute_error"] = max((abs(int(a) - int(b)) for a, b in zip(actual[~matched], expected[~matched])), default=0)
    diagnostics["mismatch_count"] = int(np.count_nonzero(~matched))
    if diagnostics["mismatch_count"]:
        diagnostics["reason"] = "numerical comparison failed"
        size = int(actual.nbytes + expected.nbytes)
        tensors = (actual.copy(), expected.copy()) if size <= FAILED_FIXTURE_LIMIT_BYTES else None
        diagnostics["fixture_retention"] = {"status": "pending" if tensors is not None else "skipped", "bytes": size, "limit_bytes": FAILED_FIXTURE_LIMIT_BYTES}
        raise SessionComparisonError("Session numerical comparison failed", diagnostics, tensors)
    return {"passed": True, "max_absolute_error": diagnostics["max_absolute_error"], "actual": diagnostics["actual"]["identity"], "reference": diagnostics["reference"]["identity"]}


def describe_session(session, np, torch):
    phases = {"once_before_observations": 0, "per_observation": 1, "once_after_observations": 2}
    order = [phases.get(invocation.cadence, -1) for invocation in session.execution_schedule]
    if session.reset != "restore_initial_inputs" or -1 in order or order != sorted(order):
        raise ValueError("Unsupported session reset or interleaved/unknown cadence")
    names = [program.name for program in session.programs]
    scheduled = [invocation.program for invocation in session.execution_schedule]
    if len(set(scheduled)) != len(scheduled) or set(scheduled) != set(names):
        raise ValueError("Schedule must declare each program exactly once")
    initial = {endpoint(binding.target): array(binding.initial, np, torch) for binding in session.input_bindings}
    kinds = {endpoint(binding.target): binding.kind for binding in session.input_bindings}
    streams = {endpoint(stream.target) for stream in session.streams}
    if len(streams) != len(session.streams):
        raise ValueError("Duplicate session stream target")
    cadences = {item.program: item.cadence for item in session.execution_schedule}
    if any(cadences[stream.target.program] != "per_observation" for stream in session.streams):
        raise ValueError("Streams require a per-observation stage")
    assigned = set()
    for route in session.routes:
        key = route.source.program, endpoint(route.target)
        if route.update != "after_source" or key in assigned or kinds[endpoint(route.target)] != "state":
            raise ValueError("Unsupported or ambiguous state route")
        assigned.add(key)
    for binding in session.input_bindings:
        if binding.kind not in ("static", "stream", "state"):
            raise ValueError("Unsupported input binding kind")
        if (endpoint(binding.target) in streams) != (binding.kind == "stream"):
            raise ValueError("Stream binding kind differs from declared stream")
    for program in session.programs:
        if program.module.training or any(module.training for module in program.module.modules()):
            raise ValueError("Session stages must already be in evaluation mode")
        for index, value in enumerate(program.inputs):
            expected = array(value, np, torch)
            actual = initial[(program.name, index)]
            if actual.dtype != expected.dtype or actual.shape != expected.shape:
                raise ValueError("Initial binding differs from stage input ABI")
    record = {"version": session.version, "kind": session.kind, "observations": session.observations, "reset": session.reset,
              "programs": [{"name": program.name, "class": type(program.module).__module__ + "." + type(program.module).__qualname__,
                            "steps": program.steps} for program in session.programs],
              "metadata": normalize(session.metadata, np, torch), "schedule": [asdict(item) for item in session.execution_schedule],
              "routes": [asdict(item) for item in session.routes], "observation_output": asdict(session.observation_output),
              "final_output": asdict(session.final_output),
              "inputs": [{"target": asdict(item.target), "kind": item.kind, "value": tensor_identity(initial[endpoint(item.target)])}
                         for item in session.input_bindings],
              "streams": [{"target": asdict(item.target), "values": tensor_identity(array(item.values, np, torch))} for item in session.streams]}
    record["sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return record, initial


def verify_session(session, compile_stage, np, torch, *, repeats=2, rtol=1e-4, atol=1e-4, max_signatures=64, adapter_trace=None):
    if not 2 <= repeats <= 10 or not 1 <= max_signatures <= 128:
        raise ValueError("Invalid session repeat/signature budget")
    description, initial = describe_session(session, np, torch)
    if adapter_trace is not None:
        if not isinstance(adapter_trace, dict) or adapter_trace:
            raise ValueError("Adapter capture requires a fresh dictionary")
        adapter_trace.update(session=description, external=[], calls=[], observation_outputs=[], final_output=None, source_reset_stable=True, host_verified=False)
    external_indices = {}
    programs = {program.name: program for program in session.programs}
    parameter_hashes = {name: parameter_digest(program.module, torch) for name, program in programs.items()}
    compiled, checks, previous = {}, [], None
    high_water = 0
    for repeat in range(repeats):
        reference = {key: value.copy() for key, value in initial.items()}
        candidate = {key: value.copy() for key, value in initial.items()}
        last, fingerprints = {}, []
        origins = {key: ("initial", *key) for key in initial}
        last_calls, ordinal = {}, 0

        def invoke(invocation, observation):
            nonlocal high_water, ordinal
            program = programs[invocation.program]
            if observation is not None:
                for stream in session.streams:
                    if stream.target.program == program.name:
                        value = array(stream.values[observation], np, torch)
                        reference[endpoint(stream.target)] = value.copy()
                        candidate[endpoint(stream.target)] = value.copy()
                        origins[endpoint(stream.target)] = ("stream", *endpoint(stream.target), observation)
            keys = [(program.name, index) for index in range(len(program.inputs))]
            expected_args, actual_args = [reference[key] for key in keys], [candidate[key] for key in keys]
            context = {"stage": invocation.cadence, "program": program.name, "repeat": repeat, "observation": observation}
            for index, (actual, expected) in enumerate(zip(actual_args, expected_args)):
                compare(actual, expected, np, rtol, atol, context={**context, "tensor": "input", "input_index": index})
            signature = tuple((str(value.dtype), tuple(value.shape)) for value in actual_args)
            key = program.name, signature
            tensors = [torch_array(value, np, torch) for value in expected_args]
            with torch.no_grad():
                expected = outputs(program.module(*tensors), np, torch)
            if any(not np.array_equal(array(value, np, torch), original) for value, original in zip(tensors, expected_args)):
                raise AssertionError("Reference stage mutated caller inputs")
            if key not in compiled:
                if len(compiled) >= max_signatures:
                    raise ValueError("Session exceeds the explicit compilation-signature budget")
                compiled[key] = compile_stage(program, tuple(actual_args), len(expected))
            copied = [value.copy() for value in actual_args]
            actual = outputs(compiled[key](tuple(copied)), np, torch, require_finite=False)
            if any(not np.array_equal(value, original) for value, original in zip(copied, actual_args)):
                raise AssertionError("Compiled stage mutated caller inputs")
            if len(actual) != len(expected):
                raise AssertionError("Stage output count differs from reference")
            values = [compare(a, b, np, rtol, atol, context={**context, "tensor": "output", "output_index": index})
                      for index, (a, b) in enumerate(zip(actual, expected))]
            if adapter_trace is not None and "capture_error" not in adapter_trace:
                identities = [tensor_identity(value) for value in expected]
                new_external = len({origins[target] for target in keys if origins[target][0] != "output" and origins[target] not in external_indices})
                if repeat == 0 and (ordinal >= 64 or sum(len(call["outputs"]) for call in adapter_trace["calls"]) + len(expected) + len(adapter_trace["external"]) + new_external > 256):
                    adapter_trace["capture_error"] = "Session exceeds bounded adapter trace call/tensor slots at invocation " + str(ordinal)
                elif repeat == 0:
                    bindings = []
                    for target, value in zip(keys, expected_args):
                        origin = origins[target]
                        if origin[0] == "output":
                            bindings.append({"output": list(origin[1:])})
                        else:
                            if origin not in external_indices:
                                external_indices[origin] = len(adapter_trace["external"])
                                adapter_trace["external"].append({"origin": list(origin), "value": value.copy()})
                            bindings.append({"fixture": external_indices[origin]})
                    adapter_trace["calls"].append({"program": program.name, "inputs": [tensor_identity(value) for value in expected_args], "routes": bindings,
                                                   "outputs": identities, "values": tuple(value.copy() for value in expected), "cadence": invocation.cadence, "observation": observation})
                elif identities != adapter_trace["calls"][ordinal]["outputs"] or [tensor_identity(value) for value in expected_args] != adapter_trace["calls"][ordinal]["inputs"]:
                    adapter_trace["source_reset_stable"] = False
            last[program.name] = actual
            last_calls[program.name] = ordinal
            routes = []
            for route in session.routes:
                if route.source.program != program.name:
                    continue
                index, target = route.source.output_index, endpoint(route.target)
                if not 0 <= index < len(actual):
                    raise ValueError("State route selects an absent output")
                value = actual[index]
                if value.dtype != initial[target].dtype or value.ndim != initial[target].ndim:
                    raise ValueError("State route changes the declared dtype or rank")
                reference[target], candidate[target] = expected[index].copy(), value.copy()
                origins[target] = ("output", ordinal, index)
                routes.append({"name": route.name, "target": asdict(route.target), "comparison": compare(candidate[target], reference[target], np, rtol, atol)})
            high_water = max(high_water, sum(value.nbytes for value in candidate.values()))
            fingerprints.append([tensor_identity(value) for value in actual])
            checks.append({"repeat": repeat, "observation": observation, "program": program.name, "outputs": values, "routed_states": routes})
            if adapter_trace is not None and repeat == 0 and "capture_error" not in adapter_trace:
                adapter_trace["calls"][-1]["state_after"] = [list(origins[endpoint(binding.target)][1:]) for binding in session.input_bindings
                                                            if binding.kind == "state" and origins[endpoint(binding.target)][0] == "output"]
            ordinal += 1

        def phase(cadence, observation=None):
            for invocation in session.execution_schedule:
                if invocation.cadence == cadence:
                    count = 1 if cadence == "per_observation" else invocation.repeats
                    for _ in range(count):
                        invoke(invocation, observation)

        phase("once_before_observations")
        observations = []
        for observation in range(session.observations):
            phase("per_observation", observation)
            selector = session.observation_output
            if selector.program not in last or not 0 <= selector.output_index < len(last[selector.program]):
                raise ValueError("Observation selector is unavailable at its declared cadence")
            observations.append(tensor_identity(last[selector.program][selector.output_index]))
            if adapter_trace is not None and repeat == 0:
                adapter_trace["observation_outputs"].append([last_calls[selector.program], selector.output_index])
        phase("once_after_observations")
        selector = session.final_output
        if selector.program not in last or not 0 <= selector.output_index < len(last[selector.program]):
            raise ValueError("Final output selector is unavailable")
        fingerprints.append([tensor_identity(last[selector.program][selector.output_index]), observations])
        if adapter_trace is not None and repeat == 0:
            adapter_trace["final_output"] = [last_calls[selector.program], selector.output_index]
        if previous is not None and fingerprints != previous:
            raise AssertionError("Session outputs differ after restoring initial inputs")
        previous = fingerprints
    for name, program in programs.items():
        if parameter_digest(program.module, torch) != parameter_hashes[name]:
            raise AssertionError("Stage parameters or registered buffers changed")
    if adapter_trace is not None:
        adapter_trace.update(host_verified=True, host_repeats=repeats, host_rtol=rtol, host_atol=atol, parameter_sha256=parameter_hashes)
    return {"session": description, "checks": checks, "repeats": repeats, "reset_verified": True,
            "compilation_signatures": len(compiled), "parameter_sha256": parameter_hashes, "retained_stage_input_high_water_bytes": high_water}


def static_signature(program, tensors):
    return program, tuple((tensor["dtype"], tuple(tensor["shape"])) for tensor in tensors)


def admit_adapter_capture(trace):
    if not isinstance(trace, dict) or trace.get("host_verified") is not True or trace.get("source_reset_stable") is not True:
        raise ValueError("Automatic adapter requires verified host gates and byte-stable source resets")
    if trace.get("capture_error"):
        raise ValueError(trace["capture_error"])
    if not 1 <= len(trace["calls"]) <= 64 or not trace["external"]:
        raise ValueError("Adapter requires bounded nonempty calls and external inputs")


def adapter_reference_sources(reference_sources):
    if not reference_sources:
        raise ValueError("Independent source reference identities are required")
    sources = []
    for source in reference_sources:
        if file_identity(source["path"], safe_path) != source:
            raise AssertionError("Independent reference source or artifact changed")
        if source not in sources:
            sources.append(source)
    return sources


def session_adapter_plan(trace, records, package):
    """Bind semantic origins to actual exported ABIs, without transforms or candidate values."""
    admit_adapter_capture(trace)
    calls, external = trace["calls"], trace["external"]
    if not 1 <= len(calls) <= 64 or not 1 <= len(records) <= 64 or not external:
        raise ValueError("Adapter requires bounded nonempty calls, signatures and external inputs")
    stages = package["stages"]
    signatures = {static_signature(record["program"], record["inputs"]): "stage_" + str(index) for index, record in enumerate(records)}
    if len(signatures) != len(records) or set(stages) != set(signatures.values()):
        raise ValueError("Exported static signature inventory differs from checked stages")
    same = lambda tensor: (tensor["dtype"], tensor["shape"], tensor["bytes"])
    route_calls, output_order, used_fixtures, used_stages = [], [], set(), set()
    for ordinal, call in enumerate(calls):
        name = signatures.get(static_signature(call["program"], call["inputs"]))
        if name is None:
            raise ValueError("Semantic invocation lacks its actual checked static signature")
        stage = stages[name]
        if [same(tensor) for tensor in stage["inputs"]] != [same(tensor) for tensor in call["inputs"]] or [same(tensor) for tensor in stage["outputs"]] != [same(tensor) for tensor in call["outputs"]]:
            raise ValueError("Exported stage input/output dtype, shape or bytes differ from source ABI")
        if len(call["routes"]) != len(call["inputs"]) or len(call["values"]) != len(call["outputs"]):
            raise ValueError("Incomplete semantic route/reference output inventory")
        for binding, tensor in zip(call["routes"], call["inputs"]):
            if not isinstance(binding, dict) or len(binding) != 1:
                raise ValueError("Malformed semantic input route")
            if "fixture" in binding:
                index = binding["fixture"]
                if type(index) is not int or not 0 <= index < len(external):
                    raise ValueError("Fixture route requires a bounded plain integer")
                origin = tensor_identity(external[index]["value"])
                used_fixtures.add(index)
            elif "output" in binding:
                source = binding["output"]
                if not isinstance(source, list) or len(source) != 2 or any(type(index) is not int for index in source) or not 0 <= source[0] < ordinal or not 0 <= source[1] < len(calls[source[0]]["outputs"]):
                    raise ValueError("State route must name an earlier invocation/output")
                origin = calls[source[0]]["outputs"][source[1]]
            else:
                raise ValueError("No transforms or inferred state are admitted")
            if origin != tensor:
                raise ValueError("Semantic input bytes do not match their explicit origin")
        if [tensor_identity(value) for value in call["values"]] != call["outputs"]:
            raise ValueError("Source reference output bytes changed after host verification")
        if any(not isinstance(selector, list) or len(selector) != 2 or any(type(index) is not int for index in selector) or not 0 <= selector[0] <= ordinal or not 0 <= selector[1] < len(calls[selector[0]]["outputs"]) for selector in call["state_after"]):
            raise ValueError("Persistent state cannot name a future or absent output")
        route_calls.append({"stage": name, "inputs": copy.deepcopy(call["routes"])})
        output_order.extend({"invocation": ordinal, "output_index": index, "tensor": tensor} for index, tensor in enumerate(call["outputs"]))
        used_stages.add(name)
    if used_fixtures != set(range(len(external))) or used_stages != set(stages) or len(external) + len(output_order) > 256:
        raise ValueError("Adapter requires dense used fixtures/stages within bounded tensor slots")
    for selector in [*trace["observation_outputs"], trace["final_output"], *(value for call in calls for value in call["state_after"])]:
        if not isinstance(selector, list) or len(selector) != 2 or any(type(index) is not int for index in selector) or not 0 <= selector[0] < len(calls) or not 0 <= selector[1] < len(calls[selector[0]]["outputs"]):
            raise ValueError("Semantic output/state selector is unavailable")
    return {"version": 1, "calls": route_calls}, output_order


def session_memory_plan(trace, package):
    """Plan immutable routed storage; this is separate from the device verifier's retained trace."""
    align = lambda size: (size + 63) // 64 * 64
    for name in ("constants_bytes", "duplicated_stage_constants_bytes", "serial_workspace_bytes"):
        if type(package.get(name)) is not int or package[name] < 0:
            raise ValueError("Memory summaries must be plain nonnegative integers")
    calls = trace["calls"]
    external_bytes = sum(align(value["value"].nbytes) for value in trace["external"])
    sizes = {(ordinal, index): align(tensor["bytes"]) for ordinal, call in enumerate(calls) for index, tensor in enumerate(call["outputs"])}
    last_use = {key: key[0] for key in sizes}
    for ordinal, call in enumerate(calls):
        for binding in call["routes"]:
            if "output" in binding:
                key = tuple(binding["output"])
                last_use[key] = max(last_use[key], ordinal)
        for selector in call["state_after"]:
            key = tuple(selector)
            last_use[key] = max(last_use[key], min(ordinal + 1, len(calls) - 1))
    # Conservatively retain selected observation/final outputs to session end.
    for selector in [*trace["observation_outputs"], trace["final_output"]]:
        last_use[tuple(selector)] = len(calls) - 1
    fixed = align(package["constants_bytes"]) + align(package["serial_workspace_bytes"]) + external_bytes
    rows = []
    for ordinal, call in enumerate(calls):
        live = [key for key in sizes if key[0] <= ordinal <= last_use[key]]
        routed = {tuple(binding["output"]) for binding in call["routes"] if "output" in binding}
        rows.append({"invocation": ordinal, "live_output_routes": [list(key) for key in live], "live_output_bytes": sum(sizes[key] for key in live),
                     "routed_prior_input_bytes": sum(sizes[key] for key in routed), "current_output_bytes": sum(sizes[(ordinal, index)] for index in range(len(call["outputs"]))),
                     "planned_tensor_bytes": fixed + sum(sizes[key] for key in live)})
    unknown = ["linked executable/read-only image excluding counted constants", "stack", "CPU math/runtime globals and startup storage", "platform reservations", "packing/conversion scratch outside exported serial workspace", "model persistent or transient state outside explicit stage ABI"]
    return {"alignment_bytes": 64, "shared_constants_including_exported_packed_weights_bytes": align(package["constants_bytes"]),
            "duplicated_stage_constants_bytes_not_allocated": package["duplicated_stage_constants_bytes"], "serial_workspace_bytes": align(package["serial_workspace_bytes"]),
            "staged_external_inputs_bytes": external_bytes, "peak_planned_tensor_bytes": max(row["planned_tensor_bytes"] for row in rows), "invocations": rows,
            "verification_trace": {"all_retained_output_bytes": sum(sizes.values()), "single_trace_tensor_bytes": fixed + sum(sizes.values()),
                                   "guarded_double_outputs_frozen_fixture_image_stack_bytes": None},
            "assumptions": ["All explicit external inputs are staged once per reset; stream delivery can use a different plan", "Read-only routed fanout shares one output allocation; every current output is disjoint from inputs", "Prior state inputs and all current outputs coexist during each invocation", "Selected observation/final outputs remain live until session end", "Released immutable output storage may be reused only after its final route/state/selector use"],
            "unknown_overheads": unknown, "complete_production_peak_bytes": None, "production_fit_established": False}


def write_session_adapter(output_dir, trace, records, package, np, *, reference_sources, package_artifact=None, memory_limit_bytes=None):
    route, output_order = session_adapter_plan(trace, records, package)
    sources = adapter_reference_sources(reference_sources)
    memory = session_memory_plan(trace, package)
    if memory_limit_bytes is not None:
        if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0 or memory["peak_planned_tensor_bytes"] > memory_limit_bytes:
            raise ValueError("Aligned live session tensor plan exceeds the explicit export budget")
        memory["known_tensor_plan_within_export_budget"] = True
        memory["tensor_export_budget_bytes"] = memory_limit_bytes
    if package_artifact is not None and file_identity(package_artifact["path"], safe_path) != package_artifact:
        raise AssertionError("Exported package identity changed")
    output_dir = safe_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for kind, values in (("input", [value["value"] for value in trace["external"]]), ("output", [value for call in trace["calls"] for value in call["values"]])):
        for index, value in enumerate(values):
            storage = value.view(np.uint16) if str(value.dtype) == "bfloat16" else value
            arrays[kind + "_" + str(index)] = np.ascontiguousarray(storage[None])
    np.savez(output_dir / "fixture.npz", **arrays)
    (output_dir / "route.json").write_text(json.dumps(route, indent=2, sort_keys=True) + "\n")
    fixture = file_identity(output_dir / "fixture.npz", safe_path)
    provenance = {"fixture_sha256": fixture["sha256"], "reference_sources": sources, "sample_ids": ["semantic-session:" + trace["session"]["sha256"] + ":reset-0"],
                  "reference_contract": "Original declared Torch stage arithmetic on source-routed inputs after all host input/output/state/reset/parameter gates. Every invocation output is retained in call/output order; BF16 is raw uint16 payload. Device acceptance remains exact bytes; host allclose does not imply device acceptance.",
                  "session": trace["session"], "exported_package": package_artifact, "fixture_byte_order": sys.byteorder,
                  "parameter_sha256": trace["parameter_sha256"], "reset": "restore_initial_inputs; no route crosses session samples or repeats",
                  "host_gate": {"rtol": trace["host_rtol"], "atol": trace["host_atol"], "repeats": trace["host_repeats"], "source_reset_bytes_stable": True},
                  "input_order": [{"origin": value["origin"], "tensor": tensor_identity(value["value"])} for value in trace["external"]],
                  "output_order": output_order, "observation_outputs": trace["observation_outputs"], "final_output": trace["final_output"],
                  "route": file_identity(output_dir / "route.json", safe_path), "stage_mapping": [{"stage": "stage_" + str(index), "program": record["program"], "compiled_example_inputs": record["inputs"]} for index, record in enumerate(records)],
                  "production_memory": memory, "device_execution": False, "quality_qualified": False, "timing_qualified": False}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"route": provenance["route"], "fixture": fixture, "fixture_provenance": file_identity(output_dir / "provenance.json", safe_path),
            "invocations": len(route["calls"]), "external_inputs": len(trace["external"]), "retained_invocation_outputs": len(output_order),
            "production_memory": memory, "static_route_generated": True, "device_admission_verified": False, "exact_byte_device_gate": "unchanged; source fixture can differ from a host-allclose compiled candidate"}


# Admission order also preserves the existing refusal messages.
SOURCE_EXPORT_POLICIES = (
    ("scalar", "scalar", "torch_cpu_language_scale960_v1", False),
    ("rotary", "rotary", "torch_cpu_denoise_rotary64_table_v1", False),
    ("flash", "Flash", "torch_cpu_flash_bf16_vision1024_v1", True),
    ("silu", "SiLU", "torch_cpu_silu_fp32_v1", True),
)


class StageCompiler:
    def __init__(self, out, mode, np, torch, onnx, tvm, exporter="legacy", opset=None, bf16_fused_export=False, export_options=None):
        if exporter not in ("legacy", "dynamo"):
            raise ValueError("Unsupported ONNX exporter selection")
        if opset is not None and (type(opset) is not int or not 1 <= opset <= onnx.defs.onnx_opset_version()):
            raise ValueError("ONNX opset must be an integer supported by the selected ONNX dependency")
        self.out, self.mode, self.np, self.torch, self.onnx, self.tvm = out, mode, np, torch, onnx, tvm
        self.exporter = exporter
        self.opset = opset if opset is not None else (18 if exporter == "dynamo" else 17)
        if type(bf16_fused_export) is not bool or bf16_fused_export and (exporter != "dynamo" or self.opset < 18):
            raise ValueError("Fused BF16 export requires explicit dynamo selection and opset18 or newer")
        self.translation_source, self.export_options = None, {}
        self.export_helper = None
        for name, *_ in SOURCE_EXPORT_POLICIES:
            setattr(self, f"source_{name}_policy", "native")
        self.bf16_fused_export = bf16_fused_export
        if export_options is not None:
            if not isinstance(export_options, dict) or set(export_options) - {"custom_translation_table", "optimize", "source_scalar_policy", "source_rotary_policy", "source_flash_policy", "source_silu_policy"}:
                raise ValueError("Factory export options may only select translations, exporter optimization and source scalar/rotary/Flash/SiLU policies")
            for name, label, selected, requires_fused in SOURCE_EXPORT_POLICIES:
                policy = export_options.get(f"source_{name}_policy", "native")
                setattr(self, f"source_{name}_policy", policy)
                if policy not in ("native", selected):
                    raise ValueError(f"Unknown source {label} export policy")
                if policy != "native":
                    if requires_fused and (exporter != "dynamo" or not bf16_fused_export):
                        raise ValueError(f"Source {label} preservation requires explicit dynamo and fused BF16 export")
                    if not requires_fused and exporter != "dynamo":
                        raise ValueError(f"Source {label} preservation requires explicit dynamo export")
        if exporter == "dynamo":
            path = safe_path(Path(__file__).with_name("bf16_export.py"))
            self.translation_source = file_identity(path, safe_path)
            helper = load_module("tvm_session_bf16_export", path)
            self.export_helper = helper
            self.export_options["custom_translation_table"] = helper.scalar_translations(torch)
        if bf16_fused_export:
            self.export_options["custom_translation_table"].update(helper.translations(torch, source_flash_policy=self.source_flash_policy, source_silu_policy=self.source_silu_policy))
            # Preserve original initializer dtypes, including BF16 bias storage.
            self.export_options["optimize"] = False
        if export_options is not None:
            translations = export_options.get("custom_translation_table", {})
            if not isinstance(translations, dict) or any(not callable(value) for value in translations.values()):
                raise ValueError("Factory ONNX translations must be a dictionary of callables")
            if translations and exporter != "dynamo":
                raise ValueError("Factory ONNX translations require explicit dynamo export")
            existing = self.export_options.get("custom_translation_table", {})
            if set(translations) & set(existing):
                raise ValueError("Factory and verifier ONNX translations overlap")
            if translations:
                self.export_options["custom_translation_table"] = {**existing, **translations}
            if "optimize" in export_options:
                if type(export_options["optimize"]) is not bool or bf16_fused_export and export_options["optimize"]:
                    raise ValueError("Invalid factory exporter optimization policy")
                self.export_options["optimize"] = export_options["optimize"]
        self.records, self.imported_modules = [], []

    def __call__(self, program, inputs, output_count):
        folder = self.out / ("stage_" + str(len(self.records)))
        folder.mkdir()
        graph_path = folder / "model.onnx"
        names = ["input_" + str(index) for index in range(len(inputs))]
        tensors = tuple(torch_array(value, self.np, self.torch) for value in inputs)
        # Tracing mutates plain Python cache attributes into FakeTensor objects.
        # Export an isolated stage so subsequent source calls and resets remain real.
        export_module = copy.deepcopy(program.module)
        preservation_records = {}
        source_ep = None
        export_source = export_module
        if any(getattr(self, f"source_{name}_policy") != "native" for name, *_ in SOURCE_EXPORT_POLICIES):
            original_parameters = parameter_digest(program.module, self.torch)
            source_ep = self.torch.export.export(export_module, tensors, strict=False)
            original_bindings = self._export_bindings(source_ep)
            original_graph, original_signature = str(source_ep.graph), str(source_ep.graph_signature)
            original_abi = self._export_abi(source_ep)
            export_source = source_ep
            # Flash admission authenticates the original graph before any rewrite.
            for name in ("flash", "scalar", "rotary", "silu"):
                export_source, preservation_records[name] = getattr(
                    self.export_helper, f"prepare_source_{name}_export")(
                        self.torch, export_source, getattr(self, f"source_{name}_policy"))
            rotary_record = preservation_records["rotary"]
            added_constants = tuple(rotary_record["added_constant_names"]) if rotary_record else ()
            if set(export_source.constants) - set(source_ep.constants) != set(added_constants) or set(source_ep.constants) - set(export_source.constants) or set(added_constants) & set(source_ep.state_dict):
                raise AssertionError("Prepared export changed the allowed constant inventory")
            if self._export_bindings(export_source, added_constants) != original_bindings or self._export_abi(export_source) != original_abi:
                raise AssertionError("Prepared export changed original bindings or public inputs/outputs")
            prepared_bindings = self._export_bindings(export_source)
            prepared_graph, prepared_signature = str(export_source.graph), str(export_source.graph_signature)
        exported = self.torch.onnx.export(export_source, tensors, str(graph_path), input_names=names,
                                         output_names=["output_" + str(index) for index in range(output_count)],
                                         opset_version=self.opset, dynamo=self.exporter == "dynamo", external_data=True, do_constant_folding=True, **self.export_options)
        if source_ep is not None:
            if self._export_bindings(source_ep) != original_bindings or self._export_bindings(export_source) != prepared_bindings or parameter_digest(program.module, self.torch) != original_parameters:
                raise AssertionError("Source-preserving export changed parameters or constants")
            if str(source_ep.graph) != original_graph or str(source_ep.graph_signature) != original_signature:
                raise AssertionError("Source-preserving export changed original graph or signature")
            if str(export_source.graph) != prepared_graph or str(export_source.graph_signature) != prepared_signature or self._export_abi(export_source) != original_abi:
                raise AssertionError("ONNX export changed the prepared graph or public inputs/outputs")
            for name, *_ in SOURCE_EXPORT_POLICIES:
                preservation_record = preservation_records[name]
                if preservation_record is not None:
                    preservation_record.update(original_parameters_sha256=original_parameters, original_EP_bindings_sha256=original_bindings,
                                               original_EP_graph_sha256=hashlib.sha256(original_graph.encode()).hexdigest(),
                                               original_EP_signature_sha256=hashlib.sha256(original_signature.encode()).hexdigest(), bindings_preserved=True, original_public_ABI_preserved=True)
        del export_module, export_source, source_ep
        graph, artifacts = load_onnx_graph(graph_path, self.onnx)
        from tvm.relax.frontend.onnx import from_onnx
        mod = from_onnx(graph, shape_dict={name: list(value.shape) for name, value in zip(names, inputs)}, keep_params_in_input=False)
        transformed, lowered = prepare_graph(mod, self.mode, self.tvm)
        record = {"program": program.name, "inputs": [tensor_identity(value) for value in inputs], "onnx": artifacts[0], "onnx_artifacts": artifacts,
                  "exporter": self.exporter, "opset": self.opset, "export_source_isolation": "deepcopy",
                  "pipelines": ["zero", "default_build"] if self.mode == "optimized" else ["default_build"]}
        if self.export_options.get("custom_translation_table"):
            targets = self.export_options["custom_translation_table"]
            record["translated_aten_operations"] = {str(target): sum(node.op == "call_function" and node.target == target for node in exported.exported_program.graph.nodes) for target in targets}
        if "optimize" in self.export_options:
            record["onnx_export_optimization"] = self.export_options["optimize"]
        for name, *_ in SOURCE_EXPORT_POLICIES:
            preservation_record = preservation_records.get(name)
            if preservation_record is not None:
                record[f"source_{name}_export"] = preservation_record
        if self.translation_source:
            if file_identity(self.translation_source["path"], safe_path) != self.translation_source:
                raise AssertionError("Selected floating export translations changed")
            record["translation_source"] = self.translation_source
            record["scalar_tensor_export"] = "DOUBLE preserves the original scalar in a typed Constant; other ordinary dtypes use the built-in translation"
        if self.bf16_fused_export:
            record["bf16_fused_arithmetic"] = "original BF16 interfaces; linear, SDPA and single-output LayerNorm use FP32 math; no native reduction/layout/statistics reproduction; unchanged source gate"
            record["bf16_attention_profile"] = "fp32-math"
            flash_record = preservation_records.get("flash")
            if flash_record is not None and flash_record["site_count"]:
                record["bf16_fused_arithmetic"] = "original BF16 interfaces; source CPU Flash for proved vision sites, generic FP32 math elsewhere; unchanged source gate"
                record["bf16_attention_profile"] = self.source_flash_policy + ";generic-fp32-math-elsewhere"
            record["onnx_export_optimization"] = False
        for name, value in (("imported", mod), ("graph", transformed), ("vm", lowered)):
            path = folder / (name + ".relax.py")
            path.write_text(value.script())
            record[name] = {"ir": file_identity(path, safe_path), "inventory": relax_inventory(value, self.tvm)}
        vm = self.tvm.relax.VirtualMachine(self.tvm.relax.build(lowered, "llvm", pipeline=None), self.tvm.cpu())
        self.records.append(record)
        self.imported_modules.append(mod)

        def run(values):
            arguments = [self.tvm.nd.array(value.copy(), self.tvm.cpu()) for value in values]
            result = vm["main"](*arguments)
            if any(not self.np.array_equal(array(value, self.np, self.torch), before) for value, before in zip(arguments, values)):
                raise AssertionError("TVM stage mutated caller inputs")
            return result
        return run

    @staticmethod
    def _export_abi(program):
        return str([spec for spec in program.graph_signature.input_specs if spec.kind.name == "USER_INPUT"]), str(program.graph_signature.output_specs), str(program.call_spec)

    def _export_bindings(self, program, ignored_constants=()):
        if len(set(ignored_constants)) != len(ignored_constants) or not set(ignored_constants) <= set(program.constants):
            raise ValueError("Invalid source export constant exclusions")
        digest = hashlib.sha256()
        for scope, values in (("state", program.state_dict), ("constant", program.constants)):
            for name, value in sorted(values.items()):
                if scope == "constant" and name in ignored_constants:
                    continue
                if not isinstance(value, self.torch.Tensor) or value.device.type != "cpu" or value.layout != self.torch.strided:
                    raise ValueError("Source-preserving export requires dense CPU tensor bindings")
                digest.update(json.dumps([scope, name, str(value.dtype), list(value.shape), list(value.stride()), value.storage_offset()]).encode())
                digest.update(value.detach().contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()


    def export_stages(self, output_dir, *, memory_limit_bytes, target=None, adapter_trace=None, reference_sources=()):
        """Package checked signatures, optionally deriving routes from a checked semantic trace."""
        if not self.records or len(self.records) != len(self.imported_modules):
            raise ValueError("No complete compiled stage inventory is available")
        if adapter_trace is not None:
            admit_adapter_capture(adapter_trace)
            if not 1 <= len(self.records) <= 64:
                raise ValueError("Adapter exceeds the bounded exported stage inventory")
            reference_sources = adapter_reference_sources(reference_sources)
        from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
        from tvm.relax.backend.contrib.gemmini_tuning import collect_gemmini_workloads
        app = safe_path(Path(self.tvm.__file__).parents[2] / "apps/gemmini")
        sys.path.insert(0, str(app))
        exporter = load_module("tvm_session_baremetal", app / "baremetal.py")
        sources = [file_identity(function.__code__.co_filename, safe_path) for function in (exporter.export_graphs, prepare_gemmini_graph, collect_gemmini_workloads)]
        modules, stages = {}, {}
        for index, (record, mod) in enumerate(zip(self.records, self.imported_modules)):
            name = "stage_" + str(index)
            modules[name] = prepare_gemmini_graph(mod, optimize=self.mode == "optimized")
            stages[name] = {"program": record["program"], "inputs": record["inputs"],
                            "onnx_artifacts": record["onnx_artifacts"], "workloads": collect_gemmini_workloads(mod)}
        bundle = exporter.export_graphs(modules, output_dir, target=target or exporter.RV64_TARGET, memory_limit_bytes=memory_limit_bytes)
        result = {"artifact": file_identity(safe_path(output_dir) / "graphs.json", safe_path), "sources": sources,
                "stages": stages, "constants_bytes": bundle["constants_bytes"],
                "duplicated_stage_constants_bytes": bundle["duplicated_stage_constants_bytes"],
                "maximum_stage_tensor_bytes": bundle["maximum_stage_tensor_bytes"],
                "serial_workspace_bytes": bundle["serial_workspace_bytes"],
                "state_routing": "caller_owned", "session_device_execution": False,
                "quality_qualified": False, "timing_qualified": False}
        if adapter_trace is not None:
            result["automatic_session_adapter"] = write_session_adapter(safe_path(output_dir) / "session", adapter_trace, self.records, bundle, self.np,
                                                                        reference_sources=reference_sources, package_artifact=result["artifact"], memory_limit_bytes=memory_limit_bytes)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "tvm-source", "tvm-build", "factory", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--factory-name", default="get_model_and_inputs")
    parser.add_argument("--factory-arguments", type=Path, help="Local JSON keyword arguments passed unchanged to the selected factory")
    parser.add_argument("--artifact", type=Path, action="append", default=[], help="Explicit local checkpoint/input/config artifacts to hash")
    parser.add_argument("--graph-mode", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--onnx-exporter", choices=("legacy", "dynamo"), default="legacy", help="Explicit modern exporter for compatible source models; dynamo defaults to opset18 and needs onnxscript")
    parser.add_argument("--opset", type=int, help="Explicit ONNX schema version; defaults to17 legacy or18 dynamo; native BF16 Conv requires22")
    parser.add_argument("--bf16-fused-export", action="store_true", help="Opt-in dynamo BF16 linear/SDPA FP32-math candidates; original BF16 ABI, unchanged source comparison, no native CPU kernel/layout equivalence claim")
    parser.add_argument("--export-baremetal", action="store_true", help="Package checked static stages, derive declared session routes and freeze source reference bytes; no device qualification")
    parser.add_argument("--memory-limit-bytes", type=int, help="Explicit export/live-session tensor budget; ELF, stack, runtime and guarded verification fixtures need separate accounting")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-signatures", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 10 or not 1 <= args.max_signatures <= 128 or not 0 <= args.rtol <= 0.1 or not 0 <= args.atol <= 0.1:
        parser.error("Invalid repeat/signature budget or numerical tolerances")
    if args.export_baremetal and (args.memory_limit_bytes is None or args.memory_limit_bytes <= 0):
        parser.error("Baremetal export requires a positive explicit --memory-limit-bytes")
    if not args.export_baremetal and args.memory_limit_bytes is not None:
        parser.error("Memory budget requires --export-baremetal")
    sys.dont_write_bytecode = True
    out = safe_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "invocation": [sys.executable, *sys.argv], "host_session_verified": False,
              "model_quality_qualified": False, "paper_validation": False, "timing_qualified": False, "device_execution": False}
    try:
        report["verifier_source"] = file_identity(__file__, safe_path)
        report["graph_helper_source"] = file_identity(prepare_graph.__code__.co_filename, safe_path)
        source, build, model_root = safe_path(args.tvm_source), safe_path(args.tvm_build), safe_path(args.model2mlir_root)
        os.environ.update(TVM_LIBRARY_PATH=str(build), TVM_FFI="ctypes")
        sys.path[:0] = [str(safe_path(source / "python")), str(model_root)]
        import numpy as np
        import torch
        import onnx
        import tvm
        from tvm._ffi.base import _LIB
        protocol = load_protocol(model_root)
        if safe_path(protocol.__file__) != model_root / "m2m/capture/external_runtime.py":
            raise ValueError("Imported session protocol differs from selected model2MLIR source")
        if not safe_path(tvm.__file__).is_relative_to(source / "python") or safe_path(_LIB._name).parent != build:
            raise ValueError("Imported TVM source/library differs from the explicit selection")
        report["protocol_source"] = file_identity(protocol.__file__, safe_path)
        report["compiler_library"] = file_identity(_LIB._name, safe_path)
        report["versions"] = {"python": sys.version, "torch": torch.__version__, "onnx": onnx.__version__, "numpy": np.__version__, "tvm": tvm.__version__}
        report["build_info"] = dict(tvm.support.libinfo())
        report["graph_mode"], report["rtol"], report["atol"] = args.graph_mode, args.rtol, args.atol
        report["factory_source"] = file_identity(args.factory, safe_path)
        report["factory_name"] = args.factory_name
        report["factory_environment"] = {name: value for name, value in os.environ.items() if name.startswith("M2M_")}
        report["declared_artifacts"] = [file_identity(path, safe_path) for path in args.artifact]
        arguments = {}
        if args.factory_arguments:
            report["factory_arguments"] = file_identity(args.factory_arguments, safe_path)
            arguments = json.loads(safe_path(args.factory_arguments).read_text())
            if not isinstance(arguments, dict):
                raise ValueError("Factory arguments must be a JSON object")
        with offline():
            factory = load_module("tvm_session_factory", args.factory)
            model, inputs = getattr(factory, args.factory_name)(**arguments)
            artifacts = factory.get_verification_artifacts(model, inputs) if callable(getattr(factory, "get_verification_artifacts", None)) else ()
            report["factory_artifacts"] = [file_identity(path, safe_path) for path in artifacts]
            metadata = factory.get_session_spec(model, inputs) if callable(getattr(factory, "get_session_spec", None)) else None
            session = protocol.external_runtime_session(model, tuple(inputs), session=metadata)
            report["session"] = describe_session(session, np, torch)[0]
            report["onnx_exporter"] = args.onnx_exporter
            hook = getattr(factory, "get_onnx_export_options", None)
            export_options = factory_export_options(factory, torch, model)
            if callable(hook):
                report["factory_export_source"] = file_identity(hook.__code__.co_filename, safe_path)
                report["factory_export_translations"] = [str(target) for target in export_options.get("custom_translation_table", {})] if isinstance(export_options, dict) else []
            compiler = StageCompiler(out, args.graph_mode, np, torch, onnx, tvm, exporter=args.onnx_exporter, opset=args.opset, bf16_fused_export=args.bf16_fused_export, export_options=export_options)
            report["bf16_fused_export"] = args.bf16_fused_export
            if compiler.translation_source:
                report["translation_source"] = compiler.translation_source
            report["compiled_stages"] = compiler.records
            adapter_trace = {} if args.export_baremetal else None
            report.update(verify_session(session, compiler, np, torch, repeats=args.repeats, rtol=args.rtol, atol=args.atol, max_signatures=args.max_signatures, adapter_trace=adapter_trace))
            report["host_session_verified"] = True
            if args.export_baremetal:
                reference_sources = [report["verifier_source"], report["protocol_source"], report["factory_source"], *report["declared_artifacts"], *report["factory_artifacts"]]
                if "factory_arguments" in report:
                    reference_sources.append(report["factory_arguments"])
                if "factory_export_source" in report:
                    reference_sources.append(report["factory_export_source"])
                report["baremetal_export"] = compiler.export_stages(out / "baremetal", memory_limit_bytes=args.memory_limit_bytes, adapter_trace=adapter_trace, reference_sources=reference_sources)
        infos = [report["verifier_source"], report["graph_helper_source"], report["protocol_source"], report["factory_source"],
                 report["compiler_library"], *report["declared_artifacts"], *report["factory_artifacts"]]
        if "factory_arguments" in report:
            infos.append(report["factory_arguments"])
        if "translation_source" in report:
            infos.append(report["translation_source"])
        if "factory_export_source" in report:
            infos.append(report["factory_export_source"])
        infos.extend(artifact for stage in report["compiled_stages"] for artifact in stage["onnx_artifacts"])
        if "baremetal_export" in report:
            infos.extend(report["baremetal_export"]["sources"])
            infos.append(report["baremetal_export"]["artifact"])
            adapter = report["baremetal_export"].get("automatic_session_adapter")
            if adapter:
                infos.extend(adapter[field] for field in ("route", "fixture", "fixture_provenance"))
        for info in infos:
            if file_identity(info["path"], safe_path) != info:
                raise AssertionError("Selected source or declared artifact changed")
        report["limitations"] = ["Host frontend/session routing only; no selected paper checkpoint, application-quality or device/timing qualification",
                                 "Explicit artifact/source hashes are not a complete runtime source closure",
                                 "Python network connections and HuggingFace downloads disabled; this is not a hermetic process sandbox",
                                 "Input-state high-water describes diagnostic retained arrays, not deployed allocator peak or model fit"]
        report.update(status="passed", host_session_verified=True)
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if isinstance(error, SessionComparisonError):
            write_failure_fixture(error, out)
            report["failure_diagnostics"] = error.diagnostics
    finally:
        (out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(out)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
