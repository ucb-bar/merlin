"""Materialize Model2MLIR's generic external-runtime protocol as ExecuTorch session artifacts.

Runs under the dedicated ExecuTorch venv.  This module intentionally contains no workload-name
dispatch: it imports the requested loader, asks Model2MLIR for its normalized tensor-index driver
plan, exports every actual PyTorch program through XNNPACK, and writes the strict manifest consumed
by :mod:`merlin.baselines.executorch_session`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

from _et_export import _load_loader


_DTYPES = {
    "torch.float32": "float32",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.bool": "bool",
}


def _install_lightweight_m2m_capture(m2m_root: str | Path) -> None:
    """Expose only the dependency-light external-runtime protocol in the ET venv.

    Model2MLIR's regular ``m2m.capture`` package imports compiler lowering modules (including xdsl)
    eagerly.  The dedicated ExecuTorch venv intentionally does not carry that compiler stack.  A
    namespace package plus the standalone protocol module lets workload loaders still import
    ``m2m.capture.causal_session`` while avoiding unrelated compiler dependencies.
    """
    root = Path(m2m_root).resolve()
    m2m_path, capture_path = root / "m2m", root / "m2m" / "capture"
    if not (capture_path / "external_runtime.py").is_file():
        raise FileNotFoundError(f"Model2MLIR external-runtime protocol is absent under {capture_path}")
    m2m = types.ModuleType("m2m")
    m2m.__path__ = [str(m2m_path)]
    capture = types.ModuleType("m2m.capture")
    capture.__path__ = [str(capture_path)]
    sys.modules["m2m"] = m2m
    sys.modules["m2m.capture"] = capture
    m2m.capture = capture
    name = "m2m.capture.external_runtime"
    spec = importlib.util.spec_from_file_location(name, capture_path / "external_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    capture.external_runtime = module
    capture.external_runtime_session = module.external_runtime_session


def _tensor_spec(value) -> dict:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"session ABI supports tensors only, got {type(value).__name__}")
    dtype = _DTYPES.get(str(value.dtype))
    if dtype is None:
        raise TypeError(f"session tensor dtype {value.dtype} is unsupported")
    if any(int(dim) <= 0 for dim in value.shape):
        raise ValueError(f"session tensor has a dynamic/empty shape: {tuple(value.shape)}")
    return {"dtype": dtype, "shape": [int(dim) for dim in value.shape]}


def _outputs(module, inputs):
    import torch

    with torch.no_grad():
        value = module(*inputs)
    values = value if isinstance(value, (tuple, list)) else (value,)
    if any(not isinstance(item, torch.Tensor) for item in values):
        raise TypeError("ExecuTorch session program outputs must be a flat tuple of tensors")
    return tuple(values)


def _export_program(program, out: Path, *, xnnpack: bool) -> tuple[dict, tuple]:
    import torch
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from torch.export import export

    module, inputs = program.module.eval(), tuple(value.contiguous() for value in program.inputs)
    outputs = _outputs(module, inputs)
    try:
        exported = export(module, inputs, strict=False)
    except Exception:  # noqa: BLE001
        exported = export(module, inputs, strict=True)
    # Constant propagation is an exact AOT transform and prevents frozen preprocessing constants
    # from becoming unnecessarily large activation arenas.
    try:
        from executorch.exir.passes.constant_prop_pass import constant_prop_pass
        exported = constant_prop_pass(exported)
    except Exception:  # noqa: BLE001
        pass
    partitioners = []
    if xnnpack:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        partitioners.append(XnnpackPartitioner())
    edge = to_edge_transform_and_lower(
        exported, partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False))
    et_program = edge.to_executorch(config=ExecutorchBackendConfig(external_constants=True))
    out.mkdir(parents=True, exist_ok=True)
    pte = out / "model.pte"
    with pte.open("wb") as stream:
        et_program.write_to_file(stream)
    try:
        et_program.write_tensor_data_to_file(str(out))
    except Exception:  # no external constants for a small program
        pass
    ptd = sorted(path.name for path in out.glob("*.ptd"))
    if len(ptd) > 1:
        raise RuntimeError(
            f"program {program.name!r} emitted multiple .ptd files; session runner ABI supports one")
    return ({"name": program.name, "pte": str(pte.relative_to(out.parent.parent)),
             "ptd": [str((out / name).relative_to(out.parent.parent)) for name in ptd],
             "method": "forward", "inputs": [_tensor_spec(v) for v in inputs]}, outputs)


def materialize(args) -> dict:
    import numpy as np
    import torch
    from m2m.capture import external_runtime_session

    torch.manual_seed(0)
    np.random.seed(0)
    loader = _load_loader(Path(args.loader))
    model, inputs = loader.get_model_and_inputs()
    session_metadata = loader.get_session_spec(model, inputs) if hasattr(loader, "get_session_spec") else None
    session = external_runtime_session(model, tuple(inputs), session=session_metadata)
    if session.observations != args.observations:
        raise ValueError(
            f"external runtime exposes {session.observations} observations, expected {args.observations}")
    if args.precision != "fp32":
        raise ValueError("stateful ExecuTorch materialization currently supports fp32 only")
    if args.paper_ready and not session.paper_ready:
        raise ValueError("paper execution requested but loader external-runtime session is not paper-ready")

    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    programs, eager_outputs = [], {}
    for program in session.programs:
        record, outputs = _export_program(
            program, root / "programs" / program.name, xnnpack=not args.no_xnnpack)
        programs.append(record)
        eager_outputs[program.name] = outputs

    bindings = []
    for binding in session.input_bindings:
        program, index = binding.target.program, binding.target.input_index
        value = binding.initial.detach().cpu().contiguous()
        rel = Path("inputs") / f"{program}-{index}-initial.bin"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.numpy().tobytes())
        bindings.append({"target": {"program": program, "index": index}, "kind": "initial",
                         "tensor": _tensor_spec(value), "file": str(rel)})
    for stream in session.streams:
        values = stream.values.detach().cpu().contiguous()
        rel = Path("streams") / f"{stream.target.program}-{stream.target.input_index}.bin"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(values.numpy().tobytes())
        bindings.append({
            "target": {"program": stream.target.program, "index": stream.target.input_index},
            "kind": "stream", "tensor": _tensor_spec(values[0]), "file": str(rel),
        })

    routes = []
    by_program = {program.name: program for program in session.programs}
    for route in session.routes:
        target_value = by_program[route.target.program].inputs[route.target.input_index]
        routes.append({
            "source": {"program": route.source.program, "index": route.source.output_index},
            "target": {"program": route.target.program, "index": route.target.input_index},
            "tensor": _tensor_spec(target_value), "update": "after_source",
        })

    # Expand cadence into a single unambiguous order.  Only the per-observation program has an
    # observation index; before/after calls are session-level.  This is what prevents accidental
    # same-buffer repetition in the native runner.
    calls = []
    for invocation in session.execution_schedule:
        if invocation.cadence == "once_before_observations":
            calls.extend({"stage": invocation.program, "program": invocation.program,
                          "observation": None, "timed": invocation.timed}
                         for _ in range(invocation.repeats))
    per_observation = [value for value in session.execution_schedule
                       if value.cadence == "per_observation"]
    for observation in range(session.observations):
        for invocation in per_observation:
            calls.append({"stage": invocation.program, "program": invocation.program,
                          "observation": observation, "timed": invocation.timed})
    for invocation in session.execution_schedule:
        if invocation.cadence == "once_after_observations":
            calls.extend({"stage": invocation.program, "program": invocation.program,
                          "observation": None, "timed": invocation.timed}
                         for _ in range(invocation.repeats))

    selector = session.observation_output
    assert selector is not None
    selected = eager_outputs[selector.program][selector.output_index]
    references = root / "references"
    references.mkdir(exist_ok=True)
    correctness = references / "correctness.npz"
    quality = references / "quality.npz"
    shutil.copy2(args.correctness, correctness)
    shutil.copy2(args.quality, quality)
    manifest = {
        "schema": "merlin.executorch.session/v1",
        "protocol_version": session.version,
        "kind": session.kind,
        "paper_ready": session.paper_ready,
        "precision": args.precision,
        "reset": session.reset,
        "observations": session.observations,
        "warmups": args.warmups,
        "measurement_repeats": args.measurement_repeats,
        "programs": programs,
        "bindings": bindings,
        "routes": routes,
        "execution_schedule": calls,
        "observation_output": {
            "source": {"program": selector.program, "index": selector.output_index},
            "tensor": _tensor_spec(selected),
        },
        "final_output": ({"program": session.final_output.program,
                          "index": session.final_output.output_index}
                         if session.final_output is not None else None),
        "correctness": str(correctness.relative_to(root)),
        "quality": str(quality.relative_to(root)),
        "logical_stages": list(session.metadata.get("stages", ()) or ()),
        "stage_schedule": [dict(value) for value in session.stage_schedule],
        "stage_attribution": ("native_programs" if session.version == 2 or
                              len(session.metadata.get("stages", ()) or ()) == 1
                              else "opaque_whole_forward"),
        "parameters": dict(session.parameters),
        "provenance": dict(session.provenance),
        "xnnpack": not args.no_xnnpack,
    }
    manifest_path = root / "executorch_session.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {"manifest": str(manifest_path), "programs": len(programs),
            "observations": session.observations, "paper_ready": session.paper_ready,
            "stage_attribution": manifest["stage_attribution"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--m2m-root", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--observations", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--measurement-repeats", type=int, required=True)
    parser.add_argument("--correctness", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--paper-ready", action="store_true")
    parser.add_argument("--no-xnnpack", action="store_true")
    args = parser.parse_args(argv)

    self_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [value for value in sys.path if value not in ("", ".", self_dir)]
    sys.path.insert(0, args.m2m_root)
    _install_lightweight_m2m_capture(args.m2m_root)
    result = materialize(args)
    print("ET_SESSION_EXPORT_JSON " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
