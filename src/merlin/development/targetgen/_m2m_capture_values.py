"""Capture loader provenance, tensor values, calibration observations and leaf constants."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import random
import sys
from pathlib import Path

from merlin.development.targetgen._m2m_capture_deployment import _mlir_dtype

_SCHEME = {
    "fp32": (None, None),
    "f32": (None, None),
    "bf16": (None, "bfloat16"),
    "fp16": (None, "float16"),
    "f16": (None, "float16"),
    "int8": ("int8_weight_only", None),
    "i8": ("int8_weight_only", None),
    "fp8": ("float8_weight_only_e4m3", None),
    "fp8_e4m3": ("float8_weight_only_e4m3", None),
}


def _load_loader(loader_py: Path):
    spec = importlib.util.spec_from_file_location("_capsule_loader", loader_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_model_and_inputs"):
        raise RuntimeError(f"{loader_py} must define get_model_and_inputs() -> (model, inputs)")
    return mod


def _loader_dependency_sources(modules_before: set[str], loader_py: Path) -> list[dict]:
    """Observe newly imported Python source after loader initialization and input creation.

    This is not a complete import/data closure or proof of the bytes Python executed.
    The hashes let later staging refuse source changes since this observation.
    """
    sources = []
    for name in sorted(set(sys.modules) - modules_before):
        source_name = getattr(sys.modules.get(name), "__file__", None)
        if not source_name:
            continue
        path = Path(source_name)
        if path.suffix != ".py" or not path.is_file() or path.resolve() == loader_py.resolve():
            continue
        sources.append(
            {
                "module": name,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return sources


def _quant_for(dtype: str, override: str | None = None):
    """The torchAO config for ``dtype``, or ``override`` when the caller names a scheme explicitly.

    WHY AN OVERRIDE EXISTS. The default int8 scheme is WEIGHT-ONLY, which emits a float matmul over
    dequantized weights -- `linalg.matmul` tagged `aten.mm.default` with an f32 datapath. That is the
    right capture for a model ladder that quantizes weights only, and the WRONG program for a capsule
    meant to exercise an integer systolic datapath: no golden substitution can fix a program that
    contains no integer contraction. `int8_dyn_act_int8_weight` emits `aten._int_mm.default`
    accumulating in i32 -- the arithmetic the mesh actually runs.

    Selectable rather than switched, because the same map serves the whole-model recaptures, and
    whether the model ladder wants W8A8 is a separate decision that must not ride along silently.
    """
    scheme = override or _SCHEME.get(dtype, (None, None))[0]
    if scheme is None:
        return None
    from m2m.capture.torchao_pipeline import QuantizationConfig

    return QuantizationConfig(scheme=scheme)


def _to_native(t):
    """A torch tensor -> JSON-safe lists without rounding integer or boolean values."""
    import torch

    if isinstance(t, torch.Tensor):
        value = t.detach().cpu()
        return (value if not (value.is_floating_point() or value.is_complex()) else value.to(torch.float64)).tolist()
    if isinstance(t, (list, tuple)):
        return [_to_native(x) for x in t]
    return t


def _scalars(mapping) -> dict:
    """The JSON-safe scalar entries of a loader-declared mapping (tensors and streams are dropped).

    A session spec carries the image/trajectory STREAM alongside its provenance; serializing that would
    put hundreds of megabytes of operand data into a meta file. Only scalars travel, so what is recorded
    is the loader's own statement about the data and never the data itself.
    """
    out = {}
    for key, value in (mapping or {}).items():
        if value is None or isinstance(value, (str, bool, int, float)):
            out[str(key)] = value
    return out


def _loader_provenance(mod, mdl, inputs) -> dict:
    """What the LOADER declares about the data this capture ran on -- never inferred here.

    A whole-model loader can be driven down more than one input path (a real, attributed dataset
    stream; a seeded synthetic one), and which path ran is a property of the loader's environment, not
    of the program. If nothing records it, a capsule that only ever proves COMPILER CORRECTNESS -- the
    compiled program reproduces the reference the same loader produced on the same inputs -- can be
    read as a statement about the model's accuracy on real data, which it is not.

    Two declaration shapes are honored because both already exist in the wild: a ``session_provenance``
    attribute on the returned module, and a ``get_session_spec`` function returning ``provenance`` +
    ``paper_ready``. FAIL CLOSED: a loader that declares neither yields status ``undeclared`` and a
    loader whose declaration RAISES yields ``error`` -- both distinct from "declared not synthetic", so
    an unknown can never be recorded as a claim.
    """
    prov: dict = {}
    ready = None
    status = "undeclared"
    error = None
    direct = getattr(mdl, "session_provenance", None)
    if isinstance(direct, dict):
        prov.update(_scalars(direct))
        status = "declared"
    if hasattr(mod, "get_session_spec"):
        try:
            spec = mod.get_session_spec(mdl, inputs)
        except Exception as exc:  # noqa: BLE001 -- a raising declaration is a fact
            error = f"{type(exc).__name__}: {exc}"
            if status != "declared":
                status = "error"
        else:
            if isinstance(spec, dict):
                declared = spec.get("provenance")
                if isinstance(declared, dict):
                    prov.update(_scalars(declared))
                    status = "declared"
                if isinstance(spec.get("paper_ready"), bool):
                    ready = bool(spec["paper_ready"])
    return {
        "loader_provenance": prov,
        "loader_paper_ready": ready,
        "loader_provenance_status": status,
        "loader_provenance_error": error,
    }


def _input_abi(inputs):
    """Flatten the loader's input pytree and preserve each tensor leaf's real shape and dtype."""
    import torch

    leaves, _spec = torch.utils._pytree.tree_flatten(inputs)
    bad = [type(x).__name__ for x in leaves if not isinstance(x, torch.Tensor)]
    if bad:
        raise RuntimeError(f"model loader inputs must have only tensor leaves; got {bad}")
    return leaves, [{"shape": list(x.shape), "dtype": _mlir_dtype(x.dtype)} for x in leaves]


def _calibration_stream_limit(RQ, recipe) -> int:
    """Samples a calibration stream must supply: what the recipe observes, and the agreement receipt."""
    observed = RQ.calibration_limit(
        recipe, inspect.signature(RQ.apply_recipe).parameters["calibration_samples"].default
    )
    return max(observed, inspect.signature(RQ.agreement).parameters["limit"].default)


def _freeze_calibration(samples, *, limit, normalize, torch):
    """Own bounded sample values before a generator can reuse its scratch storage."""
    from itertools import islice

    calibration = []
    for sample in islice(samples, limit):
        sample = normalize(sample)
        leaves, _ = _input_abi(sample)
        if any(leaf.is_complex() for leaf in leaves):
            raise RuntimeError("FP32 staging does not admit complex calibration inputs")
        calibration.append(torch.utils._pytree.tree_map(lambda leaf: leaf.detach().clone(), sample))
    return calibration


def _float_reference(mdl, inputs, torch) -> dict:
    """The untransformed model's outputs on ``inputs``, in the golden's JSON shape.

    Every RNG this worker seeds is saved and restored around the forward, so a loader whose forward
    draws random numbers produces the same golden afterwards as it would have without this run.
    """
    import numpy as np

    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        with torch.no_grad():
            y = mdl(*inputs)
        leaves, abi = _output_abi(y)
        leaves = [leaf.detach().clone() for leaf in leaves]
        values = [_to_native(x) for x in leaves]
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    return {"outputs": values[0] if len(values) == 1 else values, "output_abi": abi, "leaves": leaves}


def _output_abi(outputs):
    """Flatten model results and preserve the ABI of every tensor result.

    Whole-model captures historically recorded only the nested JSON values.  A list-shaped tensor and
    a tuple of tensors are indistinguishable in that representation, which made the parent silently
    retain only result zero.  The pytree flattening performed while torch still owns the values is the
    authoritative result cardinality and dtype record.
    """
    import torch

    leaves, _spec = torch.utils._pytree.tree_flatten(outputs)
    bad = [type(x).__name__ for x in leaves if not isinstance(x, torch.Tensor)]
    if bad:
        raise RuntimeError(f"model outputs must have only tensor leaves; got {bad}")
    return leaves, [{"shape": list(x.shape), "dtype": _mlir_dtype(x.dtype)} for x in leaves]


def _integerized_agreement(before, after, *, atol: float, rtol: float) -> dict:
    from merlin.development.targetgen._dynamic_integer_reference import compare_outputs

    return compare_outputs(before, after, atol=atol, rtol=rtol, output_abi=_output_abi)


def _fp32_stage_observation(
    original: dict, staged_leaves, original_input_abi: list[dict], staged_input_abi: list[dict]
) -> dict:
    """Measure a selected precision change; this is not an accuracy or quantization gate."""
    import torch

    before = original["leaves"]
    _, after_abi = _output_abi(tuple(staged_leaves))
    before_abi = original["output_abi"]
    if len(before) != len(staged_leaves) or not before:
        raise RuntimeError("FP32 staging changed output cardinality")
    rows = []
    for left, right, old, new in zip(before, staged_leaves, before_abi, after_abi, strict=True):
        if left.shape != right.shape:
            raise RuntimeError("FP32 staging changed output shape")
        if not left.is_floating_point():
            if right.dtype != left.dtype or not torch.equal(left, right):
                raise RuntimeError("FP32 staging changed an exact integer/bool result")
            rows.append(
                {
                    "shape": list(left.shape),
                    "original_dtype": old["dtype"],
                    "staged_dtype": new["dtype"],
                    "comparison": "exact_nonfloating",
                    "max_abs": 0.0,
                    "max_rel": 0.0,
                }
            )
            continue
        if right.dtype != torch.float32:
            raise RuntimeError("FP32 staging retained a non-FP32 floating output")
        lhs, rhs = left.detach().to(torch.float64), right.detach().to(torch.float64)
        if not bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()):
            raise RuntimeError("FP32 staging output comparison is nonfinite")
        delta = (lhs - rhs).abs()
        rows.append(
            {
                "shape": list(left.shape),
                "original_dtype": old["dtype"],
                "staged_dtype": new["dtype"],
                "comparison": "observed_floating",
                "max_abs": float(delta.max()) if delta.numel() else 0.0,
                "max_rel": float((delta / lhs.abs().clamp_min(1e-12)).max()) if delta.numel() else 0.0,
            }
        )
    return {
        "status": "observed",
        "scope": "original computation versus staged FP32; not accuracy equivalence",
        "original_input_abi": original_input_abi,
        "staged_input_abi": staged_input_abi,
        "original_output_abi": before_abi,
        "staged_output_abi": after_abi,
        "output_cardinality": len(rows),
        "output_metrics": rows,
    }


def _exported_integer_mm_count(module) -> int:
    """Count proven i8×i8→i32 contractions in the emitted, unnormalized MLIR."""
    from xdsl.dialects.builtin import IntegerType, TensorType

    def width(value) -> int | None:
        typ = getattr(value, "type", None)
        if not isinstance(typ, TensorType) or not isinstance(typ.element_type, IntegerType):
            return None
        return int(typ.element_type.width.data)

    if module is None:
        return 0
    count = 0
    for op in module.walk():
        if op.name != "linalg.generic" or len(op.operands) < 3 or len(op.results) != 1:
            continue
        prov = op.attributes.get("prov.op")
        if (
            str(getattr(prov, "data", "")) == "int_matmul"
            and [width(value) for value in op.operands[:2]] == [8, 8]
            and width(op.results[0]) == 32
        ):
            count += 1
    return count


def _write_leaf_constants(mdl, inputs, weights_path: str, dest: Path) -> str | None:
    """Write manifest-listed ``@forward`` buffers and lifted constants omitted from weights.

    Use m2m's bundle layout: ``buf::<dotted name>`` for buffers and the
    manifest name for constants. Missing leaves fail closed; another capture's
    constants cannot complete this capsule.
    """
    import numpy as np

    manifest_path = Path(weights_path + ".manifest.json")
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [meta or {} for meta in manifest.values()]
    # A buffer the externalizer STORED carries its `weight` key and is already in the weights file.
    buffers = {str(meta.get("name") or "") for meta in rows if meta.get("kind") == "buffer" and not meta.get("weight")}
    lifted = [str(meta.get("name") or "") for meta in rows if "lifted_tensor" in str(meta.get("name") or "")]
    if not buffers and not lifted:
        return None
    extra: dict = {}
    for name, tensor in mdl.named_buffers():
        if "b_" + name.replace(".", "_") in buffers:
            extra["buf::" + name] = tensor.detach().float().cpu().numpy()
    if lifted:
        from m2m.capture.bundle import _lifted_constants

        found: dict = {}
        _lifted_constants(mdl, tuple(inputs), found)
        extra.update({key: value for key, value in found.items() if key in lifted})
    recovered = {"b_" + key[len("buf::") :].replace(".", "_") for key in extra if key.startswith("buf::")}
    missing = sorted(buffers - recovered) + [name for name in lifted if name not in extra]
    if missing:
        # FAIL CLOSED: a partial file would be read as complete by a consumer that trusts it.
        raise RuntimeError(f"leaf arguments named by the manifest could not be recovered: {missing}")
    np.savez(dest, **extra)
    return str(dest)


# Preserve historical callable and record pickle lookup identities.
_load_loader.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_loader_dependency_sources.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_quant_for.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_to_native.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_scalars.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_loader_provenance.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_input_abi.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_calibration_stream_limit.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_freeze_calibration.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_float_reference.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_output_abi.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_integerized_agreement.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_fp32_stage_observation.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_exported_integer_mm_count.__module__ = "merlin.development.targetgen._m2m_capture_worker"
_write_leaf_constants.__module__ = "merlin.development.targetgen._m2m_capture_worker"
