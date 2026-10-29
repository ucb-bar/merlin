"""Capture API inspection, deterministic seeding and recorded model forwards."""
from __future__ import annotations
import hashlib
import importlib.util
import inspect
import json
import random
import shutil
from pathlib import Path
from ._m2m_capture_artifacts import _framework_cpu_dispatch

def _capture_api_report(m2m) -> dict[str, list[str]]:
    """Report selected runtime API gaps; a signature check is not capture qualification."""
    from m2m.capture.bundle import write_bundle

    bundle_args = set(inspect.signature(write_bundle).parameters)
    convert_args = set(inspect.signature(m2m.convert).parameters)
    same_conversion_missing = [
        f"m2m/capture/bundle.py:write_bundle({name})"
        for name in sorted({"source_path", "capture_trace", "conversion_result"} - bundle_args)
    ]
    if importlib.util.find_spec("m2m.capture.provenance") is None:
        same_conversion_missing.append("m2m/capture/provenance.py")
    frontend_trace_missing = [
        f"m2m/api.py:convert({name})" for name in sorted({"capture_trace", "original_frontend_snapshot"} - convert_args)
    ]
    if importlib.util.find_spec("m2m.capture.trace") is None:
        frontend_trace_missing.append("m2m/capture/trace.py")
    static_integerization_missing = [
        member
        for module, member in (
            ("m2m.capture.pt2e_integerize", "m2m/capture/pt2e_integerize.py"),
            ("m2m.capture.pt2e_integer_reference", "m2m/capture/pt2e_integer_reference.py"),
        )
        if importlib.util.find_spec(module) is None
    ]
    return {
        "same_conversion_missing": same_conversion_missing,
        "frontend_trace_missing": frontend_trace_missing,
        "static_integerization_missing": static_integerization_missing,
    }


def _diagnostic_model_copy(out: Path, loader: Path, *, capture_api: dict[str, list[str]]) -> None:
    """Expose the exact converted MLIR for inventory, never as an admitted bundle.

    This deliberately omits the capture receipt and runtime input/golden ABI. An
    older Model2MLIR may have converted the model but cannot provide the modern
    same-conversion materialization contract.
    """

    def digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if (out / "capture_receipt.json").exists() or (out / "model.mlir").exists():
        raise ValueError("diagnostic copy cannot reuse a materialized capture directory")
    shutil.copyfile(out / "linalg.mlir", out / "model.mlir")
    names = (
        "model.mlir",
        "linalg.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
        "inputs.json",
        "golden.json",
        "frontend-trace.json",
        "pytorch-opset.json",
        "meta.json",
    )
    artifacts = {name: {"bytes": (out / name).stat().st_size, "sha256": digest(out / name)} for name in names}
    record = {
        "schema": "merlin.diagnostic_m2m_model.v1",
        "status": "diagnostic_raw_conversion",
        "phase0_admission": "not_granted",
        "source_closure_verified": False,
        "materialized_abi": False,
        "reason": "raw conversion has no same-conversion Model2MLIR bundle or producer capture receipt",
        "capture_api": capture_api,
        "loader": {"path": str(loader.absolute()), "sha256": digest(loader)},
        "artifacts": artifacts,
    }
    (out / "diagnostic-capture.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


def _seed_capture(seed: int, torch, threads: int = 1) -> dict:
    """Seed loader imports as well as construction; refuse nondeterministic kernels.

    This binds the numerical capture policy, not reproducibility across different
    framework builds or unrecorded external data sources. A loader must still
    declare its data/source ownership separately.
    """
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Parallel CPU reductions split work by thread count, so reference bytes depend on it: a
    # capture runs at its selected count (one unless selected), records it, and is replayed at it.
    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise RuntimeError(f"the capture selected {threads} CPU threads but runs {torch.get_num_threads()}")
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "rngs": ["python", "numpy", "torch"],
        "deterministic_algorithms": "required",
        "cpu_threads": torch.get_num_threads(),
        "cpu_dispatch": _framework_cpu_dispatch(torch),
        "scope": "selected framework build and inputs",
    }


class _RecordedForwards:
    """The floating model's own outputs on calibration samples it already ran, for the agreement.

    A causal decode stream is the floating recurrence itself: it calls the stage on each
    ``(token, kv_cache, position)`` it yields, and the agreement receipt then calls the same stage
    on the same samples again. While the stream is read, a forward hook keeps the first ``limit``
    outputs with the exact argument tensors. :meth:`__call__` returns a kept output only for the
    same argument objects at the same version counters, on the same module whose parameters,
    buffers and train/eval flags are unchanged; any other call runs the module. The hook is removed
    before export, so the exported graph never sees it.
    """

    def __init__(self, module, *, limit: int) -> None:
        self.module, self.limit, self.kept = module, limit, []
        self._handle = None
        self._state = None

    @staticmethod
    def _tensors(values):
        import torch

        return [leaf for leaf in torch.utils._pytree.tree_flatten(values)[0] if isinstance(leaf, torch.Tensor)]

    def _fingerprint(self):
        import random

        import numpy as np
        import torch

        tensors = [*self.module.named_parameters(), *self.module.named_buffers()]
        numpy_state = np.random.get_state()
        return (
            tuple((name, id(tensor), tensor._version) for name, tensor in tensors),
            tuple(sub.training for sub in self.module.modules()),
            torch.random.get_rng_state().numpy().tobytes(),
            random.getstate(),
            (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        )

    def _record(self, _module, args, kwargs, output) -> None:
        import torch

        # Nested mutable controls and non-tensor inputs lack version counters.
        if len(self.kept) < self.limit and not kwargs and all(isinstance(arg, torch.Tensor) for arg in args):
            leaves, specification = torch.utils._pytree.tree_flatten(output)
            if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
                self.kept.append(
                    (args, [leaf._version for leaf in args], output, [leaf._version for leaf in leaves], specification)
                )

    def __enter__(self) -> _RecordedForwards:
        self._state = self._fingerprint()
        self._handle = self.module.register_forward_hook(self._record, with_kwargs=True)
        return self

    def __exit__(self, *exc) -> None:
        self._handle.remove()

    def __call__(self, *args):
        import torch

        if self._fingerprint() == self._state:
            leaves = self._tensors(args)
            for kept_args, versions, output, output_versions, output_spec in self.kept:
                kept_leaves = self._tensors(kept_args)
                current_output, specification = torch.utils._pytree.tree_flatten(output)
                if (
                    len(kept_args) == len(args)
                    and len(kept_leaves) == len(leaves)
                    and all(a is b for a, b in zip(kept_args, args, strict=True))
                    and all(a is b for a, b in zip(kept_leaves, leaves, strict=True))
                    and [leaf._version for leaf in leaves] == versions
                    and specification == output_spec
                    and all(isinstance(leaf, torch.Tensor) for leaf in current_output)
                    and [leaf._version for leaf in current_output] == output_versions
                ):
                    return output
        return self.module(*args)
