"""Small, target-agnostic gap-filler programs for exact Core ATen provenance.

These are ordinary PyTorch modules passed through the same model2MLIR capture path as the existing
capsules.  They are grouped only when a clean capture retains every requested overload.  A requested
operator that decomposes away, becomes opaque, or otherwise disappears is a failed sample and is never
credited as covered.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from merlin.targetgen.capsule_source import M2MUnavailable, PytorchRefSource
from merlin.targetgen.core_aten_cover import _provenance_ops

_ELEMENTWISE = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x, y):
        positive = torch.ops.aten.add.Scalar(torch.ops.aten.abs.default(x), 1.25)
        return (
            torch.ops.aten.abs.default(x),
            torch.ops.aten.add.Scalar(x, 0.75, 1),
            torch.ops.aten.ceil.default(x),
            torch.ops.aten.div.Scalar(x, 2.0),
            torch.ops.aten.eq.Scalar(x, 0.25),
            torch.ops.aten.floor.default(x),
            torch.ops.aten.ge.Tensor(x, y),
            torch.ops.aten.gt.Scalar(x, 0.0),
            torch.ops.aten.gt.Tensor(x, y),
            torch.ops.aten.le.Scalar(x, 0.5),
            torch.ops.aten.log.default(positive),
            torch.ops.aten.logical_not.default(torch.ops.aten.gt.Scalar(x, 0.0)),
            torch.ops.aten.lt.Tensor(x, y),
            torch.ops.aten.mul.Scalar(x, 1.5),
            torch.ops.aten.ne.Tensor(x, y),
            torch.ops.aten.pow.Tensor_Tensor(positive, torch.ops.aten.abs.default(y)),
            torch.ops.aten.relu.default(x),
            torch.ops.aten.sqrt.default(positive),
            torch.ops.aten.sub.Scalar(x, 0.5, 2),
        )

def get_model_and_inputs():
    x = torch.tensor([[-0.75, -0.25, 0.25], [0.5, 0.75, 1.0]])
    y = torch.tensor([[-0.5, 0.1, 0.5], [0.25, 1.0, 1.5]])
    return Model(), (x, y)
"""

_REDUCTIONS = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x, mask):
        max_values, max_indices = torch.ops.aten.max.dim(x, 1, False)
        return (
            torch.ops.aten.any.default(mask),
            torch.ops.aten.any.dim(mask, 1, True),
            torch.ops.aten.any.dims(mask, [0, 1], False),
            torch.ops.aten.argmax.default(x, 1, False),
            torch.ops.aten.argmin.default(x, 0, True),
            max_values,
            max_indices,
            torch.ops.aten.mean.default(x),
            torch.ops.aten.repeat.default(x, [2, 1]),
            torch.ops.aten.squeeze.dims(x.reshape(1, 2, 3, 1), [0, 3]),
        )

def get_model_and_inputs():
    x = torch.tensor([[0.5, -0.25, 1.0], [2.0, 0.75, -1.5]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    return Model(), (x, mask)
"""

_SPATIAL = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x):
        pooled, indices = torch.ops.aten.max_pool2d_with_indices.default(
            x, [2, 2], [2, 2], [0, 0], [1, 1], False
        )
        return pooled, indices, torch.ops.aten.upsample_nearest2d.vec(x, [8, 8], None)

def get_model_and_inputs():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    return Model(), (x,)
"""

_BATCH_NORM = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x, weight, bias, running_mean, running_var):
        return torch.ops.aten._native_batch_norm_legit_no_training.default(
            x, weight, bias, running_mean, running_var, 0.1, 1e-5
        )

def get_model_and_inputs():
    x = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2) / 16.0
    return Model(), (x, torch.ones(4), torch.zeros(4), torch.zeros(4), torch.ones(4))
"""

_TRANSCENDENTAL = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x, y, positive):
        return (
            torch.ops.aten.acos.default(x),
            torch.ops.aten.acosh.default(positive),
            torch.ops.aten.asin.default(x),
            torch.ops.aten.asinh.default(x),
            torch.ops.aten.atan.default(x),
            torch.ops.aten.atan2.default(x, y),
            torch.ops.aten.atanh.default(x),
            torch.ops.aten.cosh.default(x),
            torch.ops.aten.elu.default(x, 1.0, 1.0, 1.0),
            torch.ops.aten.erf.default(x),
            torch.ops.aten.expm1.default(x),
            torch.ops.aten.isnan.default(x),
            torch.ops.aten.log10.default(positive),
            torch.ops.aten.log1p.default(x),
            torch.ops.aten.log2.default(positive),
            torch.ops.aten.sign.default(x),
            torch.ops.aten.sinh.default(x),
            torch.ops.aten.tan.default(x),
            torch.ops.aten.trunc.default(x),
        )

def get_model_and_inputs():
    x = torch.tensor([[-0.75, -0.35, 0.05], [0.45, 0.70, 0.75]])
    y = torch.tensor([[0.30, -0.10, -0.50], [1.50, 1.10, 0.70]])
    positive = torch.tensor([[1.25, 1.10, 1.50], [1.75, 2.00, 2.25]])
    return Model(), (x, y, positive)
"""

_INTEGER_LOGIC_INDEX = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, lhs, rhs, x, lower, upper, index, mask, other_mask, source):
        return (
            torch.ops.aten.bitwise_and.Scalar(lhs, 3),
            torch.ops.aten.bitwise_or.Scalar(lhs, 3),
            torch.ops.aten.bitwise_or.Tensor(lhs, rhs),
            torch.ops.aten.bitwise_xor.Scalar(lhs, 3),
            torch.ops.aten.bitwise_xor.Tensor(lhs, rhs),
            torch.ops.aten.clamp.Tensor(x, lower, upper),
            torch.ops.aten.div.Scalar_mode(lhs, 2, rounding_mode="floor"),
            torch.ops.aten.div.Tensor_mode(lhs, rhs, rounding_mode="floor"),
            torch.ops.aten.flip.default(x, [1]),
            torch.ops.aten.gather.default(x, 1, index, sparse_grad=False),
            torch.ops.aten.logical_and.default(mask, other_mask),
            torch.ops.aten.logical_or.default(mask, other_mask),
            torch.ops.aten.logical_xor.default(mask, other_mask),
            torch.ops.aten.masked_scatter.default(x, mask, source),
        )

def get_model_and_inputs():
    lhs = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    rhs = torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.int64)
    x = torch.tensor([[-0.75, -0.35, 0.05], [0.45, 0.85, 1.25]])
    lower = torch.full((2, 3), -0.25)
    upper = torch.full((2, 3), 0.75)
    index = torch.tensor([[2, 1], [0, 2]], dtype=torch.int64)
    mask = torch.tensor([[False, True, False], [True, False, True]])
    other_mask = torch.tensor([[True, True, False], [False, True, True]])
    source = torch.arange(6, dtype=torch.float32)
    return Model(), (lhs, rhs, x, lower, upper, index, mask, other_mask, source)
"""

_PRODUCT_REDUCTIONS = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x):
        return (
            torch.ops.aten.prod.default(x),
            torch.ops.aten.prod.dim_int(x, 1, True),
        )

def get_model_and_inputs():
    x = torch.tensor([[1.25, 0.85, 0.55], [0.95, 1.35, 1.75]])
    return Model(), (x,)
"""

_UPSAMPLE_NEAREST = """import torch
from torch import nn

class Model(nn.Module):
    def forward(self, x):
        return torch.ops.aten.upsample_nearest2d.vec(x, [6, 6], None)

def get_model_and_inputs():
    x = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    return Model(), (x,)
"""

SAMPLE_LOADERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "elementwise": (
        _ELEMENTWISE,
        (
            "aten.abs.default",
            "aten.add.Scalar",
            "aten.ceil.default",
            "aten.div.Scalar",
            "aten.eq.Scalar",
            "aten.floor.default",
            "aten.ge.Tensor",
            "aten.gt.Scalar",
            "aten.gt.Tensor",
            "aten.le.Scalar",
            "aten.log.default",
            "aten.logical_not.default",
            "aten.lt.Tensor",
            "aten.mul.Scalar",
            "aten.ne.Tensor",
            "aten.pow.Tensor_Tensor",
            "aten.relu.default",
            "aten.sqrt.default",
            "aten.sub.Scalar",
        ),
    ),
    "reductions": (
        _REDUCTIONS,
        (
            "aten.any.default",
            "aten.any.dim",
            "aten.any.dims",
            "aten.argmax.default",
            "aten.argmin.default",
            "aten.max.dim",
            "aten.mean.default",
            "aten.repeat.default",
            "aten.squeeze.dims",
        ),
    ),
    "spatial": (
        _SPATIAL,
        ("aten.max_pool2d_with_indices.default", "aten.upsample_nearest2d.vec"),
    ),
    "batch_norm_inference": (
        _BATCH_NORM,
        ("aten._native_batch_norm_legit_no_training.default",),
    ),
    "integer_logic_index": (
        _INTEGER_LOGIC_INDEX,
        (
            "aten.bitwise_and.Scalar",
            "aten.bitwise_or.Scalar",
            "aten.bitwise_or.Tensor",
            "aten.bitwise_xor.Scalar",
            "aten.bitwise_xor.Tensor",
            "aten.clamp.Tensor",
            "aten.div.Scalar_mode",
            "aten.div.Tensor_mode",
            "aten.flip.default",
            "aten.gather.default",
            "aten.logical_and.default",
            "aten.logical_or.default",
            "aten.logical_xor.default",
            "aten.masked_scatter.default",
        ),
    ),
    "product_reductions": (
        _PRODUCT_REDUCTIONS,
        ("aten.prod.default", "aten.prod.dim_int"),
    ),
    "transcendental": (
        _TRANSCENDENTAL,
        (
            "aten.acos.default",
            "aten.acosh.default",
            "aten.asin.default",
            "aten.asinh.default",
            "aten.atan.default",
            "aten.atan2.default",
            "aten.atanh.default",
            "aten.cosh.default",
            "aten.elu.default",
            "aten.erf.default",
            "aten.expm1.default",
            "aten.isnan.default",
            "aten.log10.default",
            "aten.log1p.default",
            "aten.log2.default",
            "aten.sign.default",
            "aten.sinh.default",
            "aten.tan.default",
            "aten.trunc.default",
        ),
    ),
    "upsample_nearest2d": (
        _UPSAMPLE_NEAREST,
        ("aten.upsample_nearest2d.vec",),
    ),
}

_OPAQUE_IN_MODEL2MLIR = {
    "aten._adaptive_avg_pool2d.default",
    "aten._adaptive_avg_pool3d.default",
    "aten._cdist_forward.default",
    "aten._embedding_bag.default",
    "aten._log_softmax.default",
    "aten._native_batch_norm_legit.no_stats",
    "aten._pdist_forward.default",
    "aten.acos.default",
    "aten.acosh.default",
    "aten.adaptive_avg_pool1d.default",
    "aten.as_strided.default",
    "aten.asin.default",
    "aten.asinh.default",
    "aten.atan.default",
    "aten.atan2.default",
    "aten.atanh.default",
    "aten.avg_pool1d.default",
    "aten.avg_pool2d.default",
    "aten.avg_pool3d.default",
    "aten.bitwise_and.Scalar",
    "aten.bitwise_or.Scalar",
    "aten.bitwise_or.Tensor",
    "aten.bitwise_xor.Scalar",
    "aten.bitwise_xor.Tensor",
    "aten.clamp.Tensor",
    "aten.cosh.default",
    "aten.diagonal.default",
    "aten.div.Scalar_mode",
    "aten.div.Tensor_mode",
    "aten.elu.default",
    "aten.erf.default",
    "aten.expm1.default",
    "aten.flip.default",
    "aten.fmod.Scalar",
    "aten.fmod.Tensor",
    "aten.gather.default",
    "aten.grid_sampler_2d.default",
    "aten.hardtanh.default",
    "aten.index_select.default",
    "aten.isinf.default",
    "aten.isnan.default",
    "aten.leaky_relu.default",
    "aten.log10.default",
    "aten.log1p.default",
    "aten.log2.default",
    "aten.logical_and.default",
    "aten.logical_or.default",
    "aten.logical_xor.default",
    "aten.masked_scatter.default",
    "aten.max_pool3d_with_indices.default",
    "aten.native_dropout.default",
    "aten.native_group_norm.default",
    "aten.nonzero.default",
    "aten.prod.default",
    "aten.prod.dim_int",
    "aten.reflection_pad1d.default",
    "aten.reflection_pad2d.default",
    "aten.reflection_pad3d.default",
    "aten.remainder.Scalar",
    "aten.remainder.Tensor",
    "aten.replication_pad2d.default",
    "aten.replication_pad3d.default",
    "aten.scatter.src",
    "aten.scatter.value",
    "aten.scatter_add.default",
    "aten.scatter_reduce.two",
    "aten.sign.default",
    "aten.sinh.default",
    "aten.sort.default",
    "aten.tan.default",
    "aten.topk.default",
    "aten.trunc.default",
    "aten.var.correction",
    "aten.var.dim",
}

_LOST_DURING_CAPTURE = {
    "aten._local_scalar_dense.default",
    "aten._native_batch_norm_legit.default",
    "aten.alias.default",
    "aten.atan2.out",
    "aten.clone.default",
    "aten.empty.memory_format",
    "aten.empty_strided.default",
    "aten.fill.Scalar",
    "aten.permute.default",
    "aten.resize_.default",
    "aten.select_scatter.default",
    "aten.sym_is_contiguous.default",
    "aten.sym_numel.default",
    "aten.sym_size.int",
    "aten.sym_storage_offset.default",
    "aten.sym_stride.int",
}

_NONDETERMINISTIC = {"aten.rand.default", "aten.randn.default", "aten.randperm.default"}


def gap_reason(overload: str) -> str:
    """Concrete reason established by the bundled probe campaign for an uncovered overload."""

    if overload in _OPAQUE_IN_MODEL2MLIR:
        return "valid grouped invocation captured as an unsupported opaque op; no clean linalg capsule"
    if overload in _LOST_DURING_CAPTURE:
        return "valid invocation was eliminated, functionalized, or constant-folded; exact prov.aten did not survive"
    if overload in _NONDETERMINISTIC:
        return "no deterministic RNG-state contract for byte-stable capture and golden replay"
    if "backward" in overload:
        return "no successful clean backward-operator sample in the current capture/importer pipeline"
    return "no successful clean sample in the generated candidate pool"


def _git_revision(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def capture_gap_fillers(
    output_root: str | Path,
    *,
    m2m_dir: str | Path | None = None,
    python: str | Path | None = None,
) -> dict:
    """Capture every bundled gap filler and retain only provenance-verified successes."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source = PytorchRefSource(
        m2m_dir=Path(m2m_dir) if m2m_dir else None,
        python=Path(python) if python else None,
    )
    report: dict[str, dict] = {}
    for name, (loader_source, requested_tuple) in sorted(SAMPLE_LOADERS.items()):
        requested = set(requested_tuple)
        destination = root / name
        destination.mkdir(parents=True, exist_ok=True)
        loader = destination / "capsule.pytorch.py"
        loader.write_text(loader_source, encoding="utf-8")
        try:
            artifact = source.capture_loader(loader, "fp32", workdir=destination / "capture-work")
            mlir = destination / "capsule.linalg.mlir"
            mlir.write_text(artifact.linalg_mlir, encoding="utf-8")
            actual = _provenance_ops(mlir)
            missing = sorted(requested - actual)
            if missing:
                raise M2MUnavailable(f"capture lost requested exact overloads: {missing}")
            (destination / "inputs.json").write_text(json.dumps(artifact.inputs, indent=2) + "\n", encoding="utf-8")
            (destination / "golden.json").write_text(json.dumps(artifact.golden, indent=2) + "\n", encoding="utf-8")
            record = {
                "schema_version": 1,
                "status": "captured",
                "name": name,
                "pytorch_version": artifact.meta.get("torch_version"),
                "model2mlir_revision": _git_revision(source.m2m_dir),
                "requested_overloads": sorted(requested),
                "captured_overloads": sorted(actual),
                "capture_meta": artifact.meta,
            }
            (destination / "capture.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            report[name] = {"status": "captured", "operators": sorted(actual)}
        except Exception as exc:  # noqa: BLE001 -- each failed sample is evidence, not a batch abort
            # A recapture must never inherit a prior successful answer surface.  Keep the loader and
            # capture work directory for diagnosis, but remove every file that would make the matrix
            # treat this failed attempt as a provenance-bearing candidate.
            for stale_name in ("capsule.linalg.mlir", "inputs.json", "golden.json", "capture.json"):
                (destination / stale_name).unlink(missing_ok=True)
            report[name] = {"status": "capture_failed", "reason": f"{type(exc).__name__}: {exc}"}
    document = {
        "schema_version": 1,
        "model2mlir_root": str(source.m2m_dir.resolve()),
        "model2mlir_revision": _git_revision(source.m2m_dir),
        "python": str(source.python.resolve()),
        "samples": report,
    }
    (root / "gap_filler_capture_report.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document
