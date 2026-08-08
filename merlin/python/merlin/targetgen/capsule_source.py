"""Capsule *sources* — where a capsule's program + golden come from.

Historically a capsule's input program was a hand-templated string in the ``merlin_iface`` dialect and
its golden came from one of four in-repo engines (int / specir-fp8 / mx / simt, in
``contract/capsules/generate_corpus.py``). This module adds the two *grounded* sources the redesign is
built on, keeping everything target-agnostic (no target-name literal, no regex):

* :class:`PytorchRefSource` — a capsule defined **in PyTorch** and lowered to **linalg-on-tensors** via
  ``model2MLIR`` (``m2m``), with the **host torch-eager** result as the reference golden. Because ``torch``
  lives only in the m2m venv, the actual conversion + eager run happen in a subprocess
  (:mod:`merlin.targetgen._m2m_capture_worker`) driven by that venv's python; this class builds the op's
  loader, invokes the worker, and ingests ``linalg.mlir`` + ``golden.json`` + ``inputs.json``.

* :class:`SpecRefSource` — a capsule whose numeric contract + spec-derived golden come from a verification
  **spec** op in ``specir`` (``/scratch2/agustin/mvp-lhwir/spec``). ``specir`` is pure xDSL (no torch) so it
  runs in-process. (Scaffolded here; the fp8-matmul spec path already lives in ``generate_corpus`` and is
  folded in as these are unified.)

The op → PyTorch module mapping is a small generic vocabulary (matmul/linear, attention QK + full, rmsnorm,
softmax, layernorm, geglu/swiglu, rope). Each op is plain ``torch`` so its loader imports cleanly in the m2m
venv. Precision is a parameter (``dtype``): the SAME token a target's ``compute_units`` declares.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# ------------------------------------------------------------------------------------------------
# m2m venv resolution (torch lives there, never in the merlin venv)
# ------------------------------------------------------------------------------------------------
def _m2m_dir() -> Path:
    return Path(os.environ.get("MERLIN_M2M_DIR", "/scratch/agustin/projects/model2MLIR"))


def _m2m_python() -> Path:
    env = os.environ.get("MERLIN_M2M_PYTHON")
    if env:
        return Path(env)
    return _m2m_dir() / ".venv" / "bin" / "python"


class M2MUnavailable(RuntimeError):
    """The m2m venv (torch) or the model2MLIR repo could not be located/run — fail closed, never fake."""


# ------------------------------------------------------------------------------------------------
# op -> PyTorch loader source (generic vocabulary; plain torch, deterministic inputs)
# ------------------------------------------------------------------------------------------------
_PREAMBLE = '''"""Auto-generated capsule loader ({op}, {dtype}). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = {seed}
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0
'''

# Each op body defines `Model(nn.Module)` and `get_model_and_inputs()`. Shapes come from the spec dict.
_OP_BODIES: dict[str, str] = {
    "matmul": '''
class Model(nn.Module):
    def forward(self, a, w):
        return a @ w
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}))
''',
    "linear": '''
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear({K}, {N}, bias={bias})
    def forward(self, a):
        return self.fc(a)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "attention_qk": '''
class Model(nn.Module):
    def forward(self, q, k):
        return q @ k.transpose(-2, -1)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({N}, {K}))
''',
    "attention_full": '''
class Model(nn.Module):
    def forward(self, q, k, v):
        s = (q @ k.transpose(-2, -1)) / math.sqrt({K})
        if {causal}:
            t = q.shape[-2]
            neg = torch.finfo(s.dtype).min
            mask = torch.triu(torch.full((t, t), neg, dtype=s.dtype), 1)
            s = s + mask
        return s.softmax(-1) @ v
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({N}, {K}), _r({N}, {Dv}))
''',
    "rmsnorm": '''
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.g = nn.Parameter(_r({K}) * 0.5 + 1.0); self.eps = {eps}
    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.g
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "softmax": '''
class Model(nn.Module):
    def forward(self, x):
        return x.softmax(-1)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "layernorm": '''
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.ln = nn.LayerNorm({K}, eps={eps})
    def forward(self, x):
        return self.ln(x)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "geglu": '''
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear({K}, {N}, bias=False); self.up = nn.Linear({K}, {N}, bias=False)
    def forward(self, x):
        return torch.nn.functional.silu(self.gate(x)) * self.up(x)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "rope": '''
def _rope(x, half):
    freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(x.shape[-2], dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1); sin = torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], -1) * sin
class Model(nn.Module):
    def forward(self, x):
        return _rope(x, {K} // 2)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
}


def supported_ops() -> list[str]:
    return sorted(_OP_BODIES)


def build_loader_src(spec: dict) -> str:
    """Render the PyTorch loader source for an op spec. ``spec`` carries ``op`` + the shape fields the op
    needs (M/K/N/Dv) + optional ``eps``/``causal``/``bias`` + ``seed``/``dtype``. Fail closed on an
    unknown op (never silently emit a wrong program)."""
    op = spec["op"]
    if op not in _OP_BODIES:
        raise KeyError(f"capsule_source has no PyTorch template for op {op!r} (have {supported_ops()})")
    fields = {
        "op": op, "dtype": spec.get("dtype", "fp32"), "seed": int(spec.get("seed", 0)),
        "M": spec.get("M", 16), "K": spec.get("K", 16), "N": spec.get("N", 16),
        "Dv": spec.get("Dv", spec.get("K", 16)), "eps": spec.get("eps", 1e-5),
        "causal": bool(spec.get("causal", False)), "bias": bool(spec.get("bias", False)),
    }
    return _PREAMBLE.format(**fields) + _OP_BODIES[op].format(**fields)


# ------------------------------------------------------------------------------------------------
# artifacts
# ------------------------------------------------------------------------------------------------
@dataclass
class CapsuleArtifacts:
    """Everything a grounded capsule needs. ``pytorch_src`` + ``linalg_mlir`` are agent-VISIBLE (realistic
    lowering context); ``golden`` + ``weights_path`` are the masked answer surfaces."""
    op: str
    dtype: str
    pytorch_src: str
    linalg_mlir: str
    inputs: list
    golden: list
    weights_path: str
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------
# PyTorch source (m2m)
# ------------------------------------------------------------------------------------------------
class PytorchRefSource:
    """Capture a capsule from a PyTorch op via model2MLIR (subprocess, m2m venv)."""

    def __init__(self, m2m_dir: Path | None = None, python: Path | None = None, timeout: int = 900):
        self.m2m_dir = Path(m2m_dir) if m2m_dir else _m2m_dir()
        self.python = Path(python) if python else _m2m_python()
        self.timeout = timeout

    def available(self) -> bool:
        return self.python.exists() and (self.m2m_dir / "m2m" / "__init__.py").exists()

    def capture(self, spec: dict, *, workdir: str | Path | None = None) -> CapsuleArtifacts:
        if not self.available():
            raise M2MUnavailable(
                f"m2m venv python {self.python} or repo {self.m2m_dir} missing; set MERLIN_M2M_PYTHON / "
                f"MERLIN_M2M_DIR (torch is required and lives only in the m2m venv)")
        dtype = spec.get("dtype", "fp32")
        loader_src = build_loader_src(spec)
        tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="capsule_m2m_"))
        tmp.mkdir(parents=True, exist_ok=True)
        loader_py = tmp / "loader.py"
        loader_py.write_text(loader_src, encoding="utf-8")
        worker = Path(__file__).with_name("_m2m_capture_worker.py")

        env = dict(os.environ)
        env["MERLIN_M2M_DIR"] = str(self.m2m_dir)
        cmd = [str(self.python), str(worker), "--loader", str(loader_py),
               "--dtype", dtype, "--out", str(tmp), "--m2m-dir", str(self.m2m_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env)
        meta_p = tmp / "meta.json"
        if proc.returncode != 0 or not meta_p.exists():
            raise M2MUnavailable(
                f"m2m capture failed (rc={proc.returncode}) for op {spec.get('op')!r}/{dtype}:\n"
                f"{proc.stderr[-1500:]}")
        meta = json.loads(meta_p.read_text())
        if not meta.get("ok") or meta.get("opaque", -1) != 0:
            raise M2MUnavailable(
                f"m2m capture produced a non-clean program for op {spec.get('op')!r}/{dtype} "
                f"(ok={meta.get('ok')}, opaque={meta.get('opaque')}); a capsule input must be 0-opaque")
        return CapsuleArtifacts(
            op=spec["op"], dtype=dtype,
            pytorch_src=loader_src,
            linalg_mlir=(tmp / "linalg.mlir").read_text(encoding="utf-8"),
            inputs=json.loads((tmp / "inputs.json").read_text()),
            golden=json.loads((tmp / "golden.json").read_text()),
            weights_path=meta.get("weights", str(tmp / "weights.safetensors")),
            meta=meta)


# ------------------------------------------------------------------------------------------------
# Spec source (specir) — in-process; scaffolded (fp8-matmul spec path currently in generate_corpus)
# ------------------------------------------------------------------------------------------------
def _specir_root() -> str:
    return os.environ.get("SPECIR_ROOT", "/scratch2/agustin/mvp-lhwir/spec")


class SpecRefSource:
    """Capture an op's numeric contract + spec-derived golden from a ``specir`` verification spec.

    ``specir`` imports in-process (pure xDSL). This is the seam for pulling arbitrary spec ops
    (beyond the fp8 matmul the ``generate_corpus`` refmodel path already covers) plus their
    ``coverage_goal``. Left as a typed scaffold until the per-op ``specc testbench`` contract is folded in.
    """

    def __init__(self, root: str | None = None):
        self.root = root or _specir_root()

    def available(self) -> bool:
        return (Path(self.root) / "specir" / "__init__.py").exists()

    def capture(self, spec_ref: str, spec: dict):  # pragma: no cover - scaffold
        raise NotImplementedError(
            "SpecRefSource.capture is scaffolded; the fp8-matmul spec golden currently lives in "
            "generate_corpus._float_golden and is folded in as the sources unify")
