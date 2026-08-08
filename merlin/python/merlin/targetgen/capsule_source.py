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
    def forward(self, a, w):
        return a @ w
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}))
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
    def forward(self, x, g):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + {eps}) * g
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r(1, {K}) * 0.5 + 1.0)
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
    def forward(self, x, w, b):
        return torch.nn.functional.layer_norm(x, ({K},), w, b, {eps})
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}) * 0.5 + 1.0, _r({K}) * 0.1)
''',
    "geglu": '''
class Model(nn.Module):
    def forward(self, x, wg, wu):
        return torch.nn.functional.silu(x @ wg) * (x @ wu)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}), _r({K}, {N}))
''',
    "gelu": '''
class Model(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "silu": '''
class Model(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "add": '''
class Model(nn.Module):
    def forward(self, a, b):
        return a + b
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({M}, {K}))
''',
    "reduce_sum": '''
class Model(nn.Module):
    def forward(self, x):
        return x.sum(-1, keepdim=True)
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
    # --- elementwise / composite ops from real model graphs (all FUNCTIONAL, operands as inputs) ---
    "bias_add": '''
class Model(nn.Module):
    def forward(self, x, b):
        return x + b
def get_model_and_inputs():
    return Model(), (_r({M}, {N}), _r({N}))
''',
    "fused_matmul_bias": '''
class Model(nn.Module):
    def forward(self, x, w, b):
        return x @ w + b
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}), _r({N}))
''',
    "k_chain": '''
class Model(nn.Module):
    def forward(self, a, w1, w2):
        return (a @ w1) @ w2
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}), _r({N}, {N}))
''',
    "logit_softcap": '''
class Model(nn.Module):
    def forward(self, x):
        c = {cap}
        return c * torch.tanh(x / c)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "embed_scale": '''
class Model(nn.Module):
    def forward(self, x):
        return x * ({K} ** 0.5)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
''',
    "gemma_4norm": '''
def _rms(x, g, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * g
class Model(nn.Module):
    def forward(self, x, g1, g2):
        return _rms(_rms(x, g1, {eps}), g2, {eps})
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r(1, {K}) * 0.5 + 1.0, _r(1, {K}) * 0.5 + 1.0)
''',
    "gemv_batched": '''
class Model(nn.Module):
    def forward(self, a, v):
        return torch.bmm(a, v)
def get_model_and_inputs():
    return Model(), (_r({B}, {M}, {K}), _r({B}, {K}, 1))
''',
    "patch_embed": '''
class Model(nn.Module):
    def forward(self, x, w):
        return torch.nn.functional.conv2d(x, w, stride={P})
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}), _r({N}, {Cin}, {P}, {P}))
''',
    # --- conv/pool family that runs on the vector lanes (fused, host-eager golden) ---
    "depthwise_conv2d": '''
class Model(nn.Module):
    def forward(self, x, w):
        return torch.nn.functional.conv2d(x, w, stride=1, padding={P} // 2, groups={Cin})
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}), _r({Cin}, 1, {P}, {P}))
''',
    # pooling decomposed into reshape+reduce so it lowers to pure linalg (torch-mlir leaves the fused
    # aten pool ops opaque); kernel == stride == P, exact same numerics as F.{max,avg}_pool2d.
    "maxpool2d": '''
class Model(nn.Module):
    def forward(self, x):
        return x.reshape(1, {Cin}, {Himg} // {P}, {P}, {Wimg} // {P}, {P}).amax((3, 5))
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}),)
''',
    "avgpool2d": '''
class Model(nn.Module):
    def forward(self, x):
        return x.reshape(1, {Cin}, {Himg} // {P}, {P}, {Wimg} // {P}, {P}).mean((3, 5))
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}),)
''',
    "global_average": '''
class Model(nn.Module):
    def forward(self, x):
        return x.mean((-2, -1))
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}),)
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
        # extra shape/scalar fields for the composite ops (batch, soft-cap, conv geometry)
        "B": spec.get("B", 2), "cap": spec.get("cap", 50.0),
        "Cin": spec.get("Cin", 1), "Himg": spec.get("Himg", 8), "Wimg": spec.get("Wimg", 8),
        "P": spec.get("P", 2),
    }
    return _PREAMBLE.format(**fields) + _OP_BODIES[op].format(**fields)


# ------------------------------------------------------------------------------------------------
# artifacts
# ------------------------------------------------------------------------------------------------
@dataclass
class CapsuleArtifacts:
    """Everything a grounded capsule needs. ``pytorch_src`` + ``linalg_mlir`` are agent-VISIBLE (realistic
    lowering context). The masked answer surface is ``golden`` (the host-eager OUTPUT, written to
    ``golden.yaml`` and hidden by the sandbox's ``golden.*`` glob). ``weights_path`` is NOT an answer: for a
    whole-model capsule the externalized weights are a legitimate compile INPUT the linalg references, so they
    ship VISIBLE alongside the interface. Only ``kind == model`` capsules ever carry a ``.safetensors`` (op
    capsules write none), so a weight file can never leak an op's answer."""
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
        """Capture a generated op loader (matmul/attention/... rendered from ``spec``)."""
        if not self.available():
            raise M2MUnavailable(
                f"m2m venv python {self.python} or repo {self.m2m_dir} missing; set MERLIN_M2M_PYTHON / "
                f"MERLIN_M2M_DIR (torch is required and lives only in the m2m venv)")
        dtype = spec.get("dtype", "fp32")
        loader_src = build_loader_src(spec)
        tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="capsule_m2m_"))
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "loader.py").write_text(loader_src, encoding="utf-8")
        return self._run(tmp / "loader.py", spec["op"], dtype, workdir=tmp, src=loader_src)

    def capture_loader(self, loader_py: str | Path, dtype: str, *,
                       workdir: str | Path | None = None) -> CapsuleArtifacts:
        """Capture an EXISTING loader file (a whole-model workload) rather than a generated op loader.
        Same worker path; ``op`` is ``model`` and ``pytorch_src`` is the loader's own source."""
        loader_py = Path(loader_py)
        if not self.available():
            raise M2MUnavailable(f"m2m venv python {self.python} or repo {self.m2m_dir} missing")
        if not loader_py.is_file():
            raise M2MUnavailable(f"model loader not found: {loader_py}")
        tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="capsule_model_"))
        tmp.mkdir(parents=True, exist_ok=True)
        return self._run(loader_py, "model", dtype, workdir=tmp, src=loader_py.read_text())

    def _run(self, loader_py: Path, op: str, dtype: str, *, workdir: Path, src: str) -> CapsuleArtifacts:
        worker = Path(__file__).with_name("_m2m_capture_worker.py")
        env = dict(os.environ)
        env["MERLIN_M2M_DIR"] = str(self.m2m_dir)
        cmd = [str(self.python), str(worker), "--loader", str(loader_py),
               "--dtype", dtype, "--out", str(workdir), "--m2m-dir", str(self.m2m_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env)
        meta_p = workdir / "meta.json"
        if proc.returncode != 0 or not meta_p.exists():
            raise M2MUnavailable(f"m2m capture failed (rc={proc.returncode}) for op {op!r}/{dtype}:\n"
                                 f"{proc.stderr[-1500:]}")
        meta = json.loads(meta_p.read_text())
        if not meta.get("ok") or meta.get("opaque", -1) != 0:
            raise M2MUnavailable(
                f"m2m capture produced a non-clean program for op {op!r}/{dtype} "
                f"(ok={meta.get('ok')}, opaque={meta.get('opaque')}); a capsule input must be 0-opaque")
        return CapsuleArtifacts(
            op=op, dtype=dtype, pytorch_src=src,
            linalg_mlir=(workdir / "linalg.mlir").read_text(encoding="utf-8"),
            inputs=json.loads((workdir / "inputs.json").read_text()),
            golden=json.loads((workdir / "golden.json").read_text()),
            weights_path=meta.get("weights", str(workdir / "weights.safetensors")),
            meta=meta)


# ------------------------------------------------------------------------------------------------
# writer: a full capsule dir from a PyTorch source (interface via the existing merlin_iface builders,
# golden via host torch-eager). Only ops that map to a merlin_iface builder are written here; other
# fused ops present the linalg directly (a later step) or fall back to a direct-MLIR engine.
# ------------------------------------------------------------------------------------------------
# loader positional-input index -> the capsule input NAME the harness feeds (must match the merlin_iface
# builder's declared input names). matmul/linear loader=(a,w); attention_qk=(q,k); rmsnorm=(x,g).
_OP_INPUT_NAMES = {
    "matmul": ["A0", "W"], "linear": ["A0", "W"],
    "attention_qk": ["Q", "K"], "rmsnorm": ["X", "G"],
}

# FUSED ops have no merlin_iface builder: their interface is the m2m linalg module DIRECTLY (the agent
# compiles the linalg), fed positionally. loader input order defines the func-arg order the harness passes.
_FUSED_OP_INPUT_NAMES = {
    "attention_full": ["Q", "K", "V"], "softmax": ["X"],
    "layernorm": ["X", "W", "B"], "geglu": ["X", "WG", "WU"], "rope": ["X"],
    "gelu": ["X"], "silu": ["X"], "add": ["A", "B"], "reduce_sum": ["X"],
    # composite ops from real model graphs (linalg-as-interface, positional args)
    "bias_add": ["X", "B"], "fused_matmul_bias": ["X", "W", "B"], "k_chain": ["A0", "W", "W2"],
    "logit_softcap": ["X"], "embed_scale": ["X"], "gemma_4norm": ["X", "G1", "G2"],
    "gemv_batched": ["A0", "V"], "patch_embed": ["X", "W"],
    "depthwise_conv2d": ["X", "W"], "maxpool2d": ["X"], "avgpool2d": ["X"], "global_average": ["X"],
}


def _entry_seed(name: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) or 1


# Float tolerance for a host-eager golden. A target whose native datapath is integer (e.g. gemmini,
# exact_int) carries no atol/rtol, but a float (bf16/fp16) op still needs one — fall back to a sane
# default so a fused op on an integer-datapath target is gradeable.
_DEF_ATOL, _DEF_RTOL = 0.03125, 0.02


def _tol(binding) -> tuple[float, float]:
    atol = binding.atol if binding.atol is not None else _DEF_ATOL
    rtol = binding.rtol if binding.rtol is not None else _DEF_RTOL
    return atol, rtol


def _shape_of(nested) -> list[int]:
    s = []
    x = nested
    while isinstance(x, list):
        s.append(len(x))
        x = x[0] if x else None
    return s


def _flatten(nested) -> list:
    out: list = []
    stack = [nested]
    if not (isinstance(nested, list) and nested and isinstance(nested[0], list)):
        return list(nested)
    for row in nested:
        out.extend(_flatten(row) if isinstance(row[0], list) else row)
    return out


def _capture_spec(entry: dict, binding) -> dict:
    """Build the PytorchRefSource capture spec (op + shapes + dtype + seed) from an abstract entry."""
    dim = binding.tile_dim
    op = entry.get("op", "matmul")
    M = entry.get("M", entry.get("M_tiles", 1) * dim)
    K = entry.get("K", entry.get("K_tiles", 1) * dim)
    N = entry.get("N", entry.get("N_tiles", 1) * dim)
    if op == "attention_qk":
        N = M                                    # Q@K^T scores are [M,M]; K rows == M (matches the builder)
    spec = {"op": op, "dtype": binding.operand_dtype, "seed": _entry_seed(entry["name"]),
            "M": M, "K": K, "N": N}
    if op in ("rmsnorm", "gemma_4norm", "layernorm"):
        spec["eps"] = entry.get("eps", 1.0 / 65536.0)
    # optional per-op fields (batch / soft-cap / conv geometry) pass through verbatim when present
    for f in ("B", "cap", "Cin", "Himg", "Wimg", "P", "Dv", "causal"):
        if f in entry:
            spec[f] = entry[f]
    return spec


def _host_eager_golden(art, names, out_name, binding, *, interface: str, arg_order):
    """The host torch-eager golden.yaml payload shared by the mapped and fused paths."""
    if len(art.inputs) < len(names):
        raise M2MUnavailable(f"op {art.op!r} produced {len(art.inputs)} inputs, expected {len(names)}")
    prov = {nm: {"shape": _shape_of(art.inputs[i]), "decoded": _flatten(art.inputs[i])}
            for i, nm in enumerate(names)}
    return {
        "golden_source": "host_torch_eager",
        "oracle_provenance": {
            "engine": "model2MLIR fx_importer linalg-on-tensors + host torch-eager (frontend-faithful; "
                      "torchAO weight-only for int8/fp8)",
            "operand_dtype": binding.cap_dtype(binding.operand_dtype),
            "output_dtype": binding.cap_dtype(binding.accum_dtype),
            "note": "INDEPENDENT of the target RTL; the PyTorch frontend + host eval are the reference.",
            "grade_policy": {"compare": binding.compare, "atol": _tol(binding)[0], "rtol": _tol(binding)[1]},
            "interface": interface, "arg_order": list(arg_order),
            "pytorch_source": "capsule.pytorch.py", "linalg_mlir": "capsule.linalg.mlir",
            "path_taken": art.meta.get("path_taken"), "inputs": prov},
        "outputs": {out_name: art.golden},
    }


def _fused_capsule_yaml(entry: dict, binding, art, names: list[str]) -> dict:
    """A schema-valid capsule.yaml for a FUSED op whose interface is the linalg module directly (positional
    args). No merlin_iface builder / instruction classes — the agent compiles the standard-dialect linalg."""
    idt = binding.cap_dtype(binding.operand_dtype)
    out_name = entry.get("out", "Y0")
    inputs = [{"name": nm, "role": "input", "shape": _shape_of(art.inputs[i]), "dtype": idt}
              for i, nm in enumerate(names)]
    return {
        "name": entry["name"], "kind": entry.get("kind", "model_slice"),
        "source_role": "pytorch_model_slice", "source_reference": entry.get("source_reference", ""),
        "label": entry.get("label", "public"), "interface_mlir": "capsule.interface.mlir",
        "inputs": inputs,
        "operation": {"op": entry["op"], "attributes": {"out": out_name, "arg_order": names,
                                                        "causal": bool(entry.get("causal", False))}},
        "numeric_policy": {"compare": binding.compare, "dtype": idt,
                           "atol": _tol(binding)[0], "rtol": _tol(binding)[1]},
        "expected": {"instruction_classes": [], "modes": dict(entry.get("modes", {}))},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
        "pytorch_ref": {"op": entry["op"], "dtype": idt, "loader": "capsule.pytorch.py"},
        "linalg_mlir": "capsule.interface.mlir",
    }


def write_pytorch_capsule(entry: dict, binding, out_root, *, source: "PytorchRefSource | None" = None):
    """Materialize a full capsule dir from a PyTorch-defined op. For a merlin_iface-mapped op (matmul/
    linear/attention_qk/rmsnorm) the agent-facing interface + expected coverage come from the existing
    ``corpus_spec`` builder. For a FUSED op (attention_full/softmax/layernorm/geglu/rope) the interface IS
    the m2m linalg module (the agent compiles standard-dialect linalg), fed positionally. In BOTH cases the
    golden + canonical inputs come from the host torch-eager run and the pytorch loader + linalg are written
    as visible grounding. Unknown ops fail closed."""
    import yaml
    from merlin.targetgen import corpus_spec as CS

    op = entry.get("op", "matmul")
    if op not in _OP_INPUT_NAMES and op not in _FUSED_OP_INPUT_NAMES:
        raise ValueError(f"write_pytorch_capsule: unknown op {op!r} "
                         f"(mapped {sorted(_OP_INPUT_NAMES)}; fused {sorted(_FUSED_OP_INPUT_NAMES)})")
    src = source or PytorchRefSource()
    spec = _capture_spec(entry, binding)
    art = src.capture(spec)
    out_name = entry.get("out", "Y0")
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)

    if op in _OP_INPUT_NAMES:                                   # merlin_iface interface
        names = _OP_INPUT_NAMES[op]
        cap, mlir = CS.build(entry, binding)
        cap["source_role"] = "pytorch_model_slice"
        cap["pytorch_ref"] = {"op": op, "dtype": binding.cap_dtype(binding.operand_dtype),
                              "loader": "capsule.pytorch.py"}
        cap["linalg_mlir"] = "capsule.linalg.mlir"
        golden = _host_eager_golden(art, names, out_name, binding, interface="merlin_iface", arg_order=names)
        (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    elif op in _FUSED_OP_INPUT_NAMES:                           # linalg interface (positional)
        names = _FUSED_OP_INPUT_NAMES[op]
        cap = _fused_capsule_yaml(entry, binding, art, names)
        golden = _host_eager_golden(art, names, out_name, binding,
                                    interface="linalg_positional", arg_order=names + [out_name])
        # the linalg module IS the interface the agent compiles
        (d / "capsule.interface.mlir").write_text(art.linalg_mlir, encoding="utf-8")
    else:
        raise ValueError(f"write_pytorch_capsule: unknown op {op!r} "
                         f"(mapped {sorted(_OP_INPUT_NAMES)}; fused {sorted(_FUSED_OP_INPUT_NAMES)})")

    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8")
    (d / "capsule.pytorch.py").write_text(art.pytorch_src, encoding="utf-8")
    (d / "capsule.linalg.mlir").write_text(art.linalg_mlir, encoding="utf-8")
    (d / "golden.yaml").write_text(yaml.safe_dump(golden, sort_keys=False), encoding="utf-8")
    return d


# ------------------------------------------------------------------------------------------------
# whole-model capsule (WholeModelSource): a small representative network (e.g. small_llama) lowered end
# to end via model2MLIR, graded vs its host torch-eager output — GATED so it runs only once the op suite
# has proven itself (>= a pass fraction). The interface is the whole linalg module (positional).
# ------------------------------------------------------------------------------------------------
def _all_integral(nested) -> bool:
    return all(float(v) == int(v) for v in _flatten(nested))


# canonical operand token -> the `merlin-compile --target rvv --dtype` token the whole-model grader uses.
_COMPILE_DTYPE = {"int8": "int8", "i8": "int8", "fp8_e4m3": "fp8", "fp8": "fp8",
                  "f32": "fp32", "fp32": "fp32", "fp16": "fp16", "bf16": "fp32"}


def compile_dtype(token: str) -> str:
    return _COMPILE_DTYPE.get(token, "fp32")


def resolve_model_loader(entry: dict, m2m_dir: str | Path | None = None) -> Path:
    """A model capsule entry names either a workload (``model: small_llama``) or an explicit ``loader``."""
    root = Path(m2m_dir) if m2m_dir else _m2m_dir()
    if entry.get("loader"):
        p = Path(entry["loader"])
        return p if p.is_absolute() else (root / p)
    name = entry.get("model")
    if not name:
        raise ValueError("model capsule entry needs 'model' (workload name) or 'loader' (path)")
    return root / "workloads" / name / "loader.py"


def write_model_capsule(entry: dict, binding, out_root, *, source: "PytorchRefSource | None" = None):
    """Materialize a whole-model capsule: the model is lowered end-to-end via model2MLIR (the linalg IS
    the interface), weights are externalized alongside, and the golden is the host torch-eager output.
    A ``gate`` (default ``after_op_pass_fraction: 0.8``) defers scheduling until the op suite passes."""
    import shutil

    import yaml
    # only a whole-model capsule may ship externalized weights (a compile INPUT); guard so an op capsule
    # can never route here and leak a weight-derived answer (see CapsuleArtifacts). A model capsule is
    # identified by kind/op/cat == "model"; op capsules never carry any of those.
    if "model" not in (entry.get("kind"), entry.get("op"), entry.get("cat")):
        raise ValueError(f"write_model_capsule requires a model capsule (kind/op/cat == 'model'); got "
                         f"kind={entry.get('kind')!r} op={entry.get('op')!r} cat={entry.get('cat')!r} "
                         f"— weights ship only for whole-model capsules")
    src = source or PytorchRefSource()
    dtype = entry.get("operand_dtype") or binding.operand_dtype
    loader = resolve_model_loader(entry, src.m2m_dir)
    art = src.capture_loader(loader, dtype)

    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    in_names = [f"I{i}" for i in range(len(art.inputs))]
    idt = binding.cap_dtype(binding.operand_dtype)
    inputs = [{"name": nm, "role": "input", "shape": _shape_of(art.inputs[i]),
               "dtype": ("i64" if _all_integral(art.inputs[i]) else idt)}
              for i, nm in enumerate(in_names)]
    out_name = entry.get("out", "Y0")
    gate = entry.get("gate") or {"after_op_pass_fraction": 0.8}

    # self-contained weights: copy in + rewrite the linalg's absolute prov.weights_file to a relative name
    linalg = art.linalg_mlir
    wsrc = Path(art.weights_path)
    if wsrc.is_file():
        shutil.copyfile(wsrc, d / "capsule.weights.safetensors")
        linalg = linalg.replace(str(wsrc), "capsule.weights.safetensors")

    cap = {
        "name": entry["name"], "kind": "model", "source_role": "pytorch_model_slice",
        "source_reference": entry.get("source_reference", f"whole model {entry.get('model', '')}"),
        "label": entry.get("label", "public"), "interface_mlir": "capsule.interface.mlir",
        "inputs": inputs,
        "operation": {"op": "model", "attributes": {
            "model": entry.get("model", ""), "dtype": idt, "out": out_name,
            "compile_dtype": compile_dtype(binding.operand_dtype),
            "arg_order": in_names + [out_name], "weights": "capsule.weights.safetensors"}},
        # the whole-model golden is the host torch-eager float output, so it is graded with tolerance even
        # on an integer-datapath target (whose op capsules grade exact_int).
        "numeric_policy": {"compare": "tolerance_float", "dtype": "f32",
                           "atol": _tol(binding)[0], "rtol": _tol(binding)[1]},
        "expected": {"instruction_classes": []},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
        "gate": gate,
        "pytorch_ref": {"op": "model", "dtype": idt, "loader": "capsule.pytorch.py"},
        "linalg_mlir": "capsule.interface.mlir",
    }
    prov = {nm: {"shape": _shape_of(art.inputs[i]), "decoded": _flatten(art.inputs[i])}
            for i, nm in enumerate(in_names)}
    golden = {
        "golden_source": "host_torch_eager",
        "oracle_provenance": {
            "engine": "model2MLIR whole-model linalg-on-tensors + host torch-eager",
            "model": entry.get("model", ""), "output_dtype": "f32",
            "grade_policy": {"compare": binding.compare, "atol": _tol(binding)[0], "rtol": _tol(binding)[1]},
            "interface": "linalg_positional", "arg_order": in_names + [out_name],
            "pytorch_source": "capsule.pytorch.py", "linalg_mlir": "capsule.interface.mlir",
            "inputs": prov},
        "outputs": {out_name: art.golden},
    }
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(linalg, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8")
    (d / "capsule.pytorch.py").write_text(art.pytorch_src, encoding="utf-8")
    (d / "golden.yaml").write_text(yaml.safe_dump(golden, sort_keys=False), encoding="utf-8")
    return d


def model_gate_satisfied(cap: dict, op_pass_fraction: float) -> bool:
    """True when a (model) capsule may be scheduled: its ``gate.after_op_pass_fraction`` is met, or absent.
    ``op_pass_fraction`` is the fraction of graded OP-level capsules that passed (0..1)."""
    thr = (cap.get("gate") or {}).get("after_op_pass_fraction")
    return thr is None or op_pass_fraction >= float(thr)


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
