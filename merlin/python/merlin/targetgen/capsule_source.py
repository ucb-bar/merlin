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
  **spec** op in ``specir`` (``$SPECIR_ROOT``). ``specir`` is pure xDSL (no torch) so it
  runs in-process. (Scaffolded here; the fp8-matmul spec path already lives in ``generate_corpus`` and is
  folded in as these are unified.)

The op → PyTorch module mapping is a small generic vocabulary (matmul/linear, attention QK + full, rmsnorm,
softmax, layernorm, geglu/swiglu, rope). Each op is plain ``torch`` so its loader imports cleanly in the m2m
venv. Precision is a parameter (``dtype``): the SAME token a target's ``compute_units`` declares.
"""
from __future__ import annotations

import hashlib

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import env as _env, repo_root


# ------------------------------------------------------------------------------------------------
# m2m venv resolution (torch lives there, never in the merlin venv)
# ------------------------------------------------------------------------------------------------
def _m2m_dir() -> Path:
    """The model2MLIR checkout: ``$MERLIN_M2M_DIR``, else a sibling of this repo.

    model2MLIR is an EXTERNAL repo, so the fallback is a CONVENTION (checked out next to merlin),
    not one machine's home directory -- callers already treat a non-existent path as "not
    available" and skip, and that behaviour is preserved.
    """
    d = _env("MERLIN_M2M_DIR")
    return Path(d) if d else repo_root().parent / "model2MLIR"


def _m2m_python() -> Path:
    env = os.environ.get("MERLIN_M2M_PYTHON")
    if env:
        return Path(env)
    return _m2m_dir() / ".venv" / "bin" / "python"


def _stderr_cause(stderr: str, *, limit: int = 1800) -> str:
    """The part of ``stderr`` that explains a failure, rather than whatever came last.

    A python subprocess interleaves warnings with tracebacks, so the tail of stderr is often a
    warning emitted after the real error -- which is exactly how a capture failure came to be
    attributed to a benign LSTM contiguity UserWarning. So: prefer the LAST traceback, and fall back
    to the tail only when there is none, saying which was used either way.
    """
    text = str(stderr or "")
    if not text.strip():
        return "(the worker wrote nothing to stderr)"
    marker = "Traceback (most recent call last)"
    idx = text.rfind(marker)
    if idx < 0:
        return "--- stderr tail (no traceback found) ---\n" + text[-limit:]

    # ⚠️ TRUNCATE FROM THE MIDDLE, NEVER FROM THE HEAD. `text[idx:idx + limit]` keeps the header and
    # the stack frames and cuts off the LAST line -- which is the only line that names the error. A
    # deep traceback therefore recorded a failure reason containing no error at all, just frames, and
    # every diagnosis that started from one of these was slower than it needed to be. The exception
    # line is the point of a traceback, so it is the part that is guaranteed to survive.
    tb = text[idx:]
    if len(tb) <= limit:
        return "--- last traceback ---\n" + tb
    head = limit // 3
    tail = limit - head
    return ("--- last traceback (middle elided; the exception line is kept) ---\n"
            + tb[:head]
            + f"\n    ... {len(tb) - limit} characters of frames elided ...\n"
            + tb[-tail:])


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


#: A contraction whose weight is a PARAMETER rather than a second input.
#:
#: torchAO quantizes module WEIGHTS. Every body in ``_OP_BODIES`` takes both operands as inputs
#: (``a @ w``), so a quantization scheme has nothing to act on and is silently a no-op -- which is why
#: a capture requested at W8A8 still emitted ``aten.mm.default`` over f32 while faithfully recording
#: ``prov.quantization = "int8_dyn_act_int8_weight"``. The declared scheme was true and the program was
#: not. This body gives the quantizer a parameter to bind to; m2m externalizes it, so the captured
#: linalg still carries a two-operand contraction and the interface derivation is unchanged.
_PARAMETRIC_LINEAR = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear({K}, {N}, bias=False)
        with torch.no_grad():
            self.f.weight.copy_(_r({N}, {K}))
    def forward(self, a):
        return self.f(a)
def get_model_and_inputs():
    return Model().eval(), (_r({M}, {K}),)
"""


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
    # A QUANTIZED capture needs a weight PARAMETER for the scheme to bind to (see _PARAMETRIC_LINEAR).
    # Only the contraction ops have a meaningful weight; asking for a quantized elementwise op is a
    # request that cannot be honoured, and saying so beats emitting an unquantized program under a
    # quantized name.
    if spec.get("quant_scheme"):
        if op not in ("linear", "matmul"):
            raise ValueError(
                f"quant_scheme is set for op {op!r}, but only a contraction carries a weight for a "
                f"torchAO scheme to quantize; an unquantized program under a quantized name is worse "
                f"than a refusal")
        return _PREAMBLE.format(**fields) + _PARAMETRIC_LINEAR.format(**fields)
    return _PREAMBLE.format(**fields) + _OP_BODIES[op].format(**fields)


# ------------------------------------------------------------------------------------------------
# artifacts
# ------------------------------------------------------------------------------------------------
@dataclass
class CapsuleArtifacts:
    """Everything a grounded capsule needs. ``pytorch_src`` + ``linalg_mlir`` are agent-visible realistic
    lowering context. ``golden`` is the host-eager output and ``weights_path`` is the private model instance
    from which that output derives; both are operator-side answer surfaces masked from the agent. The weights
    still ship in the complete capsule bundle because the grader needs the exact external compile/runtime input
    named by the linalg. Only ``kind == model`` capsules ever carry a ``.safetensors``; op capsules write none."""
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
        return self._runnable(self.python)

    def _runnable(self, python: Path) -> bool:
        """Whether a capture can run under ``python`` -- the default interpreter, or the one a workload
        pins for itself. Same two conditions either way: an interpreter that exists, next to an m2m
        checkout to import."""
        return Path(python).exists() and (self.m2m_dir / "m2m" / "__init__.py").exists()

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
        return self._run(tmp / "loader.py", spec["op"], dtype, workdir=tmp, src=loader_src,
                         scheme=spec.get("quant_scheme"))

    def capture_loader(self, loader_py: str | Path, dtype: str, *,
                       workdir: str | Path | None = None,
                       scheme: str | None = None,
                       env: dict | None = None,
                       python: str | Path | None = None) -> CapsuleArtifacts:
        """Capture an EXISTING loader file (a whole-model workload) rather than a generated op loader.
        Same worker path; ``op`` is ``model`` and ``pytorch_src`` is the loader's own source.

        ``scheme`` reaches the worker exactly as it does for an op capsule. Without it the whole-model
        path took the DEFAULT for its dtype, and the default for int8 is weight-only -- a float matmul
        over dequantized weights. A model capsule declaring `compile_dtype: int8` was therefore capturing
        a program with no integer contraction in it at all, which is the wrong program for an integer
        mesh and one no golden substitution can repair.

        ``env`` is the model's OWN declared loader environment (:func:`model_capture_env`) and ``python``
        the interpreter it pins (:func:`model_capture_python`). Both used to be ignored here, and a real
        network is not capturable without them: a loader that refuses to invent inputs unless told which
        stream to use raised, and a loader whose dependency lives only in its own venv raised
        ``ModuleNotFoundError`` -- and BOTH were recorded as "this model could not be built", which reads
        as a limit of the compiler rather than of how it was invoked."""
        loader_py = Path(loader_py)
        # `available()` stays the predicate when nothing is pinned, so a caller that has established
        # availability some other way keeps saying so; a PINNED interpreter is checked on its own terms.
        interpreter = Path(python) if python else self.python
        if not (self._runnable(interpreter) if python else self.available()):
            raise M2MUnavailable(f"m2m venv python {interpreter} or repo {self.m2m_dir} missing")
        if not loader_py.is_file():
            raise M2MUnavailable(f"model loader not found: {loader_py}")
        tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="capsule_model_"))
        tmp.mkdir(parents=True, exist_ok=True)
        return self._run(loader_py, "model", dtype, workdir=tmp, src=loader_py.read_text(),
                         scheme=scheme, env=env, python=interpreter)

    def _cache_slot(self, op: str, dtype: str, src: str, scheme: str | None,
                    env: dict | None = None, python: "Path | None" = None) -> "Path | None":
        """Where a capture of exactly this input already lives, or ``None`` if caching is unavailable.

        Keyed on everything that changes the PROGRAM -- the loader's source, the format, the capture
        scheme, the op role -- so a changed loader misses rather than serving a stale module. Not keyed
        on the target: a capture is a property of the model and the format, and two targets that choose
        the same format are entitled to the same bytes.

        The declared loader ENVIRONMENT and INTERPRETER are part of the key for the same reason the
        source is: they change the program and the operands (which input stream a loader reads, which
        checkpoint it loads, which dependency stack it imports). Keyed on the source alone, a capture
        taken under one input declaration would be served for a request that declared another -- the
        silent kind of substitution this corpus exists to prevent.

        This exists because the roster axis made whole-model captures routine. Four networks per target
        re-captured on every corpus regeneration is tens of minutes of work whose result is identical
        each time, and the alternative that suggests itself -- generating the entries but not building
        them -- is the silent kind of skip this corpus is built to avoid.
        """
        try:
            from merlin.common.artifacts import cache_dir
            declared = "\x1f".join(f"{k}={v}" for k, v in sorted((env or {}).items()))
            key = hashlib.sha256(
                "\0".join((op, str(dtype), str(scheme or ""), str(self.m2m_dir), src,
                           declared, str(python or self.python))).encode()
            ).hexdigest()[:16]
            return cache_dir("model_capture") / f"{op}_{dtype}_{key}"
        except Exception:            # noqa: BLE001 -- an unavailable cache is not a failed capture
            return None

    def _run(self, loader_py: Path, op: str, dtype: str, *, workdir: Path, src: str,
             scheme: str | None = None, env: dict | None = None,
             python: "Path | None" = None) -> CapsuleArtifacts:
        worker = Path(__file__).with_name("_m2m_capture_worker.py")
        # A CACHED CAPTURE IS THE SAME CAPTURE. Reused only when the slot holds a complete, clean result
        # -- the same two conditions the fresh path checks below -- so a half-written slot from an
        # interrupted run re-captures instead of serving a truncated module.
        declared_env = {str(k): str(v) for k, v in (env or {}).items()}
        interpreter = Path(python) if python else self.python
        slot = self._cache_slot(op, dtype, src, scheme, declared_env, interpreter)
        if slot is not None and (slot / "meta.json").is_file():
            try:
                cached = json.loads((slot / "meta.json").read_text())
                # A slot written before the worker recorded input provenance cannot say what its inputs
                # were, and a capsule may not be built from a capture whose provenance is unrecoverable.
                # Re-capturing is the cheap, honest repair; serving it would launder an unknown.
                if (cached.get("ok") and cached.get("opaque", -1) == 0
                        and "loader_provenance_status" in cached):
                    return CapsuleArtifacts(
                        op=op, dtype=dtype, pytorch_src=src,
                        linalg_mlir=(slot / "linalg.mlir").read_text(encoding="utf-8"),
                        inputs=json.loads((slot / "inputs.json").read_text()),
                        golden=json.loads((slot / "golden.json").read_text()),
                        weights_path=cached.get("weights", str(slot / "weights.safetensors")),
                        meta=cached)
            except Exception:        # noqa: BLE001 -- an unreadable slot is a miss, never a failure
                pass
        if slot is not None:
            workdir = slot
            workdir.mkdir(parents=True, exist_ok=True)
        # THE MODEL'S OWN DECLARED LOADER ENVIRONMENT, applied under the ambient one. A loader states
        # what it needs next to itself (its workload's `capture.toml`, plus the curated fidelity knobs in
        # `baselines.bundle`); building the worker environment from `os.environ` alone silently dropped
        # that declaration, and every loader that refuses to guess then raised -- recorded as an unbuilt
        # model. The DECLARATION WINS over an ambient value, as it does on every other arm that replays
        # it: a capsule must capture the model its declaration names, not whatever the shell that ran
        # the generator happened to export, or two regenerations produce two different networks under
        # one name.
        env = dict(os.environ)
        env.update(declared_env)
        env["MERLIN_M2M_DIR"] = str(self.m2m_dir)
        cmd = [str(interpreter), str(worker), "--loader", str(loader_py),
               "--dtype", dtype, "--out", str(workdir), "--m2m-dir", str(self.m2m_dir)]
        if scheme:
            cmd += ["--scheme", str(scheme)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env)
        meta_p = workdir / "meta.json"
        # ⚠️ A NON-ZERO RC IS NOT NECESSARILY A CRASH, and treating it as one threw away the
        # diagnosis. The worker returns 3 for "ran fine, but the program is not clean" -- and it
        # WRITES meta.json with `opaque` and `opaque_detail` before doing so. This branch fired first
        # on rc=3 and dumped the stderr tail, which for a torch export is a wall of warnings, while
        # the block just below already had the useful message. Measured on the lstmnetvit roster model
        # at W8A8: the real answer is opaque ops from un-inlined torchAO quantization calls, and what
        # got reported was an LSTM `_flat_weights` contiguity notice.
        #
        # So the stderr path is now for a worker that produced NO diagnosis at all; if meta.json
        # exists the worker reached its own verdict and that verdict is what to report.
        if not meta_p.exists():
            # ⚠️ REPORT THE CAUSE, NOT THE TAIL. This used to print `stderr[-1500:]`, and a capture
            # that emits warnings AFTER its traceback therefore reported the warnings. Measured on
            # the lstmnetvit roster model: the failure was attributed to a benign
            # "tensor attributes self.net.lstm._flat_weights[...] are not part of a single contiguous
            # chunk of memory" UserWarning, which is not an error at all and sent a reader looking at
            # the wrong thing. The two conditions are also different failures and were reported as
            # one: a non-zero rc means the worker died, a missing meta.json after rc=0 means it
            # exited cleanly without producing a capture.
            why = ("worker exited non-zero" if proc.returncode != 0
                   else "worker exited 0 but wrote no meta.json")
            raise M2MUnavailable(f"m2m capture failed for op {op!r}/{dtype}: {why} "
                                 f"(rc={proc.returncode})\n{_stderr_cause(proc.stderr)}")
        meta = json.loads(meta_p.read_text())
        if not meta.get("ok") or meta.get("opaque", -1) != 0:
            # Name WHICH ops are opaque: "opaque=44" says a capsule cannot use this program, and the
            # detail says what to go and inline. The scheme is included because the same model is
            # clean under one quantization and not another -- lstmnetvit captures cleanly at plain
            # int8 and carries opaque torchAO calls at W8A8, which is a fact about the scheme.
            detail = meta.get("opaque_detail") or {}
            top = ", ".join(f"{k}={v}" for k, v in sorted(detail.items(),
                                                          key=lambda kv: -int(kv[1] or 0))[:6])
            raise M2MUnavailable(
                f"m2m capture produced a non-clean program for op {op!r}/{dtype} "
                f"(ok={meta.get('ok')}, opaque={meta.get('opaque')}, "
                f"scheme={meta.get('scheme')!r}, rc={proc.returncode}); a capsule input must be "
                f"0-opaque. Opaque ops: {top or '(none reported)'}")
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
    # WHICH QUANTIZATION the captured PROGRAM should carry, when the entry names one. The dtype default
    # for int8 is weight-only, which emits a float matmul over dequantized weights -- correct for a
    # model ladder, wrong for a capsule meant to exercise an integer datapath, and not fixable by
    # substituting a golden: the program itself contains no integer contraction.
    if entry.get("quant_scheme"):
        spec["quant_scheme"] = str(entry["quant_scheme"])
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
        # Same per-target tier declaration the direct-MLIR builders carry (corpus_spec.build): a capsule
        # authored through a different source must not silently lose it, or the same tier that is honestly
        # N/A for the rest of the corpus fails forever on this one.
        **({"inapplicable_oracle_tiers": dict(binding.inapplicable_tiers)}
           if getattr(binding, "inapplicable_tiers", None) else {}),
        "pytorch_ref": {"op": entry["op"], "dtype": idt, "loader": "capsule.pytorch.py"},
        "linalg_mlir": "capsule.interface.mlir",
    }


# mapped op -> a defining ``prov.op`` marker the captured lowering MUST carry for a merlin_iface interface
# to be a faithful lowering of it (matmul-family lowers to a linalg.matmul; rmsnorm decomposes to a
# reciprocal-sqrt over the mean-square). Used to derive-and-verify (not assume) the interface from the linalg.
_OP_MARKER: dict[str, tuple[str, ...]] = {
    # A contraction may arrive tagged `matmul` (a float `linalg.matmul`) or `int_matmul` (the
    # `aten._int_mm` form a W8A8 capture emits, a linalg.generic accumulating in i32). Both ARE the
    # contraction; matching only the float spelling refused the very capture that carries the integer
    # arithmetic a systolic mesh runs -- the program was right and the check was reading for the wrong
    # word.
    "matmul": ("matmul", "int_matmul"),
    "linear": ("matmul", "int_matmul"),
    "attention_qk": ("matmul", "int_matmul"),
    "rmsnorm": ("rsqrt",),
}


def _tensor_types(s: str) -> list[tuple[list[int], str]]:
    """Structurally tokenize the ``tensor<...>`` types in a fragment (str.split, NO regex): each becomes
    ``([dims], dtype)``. Dynamic/non-integer dims are skipped (a shaped op capsule is always static)."""
    out: list[tuple[list[int], str]] = []
    for chunk in s.split("tensor<")[1:]:
        body = chunk.split(">", 1)[0]                         # e.g. "16x16xf32"
        parts = body.split("x")
        if len(parts) < 2:
            continue
        *dims, dt = parts
        try:
            out.append(([int(d) for d in dims], dt))
        except ValueError:
            continue                                          # a dim that isn't an int -> not a static shape
    return out


def linalg_summary(linalg_mlir: str) -> dict:
    """Read a linalg-on-tensors module STRUCTURALLY (str tokenizer, NO regex): the ``@forward`` signature's
    operand + result tensor types and the set of ``prov.op`` / ``prov.family`` tags m2m stamps on each region.
    This is the structural view used to verify a mapped op's interface against its actual lowering."""
    prov_ops = [c.split('"', 1)[0] for c in linalg_mlir.split('prov.op = "')[1:]]
    prov_families = [c.split('"', 1)[0] for c in linalg_mlir.split('prov.family = "')[1:]]
    inputs: list[tuple[list[int], str]] = []
    output = None
    head = linalg_mlir.split("func.func @forward(", 1)
    if len(head) > 1:
        sig = head[1].split("{", 1)[0]                        # the signature line, before the body brace
        argpart, _, rest = sig.partition(") ->")
        inputs = _tensor_types(argpart)
        outs = _tensor_types(rest)
        output = outs[0] if outs else None
    return {"prov_ops": prov_ops, "prov_families": prov_families, "inputs": inputs, "output": output}


def _matmul_extents(linalg_mlir: str) -> list[tuple[int, int, int]]:
    """The (M, K, N) extent of each ``linalg.matmul`` in text order, read structurally (str tokenizer, NO
    regex): a matmul's ``ins`` operands are ``tensor<MxK...>, tensor<KxN...>``. Used to compile each
    whole-model matmul LAYER at its real shape instead of a fixed tile. Best-effort — a matmul whose ins
    types can't be read is skipped (its demand falls back to the mesh tile dim)."""
    extents: list[tuple[int, int, int]] = []
    for seg in linalg_mlir.split("linalg.matmul")[1:]:
        # Exact op name only. `linalg.matmul_transpose_b` also starts with "linalg.matmul", and its ins
        # types describe a TRANSPOSED operand pair -- so accepting it recorded the wrong (M, K, N) and,
        # because the caller joins extents positionally, shifted the shape of every later layer.
        if seg[:1].isalnum() or seg[:1] == "_":
            continue
        head = seg.split("outs", 1)[0]                       # the ins(...) clause precedes outs(...)
        ts = _tensor_types(head)
        if len(ts) >= 2 and len(ts[0][0]) >= 2 and len(ts[1][0]) >= 2:
            (m, k), (_k2, n) = ts[0][0][:2], ts[1][0][:2]
            extents.append((m, k, n))
        else:
            extents.append((0, 0, 0))                        # unreadable — sentinel, demand falls back
    return extents


def model_op_demands(linalg_mlir: str, in_fmt: str, weight_fmt: str | None = None) -> list:
    """Per-op routing demands from a captured model's linalg (structural prov.op/prov.family read, NO regex).
    A contraction op (matmul/conv — ``prov.family == "contraction"``) carries a weight format AND its real
    (M, K, N) extents (so a whole-model matmul layer compiles at its true shape); normalization / elementwise
    / reduction ops are unary. Feeds ``routing.route_target`` so a whole model can be split across a target's
    compute units (matmul tiles -> the systolic mesh, the rest -> vector/scalar lanes)."""
    from merlin.targetgen.routing import OpDemand
    summ = linalg_summary(linalg_mlir)
    ops, fams = summ["prov_ops"], summ["prov_families"]
    extents = _matmul_extents(linalg_mlir)
    wf = weight_fmt or in_fmt
    demands: list = []
    mm = 0
    for i, op in enumerate(ops):
        fam = fams[i] if i < len(fams) else ""
        if op == "fill":                                     # linalg.fill init — not a routable compute op
            continue
        m = k = n = None
        if op == "matmul" and mm < len(extents) and all(extents[mm]):
            m, k, n = extents[mm]
        if fam == "contraction" and op == "matmul":
            mm += 1
        demands.append(OpDemand(op=op, in_fmt=in_fmt,
                                weight_fmt=(wf if fam == "contraction" else None), site=op,
                                m=m, k=k, n=n))
    return demands


def linalg_to_iface(linalg_mlir: str, entry: dict, binding):
    """Derive-and-verify a mapped op's ``merlin_iface`` interface from the CAPTURED linalg rather than
    assuming the PyTorch capture matches the profile entry. Structurally confirm the lowering actually
    contains the op family the interface claims AND that its operand shapes are the ones the builder
    declares; fail closed (``M2MUnavailable``) on any mismatch — never emit an interface the lowering does
    not support. Returns ``(cap, mlir)`` from the merlin_iface builder (byte-identical to the handwritten
    form when they agree)."""
    from merlin.targetgen import corpus_spec as CS
    op = entry.get("op", "matmul")
    summ = linalg_summary(linalg_mlir)
    markers = _OP_MARKER.get(op)
    if markers is not None and not (set(markers) & set(summ["prov_ops"])) and op not in summ["prov_ops"]:
        raise M2MUnavailable(
            f"captured linalg for {entry['name']!r} lacks any {op!r} marker op {list(markers)} "
            f"(prov.op={summ['prov_ops']}) — refusing to emit a merlin_iface interface the lowering "
            f"does not support")
    cap, mlir = CS.build(entry, binding)
    # every operand shape the builder declares must appear among the captured linalg's operand shapes.
    lin_shapes = [tuple(sh) for sh, _ in summ["inputs"]]
    for inp in cap.get("inputs", []):
        want = tuple(inp["shape"])
        if lin_shapes and want not in lin_shapes:
            raise M2MUnavailable(
                f"interface operand {inp['name']!r} shape {want} for {entry['name']!r} not found in the "
                f"captured linalg operands {lin_shapes} — interface/lowering shape mismatch")
    return cap, mlir


def write_pytorch_capsule(entry: dict, binding, out_root, *, source: "PytorchRefSource | None" = None):
    """Materialize a full capsule dir from a PyTorch-defined op. For a merlin_iface-mapped op (matmul/
    linear/attention_qk/rmsnorm) the agent-facing interface + expected coverage are DERIVED-AND-VERIFIED
    from the captured linalg (``linalg_to_iface``: the lowering must actually contain the op + shapes) via
    the existing ``corpus_spec`` builder. For a FUSED op (attention_full/softmax/layernorm/geglu/rope) the
    interface IS the m2m linalg module (the agent compiles standard-dialect linalg), fed positionally. In
    BOTH cases the
    golden + canonical inputs come from the host torch-eager run and the pytorch loader + linalg are written
    as visible grounding. Unknown ops fail closed."""
    import yaml

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
        cap, mlir = linalg_to_iface(art.linalg_mlir, entry, binding)   # derive-and-verify from the lowering
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
# canonical operand token -> the `merlin-compile --target rvv --dtype` token the whole-model grader uses.
_COMPILE_DTYPE = {"int8": "int8", "i8": "int8", "fp8_e4m3": "fp8", "fp8": "fp8",
                  "f32": "fp32", "fp32": "fp32", "fp16": "fp16", "bf16": "fp32"}


def compile_dtype(token: str) -> str:
    return _COMPILE_DTYPE.get(token, "fp32")


def model_input_abi(art: CapsuleArtifacts) -> list[dict]:
    """Return the captured model's runtime-input ABI, cross-checked at both sides of capture.

    The worker records torch tensor leaf dtypes *before* JSON conversion. The linalg interface records the
    compiler-visible ABI after capture. Externalized parameters precede runtime leaves in model2MLIR's
    ``@forward`` signature, so its trailing leaves must agree exactly. Refuse generation if either record is
    absent or drifts; guessing from decoded values is invalid (an f32 tensor may contain only integral values).
    """
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    loader_abi = (art.meta or {}).get("input_abi")
    if not isinstance(loader_abi, list) or len(loader_abi) != len(art.inputs):
        raise M2MUnavailable(
            "model capture has no complete input_abi from the loader; refusing to infer dtypes from JSON "
            f"values (abi={loader_abi!r}, decoded_inputs={len(art.inputs)})")
    try:
        iface_args = parse_linalg_mlir(art.linalg_mlir)["args"]
    except Exception as e:  # noqa: BLE001 - turn any malformed captured ABI into a closed generation error
        raise M2MUnavailable(f"could not read captured model @forward ABI: {e}") from e
    if len(iface_args) < len(loader_abi):
        raise M2MUnavailable(
            f"captured @forward has {len(iface_args)} args for {len(loader_abi)} loader input leaves")
    runtime_args = iface_args[-len(loader_abi):] if loader_abi else []
    out: list[dict] = []
    for i, (decl, iface, decoded) in enumerate(zip(loader_abi, runtime_args, art.inputs, strict=True)):
        loader_shape = [int(x) for x in (decl.get("shape") or [])]
        loader_dtype = str(decl.get("dtype") or "")
        decoded_shape = _shape_of(decoded)
        iface_shape = [int(x) for x in (iface.get("shape") or [])]
        iface_dtype = str(iface.get("dtype") or "")
        if not loader_dtype or loader_shape != decoded_shape:
            raise M2MUnavailable(
                f"loader input leaf I{i} ABI {loader_shape}/{loader_dtype!r} disagrees with decoded "
                f"input shape {decoded_shape}")
        if (loader_shape, loader_dtype) != (iface_shape, iface_dtype):
            raise M2MUnavailable(
                f"loader input leaf I{i} ABI {loader_shape}/{loader_dtype} disagrees with captured "
                f"@forward runtime arg {iface_shape}/{iface_dtype}")
        out.append({"shape": loader_shape, "dtype": loader_dtype})
    return out


#: Canonical format token -> the torchAO scheme that quantizes the ACTIVATIONS as well as the weights.
#:
#: Mirrors the m2m scheme catalog (``m2m.capture.torchao_schemes.TORCHAO_SCHEMES``), whose entries carry
#: both a ``weight_dtype`` and an ``activation_dtype``; each name below is the catalog entry whose two
#: dtypes are the format on the left. Carried here rather than imported because the catalog lives in the
#: model2MLIR checkout and importing it pulls in torch, which this process does not have -- the same
#: reason ``_m2m_capture_worker._SCHEME`` mirrors ``workloads/capture.py`` instead of importing it.
#:
#: WHY IT IS NEEDED AT ALL. The default scheme for every one of these formats is WEIGHT-ONLY, which
#: emits a float matmul over dequantized weights. That is the wrong program for a datapath that consumes
#: the narrow format on BOTH operands: no golden substitution can fix a capture that contains no
#: contraction in the target's arithmetic, and the capsule then certifies the host's float math.
_ACTIVATION_QUANTIZING_SCHEME = {
    "int8": "int8_dyn_act_int8_weight", "i8": "int8_dyn_act_int8_weight",
    "fp8_e4m3": "float8_dyn_act_float8_weight",
    "mxfp4": "mx_dyn_act_mx_weight_mx4",
    "mxfp6": "mx_dyn_act_mx_weight_mx6",
    "nvfp4": "nvfp4_dyn_act_nvfp4_weight",
}

#: Formats a model already runs in, so a capture needs no quantization step at all.
_UNQUANTIZED_FORMATS = frozenset({"fp32", "f32", "bf16", "fp16", "f16"})


def activation_quantizing_scheme(fmt: str) -> str | None:
    """The capture scheme a whole-model capsule needs at ``fmt``, or ``None`` when it needs none.

    ``None`` means the format is one the model already runs in (a native float), so there is nothing to
    quantize and the default capture is already the right program.

    Raises for a format that IS quantized but whose activation-quantizing scheme is unknown here.
    Falling back to the weight-only default would be the silent version of the bug this exists to
    prevent: the capture would succeed, contain a float matmul, and certify arithmetic the target does
    not perform.
    """
    tok = str(fmt)
    if tok in _UNQUANTIZED_FORMATS:
        return None
    scheme = _ACTIVATION_QUANTIZING_SCHEME.get(tok)
    if scheme is None:
        raise ValueError(
            f"no activation-quantizing capture scheme is known for format {tok!r}; the weight-only "
            f"default would capture a float matmul over dequantized weights, which is not the "
            f"arithmetic a datapath in this format performs. Add the scheme whose catalog entry "
            f"declares {tok!r} as BOTH its weight_dtype and its activation_dtype")
    return scheme


def resolve_model_loader(entry: dict, m2m_dir: str | Path | None = None) -> Path:
    """A model capsule entry names either a workload (``model: small_llama``) or an explicit ``loader``.

    A relative ``loader`` is resolved REPO-ROOT-FIRST, then against the m2m checkout. Both roots are
    needed and the order matters: a whole-model capsule whose network is defined in THIS repo (the
    self-contained case -- the capsule's own ``capsule.pytorch.py`` is the source of record) can then be
    named by the repo-root-relative path this repo already uses for every other in-tree reference, while
    a capsule that names a model2MLIR workload keeps working unchanged. Resolving only against the m2m
    root, as this did, made the in-repo case unnameable: every whole-model capsule had to keep its
    network in an external checkout, which is exactly the dependency that leaves a capsule unbuildable
    from a clean clone.
    """
    root = Path(m2m_dir) if m2m_dir else _m2m_dir()
    if entry.get("loader"):
        p = Path(entry["loader"])
        if p.is_absolute():
            return p
        in_repo = repo_root() / p
        return in_repo if in_repo.is_file() else (root / p)
    name = entry.get("model")
    if not name:
        raise ValueError("model capsule entry needs 'model' (workload name) or 'loader' (path)")
    exact = root / "workloads" / name / "loader.py"
    if exact.is_file():
        return exact
    return _workload_named(root, name) or exact


def _workload_named(root: Path, name: str) -> "Path | None":
    """The workload directory a ROSTER name denotes, when it is not spelled the same.

    A roster declares a model (``resnet50``); a workload directory carries the checkpoint revision
    (``resnet50_v1_5``). ``dse_guidance.models`` already resolves the capture direction of this by
    longest-prefix match -- without which a fully captured ResNet reads as ABSENT -- and the capsule
    direction had no resolution at all, so a roster model with a revisioned workload simply could not be
    named by a synthesized capsule.

    Prefix, and only prefix: a revision suffix extends the model name, it never replaces it. Ambiguity
    is an error rather than a pick, because two revisions are two different networks and choosing by
    sort order would silently certify the wrong one. ``None`` when nothing matches, so the caller
    reports the path it expected rather than a guess.
    """
    wroot = root / "workloads"
    if not wroot.is_dir():
        return None
    hits = sorted(d for d in wroot.iterdir()
                  if d.is_dir() and d.name.startswith(name) and (d / "loader.py").is_file())
    if len(hits) == 1:
        return hits[0] / "loader.py"
    if len(hits) > 1:
        raise ValueError(
            f"roster model {name!r} matches {len(hits)} workload directories "
            f"({[d.name for d in hits]}); name the revision in the roster -- picking one by sort order "
            f"would certify a different network than the one declared")
    return None


def resolve_model_workload(entry: dict, m2m_dir: str | Path | None = None) -> "str | None":
    """The model2MLIR WORKLOAD DIRECTORY behind this entry's loader, or ``None`` when it has none.

    The roster names a model (``resnet50``) and the workload directory carries the checkpoint revision
    (``resnet50_v1_5``); the entry's own ``model`` string is therefore NOT the key under which the
    model's capture declaration is filed. Resolving the loader first and reading the directory back off
    it means one resolution serves both -- the loader and its declaration always come from the same
    workload, which a second lookup by name could not guarantee.

    ``None`` for an in-repo loader (a derived micro model writes its own): it is not a workload, so it
    declares no capture environment, and inventing one for it would be a fabricated fact.
    """
    root = (Path(m2m_dir) if m2m_dir else _m2m_dir()) / "workloads"
    try:
        rel = resolve_model_loader(entry, m2m_dir).resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return rel.parts[0] if len(rel.parts) >= 2 else None


def model_capture_env(workload: "str | None") -> dict:
    """The loader environment ``workload`` DECLARES for its own capture (``{}`` when it declares none).

    Delegated to :func:`merlin.baselines.bundle.loader_env`, which is where this repo already states
    the policy per MODEL -- host locations replayed from the workload's own ``capture.toml``, plus the
    curated full-fidelity knobs. One declaration, read by every arm that captures a model, so a capsule
    and a baseline cannot end up capturing two different networks under one name.
    """
    if not workload:
        return {}
    try:
        from merlin.baselines import bundle as _bundle
        return dict(_bundle.loader_env(workload))
    except Exception:                       # noqa: BLE001 -- an unreadable declaration is not a capture failure
        return {}


def model_capture_python(workload: "str | None") -> "Path | None":
    """The interpreter ``workload`` pins for its own capture, when it pins one that exists here."""
    if not workload:
        return None
    try:
        from merlin.baselines import bundle as _bundle
        return _bundle.capture_python(workload)
    except Exception:                       # noqa: BLE001
        return None


#: What a whole-model capsule GRADES. Stated on the capsule itself so the distinction cannot be lost
#: between the capture and whoever quotes the result.
_CORRECTNESS_ONLY_NOTE = (
    "This capsule grades COMPILER CORRECTNESS: whether the compiled program reproduces, on these exact "
    "inputs, the reference the same loader produced. It is not a statement about the model's accuracy "
    "on real data; only a capture whose inputs are real and attributed can support that.")


def input_provenance_record(workload: "str | None", applied_env: dict, meta: dict) -> dict:
    """What the capsule records about WHERE ITS INPUTS CAME FROM, from the loader's own declaration.

    A whole-model capsule can be captured on real, attributed dataset samples or on a seeded synthetic
    stream, and the compiled-vs-reference comparison is equally valid either way -- both sides come from
    the same loader on the same operands. What differs is what the PASS may then be quoted as: only real
    attributed inputs over a trained checkpoint can back an accuracy claim, and a capsule that does not
    say which it ran on invites the stronger reading of the weaker fact.

    FAIL CLOSED, in the direction that matters for each field separately:

    * ``synthetic_inputs`` is tri-state and stays ``"unknown"`` when the loader declared nothing -- an
      undeclared capture is not thereby a real-data one, and it is not thereby a synthetic one either.
    * ``accuracy_claim_supported`` is False unless the loader positively declares real inputs AND
      certifies the capture (``paper_ready``). An unknown withholds the claim and says so in
      ``accuracy_claim_withheld_because``, which is a different sentence from "the inputs were
      synthetic" so the two never read as one.
    """
    declared = dict(meta.get("loader_provenance") or {})
    status = str(meta.get("loader_provenance_status") or "unknown")
    raw = declared.get("synthetic_inputs")
    synthetic = raw if isinstance(raw, bool) else "unknown"
    ready = meta.get("loader_paper_ready")
    paper_ready = ready if isinstance(ready, bool) else "unknown"
    supported = (synthetic is False) and (paper_ready is True)
    if supported:
        why = ""
    elif status != "declared":
        why = (f"the loader declares no input provenance (status {status!r}), so whether these inputs "
               f"are real dataset samples is UNKNOWN -- an unknown may not be quoted as either")
    elif synthetic is True:
        why = ("the inputs are a seeded SYNTHETIC stream, not real dataset samples; the reference is "
               "self-consistent with them, which proves the compiler and says nothing about accuracy")
    elif synthetic == "unknown":
        why = "the loader's declaration does not state whether its inputs are real dataset samples"
    else:
        why = ("the loader did not certify this capture (paper_ready is "
               f"{paper_ready!r}); real inputs alone do not make the run an accuracy measurement")
    return {
        "workload": workload or "",
        "status": status,
        "synthetic_inputs": synthetic,
        "paper_ready": paper_ready,
        "accuracy_claim_supported": bool(supported),
        **({"accuracy_claim_withheld_because": why} if why else {}),
        "grades": "compiler_correctness",
        "note": _CORRECTNESS_ONLY_NOTE,
        # WHAT MERLIN APPLIED, next to WHAT THE LOADER SAID. Two independent records: the first says how
        # the capture was invoked, the second what the loader made of it, and a disagreement between
        # them is visible rather than averaged away.
        "loader_env": {str(k): str(v) for k, v in sorted((applied_env or {}).items())},
        "declared": declared,
        **({"declaration_error": meta["loader_provenance_error"]}
           if meta.get("loader_provenance_error") else {}),
    }


def model_accelerator_demand(linalg_mlir: str, binding) -> tuple[str | None, list[str]]:
    """What a whole-model capstone OWES the accelerator: its semantic family and instruction classes.

    Both DERIVED, and from two independent places: the family comes from the MODEL's own captured linalg
    (which ops it actually contains), the classes from THIS TARGET's role census. Neither is authored, so
    adding a target or swapping the model changes the demand without editing anything here.

    WHY IT EXISTS. A model capsule shipped ``instruction_classes: []`` with ``must_accelerate: false``,
    which together mean a submission that runs the entire network on the CPU and returns the right numbers
    PASSES the capstone. That is precisely the vacuity that was removed from the op capsules and never
    from the thing the whole suite builds toward -- and the capstone is the one result anybody quotes.

    Returns ``(family, classes)``. FAIL CLOSED at every step: a model whose ops the vocabulary cannot name,
    a target that declares no matching capability, or a taxonomy with no role census all yield
    ``(None, [])``, and the caller must then leave ``must_accelerate`` off rather than assert a demand it
    could not ground. An ungrounded demand fails a conformant submission, which is the one direction
    running the capsule cannot detect.
    """
    from merlin.targetgen import eligibility as _el
    from merlin.targetgen import semantic_families as _sf

    target = getattr(binding, "target", None)
    if not target or not linalg_mlir:
        return None, []
    try:
        cap_map = _el.capability_map_for_target(target)
    except Exception:                       # noqa: BLE001 — no contract yet: demand nothing
        return None, []
    if not cap_map:
        return None, []

    in_fmt = binding.cap_dtype(binding.operand_dtype)
    try:
        demands = model_op_demands(linalg_mlir, in_fmt)
    except Exception:                       # noqa: BLE001 — unreadable capture: demand nothing
        return None, []

    # Ops of the model this target's hardware is declared able to run. Asked of the eligibility oracle,
    # the same independent denominator ARR uses -- not of routing, which is the thing under test.
    eligible_ops: list[str] = []
    for d in demands:
        if d.op in eligible_ops:
            continue
        desc = _el.RegionDescriptor(source=d.site or d.op, op=d.op, in_dtype=d.in_fmt,
                                    weight_dtype=d.weight_fmt, m=d.m, k=d.k, n=d.n)
        if _el.is_eligible(desc, cap_map).eligible:
            eligible_ops.append(d.op)
    if not eligible_ops:
        return None, []

    # The capsule's family is the one the accelerator is actually FOR. A model is a composition and the
    # closed vocabulary has no name for it, so naming the family it owes -- rather than adding a bogus
    # ``model`` entry to the vocabulary -- is what lets the eligibility oracle resolve the capsule at all,
    # and therefore what lets must_accelerate mean anything. Contraction when present (every declared
    # matrix unit exists for it); otherwise the first eligible family, in program order.
    families = [f for f in (_sf.from_op(op) for op in eligible_ops) if f]
    if not families:
        return None, []
    family = "contraction" if "contraction" in families else families[0]

    # Classes come from the binding's OWN deriver -- the same callable every op capsule is built with, so
    # the capstone and the op capsules can never disagree about what this target's contraction sequence
    # is. (It resolves three regimes: a declared matrix unit, a self-hosted ISA taxonomy, or a RoCC
    # encoding map. Reaching past it to one of those directly is how the capstone came out empty on a
    # target whose op capsules carry the full eight-class sequence.)
    classes: list[str] = []
    for op in eligible_ops:
        try:
            got = binding.classes_for(op=op, output_dtype=in_fmt)
        except Exception:                   # noqa: BLE001 — an underivable class list demands nothing
            continue
        for c in got or ():
            if c not in classes:
                classes.append(c)
    return family, _in_issue_order(classes, binding, in_fmt)


def _in_issue_order(classes: list[str], binding, in_fmt: str) -> list[str]:
    """Put a UNION of per-op class lists back into the target's own issue order.

    The union above is built in first-encounter order, which was harmless only while every op returned
    the identical full sequence. Now that a movement op correctly owes no contraction classes, the op
    the model happens to mention first decides the order -- and a model whose first eligible op is a
    copy reported its store class BEFORE its multiply, i.e. a sequence the target cannot issue.

    The reference order is asked of the binding, not written here: a contraction is the op whose class
    list IS this target's full sequence, so its order is the canonical one, and using it keeps this
    agnostic to how the target spells or orders its classes. A class the reference does not contain is
    kept, at the end -- an unplaceable class is not evidence it is unnecessary, and dropping it would be
    the silent drop the parsing rule forbids.
    """
    try:
        reference = list(binding.classes_for(op="matmul", output_dtype=in_fmt) or ())
    except Exception:                       # noqa: BLE001 -- no reference: keep the order we have
        return classes
    rank = {c: i for i, c in enumerate(reference)}
    return sorted(classes, key=lambda c: (rank.get(c, len(rank)), classes.index(c)))



def _model_semantic_block(entry: dict, family: str | None, classes) -> dict:
    """The generalization-intent block for a whole-model capsule.

    Three things are decided here, and the profile entry's ``generalization`` block overrides each --
    the same authored-override channel ``corpus_spec._semantic_block`` already honours for every other
    capsule kind. The model path used to ignore it entirely, which caused two failures at once:

    * ``must_accelerate`` was FORCED false by the mere presence of ``lanes.require``. The reasoning is
      sound for a whole real model, whose norms have nowhere but the host to go -- but it is wrong for a
      capsule whose subject is the seam itself. The host-island capsule needs BOTH: its two int8 GEMMs
      must reach the mesh (``must_accelerate``) *and* its LayerNorm island must land on the host lane
      (``lanes.require``). Forcing them apart made declaring the lane contract silently WEAKEN the mesh
      assertion, so the automatic withholding is now a default an author can override, not a mandate.
    * ``not_asserted_reason`` could only ever say "the demand could not be derived". A capsule that
      withholds the assertion ON PURPOSE had no way to say so from the profile, so the reason was
      hand-written into the generated capsule and deleted by the next regeneration.
    """
    grounded = bool(family and classes)
    authored = entry.get("generalization")
    authored = dict(authored) if isinstance(authored, dict) else {}
    interop = bool((entry.get("lanes") or {}).get("require"))
    block: dict = {}
    if family:
        block["semantic_family"] = family
    block["generalization_axis"] = authored.get("generalization_axis", "model")
    block["must_accelerate"] = bool(authored.get("must_accelerate", grounded and not interop))
    # A WITHHELD ASSERTION ALWAYS SAYS WHY. The two reasons are different facts and must not be
    # conflated: "we could not derive the demand" is about our own reach, while "host-lane work is the
    # subject" is about the capsule. Deriving both means a synthesized interop capsule carries the
    # explanation without anyone typing it, and the profile text is an override rather than the only
    # source -- silence here reads as an author who simply never made a claim.
    reason = authored.get("not_asserted_reason")
    if not reason and not grounded:
        reason = ("the accelerator demand could not be derived from this model's linalg against this "
                  "target's declared capabilities and role census, so must_accelerate is withheld "
                  "rather than asserted ungrounded")
    if not reason and interop:
        reason = ("interop capsule: composition across lanes is the behaviour under test, so host-lane "
                  "work is expected and required. The accelerator demand is expressed as lanes.require, "
                  "which fails unless every named lane actually carried work.")
    if reason and not block["must_accelerate"]:
        block["not_asserted_reason"] = reason
    block["eligible"] = authored.get("eligible", "auto")
    return block

def _checked_lanes(entry: dict, binding) -> list[str]:
    """The lanes an interop capsule requires, REFUSED at generation time if this target cannot populate one.

    A required lane is a bar the submission must clear. A bar the target's declared compute units make
    unreachable is not a hard capability test, it is a wall: no compiler can put work on a lane the router
    has nothing to route there. Measured on the corpus this check was written for -- a capsule required
    ``in_contract_vector_scalar`` on a target whose every declared unit is a mesh kind, so the lane was
    empty by construction and the capsule was unpassable however good the backend was.

    Raising here (rather than shipping the capsule and failing it forever) keeps the failure where someone
    can fix it: at authoring time, naming the lane and what the target actually offers."""
    want = [str(x) for x in (entry.get("lanes") or {}).get("require") or []]
    target = getattr(binding, "target", None)
    if not target:
        return want                                   # no target to judge against: leave the declaration
    try:
        from merlin.targetgen.routing import reachable_lanes
        have = reachable_lanes(target)
    except Exception:                                 # noqa: BLE001 - never block generation on this
        return want
    missing = [ln for ln in want if ln not in have]
    if missing:
        raise ValueError(
            f"capsule {entry.get('name')!r} requires lane(s) {missing} that target {target!r} cannot "
            f"populate (its declared compute units make {sorted(have)} reachable). A required lane the "
            f"router can put nothing on is unpassable by construction -- declare a lane this target owns, "
            f"or give the target a compute unit that serves the one you want")
    return want


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
    # THE MODEL'S OWN CAPTURE DECLARATION. A roster model states next to its loader how it must be
    # captured -- which input stream to read, which checkpoint, which interpreter its dependencies live
    # in -- and this path used to invoke the loader with none of it. Every loader that declines to
    # invent its inputs then raised, and the corpus recorded the model as one it could not build: a
    # fact about the invocation, published as a fact about the compiler's reach.
    workload = resolve_model_workload(entry, src.m2m_dir)
    capture_env = model_capture_env(workload)
    # The capsule's declared scheme, not the dtype default. `quant_scheme` is how an entry says which
    # arithmetic it means by "int8"; dropping it here silently substituted weight-only quantization.
    art = src.capture_loader(loader, dtype, scheme=entry.get("quant_scheme"),
                             env=capture_env, python=model_capture_python(workload))
    # WHERE THE INPUTS CAME FROM -- recorded unconditionally, tri-state, and never inferred here.
    provenance = input_provenance_record(workload, capture_env, art.meta)

    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    input_abi = model_input_abi(art)
    in_names = [f"I{i}" for i in range(len(input_abi))]
    idt = binding.cap_dtype(binding.operand_dtype)
    inputs = [{"name": nm, "role": "input", "shape": input_abi[i]["shape"],
               "dtype": input_abi[i]["dtype"]}
              for i, nm in enumerate(in_names)]
    out_name = entry.get("out", "Y0")
    gate = entry.get("gate") or {"after_op_pass_fraction": 0.8}

    # self-contained weights: copy in + rewrite the linalg's absolute prov.weights_file to a relative name
    linalg = art.linalg_mlir
    wsrc = Path(art.weights_path)
    if wsrc.is_file():
        shutil.copyfile(wsrc, d / "capsule.weights.safetensors")
        linalg = linalg.replace(str(wsrc), "capsule.weights.safetensors")

    # What this model owes the accelerator, derived from its own captured linalg and this target's role
    # census. Without it the capstone is vacuous: no required classes and no must_accelerate means a
    # submission that ran the whole network on the CPU and got the right numbers PASSES the one capsule
    # the entire suite builds toward.
    _model_family, _model_classes = model_accelerator_demand(linalg, binding)

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
        "expected": {"instruction_classes": _model_classes},
        "required_oracle_tiers": list(binding.tiers), "vcs": "optional", "firesim": "optional",
        # Same per-target tier declaration the direct-MLIR builders carry (corpus_spec.build): a capsule
        # authored through a different source must not silently lose it, or the same tier that is honestly
        # N/A for the rest of the corpus fails forever on this one.
        **({"inapplicable_oracle_tiers": dict(binding.inapplicable_tiers)}
           if getattr(binding, "inapplicable_tiers", None) else {}),
        "gate": gate,
        # INTEROP: the lanes this capsule requires to have carried work (routing-plan keys, e.g.
        # `on_mesh` + `scalar_rvv_lane`). Present only when the profile declares it. A capsule that names
        # lanes is asserting COMPOSITION -- part of the model on the accelerator, part on the lane the
        # target also owns -- so host-lane work is the behaviour under test rather than a fallback
        # failure, and must_accelerate is withheld below for the same reason.
        **({"lanes": {"require": _checked_lanes(entry, binding)}}
           if (entry.get("lanes") or {}).get("require") else {}),
        "pytorch_ref": {"op": "model", "dtype": idt, "loader": "capsule.pytorch.py"},
        # WHAT THIS CAPSULE'S INPUTS WERE. On the capsule rather than only in the golden, because the
        # capsule is what a reader has in hand when they quote the result, and a pass on seeded
        # synthetic inputs proves the compiler reproduces the reference -- not that the model is
        # accurate. `accuracy_claim_supported` is False unless the loader positively says otherwise.
        "input_provenance": provenance,
        "linalg_mlir": "capsule.interface.mlir",
        # A model capsule carried no semantic block, so the eligibility oracle could not resolve it
        # (`model` is not a family and never will be -- a model is a composition). Naming the family it
        # OWES is what makes it resolvable, and therefore what gives must_accelerate something to bite on:
        # an eligible region that falls back to the CPU is a hard failure. Only asserted when the demand
        # was actually grounded; an ungrounded assertion would fail a conformant submission.
        "semantic": _model_semantic_block(entry, _model_family, _model_classes),
    }
    prov = {nm: {"shape": input_abi[i]["shape"], "decoded": _flatten(art.inputs[i])}
            for i, nm in enumerate(in_names)}
    golden = {
        "golden_source": "host_torch_eager",
        "oracle_provenance": {
            "engine": "model2MLIR whole-model linalg-on-tensors + host torch-eager",
            "model": entry.get("model", ""), "output_dtype": "f32",
            "grade_policy": {"compare": binding.compare, "atol": _tol(binding)[0], "rtol": _tol(binding)[1]},
            "interface": "linalg_positional", "arg_order": in_names + [out_name],
            "pytorch_source": "capsule.pytorch.py", "linalg_mlir": "capsule.interface.mlir",
            # The same record the capsule carries, on the oracle side too: what the reference was
            # computed over is part of where the reference came from, and a grader reading only the
            # golden must not have to infer it.
            "input_provenance": provenance,
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
    """The specir checkout: ``$SPECIR_ROOT``, else a sibling of this repo (same convention as
    :func:`_m2m_dir` -- external repo, machine-independent fallback, absent means unavailable)."""
    return _env("SPECIR_ROOT") or str(repo_root().parent / "spec")


class SpecProgramUnavailable(RuntimeError):
    """specir cannot emit a compiler-consumable program for this gen:op (only the RoCC/command-buffer
    families expose ``emit_command_buffer`` today; others provide golden + coverage only)."""


@dataclass
class SpecArtifacts:
    """A spec-derived capsule source: the spec's own PROGRAM (RoCC command buffer for a systolic RoCC gen,
    an MXU command sequence / SIMT warp schedule for the others), the deterministic operands + the golden
    the spec's refmodel computes over them, the issue/command sequence, and the ``coverage_goal`` /
    ``test_intent`` the spec declares. All from ``specir`` — INDEPENDENT of the target RTL (the spec is the
    reference). Operands are keyed by ROLE (``lhs`` / ``weight``) so they map onto the merlin_iface interface
    input names regardless of the gen's own tensor names; the golden is keyed ``out``."""
    gen: str
    op: str
    command_buffer: dict      # the spec-emitted program (cb / mxu sequence / warp schedule) — ships as grounding
    operands: dict            # role ("lhs"/"weight") -> rows of values (int for RoCC, float for fp8/simt)
    golden: dict              # {"out": rows} — the spec refmodel result (bit-exact int, or decoded float)
    instructions: list        # RoCC instructions / MXU commands in issue order
    coverage_goal: list       # [{node, kind, text, oracle, tolerance, covers}]
    opcode_backing: dict      # merlin opcode -> [authored RoCC commands backing it] (RoCC gens only)
    workload: tuple
    compare: str = "exact_int"   # "exact_int" for a RoCC int datapath, "tolerance_float" for fp8/simt


def _parse_spec_ref(spec_ref: str) -> tuple[str, str]:
    """``"<gen>:op.<name>"`` -> ``(gen, "op.<name>")``. The op token is the last colon-delimited field so a
    gen may carry no colon; everything before is the gen id (matched against the specir manifest)."""
    gen, _, op = spec_ref.rpartition(":")
    if not gen or not op:
        raise ValueError(f"spec_ref {spec_ref!r} must be '<gen>:op.<name>' (a gen id from the specir "
                         f"manifest, then a spec op symbol)")
    return gen, op


def _covers(node) -> list[str]:
    """The op/claim symbols a coverage node DECLARES it covers, from its ``refs`` dictionary.

    The linkage lives in node refs rather than in the flat attrs, which is why an earlier reader said
    it could not be followed and surfaced every one of the gen's declarations for every op. It can be
    followed: ``refs`` is a ``DictionaryAttr`` whose ``covers`` entry is an array of ``SymbolRefAttr``.
    """
    refs = getattr(node, "refs", None)
    table = getattr(refs, "data", None)
    # NOT `isinstance(table, dict)`: xDSL's DictionaryAttr holds an `immutabledict`, which is a Mapping
    # and NOT a dict subclass. The isinstance form looked like careful defensive coding and silently
    # returned "covers nothing" for every node, which disables the op-scoping this function exists to
    # do -- a guard that cannot pass is the same defect as a check that cannot fail.
    if table is None or not hasattr(table, "get"):
        return []
    covers = table.get("covers")
    out: list[str] = []
    for elem in getattr(covers, "data", ()) or ():
        sym = getattr(elem, "root_reference", None)
        name = getattr(sym, "data", None) if sym is not None else None
        if name:
            out.append(str(name))
    return out


def declared_ops(module) -> list[str]:
    """Every ``spec.op`` this gen declares, by name (``op.matmul``, ``op.isa_flush``, ...).

    The op vocabulary a cell cannot express: 13 for one target, 11 for another, all of them collapsed
    into ``contraction``/``movement`` by the cell key.
    """
    from specir.graph import all_nodes, name_of

    return [str(name_of(n)) for n in all_nodes(module)
            if (getattr(n, "name", "") or "") == "spec.op"]


def _coverage_goals(module, op: str) -> list[dict]:
    """The spec's ``spec.coverage_goal`` / ``spec.test_intent`` nodes that cover ``op``.

    ⚠️ OP-SCOPED, AND IT WAS NOT. Every declaration in the gen was returned for every op, so a matmul
    capsule carried a transcendental's test intent -- the wrong oracle and the wrong tolerance, stated
    with the same confidence as the right ones. The ``covers`` linkage is followed by :func:`_covers`.

    A node covering NOTHING is still returned: a gen-wide coverage goal with no ``covers`` list applies
    to the whole gen by construction, and dropping it would lose a real obligation. What is dropped is
    a node that names other ops and not this one.
    """
    from specir.graph import all_nodes, attrs_of, name_of
    goals: list[dict] = []
    for n in all_nodes(module):
        mnem = getattr(n, "name", "") or ""
        if mnem not in ("spec.coverage_goal", "spec.test_intent"):
            continue
        covers = _covers(n)
        if covers and op not in covers:
            continue                               # authored for other ops; not this capsule's contract
        a = attrs_of(n) or {}
        goals.append({"node": name_of(n), "kind": mnem, "covers": covers,
                      "text": a.get("text"), "oracle": a.get("oracle"),
                      "tolerance": a.get("tolerance"), "test_type": a.get("test_type")})
    return goals


def _role_of(meta: dict) -> str:
    """Normalize a program tensor's declared role to ``lhs`` / ``weight`` (the merlin_iface interface roles)."""
    return "weight" if str(meta.get("role", "")).startswith("weight") else "lhs"


def _cb_role_operands(cb: dict, operands: dict) -> dict:
    """RoCC path: map the command buffer's named operands (deterministic int rows) onto lhs/weight roles."""
    tmeta = cb.get("tensors", {})
    return {_role_of(tmeta.get(name, {})): rows for name, rows in operands.items()}


def _decode_program_tensors(program: dict) -> dict:
    """Decode a float program's OPERAND tensor BITS to values via specir's codec, keyed by lhs/weight role.
    Only the input operands ship as raw bits; the output tensor (role accumulator/out) has no bits — it IS
    the golden (already decoded) — so it is skipped here."""
    from specir.oracle.dtypes import decode_float, float_format
    out: dict = {}
    for meta in program.get("tensors", {}).values():
        role = str(meta.get("role", ""))
        if "bits" not in meta or role.startswith(("accum", "out", "result")):
            continue                                          # the output tensor, not an operand
        fmt = float_format(meta["dtype"])
        out[_role_of(meta)] = [[decode_float(b, fmt) for b in row] for row in meta["bits"]]
    return out


def _float_program_artifacts(gen: str, op: str, program: dict, cov: list, workload) -> "SpecArtifacts":
    """Normalize an fp8/simt program (atlas MXU sequence / radiance warp schedule) into SpecArtifacts:
    role-keyed decoded operands + the already-decoded golden + the program as grounding (float compare)."""
    return SpecArtifacts(
        gen=gen, op=op, command_buffer=program, operands=_decode_program_tensors(program),
        golden={"out": program["golden"]["out"]["values"]},
        instructions=program.get("commands", []) if isinstance(program.get("commands"), list) else [],
        coverage_goal=cov, opcode_backing={}, workload=tuple(workload), compare="tolerance_float")


class SpecRefSource:
    """Capture a capsule from a ``specir`` verification spec: the spec's own PROGRAM + the golden its refmodel
    computes + the declared coverage. ``specir`` imports in-process (pure xDSL). Program emitters exist for
    the RoCC/command-buffer family (systolic int, e.g. gemmini), the atlas MXU family (fp8->bf16), and the
    radiance SIMT-warp family; the emitters are tried in turn and the one that produces a program wins. A
    gen no emitter supports fails closed with :class:`SpecProgramUnavailable` (never a faked program)."""

    def __init__(self, root: str | None = None):
        self.root = root or _specir_root()

    def available(self) -> bool:
        return (Path(self.root) / "specir" / "__init__.py").exists()

    def capture(self, spec_ref: str, *, workload=(16, 16, 16), tile_dim: int = 16) -> SpecArtifacts:
        if not self.available():
            raise SpecProgramUnavailable(f"specir root {self.root} missing; set SPECIR_ROOT")
        import sys
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        from specir.gate import load_targets
        from specir.interface.emit_capsule import emit_command_buffer
        from specir.interface.rocc_lower import lower_buffer
        from specir.loading import parse_spec_file
        from specir.registry import _SPEC_ROOT

        gen, op = _parse_spec_ref(spec_ref)
        entry = {t.get("id"): t for t in load_targets(_SPEC_ROOT)}.get(gen)
        if entry is None:
            raise SpecProgramUnavailable(f"gen {gen!r} not registered in the specir manifest")
        spec_path = Path(_SPEC_ROOT) / entry["spec"]
        gen_dir = spec_path.parent
        module = parse_spec_file(spec_path)

        # ⚠️ THE OP TOKEN WAS IGNORED, AND THE REF FAILED OPEN. Measured: `gemmini:op.matmul`,
        # `gemmini:op.isa_flush` and `gemmini:op.TOTALLY_BOGUS` returned a BYTE-IDENTICAL command
        # buffer, golden and coverage goal, because the emitter takes `op` and never reads it. A
        # typo'd or renamed spec_ref therefore silently produced a matmul capsule and called it
        # whatever the ref said. Deriving from a source that fails open is worse than not deriving
        # from it, so the token is checked against the gen's own declared ops before anything is
        # emitted. (The emitter's own indifference to `op` is an upstream defect; this makes it
        # unreachable from here rather than pretending it is fixed.)
        ops = declared_ops(module)
        if op not in ops:
            raise SpecProgramUnavailable(
                f"{gen} declares no {op!r}; it declares {sorted(ops)}. The program emitter does not "
                f"read the op token, so an unchecked ref here would emit a DIFFERENT op's program "
                f"under this name")
        cov = _coverage_goals(module, op)

        # (1) RoCC command-buffer program (systolic int datapath, e.g. gemmini) — bit-exact int golden.
        from specir.interface.rocc_lower import RoccLoweringError
        cb, prov = emit_command_buffer(module, gen, op, gen_dir, workload=tuple(workload))
        if cb is not None:
            try:
                p = lower_buffer(module, cb, dim=tile_dim)
                return SpecArtifacts(gen=gen, op=op, command_buffer=cb, operands=_cb_role_operands(cb, dict(p.operands)),
                                     golden={"out": next(iter(p.golden.values()))} if len(p.golden) == 1
                                     else {"out": p.golden.get("Y0") or next(iter(p.golden.values()))},
                                     instructions=list(p.instructions), coverage_goal=cov,
                                     opcode_backing=prov.get("opcode_backed_by_spec_rocc", {}),
                                     workload=tuple(workload), compare="exact_int")
            except RoccLoweringError:
                pass          # not a RoCC gen — fall through to the fp8/simt emitters

        # (2) atlas MXU program (fp8 -> bf16); (3) radiance SIMT-warp program. Each returns None for a gen
        # it does not author — try both, use the one that produces a program, else fail closed.
        from specir.interface.emit_atlas_program import emit_atlas_program      # target-ok: specir program emitters tried additively (each returns None for a gen it doesn't author); pending emitter registry (OV11/2f)
        from specir.interface.emit_radiance_program import emit_radiance_program  # target-ok: specir program emitters tried additively; pending emitter registry (OV11/2f)
        aprog, _ = emit_atlas_program(module, op, gen_dir, workload=tuple(workload))
        if aprog is not None:
            return _float_program_artifacts(gen, op, aprog, cov, workload)
        rprog, _ = emit_radiance_program(module, op, gen_dir, workload=tuple(workload))
        if rprog is not None:
            return _float_program_artifacts(gen, op, rprog, cov, workload)
        raise SpecProgramUnavailable(
            f"no specir program emitter (RoCC command-buffer / MXU command-sequence / SIMT-warp) produced "
            f"a program for {gen}:{op} — this op/gen is golden+coverage only")


def write_spec_capsule(entry: dict, binding, out_root, *, source: "SpecRefSource | None" = None):
    """Materialize a capsule from a specir verification spec (``spec_ref: '<gen>:op.<name>'``): the agent
    compiles the merlin_iface interface for the op; the golden + the exact operands come from the spec's own
    command-buffer program (bit-exact refmodel, INDEPENDENT of the target RTL); the spec's command buffer +
    coverage_goal ship as grounding. Fails closed if specir has no program emitter for the gen."""
    import json

    import yaml
    from merlin.targetgen import corpus_spec as CS
    src = source or SpecRefSource()
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    art = src.capture(entry["spec_ref"], workload=(M, N, K), tile_dim=D)

    cap, mlir = CS.build(entry, binding)                     # merlin_iface interface for the op (names align)
    cap["spec_ref"] = entry["spec_ref"]
    out_name = entry.get("out", "Y0")
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    # map the spec's role-keyed operands onto the interface's input names (lhs -> the activation input,
    # weight -> the resident weight) so the grader feeds the exact values the golden was computed over.
    role_to_name = {"lhs": entry.get("lhs", "A0"), "weight": entry.get("weight", "W")}
    prov_inputs = {role_to_name.get(role, role): {"shape": _shape_of(rows), "decoded": rows}
                   for role, rows in art.operands.items()}
    golden = {
        "golden_source": f"specir_program_{art.gen}",
        "oracle_provenance": {
            "engine": "specir spec program + refmodel golden (INDEPENDENT of the target RTL; the spec is "
                      "the reference)",
            "spec_ref": entry["spec_ref"], "workload": list(art.workload), "compare": art.compare,
            "coverage_goal": art.coverage_goal, "opcode_backed_by_spec_rocc": art.opcode_backing,
            "program": "capsule.command_buffer.json",
            "inputs": prov_inputs},
        "outputs": {out_name: art.golden.get("out")},
    }
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    # the program ships as VISIBLE grounding — strip any embedded golden (the MXU/SIMT programs carry the
    # output values under "golden"; the answer lives ONLY in the masked golden.yaml, never in a visible file)
    program = {k: v for k, v in art.command_buffer.items() if k != "golden"}
    (d / "capsule.command_buffer.json").write_text(json.dumps(program, indent=1), encoding="utf-8")
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8")
    (d / "golden.yaml").write_text(yaml.safe_dump(golden, sort_keys=False), encoding="utf-8")
    return d
