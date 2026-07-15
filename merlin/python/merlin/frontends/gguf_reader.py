"""Read a GGUF checkpoint: architecture metadata + per-tensor quantization.

GGUF stores quantized weights plus architecture metadata (but no compute graph). This module is the
arch-independent foundation of GGUF ingestion: it opens a ``.gguf`` with the vendored gguf-py
``GGUFReader``, normalizes the architecture metadata a decoder needs, and classifies every tensor
against the target-agnostic :mod:`merlin.common.quant_formats` registry — with a lazy dequantization
to fp32 (via ``gguf.quants.dequantize``) that serves as the correctness reference. The graph is
reconstructed elsewhere (the model2MLIR GGUF frontend, from this metadata); the ggml-type ->
quant_ext mapping is driven purely off ``tensor.tensor_type`` and is fully architecture-independent.

gguf-py ships under ``third_party/baselines/llama.cpp/gguf-py``; :func:`_gguf` adds it to ``sys.path``
lazily so importing this module never hard-fails when the vendored tree is absent.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from merlin.common import quant_formats as qf
from merlin.common.paths import repo_root

# ggml scalar (non-block) types map straight onto the regular float/int formats.
_SCALAR_GGML_TO_FORMAT = {"F32": "fp32", "F16": "fp16", "BF16": "bf16"}

# The architecture metadata keys a Llama/Qwen2/Gemma2 decoder needs, as `<arch>.<suffix>` (plus the
# gemma2-only softcapping/window fields). Values pulled structurally via ReaderField.contents().
_ARCH_KEYS = {
    "block_count": "block_count",
    "embedding_length": "embedding_length",
    "feed_forward_length": "feed_forward_length",
    "context_length": "context_length",
    "head_count": "attention.head_count",
    "head_count_kv": "attention.head_count_kv",
    "key_length": "attention.key_length",
    "value_length": "attention.value_length",
    "rms_eps": "attention.layer_norm_rms_epsilon",
    "rope_dim": "rope.dimension_count",
    "rope_freq_base": "rope.freq_base",
    "attn_logit_softcapping": "attn_logit_softcapping",
    "final_logit_softcapping": "final_logit_softcapping",
    "attn_scale": "attention.scale",
    "sliding_window": "attention.sliding_window",
}


@lru_cache(maxsize=1)
def _gguf():
    """Import the vendored gguf-py package (added to sys.path on first use)."""
    base = repo_root() / "third_party" / "baselines" / "llama.cpp" / "gguf-py"
    if base.is_dir() and str(base) not in sys.path:
        sys.path.insert(0, str(base))
    import gguf  # noqa: PLC0415

    return gguf


@dataclass(frozen=True)
class GgufTensor:
    """One tensor in a GGUF file, classified against the quant-format registry."""

    name: str
    shape: tuple[int, ...]       # GGUF storage order (ne[0] fastest); orient at graph-build time
    ggml_type: str               # e.g. "Q6_K", "F32"
    fmt: qf.QuantFormat | None   # canonical format, or None if unsupported
    n_elements: int
    _reader_tensor: Any = None

    @property
    def is_quantized(self) -> bool:
        return self.ggml_type not in _SCALAR_GGML_TO_FORMAT

    def dequantize(self):
        """Dequantize to an fp32 numpy array (the correctness reference). Requires numpy + gguf."""
        gguf = _gguf()
        rt = self._reader_tensor
        arr = gguf.quants.dequantize(rt.data, rt.tensor_type)
        return arr.reshape(tuple(int(d) for d in reversed(rt.shape)))


@dataclass(frozen=True)
class GgufModel:
    path: Path
    arch: str
    metadata: dict[str, Any]
    tensors: tuple[GgufTensor, ...]

    def tensor(self, name: str) -> GgufTensor | None:
        return next((t for t in self.tensors if t.name == name), None)

    def quant_histogram(self) -> dict[str, int]:
        """ggml type name -> tensor count (what formats this checkpoint actually uses)."""
        hist: dict[str, int] = {}
        for t in self.tensors:
            hist[t.ggml_type] = hist.get(t.ggml_type, 0) + 1
        return hist

    def unsupported(self) -> list[str]:
        """ggml types present that have no canonical quant_formats mapping (honest gap list)."""
        return sorted({t.ggml_type for t in self.tensors if t.fmt is None})


def _format_for_ggml(ggml_name: str) -> qf.QuantFormat | None:
    scalar = _SCALAR_GGML_TO_FORMAT.get(ggml_name)
    if scalar is not None:
        return qf.get(scalar)
    return qf.from_ggml(ggml_name)


def _field_value(reader: Any, key: str):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def arch_metadata(reader: Any, arch: str) -> dict[str, Any]:
    """Normalized architecture metadata: our field name -> value (missing keys omitted)."""
    out: dict[str, Any] = {}
    for name, suffix in _ARCH_KEYS.items():
        val = _field_value(reader, f"{arch}.{suffix}")
        if val is not None:
            out[name] = val
    return out


def weight_demands(model: "GgufModel"):
    """One routing OpDemand per quantized weight tensor (op=matmul, in=weight=its format).

    Norm/embedding vectors and unquantized (scalar-float) tensors are skipped — the demands describe
    the low-precision matmul weights whose format a target must support. Tensors whose ggml type has
    no canonical format are reported as demands with their raw ggml type so routing gaps them honestly.
    """
    from merlin.targetgen.routing import OpDemand

    demands = []
    for t in model.tensors:
        if not t.is_quantized:
            continue
        fmt_name = t.fmt.name if t.fmt is not None else t.ggml_type
        demands.append(OpDemand(op="matmul", in_fmt=fmt_name, weight_fmt=fmt_name, site=t.name))
    return demands


def read(path: str | Path) -> GgufModel:
    """Open a GGUF checkpoint and return its arch metadata + classified tensors."""
    gguf = _gguf()
    reader = gguf.GGUFReader(str(path), "r")
    arch = _field_value(reader, "general.architecture") or ""
    tensors = tuple(
        GgufTensor(
            name=t.name,
            shape=tuple(int(d) for d in t.shape),
            ggml_type=t.tensor_type.name,
            fmt=_format_for_ggml(t.tensor_type.name),
            n_elements=int(t.n_elements),
            _reader_tensor=t,
        )
        for t in reader.tensors
    )
    return GgufModel(path=Path(path), arch=str(arch), metadata=arch_metadata(reader, str(arch)), tensors=tensors)
