"""GGUF frontend adapter — lift a ``.gguf`` checkpoint into the quant_ext dialect.

GGUF is the source of the INT8 (Q8_0), FP6-analogous (Q6_K), and FP4-analogous (Q4_K) weights in the
model download matrix, plus true MXFP4/NVFP4. This adapter reads a GGUF with the vendored gguf-py
``GGUFReader`` and emits the same linalg-on-tensors + ``quant_ext``-typed bundle the torch path
produces, so GGUF-quantized models flow through the identical Merlin pipeline as torchAO ones. Each
GGML quantization type maps onto a canonical :mod:`merlin.common.quant_formats` entry
(via ``QuantFormat.ggml_type``).

The reader/writer implementation lands with the P1 vertical slice; the adapter is registered now so
the frontend registry is complete and its contract is exercised.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.baselines.bundle import CaptureBundle

NAME = "gguf"


def can_handle(source: Any) -> bool:
    """True for a GGUF checkpoint path."""
    return str(source).endswith(".gguf")


@dataclass(frozen=True)
class GgufTargetReport:
    """How well a target can run a GGUF checkpoint's quantized weights."""

    target: str
    arch: str
    n_weights: int
    routable: int
    gaps: dict[str, int]          # format/ggml-type -> count of weights that gap on this target
    quant_histogram: dict[str, int]
    unsupported_types: list[str]  # ggml types with no canonical format at all

    @property
    def fully_routable(self) -> bool:
        return self.n_weights > 0 and not self.gaps


def analyze(source: Any, *, target: str) -> GgufTargetReport:
    """Read a GGUF and route its quantized weights against a target's compute_units.

    Answers 'which of this checkpoint's quantized weights can <target> actually run' without any
    graph reconstruction — a fast capability probe built on the GGUF reader + the target-agnostic
    routing tooling. Honest: a format the target does not list (e.g. a K-quant on RVV) is a gap.
    """
    from merlin.frontends import gguf_reader
    from merlin.targetgen import routing

    model = gguf_reader.read(Path(source))
    demands = gguf_reader.weight_demands(model)
    results = routing.route_target(demands, target)
    gaps: dict[str, int] = {}
    for r in results:
        if r.gap is not None:
            gaps[r.demand.in_fmt] = gaps.get(r.demand.in_fmt, 0) + 1
    return GgufTargetReport(
        target=target,
        arch=model.arch,
        n_weights=len(demands),
        routable=sum(1 for r in results if r.gap is None),
        gaps=gaps,
        quant_histogram=model.quant_histogram(),
        unsupported_types=model.unsupported(),
    )


def ingest(source: Any, *, model: str, variant: str = "gguf", seq_len: int = 8,
           out: Any = None, **_kw: Any) -> CaptureBundle:
    """Reconstruct a runnable Merlin capture bundle from a ``.gguf`` and return it.

    Drives ``m2m.frontends.gguf.capture_gguf_bundle`` in the model2MLIR venv (which has torch /
    transformers / gguf) as a subprocess — Merlin's own venv has none of those. The bundle lands under
    ``recaptures_dir()/<model>_<variant>`` and is returned as a fail-closed :class:`CaptureBundle`.
    """
    import json
    import subprocess
    from pathlib import Path

    from merlin.common.artifacts import recaptures_dir
    from merlin.llvmlower import toolchain

    # A GGUF source is always a checkpoint FILE (`can_handle` is a `.gguf` suffix test, and GGUF
    # has no repo-id form). Check it here rather than in the subprocess: transformers treats an
    # unresolvable path as a Hub repo id and goes to the network, so a simple typo surfaces as a
    # HuggingFace 404 wrapped in a CalledProcessError, minutes later and only when online.
    src = Path(str(source))
    if not src.is_file():
        raise FileNotFoundError(
            f"GGUF checkpoint not found: {src}. ingest() takes a path to a .gguf file "
            f"(GGUF has no Hub repo-id form) — check the path before re-running.")

    py = toolchain.m2m_python()
    if not Path(py).exists():
        raise RuntimeError(f"model2MLIR venv python not found at {py} (set MERLIN_M2M_DIR / venv)")
    dest = Path(out) if out is not None else recaptures_dir() / f"{model}_{variant}"
    code = (
        "import json; from m2m.frontends.gguf import capture_gguf_bundle; "
        f"print('BUNDLE_OK ' + json.dumps("
        f"capture_gguf_bundle({str(source)!r}, {str(dest)!r}, seq_len={int(seq_len)})))"
    )
    subprocess.run([str(py), "-c", code], check=True)
    return CaptureBundle(model=model, variant=variant, root=dest).require()
