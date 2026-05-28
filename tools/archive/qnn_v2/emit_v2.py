"""MLIR→QNN emitter v2 — bindings-based dispatcher.

Replaces the regex parsers in `qnn_emit.py` with a walker over the parsed
`mlir.ir.Module`. Each recognizer lives in
`tools/kernels/qnn_emit_recognizers/`; this file is the priority-ordered
dispatcher and the public entry point.

API contract (preserved from v1):

    parse_mlir(text: str, *, fp_dtype: str = "float32") -> QnnGraphDesc

The downstream pipeline (`qnn_ir.emit_qnn_cpp` → `qnn_build` →
`.qnn-ctx`) is unchanged. For inputs the v1 emitter accepted, v2 emits a
byte-identical `.qnn.cpp` (parity gate at
`tools/kernels/tests/test_qnn_emit_v2_parity.py`).

Fallback: set `MERLIN_QNN_EMIT_REGEX=1` to route `parse_mlir`/`emit`
through the legacy regex implementation. Useful while migrating
recognizers — once all v1 recognizers have v2 equivalents covered by the
parity test, the env knob and v1 file go away.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qnn_ir import QnnGraphDesc, emit_qnn_cpp  # noqa: E402

_LOG = logging.getLogger("qnn_emit_v2")


def _use_regex_fallback() -> bool:
    return os.environ.get("MERLIN_QNN_EMIT_REGEX", "") == "1"


def parse_mlir(text: str, *, fp_dtype: str = "float32") -> QnnGraphDesc:
    """Parse `text`, walk via `iree.compiler.ir`, dispatch to recognizers.

    Falls back to the legacy regex emitter when `MERLIN_QNN_EMIT_REGEX=1`.
    """
    if _use_regex_fallback():
        from qnn_emit import parse_mlir as v1_parse_mlir

        return v1_parse_mlir(text, fp_dtype=fp_dtype)

    # Late import so test environments without the bindings can still
    # import qnn_emit_v2 (e.g. for type checking) without crashing.
    from iree.compiler import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text, ctx)

    from qnn_emit_recognizers import REGISTRY

    tried: list[str] = []
    for recognizer in REGISTRY:
        name = getattr(recognizer, "NAME", recognizer.__name__)
        tried.append(name)
        result = recognizer.try_recognize(module, fp_dtype=fp_dtype)
        if result is not None:
            _LOG.debug("recognizer matched: %s", name)
            return result

    raise ValueError(
        "no v2 recognizer matched the MLIR module. Tried: "
        + ", ".join(tried)
        + ". Set MERLIN_QNN_EMIT_REGEX=1 to fall back to the legacy "
        "regex emitter while the v2 path is being expanded."
    )


def emit(
    mlir_path: pathlib.Path,
    output_cpp: pathlib.Path,
    *,
    fp_dtype: str = "float32",
) -> QnnGraphDesc:
    """Parse `mlir_path` via v2, lower to `QnnGraphDesc`, write `.qnn.cpp`.

    Returns the descriptor for downstream callers that need shape /
    dtype metadata.
    """
    text = mlir_path.read_text()
    graph = parse_mlir(text, fp_dtype=fp_dtype)
    cpp = emit_qnn_cpp(graph)
    output_cpp.parent.mkdir(parents=True, exist_ok=True)
    output_cpp.write_text(cpp)
    _LOG.info(
        "emitted %s (%d bytes; graph=%s, %d tensors, %d nodes)",
        output_cpp,
        output_cpp.stat().st_size,
        graph.name,
        len(graph.tensors),
        len(graph.nodes),
    )
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--name",
        required=True,
        help="Override the graph name (defaults to func.func name)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to write the emitted .qnn.cpp",
    )
    parser.add_argument(
        "--fp-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="On-device floating-point dtype for fp recognizers.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    graph = emit(args.mlir, args.output, fp_dtype=args.fp_dtype)
    if graph.name != args.name:
        _LOG.info(
            "note: --name=%s differs from func name %s; using func name",
            args.name,
            graph.name,
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
