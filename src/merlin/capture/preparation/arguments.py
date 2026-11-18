"""Command-line options and preparation-mode validation for capture workers."""

from __future__ import annotations

import argparse
import math
import os


def parse_arguments(argv=None) -> argparse.Namespace:
    """Read the existing worker options before importing framework packages."""
    ap = argparse.ArgumentParser(description="m2m capsule capture worker (runs in the m2m venv).")
    ap.add_argument("--loader", required=True, help="path to a .py exposing get_model_and_inputs()")
    ap.add_argument("--dtype", required=True, help="canonical dtype token (fp32/bf16/fp16/int8/fp8)")
    ap.add_argument("--out", required=True, help="output directory for the artifacts")
    ap.add_argument(
        "--m2m-dir",
        default=os.environ.get("MERLIN_M2M_DIR"),
        help="model2MLIR repo root (for sys.path if m2m isn't installed)",
    )
    ap.add_argument("--func-name", default="forward")
    ap.add_argument(
        "--scheme",
        default="",
        help="torchAO scheme name, overriding the dtype default (e.g. "
        "int8_dyn_act_int8_weight for a true W8A8 integer contraction)",
    )
    ap.add_argument(
        "--recipe",
        default="",
        help="path to a quant_recipe_v1 JSON derived from the target; when given, the "
        "model is quantized under it and --scheme / the dtype default are not used",
    )
    ap.add_argument(
        "--already-quantized",
        action="store_true",
        help="capture the loader's own materialized numeric graph without applying a TorchAO recipe or scheme",
    )
    ap.add_argument(
        "--native-integer-source",
        action="store_true",
        help="versioned unchanged native integer source; retain original frontend trace without quantization",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="Python/NumPy/Torch RNG seed applied before loader import and construction"
    )
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="intra-op CPU threads; reference bytes depend on it, so it is recorded and replayed (default 1)",
    )
    ap.add_argument("--agreement-atol", type=float, default=1e-3)
    ap.add_argument("--agreement-rtol", type=float, default=1e-3)
    ap.add_argument(
        "--stage-fp32",
        action="store_true",
        help="explicitly stage the captured floating frontend to FP32 with audited dtype operands",
    )
    ap.add_argument(
        "--materialize-bundle",
        action="store_true",
        help="also emit the full model2MLIR runtime bundle from this exact conversion and model instance",
    )
    ap.add_argument(
        "--quantize-activation-contractions",
        action="store_true",
        help="also quantize the contractions the scheme cannot reach (activation x activation "
        "matmuls, non-overlapping convolutions) in the scheme's own int8 dynamic form; see "
        "_activation_contractions.py",
    )
    ap.add_argument(
        "--integer-nonlinear",
        action="store_true",
        help="with --quantize-activation-contractions: also compute softmax, GELU and layer norm in "
        "integer arithmetic (I-BERT); a different numerical model, see _integer_nonlinear.py",
    )
    ap.add_argument(
        "--diagnostic-model-copy",
        action="store_true",
        help="copy raw converted MLIR for inventory with byte hashes, without a capture receipt or Phase 0 admission",
    )
    ap.add_argument(
        "--observe-buffer-state",
        action="store_true",
        help="observe actual prepared export effects and source storage for explicit later freeze consent",
    )
    a = argv if isinstance(argv, argparse.Namespace) else ap.parse_args(argv)
    native_integer_source = getattr(a, "native_integer_source", False)
    if native_integer_source and (
        not a.already_quantized
        or a.dtype != "int8"
        or a.recipe
        or a.scheme
        or a.stage_fp32
        or a.quantize_activation_contractions
        or a.integer_nonlinear
    ):
        ap.error("--native-integer-source requires unchanged int8 source without additional transformations")
    if a.materialize_bundle and a.diagnostic_model_copy:
        ap.error("--materialize-bundle and --diagnostic-model-copy are mutually exclusive")
    if not 0 <= a.seed < 2**32:
        ap.error("--seed must be an unsigned 32-bit integer")
    if not 1 <= a.cpu_threads <= 256:
        ap.error("--cpu-threads must be between 1 and 256")
    if not all(math.isfinite(value) and value >= 0 for value in (a.agreement_atol, a.agreement_rtol)):
        ap.error("agreement tolerances must be finite and nonnegative")
    if a.already_quantized and (a.recipe or a.scheme):
        ap.error("--already-quantized cannot be combined with --recipe or --scheme")
    if a.integer_nonlinear and not a.quantize_activation_contractions:
        # Defined only on top of the int8 contractions (the softmax's int8 numerators are the next
        # contraction's own operand), so it is refused without them rather than mixed into a float program.
        ap.error("--integer-nonlinear is defined on top of --quantize-activation-contractions only")
    if a.stage_fp32 and (
        (a.dtype not in {"fp32", "f32"} and not (a.dtype == "int8" and a.recipe))
        or a.scheme
        or a.already_quantized
        or a.quantize_activation_contractions
        or a.integer_nonlinear
    ):
        ap.error("--stage-fp32 requires FP32 or an explicit int8 recipe without additional rewrites")

    return a
