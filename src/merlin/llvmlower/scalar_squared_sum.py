"""Explicit scalar accumulators for typed ordered tensor squared sums.

Keep every source binary32 multiply and addition in increasing reduction-index
order. Tensor SSA and upstream bufferization own aliasing and destination
lifetimes; this pass adds no physical no-alias or floating reassociation facts.
"""

from __future__ import annotations

from .host_tensor_rewrites import SQUARED_SUM_SOURCE

FEATURE = "scalar_squared_sum_accumulator"
MARKER = "__merlin_scalar_squared_sum_accumulator__"


def edit_pipeline(passes: list[str]) -> list[str]:
    hits = [i for i, stage in enumerate(passes) if "one-shot-bufferize" in stage]
    if len(hits) != 1 or MARKER in passes:
        raise ValueError("scalar squared sum requires exactly one bufferization stage")
    at = hits[0]
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                edit_pipeline=edit_pipeline,
                description="Keep structurally proved binary32 squared-sum reductions in scalar SSA; preserve source multiply/add, initialization and increasing-K order. Default off; tensor ownership remains authoritative.",
            )
        )
    return FEATURE


RUNNER_PRELUDE = (
    SQUARED_SUM_SOURCE
    + r"""_SQS_MARKER = "__merlin_scalar_squared_sum_accumulator__"
_SQS_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _SQS_MARKER not in passes:
        return _SQS_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_SQS_MARKER) != 1:
        raise ValueError("scalar squared sum marker must occur once")
    at = passes.index(_SQS_MARKER)
    _SQS_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:at]), 0, (), (), (), pre_generalize)
    print('OK scalar_squared_sum_accumulator', _scalarize_squared_sums(ctx, module))
    _SQS_ORIG_RUN_STAGES(ctx, module, ','.join(passes[at + 1:]), erase, mid, late, post_openmp, ())
"""
)


def apply_for_test(mlir_text: str) -> tuple[str, int]:
    """Execute the actual shipped structural rewrite in the upstream toolchain."""
    import subprocess
    import tempfile
    from pathlib import Path

    from .toolchain import m2m_python

    with tempfile.TemporaryDirectory(prefix="merlin_squared_sum_") as directory:
        root = Path(directory)
        source, script = root / "input.mlir", root / "run.py"
        source.write_text(mlir_text)
        script.write_text(
            "import sys\nfrom torch_mlir import ir\n"
            + RUNNER_PRELUDE.split("_SQS_MARKER =", 1)[0]
            + "ctx=ir.Context()\nmodule=ir.Module.parse(open(sys.argv[1]).read(),ctx)\n"
            + "print('COUNT',_scalarize_squared_sums(ctx,module))\nprint('MODULE_BEGIN')\nprint(module)\n"
        )
        result = subprocess.run(
            [str(m2m_python()), str(script), str(source)], capture_output=True, text=True, timeout=120
        )
        if result.returncode:
            raise RuntimeError(f"scalar squared sum rewrite failed:\n{result.stdout}\n{result.stderr}")
        count = int(next(line.split()[1] for line in result.stdout.splitlines() if line.startswith("COUNT ")))
        return result.stdout.split("MODULE_BEGIN\n", 1)[1], count
